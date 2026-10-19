"""Quadrotor and ground-vehicle models for the simulator.

The quadrotor accepts world-frame velocity setpoints (what a PX4/ArduPilot offboard
interface accepts). Its inner loops turn a desired acceleration into a tilt, and the tilt
follows with a first-order lag. That keeps the coupling that matters in closed-loop flight:
to accelerate sideways the drone has to tilt, and the downward camera tilts with it.
"""
from dataclasses import dataclass, field

import numpy as np

from .geometry import euler_to_R, rot_z, wrap_angle

G = 9.81


@dataclass
class QuadParams:
    tau_vel: float = 0.45        # velocity-loop time constant [s]
    ki_vel: float = 2.0          # velocity-loop integral gain (wind rejection, as in PX4)
    tau_att: float = 0.12        # attitude response time constant [s]
    a_max_xy: float = 5.0        # [m/s^2]  (~27 deg tilt)
    a_max_z: float = 4.0
    tilt_max: float = np.radians(30.0)
    drag: float = 0.12           # linear drag coefficient [1/s]
    leg_length: float = 0.10     # CoG height above skids [m]


@dataclass
class QuadState:
    p: np.ndarray = field(default_factory=lambda: np.zeros(3))
    v: np.ndarray = field(default_factory=lambda: np.zeros(3))
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    motors_on: bool = True

    @property
    def R(self):
        return euler_to_R(self.roll, self.pitch, self.yaw)


class Quadrotor:
    def __init__(self, params: QuadParams = None, p0=(0.0, 0.0, 0.0)):
        self.prm = params or QuadParams()
        self.s = QuadState(p=np.array(p0, float))
        self.a = np.zeros(3)
        self.vi = np.zeros(3)

    def step(self, dt, v_cmd, yaw_rate_cmd=0.0, wind_acc=np.zeros(3)):
        s, prm = self.s, self.prm
        if s.motors_on:
            ev = np.asarray(v_cmd, float) - s.v
            self.vi = np.clip(self.vi + ev * dt, -1.5, 1.5)
            a_des = ev / prm.tau_vel + prm.ki_vel * self.vi
            a_xy = a_des[:2]
            n = np.linalg.norm(a_xy)
            if n > prm.a_max_xy:
                a_xy = a_xy * prm.a_max_xy / n
            a_z = float(np.clip(a_des[2], -prm.a_max_z, prm.a_max_z))
            # desired tilt in the heading frame (small-angle-exact for a level thrust model)
            a_h = rot_z(-s.yaw)[:2, :2] @ a_xy
            pitch_des = np.clip(np.arctan2(a_h[0], G + a_z), -prm.tilt_max, prm.tilt_max)
            roll_des = np.clip(-np.arctan2(a_h[1], G + a_z), -prm.tilt_max, prm.tilt_max)
            k = min(1.0, dt / prm.tau_att)
            s.roll += k * (roll_des - s.roll)
            s.pitch += k * (pitch_des - s.pitch)
            s.yaw = wrap_angle(s.yaw + yaw_rate_cmd * dt)
            R = s.R
            thrust = (G + a_z) / max(0.5, R[2, 2])
            acc = R[:, 2] * thrust - np.array([0.0, 0.0, G]) - prm.drag * s.v + wind_acc
        else:
            # motors cut: ballistic (paper Sec. III-D "turn off the rotors and go in a projectile")
            acc = np.array([0.0, 0.0, -G]) - 0.3 * prm.drag * s.v + 0.3 * wind_acc
        s.v = s.v + acc * dt
        s.p = s.p + s.v * dt
        self.a = acc
        return s


class WindModel:
    """Ornstein-Uhlenbeck gusts expressed as an acceleration disturbance."""

    def __init__(self, rng, sigma=0.6, tau=2.0, mean=(0.0, 0.0, 0.0)):
        self.rng, self.sigma, self.tau = rng, sigma, tau
        self.mean = np.array(mean, float)
        self.w = np.zeros(3)

    def step(self, dt):
        if self.sigma <= 0:
            return self.mean.copy()
        k = dt / self.tau
        self.w += -k * self.w + self.sigma * np.sqrt(2 * k) * self.rng.normal(size=3)
        self.w[2] *= 0.3
        return self.mean + self.w


# --------------------------------------------------------------------------------------
# Ground vehicle carrying the pad
# --------------------------------------------------------------------------------------

def _smooth_step(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


SCENARIOS = {
    # name: (speed(t) [m/s], yaw_rate(t) [rad/s], description)
    'static': (lambda t: 0.0, lambda t: 0.0, 'Stationary pad (baseline)'),
    'straight': (lambda t: 2.0, lambda t: 0.0, 'Constant 2 m/s, straight line'),
    'accelerating': (lambda t: 0.5 + 3.0 * _smooth_step((t - 5.0) / 20.0), lambda t: 0.0,
                     '0.5 -> 3.5 m/s ramp over 20 s'),
    'circle': (lambda t: 2.5, lambda t: 2.5 / 15.0, '2.5 m/s on a 15 m radius circle'),
    'weave': (lambda t: 2.5, lambda t: 0.35 * np.sin(2 * np.pi * t / 8.0),
              '2.5 m/s lane-change weave, 8 s period'),
    'stop_go': (lambda t: 3.0 * _smooth_step(np.sin(2 * np.pi * t / 16.0) * 2.0), lambda t: 0.0,
                'Stop-and-go traffic, 0 <-> 3 m/s'),
    'fast': (lambda t: 4.5, lambda t: 0.0, 'Constant 4.5 m/s'),
}


class GroundVehicle:
    def __init__(self, scenario='straight', deck_height=0.35, p0=(0.0, 0.0), yaw0=0.0):
        self.speed_fn, self.yaw_rate_fn, self.description = SCENARIOS[scenario]
        self.scenario = scenario
        self.deck_height = deck_height
        self.p = np.array([p0[0], p0[1], deck_height], float)
        self.yaw = yaw0
        self.t = 0.0
        self.v = np.zeros(3)

    def step(self, dt):
        sp = float(self.speed_fn(self.t))
        self.yaw = wrap_angle(self.yaw + float(self.yaw_rate_fn(self.t)) * dt)
        self.v = np.array([sp * np.cos(self.yaw), sp * np.sin(self.yaw), 0.0])
        self.p = self.p + self.v * dt
        self.t += dt

    @property
    def R_deck(self):
        # pad +y (texture "up") points along the direction of travel
        return rot_z(self.yaw - np.pi / 2.0)
