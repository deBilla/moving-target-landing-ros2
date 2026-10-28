"""Simulated world: quadrotor, ground vehicle with pad, wind, camera and contact.

Shared by the ROS `sim_world` node and the batch runner, so both run the same physics.
"""
from dataclasses import dataclass, field

import numpy as np

from .camera import PinholeCamera, SceneRenderer
from .dynamics import SCENARIOS, GroundVehicle, QuadParams, Quadrotor, Scenario, WindModel
from .geometry import P_BODY_CAM, euler_to_R
from .pad import DEFAULT_PAD, PadLayout


@dataclass
class WorldConfig:
    scenario: str = 'straight'   # name in SCENARIOS, used unless `motion` is given
    motion: Scenario | None = None  # custom vehicle motion; overrides `scenario`
    seed: int = 0
    drone_start: tuple = (-2.0, 1.5, 6.0)
    vehicle_start: tuple = (0.0, 0.0)
    wind_sigma: float = 0.6
    wind_mean: tuple = (0.0, 0.0, 0.0)
    cam_width: int = 800
    cam_height: int = 600
    cam_hfov_deg: float = 70.0
    # flight-controller state estimate noise (what the drone knows about itself)
    att_noise_deg: float = 0.4   # slowly varying attitude bias (IMU/estimator), 1-sigma
    att_jitter_deg: float = 0.05 # white attitude noise per sample
    vel_noise: float = 0.05
    pos_drift: float = 0.02      # m/sqrt(s) random walk of own-position estimate
    quad: QuadParams = field(default_factory=QuadParams)
    deck_height: float = 0.35    # deck top above ground [m]
    deck_size: tuple = (1.6, 1.3)  # deck footprint across x, along y of the pad frame [m]
    pad: PadLayout = DEFAULT_PAD

    def vehicle_motion(self) -> Scenario:
        return self.motion if self.motion is not None else SCENARIOS[self.scenario]


class World:
    def __init__(self, cfg: WorldConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.t = 0.0
        self.quad = Quadrotor(cfg.quad, p0=cfg.drone_start)
        self.vehicle = GroundVehicle(cfg.vehicle_motion(), deck_height=cfg.deck_height,
                                     p0=cfg.vehicle_start)
        self.wind = WindModel(self.rng, sigma=cfg.wind_sigma, mean=cfg.wind_mean)
        self.camera = PinholeCamera(cfg.cam_width, cfg.cam_height, cfg.cam_hfov_deg)
        self.renderer = SceneRenderer(self.camera, deck_size=cfg.deck_size, seed=cfg.seed,
                                      gain=float(self.rng.uniform(0.85, 1.1)), layout=cfg.pad)
        self.pos_bias = np.zeros(3)
        self.att_bias = np.zeros(3)
        self.outcome = None          # dict once the episode ends
        self.attached_offset = None  # drone offset in pad frame after landing

    # ---------------------------------------------------------------------------------
    def step(self, dt, v_cmd, motors_on=True):
        if self.outcome is not None and self.attached_offset is not None:
            self.vehicle.step(dt)
            self.t += dt
            R = self.vehicle.R_deck
            self.quad.s.p = self.vehicle.p + R @ self.attached_offset
            self.quad.s.v = self.vehicle.v.copy()
            return
        self.quad.s.motors_on = bool(motors_on) and self.quad.s.motors_on
        w = self.wind.step(dt)
        self.vehicle.step(dt)
        self.quad.step(dt, v_cmd, 0.0, w)
        self.t += dt
        self.pos_bias += self.rng.normal(0, self.cfg.pos_drift * np.sqrt(dt), 3) * [1, 1, 0.3]
        k = dt / 5.0
        self.att_bias += -k * self.att_bias + np.radians(self.cfg.att_noise_deg) * np.sqrt(2 * k) * self.rng.normal(size=3)
        self._contact()

    def end(self, result):
        """End the episode without contact (e.g. 'lost_target', 'timeout')."""
        if self.outcome is None:
            self.outcome = dict(result=result, t=self.t, err=np.nan, vz=np.nan, vxy=np.nan)

    def _contact(self):
        if self.outcome is not None:
            return
        s, prm = self.quad.s, self.quad.prm
        skid_z = s.p[2] - prm.leg_length
        off_pad = self.vehicle.R_deck.T @ (s.p - self.vehicle.p)
        deck_w, deck_l = self.cfg.deck_size
        on_deck = abs(off_pad[0]) < deck_w / 2.0 and abs(off_pad[1]) < deck_l / 2.0
        if on_deck and skid_z <= self.vehicle.p[2]:
            v_rel = s.v - self.vehicle.v
            on_pad = max(abs(off_pad[0]), abs(off_pad[1])) <= self.cfg.pad.size / 2.0
            err = float(np.hypot(off_pad[0], off_pad[1]))
            vz = float(-v_rel[2])
            vxy = float(np.hypot(v_rel[0], v_rel[1]))
            ok = on_pad and err <= 0.35 and vz <= 2.2 and vxy <= 0.6
            self.outcome = dict(result='landed' if ok else ('hard_landing' if on_pad else 'deck_edge'),
                                t=self.t, err=err, vz=vz, vxy=vxy,
                                offset=off_pad[:2].tolist(), motors_on=s.motors_on)
            self.attached_offset = np.array([off_pad[0], off_pad[1], prm.leg_length])
        elif skid_z <= 0.0:
            self.outcome = dict(result='crash_ground', t=self.t, err=float(
                np.hypot(*(s.p[:2] - self.vehicle.p[:2]))), vz=float(-s.v[2]), vxy=np.nan,
                motors_on=s.motors_on)

    # ---------------------------------------------------------------------------------
    def render(self):
        s = self.quad.s
        return self.renderer.render(s.R, s.p, self.vehicle.R_deck, self.vehicle.p)

    def fc_estimate(self):
        """Drone's own state as its flight controller would report it (IMU/GPS/flow)."""
        s = self.quad.s
        n = np.radians(self.cfg.att_jitter_deg)
        b = self.att_bias
        R = euler_to_R(s.roll + b[0] + self.rng.normal(0, n), s.pitch + b[1] + self.rng.normal(0, n),
                       s.yaw + b[2] + self.rng.normal(0, n))
        v = s.v + self.rng.normal(0, self.cfg.vel_noise, 3)
        return s.p + self.pos_bias, v, R

    def truth(self):
        s, veh = self.quad.s, self.vehicle
        rel = veh.p - s.p
        rel_v = veh.v - s.v
        return dict(t=self.t, drone_p=s.p.copy(), drone_v=s.v.copy(), roll=s.roll,
                    pitch=s.pitch, yaw=s.yaw, tgt_p=veh.p.copy(), tgt_v=veh.v.copy(),
                    tgt_yaw=veh.yaw, rel_p=rel, rel_v=rel_v,
                    rel_speed=float(np.hypot(rel_v[0], rel_v[1])),
                    tgt_speed=float(np.hypot(veh.v[0], veh.v[1])),
                    tgt_heading=float(np.arctan2(veh.v[1], veh.v[0])) if np.hypot(
                        veh.v[0], veh.v[1]) > 0.05 else np.nan,
                    cam_height=float(s.p[2] + P_BODY_CAM[2] - veh.p[2]),
                    motors_on=s.motors_on)
