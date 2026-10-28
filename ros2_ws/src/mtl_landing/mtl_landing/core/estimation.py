"""Relative-state estimation of the moving pad (paper Sec. III-C).

Four estimators share one interface (Estimator) so they can be compared on identical
detections:

PaperEstimator         - the Sec. III-C formulas coded literally as printed:
                           d     = sqrt((tx1^2 + ty1^2) - (tx2^2 + ty2^2))
                           v     = d / duration
                           angle = ty_i / tx_i
                           alt   = tz
PlainDistanceEstimator - Euclidean displacement and atan2, still in the camera frame and
                         differencing raw frames.
CompensatedEstimator   - tilt/frame compensation, raw finite differences.
CorrectedEstimator     - rotate tvec into a gravity-aligned world frame with the drone's
                         attitude at the image timestamp, add the drone's own position, and
                         run a constant-velocity Kalman filter on the target's world
                         position. Relative velocity = v_target - v_drone.

Notes on the step from measuring speed to closed-loop flight (see README):
  (1) Coded literally, sqrt(r1^2 - r2^2) uses the difference of squared ranges from the
      camera axis, so it is undefined when the target moves away (r2 > r1). The 2019
      Table I results imply the original code used a plain distance; this variant keeps
      the literal reading so its effect can be measured.
  (2) ty/tx gives the target's bearing in the image; the direction of motion comes from
      the velocity, via atan2.
  (3) tvec is in the camera frame, and the camera tilts with the drone (it must tilt to
      accelerate). Rotating by the drone's attitude keeps position and altitude correct
      in flight.
  (4) rvec from solvePnP is a Rodrigues axis-angle vector; cv2.Rodrigues turns it into the
      rotation used for (3).
"""
from abc import ABC, abstractmethod
from collections import deque

import cv2
import numpy as np

from .geometry import P_BODY_CAM, R_BODY_CAM, R_to_euler
from .state import DroneState, Estimate


class Estimator(ABC):
    """One way of turning pad detections into an Estimate for the controller.

    update()  - feed a detection taken at time t (drone = flight-controller state at t).
    predict() - the estimate to fly on at time t, or None when there is nothing usable.
    Each estimator decides itself when its estimate is too old to use.
    """
    name = ''

    @abstractmethod
    def update(self, t, det, drone: DroneState = None) -> Estimate | None:
        ...

    @abstractmethod
    def predict(self, t, drone: DroneState = None) -> Estimate | None:
        ...


class LastFixEstimator(Estimator):
    """Reports only at camera fixes and holds the last value, for at most max_hold s."""
    max_hold = 0.5

    def __init__(self):
        self.prev = None
        self.last = None

    def predict(self, t, drone=None):
        if self.last is None or t - self.last.t > self.max_hold:
            return None
        return self.last


class PaperEstimator(LastFixEstimator):
    name = 'paper'

    def __init__(self):
        super().__init__()
        self.n_updates = 0
        self.n_undefined = 0

    def update(self, t, det, drone=None):
        tx, ty, tz = det.tvec
        # Paper has no camera->body mapping; the minimal one for a down-looking camera
        # (camera x right = -body y, camera y down = -body x), drone assumed level.
        out = Estimate(t=t, rel_pos=np.array([-ty, -tx, -tz]), altitude=tz)
        if self.prev is not None:
            t1, (tx1, ty1, _) = self.prev
            tx2, ty2 = tx, ty
            duration = t - t1
            d2 = (tx1 ** 2 + ty1 ** 2) - (tx2 ** 2 + ty2 ** 2)
            self.n_updates += 1
            if d2 < 0:
                self.n_undefined += 1
                rel_dist = np.nan
            else:
                rel_dist = np.sqrt(d2)
            out.speed = rel_dist / duration if duration > 0 else np.nan
            ratio = ty / tx if abs(tx) > 1e-9 else np.nan
            out.angle_ratio = ratio
            out.heading = np.arctan(ratio)   # most charitable reading of "angle = ty/tx"
            if np.isfinite(out.speed) and np.isfinite(out.heading):
                # direction (cos a, sin a) in camera x/y, mapped to body x/y
                ca, sa = np.cos(out.heading), np.sin(out.heading)
                out.rel_vel = out.speed * np.array([-sa, -ca, 0.0])
                out.heading = float(np.arctan2(-ca, -sa))
        self.prev = (t, det.tvec.copy())
        self.last = out
        return out


class PlainDistanceEstimator(LastFixEstimator):
    """Plain distance and atan2, in the camera frame with raw finite differences."""
    name = 'naive'

    def update(self, t, det, drone=None):
        tx, ty, tz = det.tvec
        rel_body = np.array([-ty, -tx, -tz])
        out = Estimate(t=t, rel_pos=rel_body, altitude=tz)
        if self.prev is not None:
            t1, p1 = self.prev
            dt = t - t1
            if dt > 0:
                v = (rel_body - p1) / dt
                out.rel_vel = v
                out.speed = float(np.hypot(v[0], v[1]))
                out.heading = float(np.arctan2(v[1], v[0]))
        self.prev = (t, rel_body)
        self.last = out
        return out


class CompensatedEstimator(LastFixEstimator):
    """Ablation: tilt/frame compensation (as CorrectedEstimator) but raw finite differences."""
    name = 'compensated'

    def update(self, t, det, drone=None):
        if drone is None:
            return None
        z, _ = CorrectedEstimator.measurement(det, drone)
        rel = z - drone.p
        out = Estimate(t=t, rel_pos=rel, altitude=float(-rel[2] + P_BODY_CAM[2]))
        if self.prev is not None and t > self.prev[0]:
            v_t = (z - self.prev[1]) / (t - self.prev[0])
            out.tgt_vel = v_t
            out.rel_vel = v_t - drone.v
            out.speed = float(np.hypot(*out.rel_vel[:2]))
            out.heading = float(np.arctan2(v_t[1], v_t[0]))
        self.prev = (t, z)
        self.last = out
        return out


class DroneStateBuffer:
    """Short history of the flight controller's state, for lookup at image timestamps."""

    def __init__(self, maxlen=400):
        self.buf = deque(maxlen=maxlen)

    def add(self, t, p, v, R):
        self.buf.append(DroneState(t, np.asarray(p, float), np.asarray(v, float),
                                   np.asarray(R, float)))

    def at(self, t) -> DroneState | None:
        if not self.buf:
            return None
        return min(self.buf, key=lambda e: abs(e.t - t))

    def latest(self) -> DroneState | None:
        return self.buf[-1] if self.buf else None


class CorrectedEstimator(Estimator):
    name = 'corrected'

    def __init__(self, accel_noise=2.0, meas_base=0.02, meas_per_m=0.012, gate_chi2=16.0,
                 timeout=1.0):
        self.q = accel_noise
        self.meas_base, self.meas_per_m = meas_base, meas_per_m
        self.gate = gate_chi2
        self.timeout = timeout
        self.x = None              # [px py pz vx vy vz] of the pad in world frame
        self.P = None
        self.t = None
        self.t_meas = None
        self.pad_yaw = np.nan
        self.last = None
        self.n_rejected = 0
        self.init_buf = []

    # -- KF ------------------------------------------------------------------------------
    def _predict_to(self, t):
        dt = t - self.t
        if dt <= 0:
            return
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        q = self.q ** 2
        Q1 = np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]]) * q
        Q = np.zeros((6, 6))
        for i in range(3):
            Q[np.ix_([i, i + 3], [i, i + 3])] = Q1
        Q[2, 2] *= 0.1
        Q[5, 5] *= 0.1
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.t = t

    @staticmethod
    def measurement(det, drone: DroneState):
        """World-frame pad position from tvec and the drone pose at the image time."""
        p_rel_body = R_BODY_CAM @ det.tvec + P_BODY_CAM
        return drone.p + drone.R @ p_rel_body, drone.R

    def update(self, t, det, drone=None):
        if drone is None:
            return None
        z, R_wb = self.measurement(det, drone)
        rng = float(np.linalg.norm(det.tvec))
        sig = self.meas_base + self.meas_per_m * rng
        Rm = np.diag([sig ** 2, sig ** 2, (1.5 * sig) ** 2])
        if self.x is None or (self.t_meas is not None and t - self.t_meas > self.timeout):
            # initialise from a short least-squares fit (>= 3 fixes over >= 0.1 s), which
            # is far less noise-sensitive than differencing two frames 33 ms apart
            self.init_buf = [b for b in self.init_buf if t - b[0] < 0.4] + [(t, z)]
            ts = np.array([b[0] for b in self.init_buf])
            if len(ts) < 3 or ts[-1] - ts[0] < 0.1:
                return None
            Z = np.array([b[1] for b in self.init_buf])
            A = np.column_stack([np.ones_like(ts), ts - t])
            coef, *_ = np.linalg.lstsq(A, Z, rcond=None)
            span = ts[-1] - ts[0]
            sv = 3.5 * sig / span / np.sqrt(len(ts))
            self.x = np.concatenate([coef[0], coef[1]])
            self.P = np.diag([sig ** 2] * 3 + [sv ** 2] * 3)
            self.t = t
            self.init_buf = []
        else:
            self._predict_to(t)
            H = np.hstack([np.eye(3), np.zeros((3, 3))])
            y = z - H @ self.x
            S = H @ self.P @ H.T + Rm
            d2 = float(y @ np.linalg.solve(S, y))
            if d2 > self.gate:
                self.n_rejected += 1
                return self.predict(t, drone)
            Kg = self.P @ H.T @ np.linalg.inv(S)
            self.x = self.x + Kg @ y
            self.P = (np.eye(6) - Kg @ H) @ self.P
        self.t_meas = t
        R_cp, _ = cv2.Rodrigues(det.rvec)
        R_wp = R_wb @ R_BODY_CAM @ R_cp
        self.pad_yaw = R_to_euler(R_wp)[2]
        return self.predict(t, drone)

    def predict(self, t, drone=None):
        if self.x is None or drone is None:
            return None
        if self.t_meas is None or t - self.t_meas > self.timeout:
            return None
        if t > self.t:
            self._predict_to(t)
        p_t, v_t = self.x[:3], self.x[3:]
        rel_pos = p_t - drone.p
        rel_vel = v_t - drone.v
        out = Estimate(
            t=t, rel_pos=rel_pos, rel_vel=rel_vel, tgt_vel=v_t.copy(),
            speed=float(np.hypot(rel_vel[0], rel_vel[1])),
            heading=float(np.arctan2(v_t[1], v_t[0])),
            # report camera height (like tz) so all estimators feed the controller alike
            altitude=float(-rel_pos[2] + P_BODY_CAM[2]), pad_yaw=self.pad_yaw,
            age=t - self.t_meas, pos_std=float(np.sqrt(np.trace(self.P[:2, :2]) / 2)))
        self.last = out
        return out


ESTIMATORS = {
    'paper': PaperEstimator,
    'naive': PlainDistanceEstimator,
    'compensated': CompensatedEstimator,
    'corrected': CorrectedEstimator,
}
