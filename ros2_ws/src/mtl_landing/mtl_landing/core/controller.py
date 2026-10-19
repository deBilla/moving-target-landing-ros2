"""Landing state machine (paper Sec. III-D).

SEARCH  -> hold the search altitude until the pad is seen
TRACK   -> follow at a constant height above the pad: velocity feed-forward of the
           target's velocity plus PI on the horizontal offset ("lock the target")
DESCEND -> descend while staying aligned, with an alignment cone that narrows with height
FINAL   -> hover just above the pad and wait until a ballistic drop would land on it:
           the paper's "turn off the rotors and go in a projectile to the moving target".
           The drop is only allowed when the predicted impact point is on the pad.
CUT     -> motors off, falling
LANDED  -> contact
ABORT   -> "landing operation will be terminated and quadcopter will increase its
           altitude again" - climb back to the tracking height and retry
"""
from dataclasses import dataclass

import numpy as np

G = 9.81


@dataclass
class LandingParams:
    h_search: float = 6.0
    h_track: float = 4.0
    kp_xy: float = 1.6
    ki_xy: float = 0.25
    kp_z: float = 0.9
    v_max_xy: float = 7.0
    v_ff_max: float = 6.0
    v_desc_high: float = 0.8
    v_desc_low: float = 0.35
    align_base: float = 0.15       # allowed horizontal error = base + per_m * height
    align_per_m: float = 0.12
    settle_time: float = 1.0
    cam_to_skid: float = 0.05      # camera is 5 cm above the skids
    h_final: float = 0.45          # camera height at which FINAL starts
    h_cut: float = 0.24            # camera height to hold while waiting to drop
    drop_radius: float = 0.12      # predicted impact must be this close to pad centre
    drop_rel_speed: float = 0.35
    final_timeout: float = 5.0
    lost_timeout: float = 0.7
    abort_climb: float = 1.0
    memory_time: float = 5.0       # dead-reckon on the last pad estimate this long
    fresh_age: float = 0.15        # a drop needs a camera fix at most this old
    reacquire_age: float = 0.3     # close to the pad, climb when blind this long
    v_approach_max: float = 2.5    # cap on speed relative to the pad: limits tilt so the
                                   # down-looking camera keeps the pad in view
    desc_rel_speed: float = 0.6    # below 1.5 m only descend when this well matched


class LandingController:
    def __init__(self, params: LandingParams = None):
        self.p = params or LandingParams()
        self.state = 'SEARCH'
        self.t_state = 0.0
        self.t_aligned = None
        self.t_last_seen = None
        self.ix = np.zeros(2)
        self.n_aborts = 0
        self.t_prev = None
        self.predicted_impact = None
        self.memory = None             # (pad world position, pad velocity xy) when last seen

    def _go(self, state, t):
        if state != self.state:
            if state == 'ABORT':
                self.n_aborts += 1
            self.state = state
            self.t_state = t
            self.t_aligned = None

    def step(self, t, est, drone_p, drone_v):
        """Return (v_cmd world [3], motors_on bool)."""
        p = self.p
        dt = 0.0 if self.t_prev is None else max(0.0, t - self.t_prev)
        self.t_prev = t

        if self.state in ('CUT', 'LANDED'):
            return np.zeros(3), False

        seen = est is not None and est.get('valid', False)
        if seen:
            self.t_last_seen = t
        lost_for = np.inf if self.t_last_seen is None else t - self.t_last_seen

        if not seen:
            if self.state in ('DESCEND', 'FINAL') and lost_for > p.lost_timeout:
                self._go('ABORT', t)
            if self.state == 'TRACK' and lost_for > 2.0:
                self._go('SEARCH', t)
            # climb to widen the view and dead-reckon on the last pad estimate, so a
            # moving pad is not left behind while it is out of the frame
            vz = p.kp_z * (p.h_search - drone_p[2]) if self.state in ('SEARCH', 'ABORT') else 0.0
            v_xy = np.zeros(2)
            if self.memory is not None and lost_for < p.memory_time:
                p_t, v_t = self.memory
                e_pred = p_t[:2] + v_t * lost_for - drone_p[:2]
                v_xy = v_t + p.kp_xy * e_pred
                n = np.linalg.norm(v_xy)
                if n > p.v_max_xy:
                    v_xy *= p.v_max_xy / n
            return np.array([v_xy[0], v_xy[1], np.clip(vz, -1.0, 1.5)]), True

        if self.state == 'SEARCH':
            self._go('TRACK', t)

        e = np.nan_to_num(np.asarray(est['rel_pos'][:2], float))
        h = float(est['altitude'])
        v_ff = est.get('tgt_vel')
        if v_ff is None or not np.all(np.isfinite(v_ff[:2])):
            rv = est.get('rel_vel')
            v_ff = (np.asarray(rv[:2]) + drone_v[:2]) if rv is not None and np.all(
                np.isfinite(rv[:2])) else np.zeros(2)
        v_ff = np.asarray(v_ff[:2], float)
        n = np.linalg.norm(v_ff)
        if n > p.v_ff_max:
            v_ff *= p.v_ff_max / n
        self.memory = (np.asarray(drone_p, float) + np.concatenate([e, [0.0]]), v_ff.copy())
        rel_vel = est.get('rel_vel')
        rel_speed = float(np.linalg.norm(rel_vel[:2])) if rel_vel is not None and np.all(
            np.isfinite(rel_vel[:2])) else np.inf

        # horizontal: feed-forward + PI
        self.ix = np.clip(self.ix + e * dt, -2.0, 2.0)
        v_corr = p.kp_xy * e + p.ki_xy * self.ix
        n = np.linalg.norm(v_corr)
        if n > p.v_approach_max:
            v_corr *= p.v_approach_max / n
        v_xy = v_ff + v_corr
        n = np.linalg.norm(v_xy)
        if n > p.v_max_xy:
            v_xy *= p.v_max_xy / n

        err = float(np.linalg.norm(e))
        age = float(est.get('age', 0.0) or 0.0)    # time since the last camera fix
        allow = p.align_base + p.align_per_m * max(h, 0.0)
        vz = 0.0
        self.predicted_impact = None

        if self.state == 'TRACK':
            vz = p.kp_z * (p.h_track - h)
            if err < allow and rel_speed < 0.6:
                self.t_aligned = self.t_aligned if self.t_aligned is not None else t
                if t - self.t_aligned > p.settle_time:
                    self._go('DESCEND', t)
            else:
                self.t_aligned = None

        elif self.state == 'DESCEND':
            v_desc = p.v_desc_high if h > 1.5 else p.v_desc_low
            matched = h > 1.5 or rel_speed < p.desc_rel_speed
            if h < 1.0 and age > p.reacquire_age:
                vz = 0.3            # blind close to the pad: back off to widen the view
            elif err < allow and matched:
                vz = -v_desc * (1.0 - 0.5 * err / allow)
            elif err < 2.0 * allow:
                vz = 0.0
            else:
                vz = 0.4
            if err > 3.0 * allow + 0.5:
                self._go('ABORT', t)
            elif h < p.h_final:
                self._go('FINAL', t)

        elif self.state == 'FINAL':
            vz = p.kp_z * 1.5 * (p.h_cut - h)
            vz = float(np.clip(vz, -0.3, 0.3))
            h_skid = max(h - p.cam_to_skid, 0.0)
            v_down = max(-(drone_v[2]), 0.0)
            t_fall = (v_down + np.sqrt(v_down ** 2 + 2 * G * h_skid)) / G
            rv = np.nan_to_num(np.asarray(rel_vel[:2], float)) if rel_vel is not None else np.zeros(2)
            # where will the pad be relative to the drone when the skids touch?
            impact = e + rv * t_fall
            self.predicted_impact = impact
            if age > p.reacquire_age:
                vz = 0.3
            if (np.linalg.norm(impact) < p.drop_radius and rel_speed < p.drop_rel_speed
                    and h <= p.h_cut + 0.04 and age <= p.fresh_age):
                self._go('CUT', t)
                return np.zeros(3), False
            if t - self.t_state > p.final_timeout or err > 0.35:
                self._go('ABORT', t)

        elif self.state == 'ABORT':
            vz = p.abort_climb
            if h >= p.h_track - 0.3:
                self.ix[:] = 0.0
                self._go('TRACK', t)

        return np.array([v_xy[0], v_xy[1], float(np.clip(vz, -1.0, 1.5))]), True

    def touchdown(self, t):
        self._go('LANDED', t)
