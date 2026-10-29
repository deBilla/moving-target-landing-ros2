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

Each state with the pad in view has one handler that returns the vertical velocity command
and makes its own transitions. A new state is a new State member plus a handler.
"""
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from .geometry import G
from .state import Estimate


class State(StrEnum):
    SEARCH = 'SEARCH'
    TRACK = 'TRACK'
    DESCEND = 'DESCEND'
    FINAL = 'FINAL'
    CUT = 'CUT'
    LANDED = 'LANDED'
    ABORT = 'ABORT'


@dataclass
class LandingParams:
    h_search: float = 6.0
    h_track: float = 4.0
    kp_xy: float = 1.6
    ki_xy: float = 0.25
    ix_max: float = 2.0            # clamp on each component of the horizontal integrator
    kp_z: float = 0.9
    v_max_xy: float = 7.0
    v_ff_max: float = 6.0
    v_climb_max: float = 1.5       # vertical command limits
    v_sink_max: float = 1.0
    v_desc_high: float = 0.8
    v_desc_low: float = 0.35
    h_desc_slow: float = 1.5       # below this, descend at v_desc_low and only when matched
    desc_slowdown: float = 0.5     # descent slows by this fraction at the edge of the cone
    align_base: float = 0.15       # allowed horizontal error = base + per_m * height
    align_per_m: float = 0.12
    hold_cone: float = 2.0         # DESCEND: hold height inside this many cones, climb outside
    abort_cone: float = 3.0        # DESCEND: abort beyond abort_cone * cone + abort_margin
    abort_margin: float = 0.5
    v_realign: float = 0.4         # DESCEND: climb rate while badly misaligned
    settle_time: float = 1.0
    track_rel_speed: float = 0.6   # TRACK -> DESCEND only when this well matched
    cam_to_skid: float = 0.05      # camera is 5 cm above the skids
    h_final: float = 0.45          # camera height at which FINAL starts
    h_cut: float = 0.24            # camera height to hold while waiting to drop
    h_cut_tol: float = 0.04        # drop allowed up to h_cut + h_cut_tol
    final_kp_scale: float = 1.5    # FINAL height gain = kp_z * final_kp_scale
    final_vz_max: float = 0.3
    final_err_max: float = 0.35    # FINAL aborts beyond this horizontal error
    drop_radius: float = 0.12      # predicted impact must be this close to pad centre
    drop_rel_speed: float = 0.35
    final_timeout: float = 5.0
    lost_timeout: float = 0.7
    search_timeout: float = 2.0    # TRACK -> SEARCH after the pad is lost this long
    abort_climb: float = 1.0
    abort_rearm: float = 0.3       # ABORT -> TRACK within this of the tracking height
    memory_time: float = 5.0       # dead-reckon on the last pad estimate this long
    fresh_age: float = 0.15        # a drop needs a camera fix at most this old
    reacquire_age: float = 0.3     # close to the pad, climb when blind this long
    h_reacquire: float = 1.0       # DESCEND: below this height, climb when blind
    v_reacquire: float = 0.3       # climb rate while blind close to the pad
    v_approach_max: float = 2.5    # cap on speed relative to the pad: limits tilt so the
                                   # down-looking camera keeps the pad in view
    desc_rel_speed: float = 0.6    # below h_desc_slow only descend when this well matched


@dataclass
class _Seen:
    """Quantities every in-view handler needs, computed once per step."""
    t: float
    e: np.ndarray          # horizontal offset pad - drone [m]
    h: float               # camera height above the pad [m]
    err: float             # |e|
    allow: float           # alignment cone at this height [m]
    rel_vel: np.ndarray    # may contain NaN
    rel_speed: float       # inf when unknown
    age: float             # time since the last camera fix [s]
    drone_v: np.ndarray


def _limit(v, n_max):
    """v scaled down to norm n_max if longer (a new array; v is never modified)."""
    n = np.linalg.norm(v)
    return v * (n_max / n) if n > n_max else v


class LandingController:
    def __init__(self, params: LandingParams = None):
        self.p = params or LandingParams()
        self.state = State.SEARCH
        self.t_state = 0.0
        self.t_aligned = None
        self.t_last_seen = None
        self.ix = np.zeros(2)
        self.n_aborts = 0
        self.t_prev = None
        self.predicted_impact = None
        self.memory = None             # (pad world position, pad velocity xy) when last seen
        self._handlers = {
            State.TRACK: self._track,
            State.DESCEND: self._descend,
            State.FINAL: self._final,
            State.ABORT: self._abort,
        }

    def _go(self, state, t):
        if state != self.state:
            if state == State.ABORT:
                self.n_aborts += 1
            self.state = state
            self.t_state = t
            self.t_aligned = None

    def _vz_limit(self, vz):
        return float(np.clip(vz, -self.p.v_sink_max, self.p.v_climb_max))

    def step(self, t, est: Estimate | None, drone_p, drone_v):
        """Return (v_cmd world [3], motors_on bool)."""
        dt = 0.0 if self.t_prev is None else max(0.0, t - self.t_prev)
        self.t_prev = t

        if self.state in (State.CUT, State.LANDED):
            return np.zeros(3), False
        if est is None:
            return self._blind(t, drone_p), True
        return self._in_view(t, dt, est, drone_p, drone_v)

    def touchdown(self, t):
        self._go(State.LANDED, t)

    # -- pad out of view ---------------------------------------------------------------
    def _blind(self, t, drone_p):
        p = self.p
        lost_for = np.inf if self.t_last_seen is None else t - self.t_last_seen
        if self.state in (State.DESCEND, State.FINAL) and lost_for > p.lost_timeout:
            self._go(State.ABORT, t)
        if self.state == State.TRACK and lost_for > p.search_timeout:
            self._go(State.SEARCH, t)
        # climb to widen the view and dead-reckon on the last pad estimate, so a
        # moving pad is not left behind while it is out of the frame
        vz = p.kp_z * (p.h_search - drone_p[2]) if self.state in (State.SEARCH, State.ABORT) else 0.0
        v_xy = np.zeros(2)
        if self.memory is not None and lost_for < p.memory_time:
            p_t, v_t = self.memory
            e_pred = p_t[:2] + v_t * lost_for - drone_p[:2]
            v_xy = _limit(v_t + p.kp_xy * e_pred, p.v_max_xy)
        return np.array([v_xy[0], v_xy[1], self._vz_limit(vz)])

    # -- pad in view -------------------------------------------------------------------
    def _in_view(self, t, dt, est, drone_p, drone_v):
        p = self.p
        self.t_last_seen = t
        if self.state == State.SEARCH:
            self._go(State.TRACK, t)

        e = np.nan_to_num(np.asarray(est.rel_pos[:2], float))
        h = float(est.altitude)
        rel_vel = np.asarray(est.rel_vel, float)
        rel_ok = np.all(np.isfinite(rel_vel[:2]))
        v_ff = np.asarray(est.tgt_vel[:2], float)
        if not np.all(np.isfinite(v_ff)):
            v_ff = rel_vel[:2] + drone_v[:2] if rel_ok else np.zeros(2)
        v_ff = _limit(np.asarray(v_ff[:2], float), p.v_ff_max)
        self.memory = (np.asarray(drone_p, float) + np.concatenate([e, [0.0]]), v_ff.copy())

        # horizontal: feed-forward + PI
        self.ix = np.clip(self.ix + e * dt, -p.ix_max, p.ix_max)
        v_corr = _limit(p.kp_xy * e + p.ki_xy * self.ix, p.v_approach_max)
        v_xy = _limit(v_ff + v_corr, p.v_max_xy)

        s = _Seen(t=t, e=e, h=h, err=float(np.linalg.norm(e)),
                  allow=p.align_base + p.align_per_m * max(h, 0.0), rel_vel=rel_vel,
                  rel_speed=float(np.linalg.norm(rel_vel[:2])) if rel_ok else np.inf,
                  age=float(est.age or 0.0), drone_v=drone_v)
        self.predicted_impact = None
        vz = self._handlers[self.state](s)
        if vz is None:                                  # FINAL decided to drop
            return np.zeros(3), False
        return np.array([v_xy[0], v_xy[1], self._vz_limit(vz)]), True

    def _track(self, s: _Seen):
        p = self.p
        if s.err < s.allow and s.rel_speed < p.track_rel_speed:
            self.t_aligned = self.t_aligned if self.t_aligned is not None else s.t
            if s.t - self.t_aligned > p.settle_time:
                self._go(State.DESCEND, s.t)
        else:
            self.t_aligned = None
        return p.kp_z * (p.h_track - s.h)

    def _descend(self, s: _Seen):
        p = self.p
        v_desc = p.v_desc_high if s.h > p.h_desc_slow else p.v_desc_low
        matched = s.h > p.h_desc_slow or s.rel_speed < p.desc_rel_speed
        if s.h < p.h_reacquire and s.age > p.reacquire_age:
            vz = p.v_reacquire          # blind close to the pad: back off to widen the view
        elif s.err < s.allow and matched:
            vz = -v_desc * (1.0 - p.desc_slowdown * s.err / s.allow)
        elif s.err < p.hold_cone * s.allow:
            vz = 0.0
        else:
            vz = p.v_realign
        if s.err > p.abort_cone * s.allow + p.abort_margin:
            self._go(State.ABORT, s.t)
        elif s.h < p.h_final:
            self._go(State.FINAL, s.t)
        return vz

    def _final(self, s: _Seen):
        """Hold just above the pad; return None to cut the motors."""
        p = self.p
        vz = p.kp_z * p.final_kp_scale * (p.h_cut - s.h)
        vz = float(np.clip(vz, -p.final_vz_max, p.final_vz_max))
        h_skid = max(s.h - p.cam_to_skid, 0.0)
        v_down = max(-(s.drone_v[2]), 0.0)
        t_fall = (v_down + np.sqrt(v_down ** 2 + 2 * G * h_skid)) / G
        rv = np.nan_to_num(s.rel_vel[:2])
        # where will the pad be relative to the drone when the skids touch?
        impact = s.e + rv * t_fall
        self.predicted_impact = impact
        if s.age > p.reacquire_age:
            vz = p.v_reacquire
        if (np.linalg.norm(impact) < p.drop_radius and s.rel_speed < p.drop_rel_speed
                and s.h <= p.h_cut + p.h_cut_tol and s.age <= p.fresh_age):
            self._go(State.CUT, s.t)
            return None
        if s.t - self.t_state > p.final_timeout or s.err > p.final_err_max:
            self._go(State.ABORT, s.t)
        return vz

    def _abort(self, s: _Seen):
        p = self.p
        if s.h >= p.h_track - p.abort_rearm:
            self.ix[:] = 0.0
            self._go(State.TRACK, s.t)
        return p.abort_climb
