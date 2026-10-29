"""Typed values passed between the pipeline stages.

DroneState - the drone's own state as the flight controller reports it.
Estimate   - what an estimator tells the landing controller about the pad. Every estimator
             returns this type, and the ROS nodes send it over the wire with to_dict /
             from_dict, so the controller sees the same fields in the batch runner, the sim
             lab and the ROS graph.
"""
from dataclasses import dataclass, field, fields
from typing import NamedTuple

import numpy as np


class DroneState(NamedTuple):
    t: float
    p: np.ndarray        # world position [m]
    v: np.ndarray        # world velocity [m/s]
    R: np.ndarray        # world <- body rotation


def _nan3():
    return np.full(3, np.nan)


@dataclass(eq=False)
class Estimate:
    """Relative state of the pad. Unknown values are NaN.

    t         : time the estimate refers to [s]
    rel_pos   : pad - drone, world-aligned; the controller uses the horizontal components
    rel_vel   : relative velocity (pad - drone), world frame
    tgt_vel   : pad ground velocity (only estimators that track the pad in the world frame)
    speed     : scalar relative speed the estimator reports
    heading   : direction of pad motion [rad]
    altitude  : camera height above the pad [m]
    age       : how far the estimate is extrapolated past its last camera fix [s]. Zero for
                estimators that only report at fix times; their estimates expire instead
                (see Estimator.max_hold).
    """
    t: float
    rel_pos: np.ndarray = field(default_factory=_nan3)
    rel_vel: np.ndarray = field(default_factory=_nan3)
    tgt_vel: np.ndarray = field(default_factory=_nan3)
    speed: float = np.nan
    heading: float = np.nan
    altitude: float = np.nan
    age: float = 0.0
    pad_yaw: float = np.nan
    pos_std: float = np.nan
    angle_ratio: float = np.nan          # PaperEstimator only: the printed ty/tx

    _VECTORS = ('rel_pos', 'rel_vel', 'tgt_vel')

    def to_dict(self):
        """Full-precision JSON-safe dict (non-finite values become None)."""
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name in self._VECTORS:
                out[f.name] = [float(x) if np.isfinite(x) else None for x in v]
            else:
                out[f.name] = float(v) if np.isfinite(v) else None
        return out

    @classmethod
    def from_dict(cls, d):
        kw = {}
        for f in fields(cls):
            if f.name not in d:
                continue
            v = d[f.name]
            if f.name in cls._VECTORS:
                kw[f.name] = np.array([np.nan if x is None else x for x in v], float)
            else:
                kw[f.name] = np.nan if v is None else float(v)
        return cls(**kw)

    def summary(self):
        """Rounded fields compared across estimators in the logs."""
        return dict(speed=rounded(self.speed), heading=rounded(self.heading),
                    altitude=rounded(self.altitude), rel_pos=rounded(self.rel_pos),
                    rel_vel=rounded(self.rel_vel), tgt_vel=rounded(self.tgt_vel))


def rounded(x, nd=4):
    """JSON-safe value for logs: None for missing or non-finite, floats rounded."""
    if x is None:
        return None
    a = np.asarray(x, float)
    if a.ndim == 0:
        return None if not np.isfinite(a) else round(float(a), nd)
    return [None if not np.isfinite(v) else round(float(v), nd) for v in a.ravel()]
