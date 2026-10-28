"""Contracts between the stages: estimate wire format, staleness, pad layouts."""
import json

import numpy as np
import pytest

from mtl_landing.core.camera import PinholeCamera, SceneRenderer
from mtl_landing.core.controller import LandingController, State
from mtl_landing.core.detector import PadDetection, PadDetector
from mtl_landing.core.estimation import ESTIMATORS, Estimator, PlainDistanceEstimator
from mtl_landing.core.geometry import euler_to_R
from mtl_landing.core.pad import DEFAULT_PAD, PadLayout, PadMarker
from mtl_landing.core.state import Estimate


def test_every_estimator_implements_the_interface():
    for cls in ESTIMATORS.values():
        assert issubclass(cls, Estimator)


def test_estimate_survives_the_json_wire_format():
    est = Estimate(t=3.2, rel_pos=np.array([0.4, -0.1, -2.0]), rel_vel=np.array([0.2, np.nan, 0.0]),
                   speed=0.21, altitude=2.05, age=0.42, pad_yaw=0.3)
    back = Estimate.from_dict(json.loads(json.dumps(est.to_dict())))
    assert back.age == pytest.approx(0.42)
    assert np.allclose(back.rel_pos, est.rel_pos)
    assert np.isnan(back.rel_vel[1]) and np.all(np.isnan(back.tgt_vel))
    assert np.isnan(back.heading)


def test_controller_backs_off_when_the_fix_is_stale():
    # DESCEND close to the pad: an estimate extrapolated 0.5 s past its fix means "blind"
    ctrl = LandingController()
    ctrl.state = State.DESCEND
    est = Estimate(t=0.0, rel_pos=np.array([0.0, 0.0, -0.8]), rel_vel=np.zeros(3),
                   altitude=0.8, age=0.5)
    v_cmd, motors = ctrl.step(0.0, est, np.array([0.0, 0.0, 1.2]), np.zeros(3))
    assert motors and v_cmd[2] == pytest.approx(ctrl.p.v_reacquire)
    est.age = 0.0
    v_cmd, _ = ctrl.step(0.02, est, np.array([0.0, 0.0, 1.2]), np.zeros(3))
    assert v_cmd[2] < 0.0


def test_last_fix_estimators_expire():
    e = PlainDistanceEstimator()
    e.update(1.0, PadDetection(rvec=np.zeros(3), tvec=np.array([0.1, 0.2, 3.0])))
    assert e.predict(1.0 + e.max_hold) is not None
    assert e.predict(1.0 + e.max_hold + 0.01) is None


def test_detector_uses_the_layout_it_is_given():
    layout = PadLayout(markers=(PadMarker(7, 0.4, 0.0, 0.0, 'only'),))
    cam = PinholeCamera()
    img = SceneRenderer(cam, seed=2, layout=layout).render(
        euler_to_R(0.0, 0.0, 0.0), np.array([0.0, 0.0, 3.4]), np.eye(3), np.array([0.0, 0.0, 0.35]))
    det = PadDetector(cam.K, layout=layout).detect(img)
    assert det is not None and det.marker_ids == [7]
    # the default pad's detector does not know marker 7, and the default is untouched
    assert PadDetector(cam.K).detect(img) is None
    assert [m.marker_id for m in DEFAULT_PAD.markers] == [0, 1, 2]
