"""Properties of the Sec. III-C formulas (read literally) and of the closed-loop pipeline."""
import numpy as np
import pytest

from mtl_landing.core.camera import PinholeCamera, SceneRenderer
from mtl_landing.core.detector import PadDetection, PadDetector
from mtl_landing.core.estimation import CorrectedEstimator, PaperEstimator
from mtl_landing.core.geometry import euler_to_R, rot_z


def literal_distance(t1, t2):
    d2 = (t1[0] ** 2 + t1[1] ** 2) - (t2[0] ** 2 + t2[1] ** 2)
    return np.sqrt(d2) if d2 >= 0 else np.nan


def det_at(tvec):
    return PadDetection(rvec=np.zeros(3), tvec=np.asarray(tvec, float))


def test_literal_distance_is_undefined_when_target_recedes():
    # target moves 0.5 m away from the optical axis: real displacement 0.5 m
    assert np.isnan(literal_distance((1.0, 0.0, 4.0), (1.5, 0.0, 4.0)))


def test_literal_distance_is_zero_for_motion_around_the_axis():
    # target moves 1.41 m along a circle centred on the optical axis
    d = literal_distance((1.0, 0.0, 4.0), (0.0, 1.0, 4.0))
    assert d == pytest.approx(0.0)


def test_literal_distance_scales_with_sqrt_of_range():
    # same 0.1 m approach, at 1 m and at 4 m from the axis: true displacement identical
    near = literal_distance((1.0, 0.0, 4.0), (0.9, 0.0, 4.0))
    far = literal_distance((4.0, 0.0, 4.0), (3.9, 0.0, 4.0))
    assert near == pytest.approx(np.sqrt(0.19))       # 0.436 m, not 0.1 m
    assert far == pytest.approx(np.sqrt(0.79))        # 0.889 m, not 0.1 m


def test_angle_ratio_is_bearing_not_direction_of_motion():
    # target at bearing 45 deg moving purely along +x: the motion direction is 0 deg,
    # but ty/tx = 1 for both samples (it describes where the target is)
    est = PaperEstimator()
    est.update(0.0, det_at((1.0, 1.0, 4.0)))
    out = est.update(0.1, det_at((1.05, 1.0, 4.0)))
    assert out['angle_ratio'] == pytest.approx(1.0 / 1.05)


def test_corrected_measurement_compensates_tilt():
    # drone at 4 m, pitched 15 deg; pad straight below the CoG
    R_wb = euler_to_R(0.0, np.radians(15), 0.3)
    p_d = np.array([2.0, -1.0, 4.35])
    p_pad = np.array([2.0, -1.0, 0.35])
    from mtl_landing.core.geometry import P_BODY_CAM, R_BODY_CAM
    R_wc = R_wb @ R_BODY_CAM
    p_wc = p_d + R_wb @ P_BODY_CAM
    tvec = R_wc.T @ (p_pad - p_wc)
    # the camera-frame offset is ~1 m even though the pad is right below
    assert np.hypot(tvec[0], tvec[1]) > 0.9
    z, _ = CorrectedEstimator.measurement(det_at(tvec), (0.0, p_d, np.zeros(3), R_wb))
    assert np.allclose(z, p_pad, atol=1e-9)


@pytest.mark.parametrize('height,tilt_deg', [(1.0, 0.0), (3.0, 10.0), (5.0, 20.0)])
def test_detector_recovers_pad_pose_from_rendered_image(height, tilt_deg):
    cam = PinholeCamera()
    rnd = SceneRenderer(cam, seed=1, noise_std=2.0)
    det = PadDetector(cam.K)
    R_wb = euler_to_R(np.radians(tilt_deg) * 0.6, np.radians(tilt_deg) * 0.8, 0.4)
    p_d = np.array([0.3, -0.2, 0.35 + 0.05 + height])
    # shift the pad so it stays in view when tilted
    R_wc, p_wc = cam.pose_in_world(R_wb, p_d)
    ray = R_wc[:, 2]
    p_pad = p_wc + ray * (height / -ray[2])
    p_pad[2] = 0.35
    R_pad = rot_z(1.0)
    img = rnd.render(R_wb, p_d, R_pad, p_pad)
    d = det.detect(img)
    assert d is not None
    z, _ = CorrectedEstimator.measurement(d, (0.0, p_d, np.zeros(3), R_wb))
    assert np.linalg.norm(z - p_pad) < 0.02 + 0.01 * height
