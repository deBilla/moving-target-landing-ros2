"""ArUco pad detection and pose estimation (paper Sec. III-B and III-C).

The paper calls cv::aruco::estimatePoseSingleMarkers on one marker. That function is
deprecated in OpenCV >= 4.7, and it is simply solvePnP with IPPE_SQUARE. This module does:

  1. detect all markers and keep the ones belonging to the pad,
  2. single-marker IPPE_SQUARE pose for the largest marker in the image (the paper's method),
     moved from that marker's centre to the pad centre with the known layout,
  3. refine with every visible corner of every pad marker in one solvePnP (board pose).

Step 3 is an improvement. It matters most at handover between the large, medium and small
markers, when two of them are partly visible.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from .pad import DEFAULT_PAD, PadLayout


@dataclass
class PadDetection:
    rvec: np.ndarray                 # pad frame -> camera frame (Rodrigues)
    tvec: np.ndarray                 # pad centre in camera frame [m]
    marker_ids: list = field(default_factory=list)
    primary_id: int = -1             # marker used for the single-marker (paper) pose
    tvec_single: np.ndarray = None   # pad centre from the single marker only
    reproj_err_px: float = 0.0
    corners: list = field(default_factory=list)


def _make_detect_fn(dictionary):
    if hasattr(cv2.aruco, 'ArucoDetector'):          # OpenCV >= 4.7
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        det = cv2.aruco.ArucoDetector(dictionary, params)
        return det.detectMarkers
    params = cv2.aruco.DetectorParameters_create()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return lambda img: cv2.aruco.detectMarkers(img, dictionary, parameters=params)


class PadDetector:
    def __init__(self, K, dist=None, layout: PadLayout = DEFAULT_PAD, use_board_refinement=True,
                 min_side_px=14.0):
        self.K = np.asarray(K, float)
        self.dist = np.zeros(5) if dist is None else np.asarray(dist, float)
        self.layout = layout
        self.markers = layout.by_id
        self.dictionary = layout.dictionary()
        self._detect = _make_detect_fn(self.dictionary)
        self.use_board = use_board_refinement
        self.min_side_px = min_side_px
        self.obj = layout.object_points()

    def detect(self, gray):
        corners, ids, _ = self._detect(gray)
        if ids is None:
            return None
        found = [(int(i), c.reshape(4, 2).astype(np.float64))
                 for i, c in zip(ids.flatten(), corners) if int(i) in self.markers]
        # markers only a few pixels across give unreliable depth: drop them
        found = [(i, c) for i, c in found
                 if np.min(np.linalg.norm(c - np.roll(c, 1, axis=0), axis=1)) >= self.min_side_px]
        if not found:
            return None

        # Paper's method: single-marker pose of the most prominent marker
        areas = [cv2.contourArea(c.astype(np.float32)) for _, c in found]
        pid, pc = found[int(np.argmax(areas))]
        m = self.markers[pid]
        h = m.size / 2.0
        obj_local = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj_local, pc, self.K, self.dist,
                                      flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            return None
        R, _ = cv2.Rodrigues(rvec)
        # marker frame is the pad frame shifted by (cx, cy): pad origin = marker origin - offset
        tvec_pad = tvec.reshape(3) + R @ np.array([-m.cx, -m.cy, 0.0])
        rvec_pad = rvec.reshape(3)
        tvec_single = tvec_pad.copy()

        if self.use_board and len(found) > 1:
            obj = np.vstack([self.obj[i] for i, _ in found])
            img = np.vstack([c for _, c in found])
            ok, r2, t2 = cv2.solvePnP(obj, img, self.K, self.dist,
                                      rvec=rvec_pad.reshape(3, 1).copy(),
                                      tvec=tvec_pad.reshape(3, 1).copy(),
                                      useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
            if ok and t2[2, 0] > 0:
                rvec_pad, tvec_pad = r2.reshape(3), t2.reshape(3)

        obj = np.vstack([self.obj[i] for i, _ in found])
        img = np.vstack([c for _, c in found])
        proj, _ = cv2.projectPoints(obj, rvec_pad, tvec_pad, self.K, self.dist)
        err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))

        return PadDetection(rvec=rvec_pad, tvec=tvec_pad, marker_ids=[i for i, _ in found],
                            primary_id=pid, tvec_single=tvec_single, reproj_err_px=err,
                            corners=[c for _, c in found])


def draw_detection(gray, det: PadDetection, K, dist=None):
    """BGR debug image with marker outlines and the pad axes (like the paper's Fig. 5)."""
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if det is None:
        cv2.putText(img, 'NO TARGET', (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return img
    for mid, c in zip(det.marker_ids, det.corners):
        cv2.polylines(img, [c.astype(np.int32)], True, (0, 255, 0), 2)
        cv2.putText(img, f'id={mid}', tuple(c[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 0, 0), 2)
    dist = np.zeros(5) if dist is None else dist
    L = 0.25
    pts = np.array([[0, 0, 0], [L, 0, 0], [0, L, 0], [0, 0, L]], np.float64)
    p, _ = cv2.projectPoints(pts, det.rvec, det.tvec, K, dist)
    p = p.reshape(-1, 2).astype(int)
    for k, col in zip((1, 2, 3), ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
        cv2.line(img, tuple(p[0]), tuple(p[k]), col, 3)
    t = det.tvec
    cv2.putText(img, f'tvec=({t[0]:+.2f},{t[1]:+.2f},{t[2]:.2f}) m', (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return img
