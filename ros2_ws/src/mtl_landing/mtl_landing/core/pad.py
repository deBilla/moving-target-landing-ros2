"""Landing-pad geometry and texture (paper Sec. III-A).

The paper's pad carries three ArUco markers of different sizes so that at least one is
fully in view from the highest altitude down to touchdown:

  * large  - visible from the highest altitude
  * medium - visible for most of the flight
  * small  - visible when the drone is just above the pad

The small marker sits at the pad centre because that is where the camera ends up looking
during the final approach; the camera footprint at 0.3 m altitude is only ~0.42 x 0.31 m.

Pad frame: origin at pad centre, x to the right, y forward ("up" in the texture),
z out of the pad surface.
"""
from dataclasses import dataclass

import cv2
import numpy as np

ARUCO_DICT_NAME = 'DICT_4X4_50'
PAD_SIZE_M = 1.0


@dataclass(frozen=True)
class PadMarker:
    marker_id: int
    size: float      # black-border edge length [m]
    cx: float        # centre in pad frame [m]
    cy: float
    label: str


# The paper gives no marker sizes. The small one is sized so it stays fully in a 70 deg
# camera's footprint at 0.3 m with up to ~8 cm of misalignment (a 10 cm marker needed < 3 cm).
PAD_MARKERS = (
    PadMarker(0, 0.32, -0.27, 0.00, 'large'),
    PadMarker(1, 0.20, 0.25, 0.24, 'medium'),
    PadMarker(2, 0.16, 0.00, 0.00, 'small'),
)
MARKERS_BY_ID = {m.marker_id: m for m in PAD_MARKERS}


def aruco_dictionary(name=ARUCO_DICT_NAME):
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def marker_image(dictionary, marker_id, px):
    # OpenCV >= 4.7 renamed drawMarker -> generateImageMarker
    if hasattr(cv2.aruco, 'generateImageMarker'):
        return cv2.aruco.generateImageMarker(dictionary, marker_id, px)
    return cv2.aruco.drawMarker(dictionary, marker_id, px)


def marker_object_points(m: PadMarker) -> np.ndarray:
    """3D corners in the pad frame, in ArUco order (TL, TR, BR, BL)."""
    h = m.size / 2.0
    return np.array([
        [m.cx - h, m.cy + h, 0.0],
        [m.cx + h, m.cy + h, 0.0],
        [m.cx + h, m.cy - h, 0.0],
        [m.cx - h, m.cy - h, 0.0],
    ], dtype=np.float64)


def make_pad_texture(px_per_m=1000, dictionary=None):
    """White square board with the three markers drawn at their pad-frame positions."""
    dictionary = dictionary if dictionary is not None else aruco_dictionary()
    n = int(round(PAD_SIZE_M * px_per_m))
    tex = np.full((n, n), 255, np.uint8)
    for m in PAD_MARKERS:
        s = int(round(m.size * px_per_m))
        img = marker_image(dictionary, m.marker_id, s)
        # texture column grows with +x, row grows with -y
        u0 = int(round((m.cx - m.size / 2 + PAD_SIZE_M / 2) * px_per_m))
        v0 = int(round((PAD_SIZE_M / 2 - (m.cy + m.size / 2)) * px_per_m))
        tex[v0:v0 + s, u0:u0 + s] = img
    # thin grey outline so the pad edge is visible against a light deck
    cv2.rectangle(tex, (0, 0), (n - 1, n - 1), 120, max(2, n // 200))
    return tex


def texture_to_plane(width_m, height_m, px_per_m):
    """3x3 matrix mapping texture pixel (u, v, 1) to plane coords (x, y, 1), centred."""
    k = 1.0 / px_per_m
    return np.array([
        [k, 0.0, 0.5 * k - width_m / 2.0],
        [0.0, -k, height_m / 2.0 - 0.5 * k],
        [0.0, 0.0, 1.0],
    ])
