"""Pinhole model of the downward camera and a planar-scene renderer.

The scene is three textured planes - ground, vehicle deck, landing pad - so each frame is
just three homography warps. That is enough to produce real images for the real ArUco
detector, including perspective distortion from drone tilt, the shrinking footprint near
touchdown and background clutter, without needing Gazebo or a GPU.
"""
import cv2
import numpy as np

from .geometry import P_BODY_CAM, R_BODY_CAM
from .pad import DEFAULT_PAD, PadLayout, texture_to_plane


class PinholeCamera:
    def __init__(self, width=800, height=600, hfov_deg=70.0):
        self.width = int(width)
        self.height = int(height)
        fx = (self.width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
        self.K = np.array([
            [fx, 0.0, (self.width - 1) / 2.0],
            [0.0, fx, (self.height - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ])
        self.dist = np.zeros(5)

    def pose_in_world(self, R_world_body, p_world_body):
        R_wc = R_world_body @ R_BODY_CAM
        p_wc = p_world_body + R_world_body @ P_BODY_CAM
        return R_wc, p_wc


class _Plane:
    def __init__(self, texture, width_m, height_m, px_per_m, z_offset=0.0):
        self.texture = texture
        self.S = texture_to_plane(width_m, height_m, px_per_m)
        self.z_offset = z_offset


def _periodic_ground_tile(rng, n=400):
    """Seamless (periodic) grass/asphalt-like texture plus some square clutter."""
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.fftfreq(n)[None, :]
    f = np.sqrt(fx ** 2 + fy ** 2)
    f[0, 0] = 1.0
    spec = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / f ** 1.4
    spec[0, 0] = 0.0
    tile = np.real(np.fft.ifft2(spec))
    tile = (tile - tile.mean()) / (tile.std() + 1e-9)
    tile = np.ascontiguousarray(np.clip(128 + 22 * tile, 40, 215).astype(np.uint8))
    # dark and light rectangles: square-ish false candidates for the marker detector
    for _ in range(14):
        w, h = rng.integers(8, 30, size=2)
        u, v = rng.integers(10, n - 40, size=2)
        val = int(rng.choice([25, 235]))
        cv2.rectangle(tile, (int(u), int(v)), (int(u + w), int(v + h)), val, -1)
    return tile


class SceneRenderer:
    GROUND_M_PER_PX = 0.025
    GROUND_TILE_PX = 400        # tile period: 10 m

    def __init__(self, camera: PinholeCamera, deck_size=(1.6, 1.3), seed=0,
                 noise_std=3.0, blur_sigma=0.6, gain=1.0, layout: PadLayout = DEFAULT_PAD):
        self.cam = camera
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.noise_std = noise_std
        self.blur_sigma = blur_sigma
        self.gain = gain

        tile = _periodic_ground_tile(rng, self.GROUND_TILE_PX)
        self.ground_tex = np.tile(tile, (3, 3))
        self.tile_m = self.GROUND_TILE_PX * self.GROUND_M_PER_PX
        gw = 3 * self.tile_m
        self.ground = _Plane(self.ground_tex, gw, gw, 1.0 / self.GROUND_M_PER_PX)

        dpx = 100
        dw, dh = deck_size
        deck_tex = np.full((int(dh * dpx), int(dw * dpx)), 70, np.uint8)
        cv2.rectangle(deck_tex, (0, 0), (deck_tex.shape[1] - 1, deck_tex.shape[0] - 1), 35, 6)
        self.deck = _Plane(deck_tex, dw, dh, dpx)

        ppm = 1000
        self.pad = _Plane(layout.texture(ppm), layout.size, layout.size, ppm)

    def _H(self, R_wc, p_wc, R_wp, p_wp, S):
        R_cp = R_wc.T @ R_wp
        t_cp = R_wc.T @ (p_wp - p_wc)
        M = np.column_stack((R_cp[:, 0], R_cp[:, 1], t_cp))
        return self.cam.K @ M @ S

    def render(self, R_world_body, p_world_body, R_world_deck, p_world_deck):
        """Grayscale uint8 image seen by the downward camera.

        p_world_deck is the deck centre at deck-top height; the pad lies on top of it.
        """
        R_wc, p_wc = self.cam.pose_in_world(R_world_body, p_world_body)
        size = (self.cam.width, self.cam.height)

        # Ground: re-anchor the periodic texture under the camera so it never runs out.
        anchor = np.floor(p_wc[:2] / self.tile_m) * self.tile_m + self.tile_m / 2.0
        p_ground = np.array([anchor[0], anchor[1], 0.0])
        img = np.full((self.cam.height, self.cam.width), 128, np.uint8)
        if p_wc[2] > 0.02:
            H = self._H(R_wc, p_wc, np.eye(3), p_ground, self.ground.S)
            img = cv2.warpPerspective(self.ground.texture, H, size, flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=128)

        # Deck and pad (only when the camera is above them)
        cam_above_deck = (R_world_deck.T @ (p_wc - p_world_deck))[2]
        if cam_above_deck > 0.02:
            H = self._H(R_wc, p_wc, R_world_deck, p_world_deck, self.deck.S)
            cv2.warpPerspective(self.deck.texture, H, size, dst=img, flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_TRANSPARENT)
            p_pad = p_world_deck + R_world_deck @ np.array([0.0, 0.0, 0.002])
            H = self._H(R_wc, p_wc, R_world_deck, p_pad, self.pad.S)
            cv2.warpPerspective(self.pad.texture, H, size, dst=img, flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_TRANSPARENT)

        out = img.astype(np.float32) * self.gain
        if self.blur_sigma > 0:
            out = cv2.GaussianBlur(out, (0, 0), self.blur_sigma)
        if self.noise_std > 0:
            out += self.rng.normal(0.0, self.noise_std, out.shape).astype(np.float32)
        return np.clip(out, 0, 255).astype(np.uint8)
