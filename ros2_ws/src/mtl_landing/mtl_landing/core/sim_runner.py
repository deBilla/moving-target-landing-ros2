"""Closed-loop simulation without ROS, for Monte Carlo runs.

Same World / PadDetector / estimators / LandingController as the ROS nodes, driven by
LandingPipeline. The ROS graph's asynchrony is replaced by an explicit camera rate and
processing latency, and all estimators run in parallel on every detection so their errors
can be compared against ground truth on identical images. Only the selected `mode` drives
the controller.
"""
import argparse
import json
import time

import cv2
import numpy as np

from .detector import draw_detection
from .estimation import ESTIMATORS
from .pipeline import LandingPipeline, PipelineConfig, PipelineObserver
from .state import rounded as _f
from .world import World, WorldConfig


class _EpisodeRecorder(PipelineObserver):
    """Per-frame and per-detection logs of one episode."""

    def __init__(self, camera_K, snapshot_every=0, snapshot_width=240):
        self.K = camera_K
        self.snapshot_every, self.snapshot_width = snapshot_every, snapshot_width
        self.est_log, self.snaps, self.frames_log = [], [], []
        self.n_frames = self.n_det = 0
        self.marker_use = {}

    def on_frame(self, t, img, det, truth):
        self.n_frames += 1
        tr_img = _truth_row(truth)
        self.frames_log.append((round(t, 3), tr_img['cam_height'], tr_img['tilt_deg'],
                                det.marker_ids if det is not None else []))
        if det is not None:
            self.n_det += 1
            key = f'{det.primary_id}'
            self.marker_use[key] = self.marker_use.get(key, 0) + 1
        if self.snapshot_every and self.n_frames % self.snapshot_every == 0:
            dbg = draw_detection(img, det, self.K)
            hgt = int(self.snapshot_width * dbg.shape[0] / dbg.shape[1])
            small = cv2.resize(dbg, (self.snapshot_width, hgt), interpolation=cv2.INTER_AREA)
            ok, jpg = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 60])
            self.snaps.append(dict(t=round(t, 3), jpg=jpg.tobytes(),
                                   ids=det.marker_ids if det else []))

    def on_estimates(self, t_img, det, outs, truth):
        row = dict(t=round(t_img, 4), ids=det.marker_ids, primary=det.primary_id,
                   reproj=round(det.reproj_err_px, 3),
                   tvec=_f(det.tvec), tvec_single=_f(det.tvec_single))
        for k, out in outs.items():
            row[k] = out.summary()
        row['truth'] = _truth_row(truth)
        self.est_log.append(row)


def run_episode(scenario='straight', mode='corrected', seed=0, t_max=90.0, cam_hz=30.0,
                latency=0.05, ctrl_hz=50.0, dt=0.005, wind_sigma=0.6, snapshot_every=0,
                snapshot_width=240, world_cfg=None, verbose=False):
    cfg = world_cfg or WorldConfig(scenario=scenario, seed=seed, wind_sigma=wind_sigma)
    world = World(cfg)
    rec = _EpisodeRecorder(world.camera.K, snapshot_every, snapshot_width)
    pipe = LandingPipeline(world, PipelineConfig(mode=mode, cam_hz=cam_hz, latency=latency,
                                                 ctrl_hz=ctrl_hz, dt=dt), observers=[rec])
    ctrl = pipe.controller
    log = []
    next_log = 0.0
    wall0 = time.time()

    while world.t < t_max:
        t = world.t
        pipe.step()
        if t >= next_log - 1e-9:
            next_log += 0.05
            tr = world.truth()
            log.append(dict(t=round(t, 3), state=ctrl.state,
                            dp=_f(tr['drone_p']), dv=_f(tr['drone_v']),
                            att=_f([tr['roll'], tr['pitch'], tr['yaw']]),
                            tp=_f(tr['tgt_p']), tv=_f(tr['tgt_v']), tyaw=_f(tr['tgt_yaw']),
                            cmd=_f(pipe.v_cmd), motors=bool(pipe.motors), hcam=_f(tr['cam_height']),
                            impact=_f(ctrl.predicted_impact)))
        # after a touchdown keep simulating 1 s so the logs show it
        if pipe.t_end is not None and world.t >= pipe.t_end + (1.0 if pipe.touched_down else 0.0):
            break

    world.end('timeout')
    summary = dict(scenario=cfg.scenario, mode=mode, seed=cfg.seed, outcome=world.outcome,
                   aborts=ctrl.n_aborts, frames=rec.n_frames, detections=rec.n_det,
                   detection_rate=rec.n_det / max(1, rec.n_frames), marker_use=rec.marker_use,
                   wall_s=round(time.time() - wall0, 1))
    if verbose:
        print(json.dumps(summary, default=float))
    return dict(summary=summary, log=log, est_log=rec.est_log, snaps=rec.snaps,
                frames_log=rec.frames_log)


def _truth_row(tr):
    dp, tp, dv, tv = tr['drone_p'], tr['tgt_p'], tr['drone_v'], tr['tgt_v']
    rel_v = tv - dv
    return dict(rel_p=_f(tp - dp), rel_v=_f(rel_v),
                rel_speed=_f(np.hypot(rel_v[0], rel_v[1])),
                tgt_v=_f(tv), tgt_speed=_f(np.hypot(tv[0], tv[1])),
                tgt_heading=_f(np.arctan2(tv[1], tv[0])) if np.hypot(tv[0], tv[1]) > 0.05 else None,
                cam_height=_f(tr['cam_height']), tilt_deg=_f(np.degrees(np.hypot(tr['roll'], tr['pitch']))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='straight')
    ap.add_argument('--mode', default='corrected', choices=list(ESTIMATORS))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--t-max', type=float, default=90.0)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    r = run_episode(a.scenario, a.mode, a.seed, a.t_max, verbose=True)
    if a.out:
        r.pop('snaps')
        with open(a.out, 'w') as fh:
            json.dump(r, fh, default=float)


if __name__ == '__main__':
    main()
