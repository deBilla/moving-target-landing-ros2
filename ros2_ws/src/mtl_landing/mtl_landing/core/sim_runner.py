"""Closed-loop simulation without ROS, for Monte Carlo runs.

Same World / PadDetector / estimators / LandingController as the ROS nodes. The ROS
graph's asynchrony is replaced by an explicit camera rate and processing latency, and
all three estimators run in parallel on every detection so their errors can be compared
against ground truth on identical images. Only the selected `mode` drives the controller.
"""
import argparse
import json
import time

import cv2
import numpy as np

from .controller import LandingController, LandingParams
from .detector import PadDetector, draw_detection
from .estimation import ESTIMATORS, DroneStateBuffer
from .world import World, WorldConfig


def _f(x):
    """JSON-safe float / list."""
    if x is None:
        return None
    a = np.asarray(x, float)
    if a.ndim == 0:
        return None if not np.isfinite(a) else round(float(a), 4)
    return [None if not np.isfinite(v) else round(float(v), 4) for v in a]


def run_episode(scenario='straight', mode='corrected', seed=0, t_max=90.0, cam_hz=30.0,
                latency=0.05, ctrl_hz=50.0, dt=0.005, wind_sigma=0.6, snapshot_every=0,
                snapshot_width=240, world_cfg=None, verbose=False):
    cfg = world_cfg or WorldConfig(scenario=scenario, seed=seed, wind_sigma=wind_sigma)
    world = World(cfg)
    detector = PadDetector(world.camera.K, world.camera.dist)
    ests = {k: cls() for k, cls in ESTIMATORS.items()}
    ctrl = LandingController(LandingParams())
    buf = DroneStateBuffer()

    pending = []            # (t_ready, t_img, det, truth at t_img)
    log, est_log, snaps, frames_log = [], [], [], []
    v_cmd, motors = np.zeros(3), True
    next_cam, next_ctrl, next_log = 0.0, 0.0, 0.0
    frame_idx = 0
    n_frames = n_det = 0
    marker_use = {}
    wall0 = time.time()
    t_end = None

    while world.t < t_max:
        t = world.t
        p_fc, v_fc, R_fc = world.fc_estimate()
        buf.add(t, p_fc, v_fc, R_fc)
        drone_now = (t, p_fc, v_fc, R_fc)

        # camera
        if t >= next_cam - 1e-9:
            next_cam += 1.0 / cam_hz
            img = world.render()
            det = detector.detect(img)
            n_frames += 1
            frame_idx += 1
            tr_img = _truth_row(world.truth())
            frames_log.append((round(t, 3), tr_img['cam_height'], tr_img['tilt_deg'],
                               det.marker_ids if det is not None else []))
            if det is not None:
                n_det += 1
                key = f'{det.primary_id}'
                marker_use[key] = marker_use.get(key, 0) + 1
            pending.append((t + latency, t, det, tr_img))
            if snapshot_every and frame_idx % snapshot_every == 0:
                dbg = draw_detection(img, det, world.camera.K)
                hgt = int(snapshot_width * dbg.shape[0] / dbg.shape[1])
                small = cv2.resize(dbg, (snapshot_width, hgt), interpolation=cv2.INTER_AREA)
                ok, jpg = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 60])
                snaps.append(dict(t=round(t, 3), jpg=jpg.tobytes(),
                                  ids=det.marker_ids if det else []))

        # detections become available after the processing latency
        while pending and pending[0][0] <= t + 1e-9:
            _, t_img, det, tr = pending.pop(0)
            if det is None:
                continue
            drone_img = buf.at(t_img)
            row = dict(t=round(t_img, 4), ids=det.marker_ids, primary=det.primary_id,
                       reproj=round(det.reproj_err_px, 3),
                       tvec=_f(det.tvec), tvec_single=_f(det.tvec_single))
            for k, e in ests.items():
                out = e.update(t_img, det, drone_img)
                if out is not None:
                    row[k] = dict(speed=_f(out['speed']), heading=_f(out['heading']),
                                  altitude=_f(out['altitude']), rel_pos=_f(out['rel_pos']),
                                  rel_vel=_f(out['rel_vel']), tgt_vel=_f(out['tgt_vel']))
            row['truth'] = tr
            est_log.append(row)

        # controller
        if t >= next_ctrl - 1e-9:
            next_ctrl += 1.0 / ctrl_hz
            est = ests[mode].predict(t, drone_now)
            if est is not None and mode != 'corrected':
                # paper/naive keep their last value; treat >0.5 s old as lost
                if t - est['t'] > 0.5:
                    est = None
            if world.outcome is None:
                v_cmd, motors = ctrl.step(t, est, p_fc, v_fc)

        world.step(dt, v_cmd, motors)

        # give up once the pad has been out of sight for 10 s after it was first tracked
        if (world.outcome is None and ctrl.state == 'SEARCH' and ctrl.t_last_seen is not None
                and t - ctrl.t_last_seen > 10.0):
            world.outcome = dict(result='lost_target', t=world.t, err=np.nan, vz=np.nan, vxy=np.nan)
            t_end = world.t
        if world.outcome is not None and t_end is None:
            ctrl.touchdown(world.t)
            t_end = world.t + 1.0

        if t >= next_log - 1e-9:
            next_log += 0.05
            tr = world.truth()
            log.append(dict(t=round(t, 3), state=ctrl.state,
                            dp=_f(tr['drone_p']), dv=_f(tr['drone_v']),
                            att=_f([tr['roll'], tr['pitch'], tr['yaw']]),
                            tp=_f(tr['tgt_p']), tv=_f(tr['tgt_v']), tyaw=_f(tr['tgt_yaw']),
                            cmd=_f(v_cmd), motors=bool(motors), hcam=_f(tr['cam_height']),
                            impact=_f(ctrl.predicted_impact) if ctrl.predicted_impact is not None else None))
        if t_end is not None and world.t >= t_end:
            break

    outcome = world.outcome or dict(result='timeout', t=world.t, err=np.nan, vz=np.nan, vxy=np.nan)
    summary = dict(scenario=cfg.scenario, mode=mode, seed=cfg.seed, outcome=outcome,
                   aborts=ctrl.n_aborts, frames=n_frames, detections=n_det,
                   detection_rate=n_det / max(1, n_frames), marker_use=marker_use,
                   wall_s=round(time.time() - wall0, 1))
    if verbose:
        print(json.dumps(summary, default=float))
    return dict(summary=summary, log=log, est_log=est_log, snaps=snaps, frames_log=frames_log)


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
