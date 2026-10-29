"""Validation experiments (run inside the mtl_landing Docker image).

  python3 tools/experiments.py speed    # E1: paper Table I analogue (hovering drone, pad drives past)
  python3 tools/experiments.py mc       # E2: closed-loop Monte Carlo, all scenarios x estimator modes
  python3 tools/experiments.py all

E1 reproduces the paper's relative-velocity experiment: the drone hovers, the vehicle drives
underneath at a known speed (the paper used a phone speedometer for ground truth), and each
estimator's speed is compared with the truth on identical images.

E2 closes the loop: every estimator mode drives the landing controller in every scenario
over several seeds. In the corrected-mode runs all estimators also run in shadow on the
same detections, so their errors can be compared against tilt and height.
"""
import json
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np

from mtl_landing.core.dynamics import SCENARIOS as SCENARIO_DEFS, Scenario
from mtl_landing.core.estimation import ESTIMATORS
from mtl_landing.core.pipeline import LandingPipeline, PipelineConfig, PipelineObserver
from mtl_landing.core.sim_runner import run_episode
from mtl_landing.core.world import World, WorldConfig

OUT = os.environ.get('RESULTS_DIR', '/results')
MODES = list(ESTIMATORS)
SCENARIOS = list(SCENARIO_DEFS)


# ---------------------------------------------------------------------------------------
# E1 - speed sweep
# ---------------------------------------------------------------------------------------
class _SpeedRows(PipelineObserver):
    """Each estimator's speed and heading per detection, with the truth at image time."""

    def __init__(self):
        self.rows = []

    def on_estimates(self, t_img, det, outs, tr):
        row = dict(truth_rel_speed=tr['rel_speed'], truth_tgt_speed=tr['tgt_speed'],
                   tilt=float(np.degrees(np.hypot(tr['roll'], tr['pitch']))))
        for k in ESTIMATORS:
            o = outs.get(k)
            row[k] = (np.nan if o is None else float(o.speed),
                      np.nan if o is None else float(o.heading))
        self.rows.append(row)


def speed_trial(args):
    speed, seed, height = args
    cfg = WorldConfig(scenario='straight', motion=Scenario.constant(speed), seed=seed,
                      drone_start=(0.0, 0.0, height + 0.45), vehicle_start=(-8.0, 0.6),
                      wind_sigma=0.6)
    w = World(cfg)
    rec = _SpeedRows()
    pipe = LandingPipeline(w, PipelineConfig(give_up_after=None), observers=[rec])
    hold = np.array(cfg.drone_start)
    while w.vehicle.p[0] < 8.0 and w.t < 60:
        # position hold (the paper's drone hovered while the vehicle drove past)
        pipe.step(command=(1.5 * (hold - w.quad.s.p), True))
    rows = rec.rows
    out = dict(speed=speed, seed=seed, n=len(rows))
    tgt_heading = 0.0
    for k in ESTIMATORS:
        s = np.array([r[k][0] for r in rows[1:]])
        h = np.array([r[k][1] for r in rows[1:]])
        truth = np.array([r['truth_rel_speed'] for r in rows[1:]])
        fin = np.isfinite(s)
        herr = np.abs((h[np.isfinite(h)] - tgt_heading + np.pi) % (2 * np.pi) - np.pi)
        out[k] = dict(
            mean=float(np.mean(s[fin])) if fin.any() else None,
            median=float(np.median(s[fin])) if fin.any() else None,
            rmse=float(np.sqrt(np.mean((s[fin] - truth[fin]) ** 2))) if fin.any() else None,
            nan_frac=float(1 - fin.mean()) if len(fin) else None,
            heading_err_deg=float(np.degrees(np.median(herr))) if len(herr) else None)
    return out


def exp_speed():
    speeds = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    jobs = [(s, seed, 3.5) for s in speeds for seed in range(3)]
    with Pool(initializer=cv2.setNumThreads, initargs=(1,)) as pool:
        res = pool.map(speed_trial, jobs)
    agg = []
    for s in speeds:
        rs = [r for r in res if r['speed'] == s]
        row = dict(speed=s, n=int(np.mean([r['n'] for r in rs])))
        for k in ESTIMATORS:
            def m(key):
                vals = [r[k][key] for r in rs if r[k][key] is not None]
                return float(np.mean(vals)) if vals else None
            row[k] = dict(mean=m('mean'), median=m('median'), rmse=m('rmse'),
                          nan_frac=m('nan_frac'), heading_err_deg=m('heading_err_deg'))
        agg.append(row)
        print(f"v={s:.0f}  " + "  ".join(
            f"{k}: mean={row[k]['mean'] if row[k]['mean'] is None else round(row[k]['mean'], 2)} "
            f"rmse={row[k]['rmse'] if row[k]['rmse'] is None else round(row[k]['rmse'], 2)} "
            f"nan={row[k]['nan_frac'] if row[k]['nan_frac'] is None else round(row[k]['nan_frac'], 2)}"
            for k in ESTIMATORS), flush=True)
    with open(os.path.join(OUT, 'speed_sweep.json'), 'w') as fh:
        json.dump(dict(height=3.5, lateral_offset=0.6, rows=agg, raw=res), fh)


# ---------------------------------------------------------------------------------------
# E2 - closed-loop Monte Carlo
# ---------------------------------------------------------------------------------------
def mc_trial(args):
    scenario, mode, seed = args
    r = run_episode(scenario, mode, seed, t_max=60.0)
    s = r['summary']
    res = dict(scenario=scenario, mode=mode, seed=seed, outcome=s['outcome'],
               aborts=s['aborts'], detection_rate=s['detection_rate'],
               marker_use=s['marker_use'])
    if mode == 'corrected':
        res['est_rows'] = [_err_row(e) for e in r['est_log']]
        res['frames'] = r['frames_log']
    return res


def _err_row(e):
    tr = e['truth']
    row = dict(t=e['t'], h=tr['cam_height'], tilt=tr['tilt_deg'], ids=e['ids'])
    rp = np.array(tr['rel_p'], float)
    for k in ESTIMATORS:
        if k not in e:
            continue
        o = e[k]
        pe = np.nan
        if o['rel_pos'] is not None and None not in o['rel_pos'][:2]:
            pe = float(np.hypot(*(np.array(o['rel_pos'][:2], float) - rp[:2])))
        se = np.nan if o['speed'] is None else float(o['speed'] - tr['rel_speed'])
        ae = np.nan if o['altitude'] is None else float(o['altitude'] - tr['cam_height'])
        tv = np.nan
        if o['tgt_vel'] is not None and None not in o['tgt_vel'][:2]:
            tv = float(np.hypot(*(np.array(o['tgt_vel'][:2]) - np.array(tr['tgt_v'][:2]))))
        row[k] = (pe, se, ae, tv, o['speed'] is None)
    return row


def exp_mc(seeds=5):
    jobs = [(sc, m, sd) for sc in SCENARIOS for m in MODES for sd in range(seeds)]
    with Pool(initializer=cv2.setNumThreads, initargs=(1,)) as pool:
        res = []
        for i, r in enumerate(pool.imap_unordered(mc_trial, jobs)):
            o = r['outcome']
            print(f"[{i + 1}/{len(jobs)}] {r['scenario']:12s} {r['mode']:11s} seed={r['seed']} "
                  f"-> {o['result']:12s} t={o['t']:.1f} err={o.get('err', float('nan')):.3f} "
                  f"aborts={r['aborts']}", flush=True)
            res.append(r)

    # outcome table
    table = {}
    for sc in SCENARIOS:
        for m in MODES:
            rs = [r for r in res if r['scenario'] == sc and r['mode'] == m]
            ok = [r for r in rs if r['outcome']['result'] == 'landed']
            table[f'{sc}|{m}'] = dict(
                n=len(rs), landed=len(ok),
                results={k: sum(1 for r in rs if r['outcome']['result'] == k) for k in
                         set(r['outcome']['result'] for r in rs)},
                err_mean=float(np.mean([r['outcome']['err'] for r in ok])) if ok else None,
                err_max=float(np.max([r['outcome']['err'] for r in ok])) if ok else None,
                vz_mean=float(np.mean([r['outcome']['vz'] for r in ok])) if ok else None,
                vxy_mean=float(np.mean([r['outcome']['vxy'] for r in ok])) if ok else None,
                t_mean=float(np.mean([r['outcome']['t'] for r in ok])) if ok else None,
                aborts_mean=float(np.mean([r['aborts'] for r in rs])))

    # estimator error vs tilt / height (shadow estimators in corrected-mode runs)
    rows = [row for r in res if r['mode'] == 'corrected' for row in r['est_rows']]
    tilt_bins = [0, 2, 5, 10, 15, 20, 30]
    by_tilt, overall = {}, {}
    for k in ESTIMATORS:
        arr = np.array([row[k][:4] for row in rows if k in row], float)
        tilts = np.array([row['tilt'] for row in rows if k in row], float)
        nan_speed = np.array([row[k][4] for row in rows if k in row], bool)
        overall[k] = dict(
            pos_med=float(np.nanmedian(arr[:, 0])), pos_p95=float(np.nanpercentile(arr[:, 0], 95)),
            speed_rmse=float(np.sqrt(np.nanmean(arr[:, 1] ** 2))),
            speed_mae=float(np.nanmean(np.abs(arr[:, 1]))),
            alt_rmse=float(np.sqrt(np.nanmean(arr[:, 2] ** 2))),
            tgtvel_med=float(np.nanmedian(arr[:, 3])) if np.isfinite(arr[:, 3]).any() else None,
            speed_undefined_frac=float(nan_speed.mean()), n=int(len(arr)))
        by_tilt[k] = []
        for lo, hi in zip(tilt_bins[:-1], tilt_bins[1:]):
            sel = (tilts >= lo) & (tilts < hi)
            if sel.sum() < 5:
                continue
            by_tilt[k].append(dict(lo=lo, hi=hi, n=int(sel.sum()),
                                   pos_med=float(np.nanmedian(arr[sel, 0])),
                                   speed_mae=float(np.nanmean(np.abs(arr[sel, 1]))),
                                   alt_mae=float(np.nanmean(np.abs(arr[sel, 2])))))

    # pad visibility vs height (validates the paper's 3-marker pad design)
    frames = [f for r in res if r['mode'] == 'corrected' for f in r['frames']]
    hb = [0, 0.3, 0.5, 0.8, 1.2, 2, 3, 4, 5, 6.5]
    vis = []
    for lo, hi in zip(hb[:-1], hb[1:]):
        sel = [f for f in frames if lo <= f[1] < hi]
        if not sel:
            continue
        n = len(sel)
        vis.append(dict(lo=lo, hi=hi, n=n, any=sum(1 for f in sel if f[3]) / n,
                        large=sum(1 for f in sel if 0 in f[3]) / n,
                        medium=sum(1 for f in sel if 1 in f[3]) / n,
                        small=sum(1 for f in sel if 2 in f[3]) / n))

    summary = dict(seeds=seeds, scenarios=SCENARIOS, modes=MODES, table=table,
                   estimator_overall=overall, estimator_by_tilt=by_tilt, visibility=vis,
                   runs=[{k: v for k, v in r.items() if k not in ('est_rows', 'frames')} for r in res])
    with open(os.path.join(OUT, 'monte_carlo.json'), 'w') as fh:
        json.dump(summary, fh, default=float)
    print(json.dumps(dict(overall=overall, visibility=vis), indent=1, default=float))
    for key, v in table.items():
        print(f"{key:28s} landed {v['landed']}/{v['n']}  err={v['err_mean']}  results={v['results']}")


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('speed', 'all'):
        exp_speed()
    if what in ('mc', 'all'):
        exp_mc(int(sys.argv[2]) if len(sys.argv) > 2 else 5)
