"""Figures for the paper, generated only from the files in results/.

  python3 paper/make_figures.py
"""
import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(ROOT, 'results')
OUT = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.linewidth': 0.6,
                     'lines.linewidth': 1.1, 'savefig.bbox': 'tight', 'savefig.dpi': 300})
W = 3.45  # IEEE column width [in]

VARIANTS = [('naive', 'Camera frame', '#d95f02', 's'),
            ('compensated', 'Tilt-compensated', '#7570b3', '^'),
            ('corrected', 'Tilt-compensated + KF', '#1b9e77', 'o')]


def fig_speed():
    d = json.load(open(os.path.join(RES, 'speed_sweep.json')))
    v = np.array([r['speed'] for r in d['rows']])
    fig, (a, b) = plt.subplots(1, 2, figsize=(W * 2, 2.1))
    a.plot([0, 10], [0, 10], color='0.6', lw=0.8, ls='--', label='Ground truth')
    for k, lab, c, m in VARIANTS:
        a.plot(v, [r[k]['mean'] for r in d['rows']], marker=m, ms=3.5, color=c, label=lab)
        b.plot(v, [r[k]['rmse'] for r in d['rows']], marker=m, ms=3.5, color=c, label=lab)
    a.set(xlabel='True pad speed [m/s]', ylabel='Mean estimated speed [m/s]', xlim=(0.5, 9.5),
          ylim=(0, 10))
    b.set(xlabel='True pad speed [m/s]', ylabel='Speed RMSE [m/s]', xlim=(0.5, 9.5), ylim=(0, 1.1))
    a.legend(frameon=False, fontsize=7)
    for ax in (a, b):
        ax.grid(alpha=0.3, lw=0.4)
    fig.savefig(os.path.join(OUT, 'speed_sweep.pdf'))


def fig_tilt():
    d = json.load(open(os.path.join(RES, 'monte_carlo.json')))
    fig, (a, b) = plt.subplots(1, 2, figsize=(W * 2, 2.1))
    for k, lab, c, m in VARIANTS:
        rows = d['estimator_by_tilt'][k]
        x = [(r['lo'] + r['hi']) / 2 for r in rows]
        a.semilogy(x, [100 * r['pos_med'] for r in rows], marker=m, ms=3.5, color=c, label=lab)
        b.plot(x, [r['speed_mae'] for r in rows], marker=m, ms=3.5, color=c, label=lab)
    a.set(xlabel='Drone tilt [deg]', ylabel='Median position error [cm]')
    b.set(xlabel='Drone tilt [deg]', ylabel='Relative-speed MAE [m/s]')
    a.legend(frameon=False, fontsize=7)
    for ax in (a, b):
        ax.grid(alpha=0.3, lw=0.4, which='both')
    fig.savefig(os.path.join(OUT, 'error_vs_tilt.pdf'))


def fig_visibility():
    d = json.load(open(os.path.join(RES, 'monte_carlo.json')))
    rows = d['visibility']
    labels = [f"{r['lo']:g}–{r['hi']:g}" for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(W, 2.0))
    ax.plot(x, [100 * r['any'] for r in rows], color='k', marker='o', ms=3.5, label='Any marker')
    for key, lab, c, m in (('large', 'Large (32 cm)', '#e7298a', 's'),
                           ('medium', 'Medium (20 cm)', '#66a61e', '^'),
                           ('small', 'Small (16 cm)', '#e6ab02', 'v')):
        ax.plot(x, [100 * r[key] for r in rows], color=c, marker=m, ms=3, lw=0.9, label=lab)
    ax.set_xticks(x, labels, rotation=35)
    ax.set(xlabel='Camera height above pad [m]', ylabel='Frames with detection [%]', ylim=(0, 100))
    ax.legend(frameon=False, fontsize=6.5, loc='upper center', bbox_to_anchor=(0.5, -0.42), ncol=4)
    ax.grid(alpha=0.3, lw=0.4)
    fig.savefig(os.path.join(OUT, 'visibility.pdf'))


STATE_COLOURS = {'SEARCH': '#eeeeee', 'TRACK': '#deebf7', 'DESCEND': '#c6dbef',
                 'FINAL': '#fdd0a2', 'CUT': '#fc9272', 'LANDED': '#c7e9c0', 'ABORT': '#fbb4b9'}


def fig_run(name='straight_corrected'):
    d = json.load(open(os.path.join(RES, f'ros_run_{name}.json')))
    L = d['log']
    t = np.array([r['t'] for r in L])
    h = np.array([r['hcam'] for r in L])
    off = np.array([math.hypot(r['tp'][0] - r['dp'][0], r['tp'][1] - r['dp'][1]) for r in L])
    fig, (a, b) = plt.subplots(2, 1, figsize=(W, 2.6), sharex=True)
    seen = set()
    start = 0
    for i in range(1, len(L) + 1):
        if i == len(L) or L[i]['state'] != L[start]['state']:
            s = L[start]['state']
            for ax in (a, b):
                ax.axvspan(t[start], t[min(i, len(L) - 1)], color=STATE_COLOURS.get(s, 'w'), lw=0,
                           label=s if (ax is a and s not in seen) else None)
            seen.add(s)
            start = i
    a.plot(t, h, color='k')
    b.plot(t, 100 * off, color='k')
    a.set(ylabel='Camera height [m]')
    b.set(ylabel='True horizontal\noffset [cm]', xlabel='Time [s]', yscale='log')
    a.legend(frameon=False, fontsize=6.5, ncol=4, loc='lower left', bbox_to_anchor=(0, 1.0))
    fig.savefig(os.path.join(OUT, 'landing_run.pdf'))


def fig_frames(name='straight_corrected', files=('0024.jpg', '0036.jpg', '0051.jpg')):
    d = json.load(open(os.path.join(RES, f'ros_run_{name}.json')))
    L = d['log']
    ts = np.array([r['t'] for r in L])
    times = {f['file']: f['t'] for f in d['frames']}
    fig, axes = plt.subplots(1, len(files), figsize=(W * 2, 1.75))
    for ax, f in zip(axes, files):
        img = plt.imread(os.path.join(RES, f'frames_{name}', f))
        ax.imshow(img, cmap='gray')
        r = L[int(np.argmin(np.abs(ts - times[f])))]
        ax.set_title(f"t = {times[f]:.1f} s, height {r['hcam']:.2f} m ({r['state']})", fontsize=7)
        ax.axis('off')
    fig.savefig(os.path.join(OUT, 'camera_frames.pdf'))


if __name__ == '__main__':
    fig_speed()
    fig_tilt()
    fig_visibility()
    fig_run()
    fig_frames()
    print('figures written to', OUT)
