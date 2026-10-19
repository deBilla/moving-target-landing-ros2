"""Export ROS replays for the blog's LandingReplay component.

  python3 tools/export_blog_replay.py <out_dir> run1 run2 ...
  (run names like straight_corrected; reads results/ros_run_<run>.json)

Writes <out_dir>/replay.json and <out_dir>/frames/<run>/NNNN.jpg (frames referenced by
file name, so the page only downloads what it shows).
"""
import json
import math
import os
import shutil
import sys

ROOT = os.path.join(os.path.dirname(__file__), '..', 'results')


def quat_tilt_deg(q):
    x, y, z, w = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1, min(1, 2 * (w * y - z * x))))
    return round(math.degrees(math.hypot(roll, pitch)), 1)


def export_run(name, out_dir, frame_stride=2):
    d = json.load(open(os.path.join(ROOT, f'ros_run_{name}.json')))
    rows, yaw = [], 0.0
    for r in d['log'][::2]:
        tv = r['tv']
        if math.hypot(tv[0], tv[1]) > 0.1:
            yaw = math.atan2(tv[1], tv[0])
        rows.append([r['t'], r['state'], *[round(v, 3) for v in r['dp']],
                     *[round(v, 3) for v in r['tp']], round(yaw, 3), quat_tilt_deg(r['q']),
                     r['hcam'], r.get('impact')])
    src = os.path.join(ROOT, f'frames_{name}')
    dst = os.path.join(out_dir, 'frames', name)
    os.makedirs(dst, exist_ok=True)
    frames = []
    for f in d['frames'][::frame_stride]:
        shutil.copyfile(os.path.join(src, f['file']), os.path.join(dst, f['file']))
        frames.append([f['t'], f['file']])
    est = []
    for e in d['est_log'][::3]:
        tr = e.get('truth')
        if tr:
            est.append([e['t'], tr['rel_speed'], (e.get('paper') or {}).get('speed'),
                        (e.get('corrected') or {}).get('speed')])
    return dict(meta=d['meta'], outcome=d['outcome'], rows=rows, frames=frames, est=est)


def main():
    out_dir, runs = sys.argv[1], sys.argv[2:]
    if os.path.isdir(os.path.join(out_dir, 'frames')):
        shutil.rmtree(os.path.join(out_dir, 'frames'))
    os.makedirs(out_dir, exist_ok=True)
    data = {'cols': ['t', 'state', 'dx', 'dy', 'dz', 'tx', 'ty', 'tz', 'tyaw', 'tilt', 'hcam', 'impact'],
            'runs': {r: export_run(r, out_dir) for r in runs}}
    path = os.path.join(out_dir, 'replay.json')
    with open(path, 'w') as fh:
        json.dump(data, fh, separators=(',', ':'))
    total = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(out_dir) for f in fs)
    print(path, os.path.getsize(path) // 1024, 'KB json,', total // 1024, 'KB total')


if __name__ == '__main__':
    main()
