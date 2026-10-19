"""Quick diagnostic: state transitions, estimation error, final approach details."""
import sys
import numpy as np
from mtl_landing.core.sim_runner import run_episode

sc = sys.argv[1] if len(sys.argv) > 1 else 'static'
mode = sys.argv[2] if len(sys.argv) > 2 else 'corrected'
tmax = float(sys.argv[3]) if len(sys.argv) > 3 else 60
r = run_episode(sc, mode, 0, tmax)
print(r['summary'])
prev = None
for row in r['log']:
    if row['state'] != prev:
        dp, tp = np.array(row['dp']), np.array(row['tp'])
        print(f"t={row['t']:6.2f} {row['state']:8s} hcam={row['hcam']:.2f} horiz_err={np.hypot(*(tp-dp)[:2]):.2f} cmd={row['cmd']}")
        prev = row['state']
# estimator error
E = r['est_log']
for k in ('corrected', 'naive', 'paper'):
    pe = [np.hypot(*(np.array(e[k]['rel_pos'][:2]) - np.array(e['truth']['rel_p'][:2]))) for e in E if k in e]
    print(k, 'rel_pos horiz err median/95%', np.round(np.median(pe), 3), np.round(np.percentile(pe, 95), 3))
# sample of log every 2s
for row in r['log'][::40]:
    dp, tp = np.array(row['dp']), np.array(row['tp'])
    print(f"t={row['t']:5.1f} {row['state']:8s} h={row['hcam']:.2f} e=({(tp-dp)[0]:+.2f},{(tp-dp)[1]:+.2f}) dv={row['dv']} tv={row['tv']} cmd={row['cmd']} imp={row['impact']}")
