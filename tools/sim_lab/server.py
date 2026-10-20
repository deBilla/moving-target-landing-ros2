"""Interactive sim lab: the landing simulator with every parameter on a slider.

Runs the same World / PadDetector / estimators / LandingController as the batch runner,
in real time (or faster/slower), and serves a browser UI on http://localhost:8765.

  docker/run_lab.sh            # then open http://localhost:8765

Endpoints
  GET  /            the UI
  GET  /schema      parameter groups, ranges and defaults
  GET  /stream      server-sent events: telemetry (~15 Hz) + camera frames
  POST /params      {"key": value, ...}  live update (some keys apply on reset)
  POST /cmd         {"action": "play"|"pause"|"step"|"reset"|"defaults"}
  GET  /health      liveness probe

The simulation only advances while at least one browser has /stream open. With
LAB_PUBLIC=1 the most expensive knobs are capped, since every visitor shares one run.
"""
import base64
import json
import math
import os
import threading
import time
from dataclasses import fields
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from mtl_landing.core import pad as padmod
from mtl_landing.core.controller import LandingController, LandingParams
from mtl_landing.core.detector import PadDetector, draw_detection
from mtl_landing.core.dynamics import SCENARIOS
from mtl_landing.core.estimation import ESTIMATORS, DroneStateBuffer
from mtl_landing.core.world import World, WorldConfig

HERE = os.path.dirname(os.path.abspath(__file__))
DT = 0.005
# LAB_PUBLIC=1 when exposed to the internet: one shared run, so cap the costly knobs
PUBLIC = os.environ.get('LAB_PUBLIC') == '1'
MAX_CLIENTS = int(os.environ.get('LAB_MAX_CLIENTS', '20'))

# --------------------------------------------------------------------------------------
# Parameter schema: one source of truth for the UI and the simulator.
# (key, label, min, max, step, default, reset_required, help)
# --------------------------------------------------------------------------------------
LP = LandingParams()


def S(key, label, lo, hi, step, default, reset=False, help=''):
    return dict(key=key, label=label, min=lo, max=hi, step=step, default=default,
                reset=reset, help=help, type='range')


def SEL(key, label, options, default, reset=False, help=''):
    return dict(key=key, label=label, options=options, default=default, reset=reset,
                help=help, type='select')


SCHEMA = [
    dict(group='Simulation', items=[
        S('sim_speed', 'Speed (× real time)', 0.1, 2.0 if PUBLIC else 4.0, 0.1, 1.0, help='How fast simulated time runs'),
        S('seed', 'Random seed', 0, 99, 1, 0, reset=True, help='Wind, noise and ground texture'),
        S('auto_reset', 'Auto-reset after landing (s, 0 = off)', 0, 10, 1, 0),
    ]),
    dict(group='Vehicle', items=[
        SEL('scenario', 'Motion', ['manual'] + list(SCENARIOS), 'straight', reset=True,
            help='"manual" lets you drive with the two sliders below'),
        S('speed_scale', 'Speed multiplier (scenarios)', 0.0, 3.0, 0.05, 1.0),
        S('manual_speed', 'Manual speed (m/s)', 0.0, 8.0, 0.1, 2.0),
        S('manual_turn', 'Manual turn rate (deg/s)', -30, 30, 1, 0),
        S('deck_height', 'Deck height (m)', 0.0, 1.5, 0.05, 0.35, reset=True),
    ]),
    dict(group='Environment', items=[
        S('wind_sigma', 'Gust strength (m/s²)', 0.0, 3.0, 0.05, 0.6),
        S('wind_x', 'Steady wind push x (m/s²)', -2.0, 2.0, 0.05, 0.0),
        S('wind_y', 'Steady wind push y (m/s²)', -2.0, 2.0, 0.05, 0.0),
    ]),
    dict(group='Drone', items=[
        S('start_alt', 'Start altitude (m)', 1.0, 12.0, 0.5, 6.0, reset=True),
        S('tau_vel', 'Velocity response time (s)', 0.1, 1.5, 0.05, 0.45),
        S('ki_vel', 'Velocity integral gain', 0.0, 5.0, 0.1, 2.0),
        S('tilt_max', 'Max tilt (deg)', 5, 45, 1, 30),
        S('a_max_xy', 'Max horizontal accel (m/s²)', 1.0, 10.0, 0.5, 5.0),
        S('att_noise', 'Attitude bias (deg, 1σ)', 0.0, 3.0, 0.05, 0.4),
        S('vel_noise', 'Velocity noise (m/s)', 0.0, 0.5, 0.01, 0.05),
    ]),
    dict(group='Camera', items=[
        S('hfov', 'Horizontal field of view (deg)', 30, 120, 1, 70, reset=True),
        SEL('resolution', 'Resolution', ['320x240', '640x480', '800x600'] + ([] if PUBLIC else ['1280x960']),
            '800x600', reset=True),
        S('cam_hz', 'Frame rate (Hz)', 5, 60, 1, 30),
        S('latency', 'Processing latency (ms)', 0, 400, 5, 50),
        S('img_noise', 'Image noise (grey levels)', 0, 30, 0.5, 3.0),
        S('blur', 'Blur (px)', 0.0, 4.0, 0.1, 0.6),
        S('min_side_px', 'Ignore markers smaller than (px)', 4, 40, 1, 14),
    ]),
    dict(group='Landing pad', items=[
        S('m_large', 'Large marker (cm)', 10, 45, 1, 32, reset=True),
        S('m_medium', 'Medium marker (cm)', 5, 30, 1, 20, reset=True),
        S('m_small', 'Centre marker (cm)', 4, 25, 1, 16, reset=True),
    ]),
    dict(group='Estimation', items=[
        SEL('mode', 'Variant flying the drone', list(ESTIMATORS), 'corrected',
            help='paper = printed formulas read literally; naive = plain distance, camera frame; '
                 'compensated = tilt compensation; corrected = tilt compensation + Kalman filter'),
        S('kf_q', 'Kalman accel noise (m/s²)', 0.1, 8.0, 0.1, 2.0),
        S('kf_meas', 'Kalman measurement noise per metre', 0.001, 0.05, 0.001, 0.012),
        S('kf_gate', 'Outlier gate (χ²)', 4, 100, 1, 16),
    ]),
    dict(group='Landing controller', items=[
        S('h_track', 'Tracking height (m)', 1.0, 8.0, 0.1, LP.h_track),
        S('kp_xy', 'Position gain', 0.2, 4.0, 0.05, LP.kp_xy),
        S('ki_xy', 'Position integral gain', 0.0, 1.0, 0.01, LP.ki_xy),
        S('v_approach_max', 'Max approach speed vs pad (m/s)', 0.5, 6.0, 0.1, LP.v_approach_max),
        S('v_desc_high', 'Descent speed high (m/s)', 0.1, 2.0, 0.05, LP.v_desc_high),
        S('v_desc_low', 'Descent speed low (m/s)', 0.05, 1.0, 0.05, LP.v_desc_low),
        S('align_base', 'Alignment cone base (m)', 0.05, 0.6, 0.01, LP.align_base),
        S('align_per_m', 'Alignment cone per metre', 0.0, 0.4, 0.01, LP.align_per_m),
        S('h_final', 'Final-approach height (m)', 0.2, 1.5, 0.05, LP.h_final),
        S('h_cut', 'Drop height, camera (m)', 0.1, 0.8, 0.01, LP.h_cut),
        S('drop_radius', 'Drop only if impact within (m)', 0.02, 0.4, 0.01, LP.drop_radius),
        S('drop_rel_speed', 'Drop only if rel. speed below (m/s)', 0.05, 1.5, 0.05, LP.drop_rel_speed),
        S('final_timeout', 'Abort final after (s)', 1.0, 15.0, 0.5, LP.final_timeout),
        S('memory_time', 'Dead-reckon when lost for (s)', 0.0, 10.0, 0.5, LP.memory_time),
    ]),
]
ITEMS = {it['key']: it for g in SCHEMA for it in g['items']}
CTRL_KEYS = {f.name for f in fields(LandingParams)}


class Lab:
    def __init__(self):
        self.p = {k: it['default'] for k, it in ITEMS.items()}
        self.lock = threading.RLock()
        self.running = True
        self.step_once = False
        self.clients = 0          # open /stream connections; nothing is simulated without one
        self.reset()

    # -- setup ---------------------------------------------------------------------------
    def reset(self):
        with self.lock:
            p = self.p
            self._set_pad(p['m_large'] / 100, p['m_medium'] / 100, p['m_small'] / 100)
            w, h = (int(x) for x in p['resolution'].split('x'))
            scen = p['scenario']
            cfg = WorldConfig(scenario='static' if scen == 'manual' else scen, seed=int(p['seed']),
                              drone_start=(-2.0, 1.5, p['start_alt']), cam_width=w, cam_height=h,
                              cam_hfov_deg=p['hfov'])
            self.world = World(cfg)
            veh = self.world.vehicle
            veh.deck_height = p['deck_height']
            veh.p[2] = p['deck_height']
            if scen == 'manual':
                veh.speed_fn = lambda t: self.p['manual_speed']
                veh.yaw_rate_fn = lambda t: math.radians(self.p['manual_turn'])
            else:
                base = SCENARIOS[scen][0]
                veh.speed_fn = lambda t, base=base: self.p['speed_scale'] * base(t)
            self.detector = PadDetector(self.world.camera.K, self.world.camera.dist)
            self.ests = {k: c() for k, c in ESTIMATORS.items()}
            self.ctrl = LandingController(LandingParams())
            self.buf = DroneStateBuffer()
            self.pending = []
            self.v_cmd, self.motors = np.zeros(3), True
            self.next_cam = self.next_ctrl = 0.0
            self.last_det, self.last_ids = None, []
            self.jpeg, self.jpeg_seq = None, 0
            self.est_out = {}
            self.t_end = None
            self.n_frames = self.n_det = 0
            self.apply_live()

    @staticmethod
    def _set_pad(large, medium, small):
        # keep markers from overlapping: large on the left, medium top-right, small centred
        gap = 0.02
        cx_large = -(small / 2 + gap + large / 2)
        markers = (
            padmod.PadMarker(0, large, max(cx_large, -0.5 + large / 2), 0.0, 'large'),
            padmod.PadMarker(1, medium, min(0.5 - medium / 2, small / 2 + gap + medium / 2),
                             min(0.5 - medium / 2, small / 2 + gap + medium / 2), 'medium'),
            padmod.PadMarker(2, small, 0.0, 0.0, 'small'),
        )
        padmod.PAD_MARKERS = markers
        padmod.MARKERS_BY_ID.clear()
        padmod.MARKERS_BY_ID.update({m.marker_id: m for m in markers})

    def apply_live(self):
        p, w = self.p, self.world
        q = w.quad.prm
        q.tau_vel, q.ki_vel, q.a_max_xy = p['tau_vel'], p['ki_vel'], p['a_max_xy']
        q.tilt_max = math.radians(p['tilt_max'])
        w.wind.sigma = p['wind_sigma']
        w.wind.mean = np.array([p['wind_x'], p['wind_y'], 0.0])
        w.cfg.att_noise_deg, w.cfg.vel_noise = p['att_noise'], p['vel_noise']
        w.renderer.noise_std, w.renderer.blur_sigma = p['img_noise'], p['blur']
        self.detector.min_side_px = p['min_side_px']
        kf = self.ests['corrected']
        kf.q, kf.meas_per_m, kf.gate = p['kf_q'], p['kf_meas'], p['kf_gate']
        for k in CTRL_KEYS & set(p):
            setattr(self.ctrl.p, k, float(p[k]))

    def set_params(self, upd):
        need_reset = False
        with self.lock:
            for k, v in upd.items():
                if k not in ITEMS:
                    continue
                it = ITEMS[k]
                if it['type'] == 'select':
                    if v not in it['options']:
                        continue
                    self.p[k] = v
                else:
                    try:
                        self.p[k] = min(max(float(v), it['min']), it['max'])
                    except (TypeError, ValueError):
                        continue
                need_reset |= it['reset']
            self.apply_live()
        return need_reset

    # -- one physics step (same pipeline as core/sim_runner.py) ---------------------------
    def step(self):
        w, p = self.world, self.p
        t = w.t
        p_fc, v_fc, R_fc = w.fc_estimate()
        self.buf.add(t, p_fc, v_fc, R_fc)
        drone_now = (t, p_fc, v_fc, R_fc)

        if t >= self.next_cam - 1e-9:
            self.next_cam = t + 1.0 / p['cam_hz']
            img = w.render()
            det = self.detector.detect(img)
            self.n_frames += 1
            self.n_det += det is not None
            self.pending.append((t + p['latency'] / 1000.0, t, det))
            self.last_ids = det.marker_ids if det else []
            dbg = draw_detection(img, det, w.camera.K)
            if dbg.shape[1] > 640:
                dbg = cv2.resize(dbg, (640, int(640 * dbg.shape[0] / dbg.shape[1])), interpolation=cv2.INTER_AREA)
            ok, jpg = cv2.imencode('.jpg', dbg, [cv2.IMWRITE_JPEG_QUALITY, 70])
            self.jpeg, self.jpeg_seq = jpg.tobytes(), self.jpeg_seq + 1

        while self.pending and self.pending[0][0] <= t + 1e-9:
            _, t_img, det = self.pending.pop(0)
            if det is None:
                continue
            d_img = self.buf.at(t_img)
            for k, e in self.ests.items():
                out = e.update(t_img, det, d_img)
                if out is not None:
                    self.est_out[k] = out

        if t >= self.next_ctrl - 1e-9:
            self.next_ctrl = t + 0.02
            mode = p['mode']
            est = self.ests[mode].predict(t, drone_now)
            if est is not None and mode != 'corrected' and t - est['t'] > 0.5:
                est = None
            if w.outcome is None:
                self.v_cmd, self.motors = self.ctrl.step(t, est, p_fc, v_fc)

        w.step(DT, self.v_cmd, self.motors)
        if w.outcome is not None and self.t_end is None:
            self.ctrl.touchdown(w.t)
            self.t_end = w.t

    def run(self):
        last = time.perf_counter()
        while True:
            now = time.perf_counter()
            wall_dt, last = now - last, now
            if self.clients == 0:
                time.sleep(0.2)
                last = time.perf_counter()
                continue
            with self.lock:
                if self.running or self.step_once:
                    n = max(1, int(round(min(wall_dt, 0.1) * self.p['sim_speed'] / DT))) if self.running else int(0.1 / DT)
                    for _ in range(n):
                        self.step()
                    self.step_once = False
                ar = self.p['auto_reset']
                if ar and self.t_end is not None and self.world.t - self.t_end > ar:
                    self.reset()
            time.sleep(0.005)

    # -- telemetry -----------------------------------------------------------------------
    def snapshot(self):
        with self.lock:
            w, s, v = self.world, self.world.quad.s, self.world.vehicle
            tr = w.truth()
            ests = {}
            for k, o in self.est_out.items():
                rp, sp = o.get('rel_pos'), o.get('speed')
                ests[k] = dict(rel=[_f(rp[0]), _f(rp[1])] if rp is not None else None, speed=_f(sp),
                               alt=_f(o.get('altitude')))
            return dict(
                t=round(w.t, 3), running=self.running, state=self.ctrl.state, aborts=self.ctrl.n_aborts,
                drone=dict(p=_l(s.p), v=_l(s.v), roll=_f(math.degrees(s.roll)), pitch=_f(math.degrees(s.pitch)),
                           motors=s.motors_on),
                pad=dict(p=_l(v.p), v=_l(v.v), yaw=_f(v.yaw),
                         markers=[[m.size, m.cx, m.cy] for m in padmod.PAD_MARKERS]),
                rel=_l(tr['rel_p']), rel_speed=_f(tr['rel_speed']), tgt_speed=_f(tr['tgt_speed']),
                hcam=_f(tr['cam_height']), tilt=_f(math.degrees(math.hypot(s.roll, s.pitch))),
                cmd=_l(self.v_cmd), ids=self.last_ids, ests=ests,
                impact=_l(self.ctrl.predicted_impact) if self.ctrl.predicted_impact is not None else None,
                outcome=w.outcome, det_rate=_f(self.n_det / max(1, self.n_frames)),
                mode=self.p['mode'], frame_seq=self.jpeg_seq, viewers=self.clients, public=PUBLIC)


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, 4) if math.isfinite(x) else None


def _l(a):
    return [_f(x) for x in np.asarray(a, float).ravel()]


LAB = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            with open(os.path.join(HERE, 'index.html'), 'rb') as fh:
                return self._send(200, fh.read(), 'text/html; charset=utf-8')
        if self.path == '/schema':
            return self._send(200, json.dumps(dict(schema=SCHEMA, values=LAB.p)).encode(), 'application/json')
        if self.path == '/health':
            return self._send(200, b'ok', 'text/plain')
        if self.path == '/stream':
            return self._stream()
        self._send(404, b'not found', 'text/plain')

    def _stream(self):
        with LAB.lock:
            if LAB.clients >= MAX_CLIENTS:
                return self._send(503, b'too many viewers, try again later', 'text/plain')
            LAB.clients += 1
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Accel-Buffering', 'no')
        self.send_header('Connection', 'keep-alive')
        self.end_headers()
        sent_seq = -1
        try:
            while True:
                snap = LAB.snapshot()
                if snap['frame_seq'] != sent_seq and LAB.jpeg is not None:
                    snap['frame'] = base64.b64encode(LAB.jpeg).decode()
                    sent_seq = snap['frame_seq']
                self.wfile.write(f'data: {json.dumps(snap)}\n\n'.encode())
                self.wfile.flush()
                time.sleep(1 / 15)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass
        finally:
            with LAB.lock:
                LAB.clients -= 1

    def do_POST(self):
        n = int(self.headers.get('Content-Length') or 0)
        if n > 4096:
            return self._send(413, b'too large', 'text/plain')
        try:
            body = json.loads(self.rfile.read(n) or b'{}')
        except json.JSONDecodeError:
            return self._send(400, b'bad json', 'text/plain')
        if not isinstance(body, dict):
            return self._send(400, b'bad json', 'text/plain')
        if self.path == '/params':
            need = LAB.set_params(body)
            return self._send(200, json.dumps(dict(ok=True, needs_reset=need, values=LAB.p)).encode(), 'application/json')
        if self.path == '/cmd':
            a = body.get('action')
            with LAB.lock:
                if a == 'play':
                    LAB.running = True
                elif a == 'pause':
                    LAB.running = False
                elif a == 'step':
                    LAB.running, LAB.step_once = False, True
                elif a == 'reset':
                    LAB.reset()
                elif a == 'defaults':
                    LAB.p = {k: it['default'] for k, it in ITEMS.items()}
                    LAB.reset()
            return self._send(200, json.dumps(dict(ok=True, values=LAB.p)).encode(), 'application/json')
        self._send(404, b'not found', 'text/plain')


def main():
    global LAB
    cv2.setNumThreads(2)
    LAB = Lab()
    threading.Thread(target=LAB.run, daemon=True).start()
    port = int(os.environ.get('PORT', 8765))
    print(f'sim lab on http://localhost:{port} (public={PUBLIC})', flush=True)
    ThreadingHTTPServer(('0.0.0.0', port), Handler).serve_forever()


if __name__ == '__main__':
    main()
