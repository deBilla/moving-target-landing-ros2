"""One closed-loop step: camera -> detector -> estimators -> controller -> physics.

The batch runner, the sim lab and the speed-sweep experiment all drive this class, so
they share one implementation of the camera rate, the processing latency, the drone-state
lookup at image time and the controller rate. Front-ends attach a PipelineObserver to see
frames and estimates instead of copying the loop.

The ROS graph runs the same stages as separate nodes; there the asynchrony comes from the
transport instead of the explicit rates and latency here.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np

from .controller import LandingController, State
from .detector import PadDetector
from .estimation import ESTIMATORS, DroneStateBuffer


@dataclass
class PipelineConfig:
    mode: str = 'corrected'            # estimator that flies the drone
    cam_hz: float = 30.0
    latency: float = 0.05              # detection processing delay [s]
    ctrl_hz: float = 50.0
    dt: float = 0.005                  # physics step [s]
    give_up_after: float | None = 10.0  # end the episode once the pad, after being tracked,
                                        # has been out of sight this long (None = never)


class PipelineObserver:
    """Hooks for front-ends. Override what you need; both are called inside step()."""

    def on_frame(self, t, img, det, truth):
        """A camera frame was rendered and run through the detector (det may be None)."""

    def on_estimates(self, t_img, det, outs, truth):
        """A detection cleared the latency; outs = {estimator name: Estimate}."""


class LandingPipeline:
    def __init__(self, world, cfg: PipelineConfig = None, detector=None, estimators=None,
                 controller=None, observers=()):
        self.world = world
        self.cfg = cfg or PipelineConfig()
        cam = world.camera
        self.detector = detector or PadDetector(cam.K, cam.dist, layout=world.cfg.pad)
        self.estimators = estimators if estimators is not None else {
            k: cls() for k, cls in ESTIMATORS.items()}
        self.controller = controller or LandingController()
        self.observers = list(observers)
        self.buf = DroneStateBuffer()
        self.pending = deque()           # (t_ready, t_img, det, truth at t_img)
        self.v_cmd, self.motors = np.zeros(3), True
        self.next_cam = self.next_ctrl = 0.0
        self.t_end = None                # world time the episode ended
        self.touched_down = False        # ended by contact (not by giving up)

    def step(self, command=None):
        """Advance one physics step.

        command = (v_cmd, motors_on) flies the drone directly and bypasses the controller
        (e.g. a position hold while measuring the estimators open loop).
        """
        w, c = self.world, self.cfg
        t = w.t
        p_fc, v_fc, R_fc = w.fc_estimate()
        self.buf.add(t, p_fc, v_fc, R_fc)

        if t >= self.next_cam - 1e-9:
            self.next_cam += 1.0 / c.cam_hz
            img = w.render()
            det = self.detector.detect(img)
            truth = w.truth()
            self.pending.append((t + c.latency, t, det, truth))
            for o in self.observers:
                o.on_frame(t, img, det, truth)

        # detections become available after the processing latency
        while self.pending and self.pending[0][0] <= t + 1e-9:
            _, t_img, det, truth = self.pending.popleft()
            if det is None:
                continue
            drone_img = self.buf.at(t_img)
            outs = {}
            for k, e in self.estimators.items():
                out = e.update(t_img, det, drone_img)
                if out is not None:
                    outs[k] = out
            for o in self.observers:
                o.on_estimates(t_img, det, outs, truth)

        if command is not None:
            self.v_cmd, self.motors = command
        elif t >= self.next_ctrl - 1e-9:
            self.next_ctrl += 1.0 / c.ctrl_hz
            est = self.estimators[c.mode].predict(t, self.buf.latest())
            if w.outcome is None:
                self.v_cmd, self.motors = self.controller.step(t, est, p_fc, v_fc)

        w.step(c.dt, self.v_cmd, self.motors)
        self._check_end(t)

    def _check_end(self, t):
        w, ctrl = self.world, self.controller
        if (self.cfg.give_up_after is not None and w.outcome is None
                and ctrl.state == State.SEARCH and ctrl.t_last_seen is not None
                and t - ctrl.t_last_seen > self.cfg.give_up_after):
            w.end('lost_target')
            self.t_end = w.t
        if w.outcome is not None and self.t_end is None:
            ctrl.touchdown(w.t)
            self.t_end = w.t
            self.touched_down = True
