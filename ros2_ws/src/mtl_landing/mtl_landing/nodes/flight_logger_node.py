"""Records a run for offline validation, then exits (which ends the launch).

Writes <output_dir>/ros_run_<scenario>_<mode>.json with the 20 Hz truth/state log, every
estimator comparison row, the detections, the outcome, and a few debug camera frames
(<output_dir>/frames_<scenario>_<mode>/*.jpg).
"""
import json
import os

import cv2
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String


def stamp_s(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def odom_arrays(m):
    p, v = m.pose.pose.position, m.twist.twist.linear
    return [round(p.x, 4), round(p.y, 4), round(p.z, 4)], [round(v.x, 4), round(v.y, 4), round(v.z, 4)]


class FlightLogger(Node):
    def __init__(self):
        super().__init__('flight_logger')
        self.declare_parameter('output_dir', '/results')
        self.declare_parameter('scenario', 'straight')
        self.declare_parameter('mode', 'corrected')
        self.declare_parameter('t_max', 90.0)
        self.declare_parameter('frame_every', 0.25)
        gp = lambda n: self.get_parameter(n).value  # noqa: E731
        self.out_dir = gp('output_dir')
        self.tag = f"{gp('scenario')}_{gp('mode')}"
        self.t_max = float(gp('t_max'))
        self.frame_every = float(gp('frame_every'))
        self.frames_dir = os.path.join(self.out_dir, f'frames_{self.tag}')
        os.makedirs(self.frames_dir, exist_ok=True)
        self.meta = dict(scenario=gp('scenario'), mode=gp('mode'), source='ros2')

        self.drone = self.target = None
        self.state = {}
        self.log, self.cmp, self.dets, self.frames = [], [], [], []
        self.outcome = None
        self.t_outcome = None
        self.next_frame = 0.0
        self.done = False

        self.create_subscription(Odometry, '/truth/drone', lambda m: setattr(self, 'drone', m), 20)
        self.create_subscription(Odometry, '/truth/target', lambda m: setattr(self, 'target', m), 20)
        self.create_subscription(String, '/landing/state', lambda m: setattr(self, 'state', json.loads(m.data)), 20)
        self.create_subscription(String, '/estimator/comparison', lambda m: self.cmp.append(
            self._with_truth(json.loads(m.data))), 50)
        self.create_subscription(String, '/pad/detection', lambda m: self.dets.append(json.loads(m.data)), 50)
        self.create_subscription(String, '/sim/outcome', self._on_outcome, 10)
        self.create_subscription(Image, '/camera/debug', self._on_debug,
                                 QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.create_timer(0.05, self._tick)

    def _with_truth(self, row):
        if self.drone is not None and self.target is not None:
            dp, dv = odom_arrays(self.drone)
            tp, tv = odom_arrays(self.target)
            rel_v = np.subtract(tv, dv)
            row['truth'] = dict(rel_p=np.round(np.subtract(tp, dp), 4).tolist(),
                                rel_v=np.round(rel_v, 4).tolist(),
                                rel_speed=round(float(np.hypot(*rel_v[:2])), 4),
                                tgt_v=tv, tgt_speed=round(float(np.hypot(*tv[:2])), 4),
                                cam_height=round(dp[2] - 0.05 - tp[2], 4))
        return row

    def _on_outcome(self, m):
        self.outcome = json.loads(m.data)
        self.t_outcome = self.get_clock().now().nanoseconds * 1e-9

    def _on_debug(self, m):
        t = stamp_s(m.header.stamp)
        if t < self.next_frame:
            return
        self.next_frame = t + self.frame_every
        img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
        small = cv2.resize(img, (320, int(320 * m.height / m.width)), interpolation=cv2.INTER_AREA)
        name = f'{len(self.frames):04d}.jpg'
        cv2.imwrite(os.path.join(self.frames_dir, name), small, [cv2.IMWRITE_JPEG_QUALITY, 70])
        self.frames.append(dict(t=round(t, 3), file=name))

    def _tick(self):
        if self.done or self.drone is None or self.target is None:
            return
        t = self.get_clock().now().nanoseconds * 1e-9
        dp, dv = odom_arrays(self.drone)
        tp, tv = odom_arrays(self.target)
        q = self.drone.pose.pose.orientation
        self.log.append(dict(t=round(t, 3), state=self.state.get('state', 'INIT'),
                             aborts=self.state.get('aborts', 0), impact=self.state.get('impact'),
                             dp=dp, dv=dv, tp=tp, tv=tv, q=[round(q.x, 4), round(q.y, 4), round(q.z, 4), round(q.w, 4)],
                             hcam=round(dp[2] - 0.05 - tp[2], 3)))
        if (self.t_outcome is not None and t - self.t_outcome > 1.5) or t > self.t_max:
            self._write()

    def _write(self):
        self.done = True
        out = dict(meta=self.meta, outcome=self.outcome or dict(result='timeout'),
                   log=self.log, est_log=self.cmp, detections=self.dets, frames=self.frames)
        path = os.path.join(self.out_dir, f'ros_run_{self.tag}.json')
        with open(path, 'w') as fh:
            json.dump(out, fh)
        self.get_logger().info(f'wrote {path}: outcome={out["outcome"]}, '
                               f'{len(self.log)} log rows, {len(self.cmp)} estimates, {len(self.frames)} frames')
        raise SystemExit(0)


def main():
    rclpy.init()
    node = FlightLogger()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
