"""Relative-state estimator of the moving pad.

Runs the printed-formula, plain-distance, tilt-compensated and filtered estimators side by side on
every detection; `mode` selects which one drives the controller.

Subscribes /pad/pose (camera frame), /drone/odom (flight-controller state).
Publishes
  /target/estimate         std_msgs/String    JSON {stamp, estimate}: the full Estimate the
                                              controller flies on (core.state.Estimate.to_dict)
  /target/relative         nav_msgs/Odometry  pose.position = pad - drone (x,y world-aligned),
                                              z = -camera height; twist.linear = relative vel.
                                              For RViz and other consumers.
  /target/ground_velocity  geometry_msgs/Vector3Stamped (when the estimator tracks it)
  /estimator/comparison    std_msgs/String    JSON with every estimator per detection
"""
import json

import cv2
import numpy as np
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String

from mtl_landing.core.detector import PadDetection
from mtl_landing.core.estimation import ESTIMATORS, DroneStateBuffer
from mtl_landing.core.geometry import R_to_quat, quat_to_R, rot_z
from mtl_landing.core.state import rounded
from mtl_landing.nodes.common import spin, stamp_s


class TargetEstimatorNode(Node):
    def __init__(self):
        super().__init__('target_estimator')
        self.declare_parameter('mode', 'corrected')
        self.declare_parameter('publish_hz', 50.0)
        self.mode = self.get_parameter('mode').value
        self.ests = {k: cls() for k, cls in ESTIMATORS.items()}
        self.buf = DroneStateBuffer()
        self.create_subscription(Odometry, '/drone/odom', self._on_odom, 50)
        self.create_subscription(PoseStamped, '/pad/pose', self._on_pose, 10)
        self.pub_est = self.create_publisher(String, '/target/estimate', 10)
        self.pub_rel = self.create_publisher(Odometry, '/target/relative', 10)
        self.pub_gv = self.create_publisher(Vector3Stamped, '/target/ground_velocity', 10)
        self.pub_cmp = self.create_publisher(String, '/estimator/comparison', 10)
        self.create_timer(1.0 / float(self.get_parameter('publish_hz').value), self._publish)
        self.get_logger().info(f'estimator mode = {self.mode}')

    def _on_odom(self, m):
        p = m.pose.pose.position
        o = m.pose.pose.orientation
        v = m.twist.twist.linear
        self.buf.add(stamp_s(m.header.stamp), [p.x, p.y, p.z], [v.x, v.y, v.z],
                     quat_to_R([o.x, o.y, o.z, o.w]))

    def _on_pose(self, m):
        t = stamp_s(m.header.stamp)
        q = m.pose.orientation
        rvec, _ = cv2.Rodrigues(quat_to_R([q.x, q.y, q.z, q.w]))
        det = PadDetection(rvec=rvec.reshape(3),
                           tvec=np.array([m.pose.position.x, m.pose.position.y, m.pose.position.z]))
        drone = self.buf.at(t)
        row = dict(t=round(t, 4), tvec=rounded(det.tvec))
        for k, e in self.ests.items():
            out = e.update(t, det, drone)
            if out is not None:
                row[k] = out.summary()
        self.pub_cmp.publish(String(data=json.dumps(row)))

    def _publish(self):
        drone = self.buf.latest()
        if drone is None:
            return
        est = self.ests[self.mode].predict(drone.t, drone)
        if est is None:
            return
        now = self.get_clock().now()
        self.pub_est.publish(String(data=json.dumps(dict(
            stamp=now.nanoseconds * 1e-9, estimate=est.to_dict()))))
        m = Odometry()
        m.header.stamp = now.to_msg()
        m.header.frame_id = 'base_level'
        m.child_frame_id = 'pad'
        rp = np.nan_to_num(est.rel_pos)
        m.pose.pose.position.x, m.pose.pose.position.y = float(rp[0]), float(rp[1])
        m.pose.pose.position.z = -float(est.altitude)
        if np.isfinite(est.pad_yaw):
            q = R_to_quat(rot_z(est.pad_yaw))
            m.pose.pose.orientation.x, m.pose.pose.orientation.y, m.pose.pose.orientation.z, m.pose.pose.orientation.w = map(float, q)
        m.twist.twist.linear.x, m.twist.twist.linear.y, m.twist.twist.linear.z = map(float, est.rel_vel)
        self.pub_rel.publish(m)
        tv = est.tgt_vel
        if np.all(np.isfinite(tv)):
            g = Vector3Stamped()
            g.header = m.header
            g.header.frame_id = 'world'
            g.vector.x, g.vector.y, g.vector.z = map(float, tv)
            self.pub_gv.publish(g)


def main():
    spin(TargetEstimatorNode)


if __name__ == '__main__':
    main()
