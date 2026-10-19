"""Landing state machine node (paper Sec. III-D).

Subscribes /target/relative, /target/ground_velocity, /drone/odom, /sim/outcome.
Publishes  /drone/cmd_vel (Twist, world frame), /drone/motors (Bool), /landing/state (String).
"""
import json

import numpy as np
import rclpy
from geometry_msgs.msg import Twist, Vector3Stamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool, String

from mtl_landing.core.controller import LandingController, LandingParams


def stamp_s(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class LandingControllerNode(Node):
    def __init__(self):
        super().__init__('landing_controller')
        self.declare_parameter('rate_hz', 50.0)
        self.declare_parameter('estimate_timeout', 0.3)
        defaults = LandingParams()
        for k, v in vars(defaults).items():
            self.declare_parameter(k, float(v))
        prm = LandingParams(**{k: float(self.get_parameter(k).value) for k in vars(defaults)})
        self.ctrl = LandingController(prm)
        self.timeout = float(self.get_parameter('estimate_timeout').value)
        self.rel = None
        self.gv = None
        self.drone = None
        self.create_subscription(Odometry, '/target/relative', self._on_rel, 10)
        self.create_subscription(Vector3Stamped, '/target/ground_velocity', self._on_gv, 10)
        self.create_subscription(Odometry, '/drone/odom', self._on_odom, 20)
        self.create_subscription(String, '/sim/outcome', self._on_outcome, 10)
        self.pub_cmd = self.create_publisher(Twist, '/drone/cmd_vel', 10)
        self.pub_motors = self.create_publisher(Bool, '/drone/motors', 10)
        self.pub_state = self.create_publisher(String, '/landing/state', 10)
        self.create_timer(1.0 / float(self.get_parameter('rate_hz').value), self._tick)
        self.last_state = None

    def _on_rel(self, m):
        self.rel = m

    def _on_gv(self, m):
        self.gv = m

    def _on_odom(self, m):
        p, v = m.pose.pose.position, m.twist.twist.linear
        self.drone = (np.array([p.x, p.y, p.z]), np.array([v.x, v.y, v.z]))

    def _on_outcome(self, m):
        self.ctrl.touchdown(self.get_clock().now().nanoseconds * 1e-9)
        self.get_logger().info(f'touchdown: {json.loads(m.data)}')

    def _tick(self):
        if self.drone is None:
            return
        t = self.get_clock().now().nanoseconds * 1e-9
        est = None
        if self.rel is not None and t - stamp_s(self.rel.header.stamp) < self.timeout:
            pp, tw = self.rel.pose.pose.position, self.rel.twist.twist.linear
            est = dict(valid=True, rel_pos=np.array([pp.x, pp.y, pp.z]),
                       altitude=-pp.z, rel_vel=np.array([tw.x, tw.y, tw.z]), tgt_vel=None)
            if self.gv is not None and abs(stamp_s(self.gv.header.stamp) - stamp_s(self.rel.header.stamp)) < 0.05:
                est['tgt_vel'] = np.array([self.gv.vector.x, self.gv.vector.y, self.gv.vector.z])
        v_cmd, motors = self.ctrl.step(t, est, *self.drone)
        cmd = Twist()
        cmd.linear.x, cmd.linear.y, cmd.linear.z = map(float, v_cmd)
        self.pub_cmd.publish(cmd)
        self.pub_motors.publish(Bool(data=bool(motors)))
        info = dict(state=self.ctrl.state, aborts=self.ctrl.n_aborts)
        if self.ctrl.predicted_impact is not None:
            info['impact'] = [round(float(x), 3) for x in self.ctrl.predicted_impact]
        self.pub_state.publish(String(data=json.dumps(info)))
        if self.ctrl.state != self.last_state:
            self.get_logger().info(f't={t:6.2f}  -> {self.ctrl.state}')
            self.last_state = self.ctrl.state


def main():
    rclpy.init()
    node = LandingControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
