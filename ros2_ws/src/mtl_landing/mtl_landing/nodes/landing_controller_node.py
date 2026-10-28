"""Landing state machine node (paper Sec. III-D).

Subscribes /target/estimate, /drone/odom, /sim/outcome.
Publishes  /drone/cmd_vel (Twist, world frame), /drone/motors (Bool), /landing/state (String).
"""
import json

import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool, String

from mtl_landing.core.controller import LandingController, LandingParams
from mtl_landing.core.state import Estimate
from mtl_landing.nodes.common import spin


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
        self.est = None              # (publish time, Estimate)
        self.drone = None
        self.create_subscription(String, '/target/estimate', self._on_est, 10)
        self.create_subscription(Odometry, '/drone/odom', self._on_odom, 20)
        self.create_subscription(String, '/sim/outcome', self._on_outcome, 10)
        self.pub_cmd = self.create_publisher(Twist, '/drone/cmd_vel', 10)
        self.pub_motors = self.create_publisher(Bool, '/drone/motors', 10)
        self.pub_state = self.create_publisher(String, '/landing/state', 10)
        self.create_timer(1.0 / float(self.get_parameter('rate_hz').value), self._tick)
        self.last_state = None

    def _on_est(self, m):
        d = json.loads(m.data)
        self.est = (float(d['stamp']), Estimate.from_dict(d['estimate']))

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
        # drop the estimate if the estimator node has stopped publishing
        est = self.est[1] if self.est is not None and t - self.est[0] < self.timeout else None
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
    spin(LandingControllerNode)


if __name__ == '__main__':
    main()
