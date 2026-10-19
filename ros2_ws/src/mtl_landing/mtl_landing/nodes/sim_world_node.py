"""Simulated world: physics, ground vehicle, rendered downward camera, /clock.

Publishes
  /clock                     rosgraph_msgs/Clock      (this node is the time source)
  /camera/image_raw          sensor_msgs/Image        mono8, camera_hz
  /camera/camera_info        sensor_msgs/CameraInfo
  /drone/odom                nav_msgs/Odometry        flight-controller estimate (noisy);
                                                      twist is in the world frame
  /truth/drone, /truth/target nav_msgs/Odometry       ground truth, for logging only
  /sim/outcome               std_msgs/String          JSON, once, on touchdown/crash
  /tf                        world->base_link, world->pad
Subscribes
  /drone/cmd_vel             geometry_msgs/Twist      world-frame velocity setpoint
  /drone/motors              std_msgs/Bool            False = cut motors
"""
import json

import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String
from tf2_ros import TransformBroadcaster

from mtl_landing.core.geometry import R_to_quat
from mtl_landing.core.world import World, WorldConfig


def to_time(t):
    sec = int(t)
    return Time(sec=sec, nanosec=int(round((t - sec) * 1e9)))


def fill_odom(msg, stamp, frame, child, p, v, R):
    msg.header.stamp = stamp
    msg.header.frame_id = frame
    msg.child_frame_id = child
    msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z = map(float, p)
    q = R_to_quat(R)
    o = msg.pose.pose.orientation
    o.x, o.y, o.z, o.w = map(float, q)
    msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z = map(float, v)
    return msg


class SimWorld(Node):
    def __init__(self):
        super().__init__('sim_world')
        self.declare_parameter('scenario', 'straight')
        self.declare_parameter('seed', 0)
        self.declare_parameter('wind_sigma', 0.6)
        self.declare_parameter('physics_dt', 0.005)
        self.declare_parameter('substeps', 2)
        self.declare_parameter('camera_hz', 30.0)
        self.declare_parameter('realtime_factor', 1.0)
        gp = lambda n: self.get_parameter(n).value  # noqa: E731

        self.world = World(WorldConfig(scenario=gp('scenario'), seed=int(gp('seed')),
                                       wind_sigma=float(gp('wind_sigma'))))
        self.dt = float(gp('physics_dt'))
        self.substeps = int(gp('substeps'))
        self.cam_period = 1.0 / float(gp('camera_hz'))
        self.next_cam = 0.0
        self.v_cmd = np.zeros(3)
        self.motors = True
        self.outcome_sent = False

        img_qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.pub_clock = self.create_publisher(Clock, '/clock', 10)
        self.pub_img = self.create_publisher(Image, '/camera/image_raw', img_qos)
        self.pub_info = self.create_publisher(CameraInfo, '/camera/camera_info', 10)
        self.pub_odom = self.create_publisher(Odometry, '/drone/odom', 20)
        self.pub_tdrone = self.create_publisher(Odometry, '/truth/drone', 20)
        self.pub_ttarget = self.create_publisher(Odometry, '/truth/target', 20)
        self.pub_outcome = self.create_publisher(String, '/sim/outcome', 10)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(Twist, '/drone/cmd_vel', self._on_cmd, 10)
        self.create_subscription(Bool, '/drone/motors', self._on_motors, 10)

        cam = self.world.camera
        self.info = CameraInfo(width=cam.width, height=cam.height, distortion_model='plumb_bob')
        self.info.header.frame_id = 'camera'
        self.info.k = [float(x) for x in cam.K.flatten()]
        self.info.d = [0.0] * 5
        self.info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.info.p = [float(cam.K[0, 0]), 0.0, float(cam.K[0, 2]), 0.0,
                       0.0, float(cam.K[1, 1]), float(cam.K[1, 2]), 0.0, 0.0, 0.0, 1.0, 0.0]

        period = self.dt * self.substeps / float(gp('realtime_factor'))
        self.create_timer(period, self._tick)
        self.get_logger().info(
            f"scenario={gp('scenario')} ({self.world.vehicle.description}), seed={gp('seed')}")

    def _on_cmd(self, msg):
        self.v_cmd = np.array([msg.linear.x, msg.linear.y, msg.linear.z])

    def _on_motors(self, msg):
        self.motors = bool(msg.data)

    def _tick(self):
        w = self.world
        for _ in range(self.substeps):
            w.step(self.dt, self.v_cmd, self.motors)
        stamp = to_time(w.t)
        self.pub_clock.publish(Clock(clock=stamp))

        p, v, R = w.fc_estimate()
        self.pub_odom.publish(fill_odom(Odometry(), stamp, 'world', 'base_link', p, v, R))
        s, veh = w.quad.s, w.vehicle
        self.pub_tdrone.publish(fill_odom(Odometry(), stamp, 'world', 'base_link_truth', s.p, s.v, s.R))
        self.pub_ttarget.publish(fill_odom(Odometry(), stamp, 'world', 'pad', veh.p, veh.v, veh.R_deck))
        self._tf(stamp, 'base_link', s.p, s.R)
        self._tf(stamp, 'pad', veh.p, veh.R_deck)

        if w.t >= self.next_cam - 1e-9:
            self.next_cam += self.cam_period
            img = w.render()
            m = Image(height=img.shape[0], width=img.shape[1], encoding='mono8',
                      is_bigendian=0, step=img.shape[1], data=img.tobytes())
            m.header.stamp = stamp
            m.header.frame_id = 'camera'
            self.info.header.stamp = stamp
            self.pub_img.publish(m)
            self.pub_info.publish(self.info)

        if w.outcome is not None and not self.outcome_sent:
            self.outcome_sent = True
            self.pub_outcome.publish(String(data=json.dumps(w.outcome, default=float)))
            self.get_logger().info(f'OUTCOME {w.outcome}')

    def _tf(self, stamp, child, p, R):
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'world'
        t.child_frame_id = child
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = map(float, p)
        q = R_to_quat(R)
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = map(float, q)
        self.tf.sendTransform(t)


def main():
    rclpy.init()
    node = SimWorld()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
