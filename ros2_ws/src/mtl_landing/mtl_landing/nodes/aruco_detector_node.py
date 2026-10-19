"""ArUco pad detector (paper Sec. III-B/C).

Subscribes /camera/image_raw + /camera/camera_info.
Publishes
  /pad/pose          geometry_msgs/PoseStamped  pad centre in the camera frame (tvec, rvec),
                                                stamped with the image time
  /pad/detection     std_msgs/String            JSON: marker ids, primary id, reprojection error
  /camera/debug      sensor_msgs/Image          bgr8 overlay (marker outlines + pad axes)
"""
import json

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from mtl_landing.core.detector import PadDetector, draw_detection
from mtl_landing.core.geometry import R_to_quat


class ArucoDetectorNode(Node):
    def __init__(self):
        super().__init__('aruco_detector')
        self.declare_parameter('debug_every', 3)
        self.debug_every = int(self.get_parameter('debug_every').value)
        self.detector = None
        self.n = 0
        qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(CameraInfo, '/camera/camera_info', self._on_info, 10)
        self.create_subscription(Image, '/camera/image_raw', self._on_image, qos)
        self.pub_pose = self.create_publisher(PoseStamped, '/pad/pose', 10)
        self.pub_det = self.create_publisher(String, '/pad/detection', 10)
        self.pub_dbg = self.create_publisher(Image, '/camera/debug', qos)

    def _on_info(self, msg):
        if self.detector is None:
            K = np.array(msg.k, float).reshape(3, 3)
            self.detector = PadDetector(K, np.array(msg.d, float) if msg.d else None)
            self.get_logger().info(f'camera fx={K[0, 0]:.1f}, detector ready')

    def _on_image(self, msg):
        if self.detector is None:
            return
        gray = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width]
        det = self.detector.detect(gray)
        self.n += 1
        if det is not None:
            p = PoseStamped()
            p.header = msg.header
            p.pose.position.x, p.pose.position.y, p.pose.position.z = map(float, det.tvec)
            R, _ = cv2.Rodrigues(det.rvec)
            q = R_to_quat(R)
            p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z, p.pose.orientation.w = map(float, q)
            self.pub_pose.publish(p)
            self.pub_det.publish(String(data=json.dumps(dict(
                t=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                ids=det.marker_ids, primary=det.primary_id,
                reproj=round(det.reproj_err_px, 3),
                tvec_single=[round(float(x), 4) for x in det.tvec_single]))))
        if self.debug_every and self.n % self.debug_every == 0 and self.pub_dbg.get_subscription_count():
            dbg = draw_detection(gray, det, self.detector.K)
            m = Image(height=dbg.shape[0], width=dbg.shape[1], encoding='bgr8', is_bigendian=0,
                      step=dbg.shape[1] * 3, data=dbg.tobytes())
            m.header = msg.header
            self.pub_dbg.publish(m)


def main():
    rclpy.init()
    node = ArucoDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
