"""Helpers shared by the ROS nodes."""
import rclpy


def stamp_s(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def spin(node_cls):
    """Run one node until Ctrl-C (or SystemExit from the node), then shut down cleanly."""
    rclpy.init()
    node = node_cls()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
