#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from top_cctv_interfaces.srv import GetClosestPose


class DummyClient(Node):
    def __init__(self):
        super().__init__("top_cctv_dummy_client")
        self.declare_parameter("service_name", "/top_cctv1/get_closest_pose")
        service_name = self.get_parameter("service_name").value
        self.cli = self.create_client(GetClosestPose, service_name)
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("waiting for service...")

        self.declare_parameter("target_class", -1)
        target_class = int(self.get_parameter("target_class").value)

        req = GetClosestPose.Request()
        req.target_class = target_class

        future = self.cli.call_async(req)
        future.add_done_callback(self.on_done)

    def on_done(self, future):
        try:
            res = future.result()
            if res.found:
                self.get_logger().info(
                    f"FOUND: x={res.x:.1f}, y={res.y:.1f}, theta={res.theta:.3f}, conf={res.conf:.2f}"
                )
            else:
                self.get_logger().info("NOT FOUND")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")
        finally:
            rclpy.shutdown()


def main():
    rclpy.init()
    node = DummyClient()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
