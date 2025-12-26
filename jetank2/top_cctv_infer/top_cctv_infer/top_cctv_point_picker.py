#!/usr/bin/env python3
import argparse
import threading

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class TopCctvPointPicker(Node):
    def __init__(self, image_topic: str):
        super().__init__("top_cctv_point_picker")
        self.bridge = CvBridge()
        self.image_topic = image_topic
        self.latest_frame = None
        self.lock = threading.Lock()
        self.points = []
        self.sub = self.create_subscription(
            Image, self.image_topic, self.on_image, qos_profile_sensor_data
        )
        self.get_logger().info(f"Subscribed: {self.image_topic}")

    def on_image(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return
        with self.lock:
            self.latest_frame = frame

    def get_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def add_point(self, x: int, y: int) -> None:
        self.points.append((int(x), int(y)))
        self.print_points()

    def pop_point(self) -> None:
        if self.points:
            self.points.pop()
            self.print_points()

    def clear_points(self) -> None:
        self.points = []
        self.print_points()

    def print_points(self) -> None:
        if not self.points:
            self.get_logger().info("px_points cleared")
            return
        points_text = ";".join([f"{x},{y}" for x, y in self.points])
        self.get_logger().info(f'px_points="{points_text}"')


def main() -> None:
    parser = argparse.ArgumentParser(description="Top-CCTV point picker (pixel coords)")
    parser.add_argument("--image-topic", default="/jetank/top_cctv1")
    parser.add_argument("--window", default="top_cctv_point_picker")
    parser.add_argument("--max-points", type=int, default=0, help="0 = unlimited")
    parser.add_argument("--scale", type=float, default=1.0, help="display scale (<=1.0 recommended)")
    args = parser.parse_args()

    rclpy.init()
    node = TopCctvPointPicker(args.image_topic)

    window_name = args.window
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        scale = float(args.scale)
        if scale <= 0:
            scale = 1.0
        px = int(x / scale)
        py = int(y / scale)
        node.add_point(px, py)
        if args.max_points > 0 and len(node.points) >= args.max_points:
            rclpy.shutdown()

    cv2.setMouseCallback(window_name, on_mouse)

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            frame = node.get_frame()
            if frame is None:
                continue

            display = frame
            scale = float(args.scale)
            if scale > 0 and abs(scale - 1.0) > 1e-3:
                h, w = frame.shape[:2]
                display = cv2.resize(frame, (int(w * scale), int(h * scale)))

            for idx, (px, py) in enumerate(node.points):
                dx = int(px * scale)
                dy = int(py * scale)
                cv2.circle(display, (dx, dy), 4, (0, 0, 255), -1)
                cv2.putText(
                    display,
                    str(idx + 1),
                    (dx + 6, dy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("c"):
                node.clear_points()
            if key == ord("u"):
                node.pop_point()
    finally:
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()
        node.destroy_node()


if __name__ == "__main__":
    main()
