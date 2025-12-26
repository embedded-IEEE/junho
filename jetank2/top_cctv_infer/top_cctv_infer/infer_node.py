#!/usr/bin/env python3
import time
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose2D
from cv_bridge import CvBridge
from ultralytics import YOLO
import os

def abs_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))

class TopCctvInfer(Node):
    def __init__(self):
        super().__init__("top_cctv_infer")

        # ====== basic params ======
        self.declare_parameter("image_topic", "/jetank/top_cctv1")
        self.declare_parameter("annot_topic", "/top_cctv1/annotated")
        self.declare_parameter("pose_topic", "/top_cctv1/closest_pose")
        #self.declare_parameter("weights", "/home/jwson/models/rectangle_obb/best.pt")
        self.declare_parameter("weights", abs_path("~/jetank_ws/src/top_cctv_infer/top_cctv_infer/best.pt"))
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("device", "cuda:0")  # cpu

        # ====== selection params ======
        # ref_mode: left_middle | center | custom
        self.declare_parameter("ref_mode", "left_middle")
        self.declare_parameter("ref_x", 0.0)  # used only if ref_mode == custom
        self.declare_parameter("ref_y", 0.0)  # used only if ref_mode == custom

        # ====== ROI params ======
        self.declare_parameter("roi_enabled", True)
        self.declare_parameter("roi_mode", "auto")  # auto|on|off|param
        self.declare_parameter("roi_xmin_ratio", 0.22)
        self.declare_parameter("roi_xmax_ratio", 0.40)
        self.declare_parameter("roi_ymin_ratio", 0.42)
        self.declare_parameter("roi_ymax_ratio", 0.58)

        self.declare_parameter("target_class", -1)   # -1 = all classes
        self.declare_parameter("theta_unit", "rad")  # rad | deg

        # ====== topics ======
        image_topic = self.get_parameter("image_topic").value
        self.image_topic = image_topic
        self.annot_topic = self.get_parameter("annot_topic").value
        pose_topic = self.get_parameter("pose_topic").value

        weights = self.get_parameter("weights").value
        self.conf = float(self.get_parameter("conf").value)
        self.device = self.get_parameter("device").value

        self.bridge = CvBridge()
        self.model = YOLO(weights)

        self.sub = self.create_subscription(
            Image, image_topic, self.on_image, qos_profile_sensor_data
        )
        self.pub_annot = self.create_publisher(Image, self.annot_topic, 10)
        self.pub_pose = self.create_publisher(Pose2D, pose_topic, 10)

        self.last_t = time.time()
        self.cnt = 0

        self.get_logger().info(f"Subscribed: {image_topic}")
        self.get_logger().info(f"Annotated pub: {self.annot_topic}")
        self.get_logger().info(f"Weights: {weights} / device={self.device} / conf={self.conf}")

    def _compute_ref(self, frame):
        """Return (ref_x, ref_y) in pixel coordinates."""
        h, w = frame.shape[:2]
        ref_mode = str(self.get_parameter("ref_mode").value)

        if ref_mode == "left_middle":
            return 0.0, h / 2.0
        if ref_mode == "center":
            return w / 2.0, h / 2.0
        if ref_mode == "right_middle":
            return float(w), h / 2.0

        # custom
        return float(self.get_parameter("ref_x").value), float(self.get_parameter("ref_y").value)

    def _roi_bounds(self, frame_w: int, frame_h: int):
        xmin_r = float(self.get_parameter("roi_xmin_ratio").value)
        xmax_r = float(self.get_parameter("roi_xmax_ratio").value)
        ymin_r = float(self.get_parameter("roi_ymin_ratio").value)
        ymax_r = float(self.get_parameter("roi_ymax_ratio").value)
        x_min = frame_w * xmin_r
        x_max = frame_w * xmax_r
        y_min = frame_h * ymin_r
        y_max = frame_h * ymax_r
        if x_min > x_max:
            x_min, x_max = x_max, x_min
        if y_min > y_max:
            y_min, y_max = y_max, y_min
        return x_min, x_max, y_min, y_max

    def _roi_center(self, bounds):
        x_min, x_max, y_min, y_max = bounds
        return (x_min + x_max) / 2.0, (y_min + y_max) / 2.0

    def _resolve_roi_enabled(self) -> bool:
        mode = str(self.get_parameter("roi_mode").value).strip().lower()
        if mode == "auto":
            return "top_cctv2" in self.image_topic
        if mode == "on":
            return True
        if mode == "off":
            return False
        return bool(self.get_parameter("roi_enabled").value)

    def on_image(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge failed: {e}")
            return

        ref_x, ref_y = self._compute_ref(frame)
        roi_bounds = None
        roi_enabled = self._resolve_roi_enabled()
        if roi_enabled:
            roi_bounds = self._roi_bounds(frame.shape[1], frame.shape[0])
            ref_x, ref_y = self._roi_center(roi_bounds)
        target_class = int(self.get_parameter("target_class").value)
        theta_unit = str(self.get_parameter("theta_unit").value)

        # ---- inference (OBB) ----
        result = self.model.predict(
            frame, conf=self.conf, device=self.device, verbose=False
        )[0]

        obb = getattr(result, "obb", None)
        annotated = frame.copy()

        if roi_bounds is not None:
            x_min, x_max, y_min, y_max = roi_bounds
            cv2.rectangle(
                annotated,
                (int(x_min), int(y_min)),
                (int(x_max), int(y_max)),
                (0, 255, 255),
                2,
            )
        # 기준점(ROI 중심 or ref_mode)을 화면에 표시 (빨간 점)
        cv2.circle(annotated, (int(ref_x), int(ref_y)), 8, (0, 0, 255), -1)

        if obb is None or obb.xywhr is None or len(obb.xywhr) == 0:
            # 검출이 없으면 기준점만 표시하고 내보냄
            out = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            out.header = msg.header
            self.pub_annot.publish(out)
            return

        xywhr = obb.xywhr.cpu().numpy()  # (N,5) = [cx, cy, w, h, theta]
        cls = obb.cls.cpu().numpy().astype(int) if obb.cls is not None else None
        conf = obb.conf.cpu().numpy() if obb.conf is not None else None

        idxs = np.arange(xywhr.shape[0])
        if cls is not None and target_class >= 0:
            idxs = idxs[cls == target_class]

        if roi_bounds is not None and idxs.size > 0:
            x_min, x_max, y_min, y_max = roi_bounds
            centers = xywhr[idxs, 0:2]
            roi_mask = (
                (centers[:, 0] >= x_min)
                & (centers[:, 0] <= x_max)
                & (centers[:, 1] >= y_min)
                & (centers[:, 1] <= y_max)
            )
            idxs = idxs[roi_mask]

        if idxs.size == 0:
            # 대상 클래스가 없으면 기준점만 표시하고 내보냄
            out = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            out.header = msg.header
            self.pub_annot.publish(out)
            return

        centers = xywhr[idxs, 0:2]  # (K,2)
        d2 = (centers[:, 0] - ref_x) ** 2 + (centers[:, 1] - ref_y) ** 2
        best_k = int(np.argmin(d2))
        best_i = int(idxs[best_k])

        cx, cy, w, h, theta = xywhr[best_i].tolist()
        best_conf = float(conf[best_i]) if conf is not None else -1.0
        best_cls = int(cls[best_i]) if cls is not None else -1

        theta_out = float(theta)
        if theta_unit == "deg":
            theta_out = float(theta) * 180.0 / np.pi

        # ---- highlight only the closest OBB ----
        poly = obb.xyxyxyxy[best_i].cpu().numpy().reshape(4, 2).astype(np.int32)
        cv2.polylines(annotated, [poly], isClosed=True, color=(0, 255, 255), thickness=3)
        cv2.circle(annotated, (int(cx), int(cy)), 5, (0, 255, 255), -1)

        label = f"cls={best_cls} conf={best_conf:.2f} cx={cx:.1f} cy={cy:.1f} th={theta_out:.3f}{theta_unit}"
        cv2.putText(
            annotated,
            label,
            (int(cx) + 10, int(cy) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        # ---- publish pose ----
        pose = Pose2D()
        pose.x = float(cx)
        pose.y = float(cy)
        pose.theta = float(theta_out)
        self.pub_pose.publish(pose)

        # ---- publish annotated image ----
        out = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        out.header = msg.header
        self.pub_annot.publish(out)

        # ---- FPS log ----
        self.cnt += 1
        now = time.time()
        if now - self.last_t >= 1.0:
            self.get_logger().info(
                f"FPS: {self.cnt/(now-self.last_t):.1f} | Closest: ({cx:.1f},{cy:.1f}) theta={theta_out:.3f}{theta_unit}"
            )
            self.last_t = now
            self.cnt = 0


def main():
    rclpy.init()
    node = TopCctvInfer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
