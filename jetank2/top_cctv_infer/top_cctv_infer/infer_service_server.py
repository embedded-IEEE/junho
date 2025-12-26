#!/usr/bin/env python3
import time
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from ament_index_python.packages import get_package_share_directory

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

from top_cctv_interfaces.srv import GetClosestPose
import os

class InferServiceServer(Node):
    def __init__(
        self,
        node_name: str = "top_cctv_infer_service_server",
        image_topic: str = "/jetank/top_cctv1",
        service_name: str = "/top_cctv1/get_closest_pose",
    ):
        super().__init__(node_name)

        # ===== params =====
        self.declare_parameter("image_topic", image_topic)
        self.declare_parameter("service_name", service_name)
        #self.declare_parameter("weights", "/home/jwson/rectangle_obb/best.pt")
        self.declare_parameter(
            "weights",
            os.path.join(get_package_share_directory("top_cctv_infer"), "best.pt"),
        )
        self.declare_parameter("conf", 0.50)    
        self.declare_parameter("device", "cuda:0")  # cpu
        self.declare_parameter("ref_mode", "left_middle")  # left_middle|center|custom
        self.declare_parameter("ref_x", 0.0)
        self.declare_parameter("ref_y", 0.0)
        self.declare_parameter("theta_unit", "rad")  # rad|deg
        self.declare_parameter("debug", False)
        self.declare_parameter("debug_mode", "auto")  # auto|on|off|param
        self.declare_parameter("debug_topic", "")
        self.declare_parameter("debug_draw_all", False)
        self.declare_parameter("debug_draw_ref", True)
        self.declare_parameter("roi_enabled", True)
        self.declare_parameter("roi_mode", "auto")  # auto|on|off|param
        self.declare_parameter("roi_xmin_ratio", 0.22)
        self.declare_parameter("roi_xmax_ratio", 0.40)
        self.declare_parameter("roi_ymin_ratio", 0.42)
        self.declare_parameter("roi_ymax_ratio", 0.58)

        image_topic = self.get_parameter("image_topic").value
        service_name = self.get_parameter("service_name").value
        weights = self.get_parameter("weights").value
        self.conf = float(self.get_parameter("conf").value)
        self.device = self.get_parameter("device").value
        self.theta_unit = str(self.get_parameter("theta_unit").value)
        self.debug = self._resolve_debug_enabled(image_topic, service_name)
        self.debug_draw_all = bool(self.get_parameter("debug_draw_all").value)
        self.debug_draw_ref = bool(self.get_parameter("debug_draw_ref").value)
        self.roi_enabled = self._resolve_roi_enabled(image_topic, service_name)

        self.bridge = CvBridge()
        self.model = YOLO(weights)

        # 최신 결과 캐시 (class별로도 저장)
        # cache[target_class] = (found, x, y, theta, conf)
        self.cache = {}

        # 구독
        self.sub = self.create_subscription(
            Image, image_topic, self.on_image, qos_profile_sensor_data
        )

        # 서비스 서버
        self.srv = self.create_service(
            GetClosestPose, service_name, self.on_request
        )
        self.debug_pub = None
        if self.debug:
            debug_topic = str(self.get_parameter("debug_topic").value)
            if not debug_topic:
                debug_topic = self._default_debug_topic(image_topic, service_name)
            self.debug_pub = self.create_publisher(Image, debug_topic, 10)
            self.get_logger().info(f"Debug image: {debug_topic}")

        self.get_logger().info(f"Subscribed: {image_topic}")
        self.get_logger().info(f"Service: {service_name}")
        self.get_logger().info(f"ROI enabled: {self.roi_enabled}")
        self.get_logger().info(f"Debug enabled: {self.debug}")
        self.get_logger().info(f"Weights: {weights} / device={self.device} / conf={self.conf}")

    def _resolve_roi_enabled(self, image_topic: str, service_name: str) -> bool:
        mode = str(self.get_parameter("roi_mode").value).strip().lower()
        if mode == "auto":
            return ("top_cctv2" in image_topic) or ("top_cctv2" in service_name)
        if mode == "on":
            return True
        if mode == "off":
            return False
        return bool(self.get_parameter("roi_enabled").value)

    def _resolve_debug_enabled(self, image_topic: str, service_name: str) -> bool:
        mode = str(self.get_parameter("debug_mode").value).strip().lower()
        if mode == "auto":
            return ("top_cctv1" in image_topic) or ("top_cctv1" in service_name) or ("top_cctv2" in image_topic) or ("top_cctv2" in service_name)
        if mode == "on":
            return True
        if mode == "off":
            return False
        return bool(self.get_parameter("debug").value)

    def _default_debug_topic(self, image_topic: str, service_name: str) -> str:
        if "top_cctv2" in image_topic or "top_cctv2" in service_name:
            return "/top_cctv2/annotated"
        if "top_cctv1" in image_topic or "top_cctv1" in service_name:
            return "/top_cctv1/annotated"
        return "/top_cctv/annotated"

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

    def _compute_ref(self, frame):
        h, w = frame.shape[:2]
        ref_mode = str(self.get_parameter("ref_mode").value)
        if ref_mode == "left_middle":
            return 0.0, h / 2.0
        if ref_mode == "center":
            return w / 2.0, h / 2.0
        if ref_mode == "right_middle":
            return float(w), h / 2.0
        return float(self.get_parameter("ref_x").value), float(self.get_parameter("ref_y").value)

    def on_image(self, msg: Image):
        # 이미지 수신마다 최신 결과를 캐시
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return

        ref_x, ref_y = self._compute_ref(frame)
        roi_bounds = None
        roi_idxs = None
        if self.roi_enabled:
            roi_bounds = self._roi_bounds(frame.shape[1], frame.shape[0])
            ref_x, ref_y = self._roi_center(roi_bounds)
        annotated = None
        if self.debug_pub:
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
            if self.debug_draw_ref:
                cv2.circle(annotated, (int(ref_x), int(ref_y)), 6, (0, 0, 255), -1)

        result = self.model.predict(
            frame, conf=self.conf, device=self.device, verbose=False
        )[0]
        obb = getattr(result, "obb", None)
        if obb is None or obb.xywhr is None or len(obb.xywhr) == 0:
            # 어떤 클래스든 found=False 업데이트(요청시 없다고 응답)
            self.cache[-1] = (False, 0.0, 0.0, 0.0, 0.0)
            if annotated is not None:
                self._publish_debug(annotated, msg)
            return

        xywhr = obb.xywhr.cpu().numpy()  # [cx,cy,w,h,theta]
        cls = obb.cls.cpu().numpy().astype(int) if obb.cls is not None else None
        conf = obb.conf.cpu().numpy() if obb.conf is not None else None
        if roi_bounds is not None:
            x_min, x_max, y_min, y_max = roi_bounds
            cx = xywhr[:, 0]
            cy = xywhr[:, 1]
            roi_mask = (cx >= x_min) & (cx <= x_max) & (cy >= y_min) & (cy <= y_max)
            roi_idxs = np.where(roi_mask)[0]

        # -1(전체 대상)에 대한 closest도 계산
        self.cache[-1] = self._closest_for_class(
            xywhr, cls, conf, target_class=-1, ref_x=ref_x, ref_y=ref_y, valid_idxs=roi_idxs
        )

        # 등장한 클래스들에 대해서도 각각 closest 계산(원하면)
        if cls is not None:
            for c in np.unique(cls):
                self.cache[int(c)] = self._closest_for_class(
                    xywhr, cls, conf, target_class=int(c), ref_x=ref_x, ref_y=ref_y, valid_idxs=roi_idxs
                )

        if annotated is not None:
            self._draw_debug(annotated, obb, xywhr, cls, conf, ref_x, ref_y, valid_idxs=roi_idxs)
            self._publish_debug(annotated, msg)

    def _closest_for_class(self, xywhr, cls, conf, target_class, ref_x, ref_y, valid_idxs=None):
        if valid_idxs is None:
            idxs = np.arange(xywhr.shape[0])
        else:
            idxs = np.asarray(valid_idxs, dtype=np.int64)
        if cls is not None and target_class >= 0:
            idxs = idxs[cls[idxs] == target_class]
        if idxs.size == 0:
            return (False, 0.0, 0.0, 0.0, 0.0)

        centers = xywhr[idxs, 0:2]
        d2 = (centers[:, 0] - ref_x) ** 2 + (centers[:, 1] - ref_y) ** 2
        best_i = int(idxs[int(np.argmin(d2))])

        cx, cy, _, _, theta = xywhr[best_i].tolist()
        c = float(conf[best_i]) if conf is not None else 0.0

        theta_out = float(theta)
        if self.theta_unit == "deg":
            theta_out = float(theta) * 180.0 / np.pi

        return (True, float(cx), float(cy), float(theta_out), float(c))

    def _draw_debug(self, annotated, obb, xywhr, cls, conf, ref_x, ref_y, valid_idxs=None):
        if valid_idxs is None:
            idxs = np.arange(xywhr.shape[0])
        else:
            idxs = np.asarray(valid_idxs, dtype=np.int64)
        if self.debug_draw_all and obb.xyxyxyxy is not None and len(obb.xyxyxyxy) > 0:
            for i in idxs:
                poly = obb.xyxyxyxy[i].cpu().numpy().reshape(4, 2).astype(np.int32)
                cv2.polylines(annotated, [poly], isClosed=True, color=(255, 0, 0), thickness=1)

        if idxs.size == 0:
            return

        centers = xywhr[idxs, 0:2]
        d2 = (centers[:, 0] - ref_x) ** 2 + (centers[:, 1] - ref_y) ** 2
        best_k = int(np.argmin(d2))
        best_i = int(idxs[best_k])

        if obb.xyxyxyxy is not None and len(obb.xyxyxyxy) > 0:
            poly = obb.xyxyxyxy[best_i].cpu().numpy().reshape(4, 2).astype(np.int32)
            cv2.polylines(annotated, [poly], isClosed=True, color=(0, 255, 255), thickness=2)

        cx, cy, _, _, theta = xywhr[best_i].tolist()
        best_conf = float(conf[best_i]) if conf is not None else -1.0
        best_cls = int(cls[best_i]) if cls is not None else -1

        label = f"cls={best_cls} conf={best_conf:.2f} cx={cx:.1f} cy={cy:.1f} th={theta:.3f}"
        cv2.circle(annotated, (int(cx), int(cy)), 4, (0, 255, 255), -1)
        cv2.putText(
            annotated,
            label,
            (int(cx) + 8, int(cy) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
        )

    def _publish_debug(self, annotated, msg: Image) -> None:
        if self.debug_pub is None:
            return
        out = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        out.header = msg.header
        self.debug_pub.publish(out)

    def on_request(self, request, response):
        # 요청 클래스에 맞는 최신 캐시값 반환
        key = int(request.target_class)
        if key not in self.cache:
            # 해당 클래스 캐시가 없으면 전체(-1)로 fallback
            key = -1

        found, x, y, theta, conf = self.cache.get(key, (False, 0.0, 0.0, 0.0, 0.0))
        response.found = bool(found)
        response.x = float(x)
        response.y = float(y)
        response.theta = float(theta)
        response.conf = float(conf)
        return response


def main():
    rclpy.init()
    node = InferServiceServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
