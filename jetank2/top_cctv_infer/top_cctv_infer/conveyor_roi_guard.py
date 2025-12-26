#!/usr/bin/env python3
import os
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from ament_index_python.packages import get_package_share_directory

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

from std_srvs.srv import SetBool


class ConveyorRoiGuard(Node):
    """
    기존 TopCctvInfer / InferServiceServer 패턴을 유지:
    - Image 구독
    - YOLO(OBB) 추론
    - 최신 결과(ROI hit 여부) 캐시
    - 상태 전환 시에만 /conveyor/power(SetBool) 호출
    """

    def __init__(self):
        super().__init__("conveyor_roi_guard")

        # ===== params (기존 노드 스타일 유지) =====
        self.declare_parameter("infer_every_n", 6)  # 30Hz면 6 → 5Hz 추론
        self.frame_count = 0
        self.declare_parameter("image_topic", "/jetank/top_cctv2")
        self.declare_parameter(
            "weights",
            os.path.join(get_package_share_directory("top_cctv_infer"), "best.pt"),
        )
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("device", "cuda:0")  # or "cpu"

        # ROI 설정: 중앙 절반의 좌측 = [W/4, W/2]
        self.declare_parameter("roi_xmin_ratio", 0.25)
        self.declare_parameter("roi_xmax_ratio", 0.50)

        # 클래스 필터(-1=all) / 디바운스 / 오검출 방지
        self.declare_parameter("target_class", -1)
        self.declare_parameter("min_area", 0)           # w*h(픽셀^2) 너무 작은 박스 제거
        self.declare_parameter("stop_consecutive", 2)   # ROI hit 연속 N프레임이면 정지
        self.declare_parameter("start_consecutive", 3)  # ROI miss 연속 N프레임이면 재개

        # ===== load params =====
        self.image_topic = self.get_parameter("image_topic").value
        weights = self.get_parameter("weights").value
        self.conf = float(self.get_parameter("conf").value)
        self.device = self.get_parameter("device").value

        self.bridge = CvBridge()
        self.model = YOLO(weights)

        # ===== subscriber =====
        self.sub = self.create_subscription(
            Image, self.image_topic, self.on_image, qos_profile_sensor_data
        )

        # ===== service client (/conveyor/power) =====
        self.cli = self.create_client(SetBool, "/conveyor/power")
        # 서비스가 늦게 뜰 수 있으니 on_image에서 readiness 체크하고 호출

        # ===== state/cache =====
        self.cache_roi_hit = False
        self.stop_count = 0
        self.start_count = 0
        self.power_on = None  # unknown(None) / True / False
        self.last_call_t = 0.0

        self.get_logger().info(f"Subscribed: {self.image_topic}")
        self.get_logger().info(f"Weights: {weights} / device={self.device} / conf={self.conf}")
        self.get_logger().info("ROI: x in [W*roi_xmin_ratio, W*roi_xmax_ratio] within full image width")

    def _roi_hit_any(self, frame_w: int, frame_h: int, centers_xy: np.ndarray) -> bool:
        xmin_r = float(self.get_parameter("roi_xmin_ratio").value)
        xmax_r = float(self.get_parameter("roi_xmax_ratio").value)
        x_min = frame_w * xmin_r
        x_max = frame_w * xmax_r

        # centers_xy: (N,2) = [cx, cy]
        if centers_xy.size == 0:
            return False

        cx = centers_xy[:, 0]
        cy = centers_xy[:, 1]
        hit = (cx >= x_min) & (cx <= x_max) & (cy >= 0) & (cy <= frame_h)
        return bool(np.any(hit))

    def _call_conveyor_power(self, on: bool):
        # 이미 원하는 상태면 호출 생략
        if self.power_on is not None and self.power_on == on:
            return

        # 과도 호출 방지
        now = time.time()
        if now - self.last_call_t < 0.2:
            return

        if not self.cli.service_is_ready():
            # 서비스가 아직 안 뜬 상태
            return

        req = SetBool.Request()
        req.data = bool(on)

        fut = self.cli.call_async(req)
        self.last_call_t = now

        def _done_cb(f):
            try:
                resp = f.result()
                if resp is not None and resp.success:
                    self.power_on = on
                    self.get_logger().info(f"/conveyor/power -> {on} (ok) msg={resp.message}")
                else:
                    self.get_logger().warn(f"/conveyor/power -> {on} (fail) msg={resp.message if resp else 'None'}")
            except Exception as e:
                self.get_logger().error(f"service call exception: {e}")

        fut.add_done_callback(_done_cb)

    def on_image(self, msg: Image):
        self.frame_count += 1
        n = int(self.get_parameter("infer_every_n").value)
        if n > 1 and (self.frame_count % n) != 0:
            return
        # 1) image
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return

        h, w = frame.shape[:2]

        # 2) inference (OBB) - 기존 노드들과 동일 방식 
        result = self.model.predict(frame, conf=self.conf, device=self.device, verbose=False)[0]
        obb = getattr(result, "obb", None)

        roi_hit = False

        if obb is not None and obb.xywhr is not None and len(obb.xywhr) > 0:
            xywhr = obb.xywhr.cpu().numpy()  # (N,5) [cx,cy,w,h,theta]
            cls = obb.cls.cpu().numpy().astype(int) if obb.cls is not None else None

            target_class = int(self.get_parameter("target_class").value)
            min_area = int(self.get_parameter("min_area").value)

            idxs = np.arange(xywhr.shape[0])

            if cls is not None and target_class >= 0:
                idxs = idxs[cls == target_class]

            if idxs.size > 0:
                # area filter
                ww = xywhr[idxs, 2]
                hh = xywhr[idxs, 3]
                area = ww * hh
                if min_area > 0:
                    idxs = idxs[area >= float(min_area)]

            if idxs.size > 0:
                centers = xywhr[idxs, 0:2]  # (K,2)
                roi_hit = self._roi_hit_any(w, h, centers)

        # 3) cache
        self.cache_roi_hit = roi_hit

        # 4) debounce -> service call
        stop_n = int(self.get_parameter("stop_consecutive").value)
        start_n = int(self.get_parameter("start_consecutive").value)

        if roi_hit:
            self.stop_count += 1
            self.start_count = 0
        else:
            self.start_count += 1
            self.stop_count = 0

        # ROI hit => stop(false)
        if self.stop_count >= stop_n:
            self._call_conveyor_power(False)

        # ROI miss => start(true)
        if self.start_count >= start_n:
            self._call_conveyor_power(True)


def main():
    rclpy.init()
    node = ConveyorRoiGuard()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
