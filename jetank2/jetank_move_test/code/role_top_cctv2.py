#!/usr/bin/env python3
"""Top CCTV 2 inference + conveyor ROI guard role."""

# top_cctv2 카메라 추론 + ROI 감지로 컨베이어를 제어하는 역할.
# ROI guard는 /conveyor/request_power를 호출하고, 정지 시 자동 비활성화됨.
# 사용 예:
#   python3 src/jetank_move_test/code/role_top_cctv2.py \\
#     --ros-args -p roi_xmin_ratio:=0.22 -p roi_xmax_ratio:=0.40 -p roi_ymin_ratio:=0.42 -p roi_ymax_ratio:=0.58
#   ros2 service call /top_cctv2/roi_guard_enable std_srvs/srv/SetBool "{data: true}"

import time

import rclpy
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import SetBool

from top_cctv_infer.conveyor_roi_guard import ConveyorRoiGuard
from top_cctv_infer.infer_service_server import InferServiceServer


class ConveyorRoiGuardSwitch(ConveyorRoiGuard):
    """ROI 감지로 컨베이어 제어 + 원격 활성/비활성 스위치."""
    def __init__(
        self,
        power_service: str = "/conveyor/request_power",
        enable_service: str = "/top_cctv2/roi_guard_enable",
        auto_disable_on_stop: bool = True,
    ):
        """ROI guard에 스위치 기능과 전원 서비스 연결을 추가."""
        super().__init__()
        # ROI y축 범위도 파라미터로 허용
        self.declare_parameter("roi_ymin_ratio", 0.42)
        self.declare_parameter("roi_ymax_ratio", 0.58)
        self.power_service = power_service
        self.auto_disable_on_stop = auto_disable_on_stop
        self.enabled = True
        self.cli = self.create_client(SetBool, self.power_service)
        self.enable_srv = self.create_service(SetBool, enable_service, self.on_enable)
        self.get_logger().info(f"ROI guard power service: {self.power_service}")
        self.get_logger().info(f"ROI guard enable service: {enable_service}")

    def on_enable(self, request, response):
        """외부 서비스로 ROI guard 활성 상태를 변경."""
        # 외부 요청으로 ROI guard 활성/비활성
        self._set_enabled(bool(request.data), reason="service")
        response.success = True
        response.message = f"enabled={self.enabled}"
        return response

    def _set_enabled(self, enabled: bool, reason: str):
        """ROI guard 상태와 카운터를 리셋."""
        self.enabled = enabled
        self.stop_count = 0
        self.start_count = 0
        self.cache_roi_hit = False
        self.get_logger().info(f"ROI guard enabled={self.enabled} ({reason})")

    def on_image(self, msg):
        """이미지 콜백: 활성 상태일 때만 처리."""
        # 비활성 상태면 추론/제어 수행하지 않음
        if not self.enabled:
            return
        super().on_image(msg)

    def _roi_hit_any(self, frame_w: int, frame_h: int, centers_xy):
        """ROI 영역 내 중심점 존재 여부를 판단."""
        # ROI 영역 히트 여부 판단(가로+세로 비율)
        xmin_r = float(self.get_parameter("roi_xmin_ratio").value)
        xmax_r = float(self.get_parameter("roi_xmax_ratio").value)
        ymin_r = float(self.get_parameter("roi_ymin_ratio").value)
        ymax_r = float(self.get_parameter("roi_ymax_ratio").value)
        x_min = frame_w * xmin_r
        x_max = frame_w * xmax_r
        y_min = frame_h * ymin_r
        y_max = frame_h * ymax_r

        if centers_xy.size == 0:
            return False

        cx = centers_xy[:, 0]
        cy = centers_xy[:, 1]
        hit = (cx >= x_min) & (cx <= x_max) & (cy >= y_min) & (cy <= y_max)
        return bool(hit.any())

    def _call_conveyor_power(self, on: bool):
        """컨베이어 전원 요청 서비스 호출."""
        # ROI guard가 /conveyor/request_power 호출
        if self.power_on is not None and self.power_on == on:
            return

        now = time.time()
        if now - self.last_call_t < 0.2:
            return

        if not self.cli.service_is_ready():
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
                    self.get_logger().info(f"{self.power_service} -> {on} (ok) msg={resp.message}")
                    if self.auto_disable_on_stop and not on:
                        # 컨베이어 정지 시 자동 비활성화
                        self._set_enabled(False, reason="auto-disable after stop")
                else:
                    self.get_logger().warn(
                        f"{self.power_service} -> {on} (fail) msg={resp.message if resp else 'None'}"
                    )
            except Exception as exc:
                self.get_logger().error(f"service call exception: {exc}")

        fut.add_done_callback(_done_cb)


def main() -> None:
    """top_cctv2 추론 서비스 + ROI guard 노드를 함께 실행."""
    rclpy.init()
    # top_cctv2 추론 서비스와 ROI guard를 동시에 실행
    infer_node = InferServiceServer(
        node_name="top_cctv2_infer_service",
        image_topic="/jetank/top_cctv2",
        service_name="/top_cctv2/get_closest_pose",
    )
    guard_node = ConveyorRoiGuardSwitch()

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(infer_node)
    executor.add_node(guard_node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        guard_node.destroy_node()
        infer_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
