#!/usr/bin/env python3

# 입력(외부): /conveyor/request_power (SetBool, data=True -> ON, False -> OFF
# 출력(하드웨어/시뮬): /conveyor/power (SetBool)
# 출력(로봇 트리거):
    #   ON -> /jetank1/start_pick (Trigger)
    #   OFF -> /jetank2/start_pick (Trigger)

import time
import threading

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_srvs.srv import SetBool, Trigger


class ConveyorCoordinator(Node):
    """컨베이어 전원 요청을 받아 제어하고 Jetank를 트리거하는 노드."""
    def __init__(self):
        """서비스/클라이언트 초기화 및 상태 변수 설정."""
        super().__init__("conveyor_coordinator")
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

        self.declare_parameter("power_service", "/conveyor/power")
        self.declare_parameter("request_service", "/conveyor/request_power")
        self.declare_parameter("jetank1_start_service", "/jetank1/start_pick")
        self.declare_parameter("jetank2_start_service", "/jetank2/start_pick")
        self.declare_parameter("power_timeout_sec", 3.0)
        self.declare_parameter("notify_timeout_sec", 3.0)

        self.power_service = self.get_parameter("power_service").value
        self.request_service = self.get_parameter("request_service").value
        self.jetank1_service = self.get_parameter("jetank1_start_service").value
        self.jetank2_service = self.get_parameter("jetank2_start_service").value
        self.power_timeout = float(self.get_parameter("power_timeout_sec").value)
        self.notify_timeout = float(self.get_parameter("notify_timeout_sec").value)

        # 컨베이어 전원 서비스 및 Jetank 시작 트리거 클라이언트
        self.power_client = self.create_client(SetBool, self.power_service)
        self.jetank1_client = self.create_client(Trigger, self.jetank1_service)
        self.jetank2_client = self.create_client(Trigger, self.jetank2_service)

        self.srv = self.create_service(SetBool, self.request_service, self.on_request_power)

        self._lock = threading.Lock()
        self.power_on = None
        self.get_logger().info(f"Conveyor coordinator ready: {self.request_service} -> {self.power_service}")

    def _wait_future(self, future, timeout_sec: float) -> bool:
        """future 완료까지 대기하고 타임아웃을 적용."""
        # 간단한 future 타임아웃 대기
        deadline = time.monotonic() + float(timeout_sec)
        while time.monotonic() < deadline:
            if future.done():
                return True
            time.sleep(0.01)
        return future.done()

    def _call_setbool(self, client, on: bool, timeout_sec: float):
        """SetBool 서비스 호출 공통 래퍼."""
        # SetBool 서비스 호출 래퍼
        if not client.service_is_ready():
            client.wait_for_service(timeout_sec=timeout_sec)
        req = SetBool.Request()
        req.data = bool(on)
        future = client.call_async(req)
        if not self._wait_future(future, timeout_sec):
            return False, "timeout"
        res = future.result()
        if res is None:
            return False, "no response"
        return bool(res.success), str(res.message)

    def _call_trigger(self, client, timeout_sec: float):
        """Trigger 서비스 호출 공통 래퍼."""
        # Trigger 서비스 호출 래퍼
        if not client.service_is_ready():
            client.wait_for_service(timeout_sec=timeout_sec)
        req = Trigger.Request()
        future = client.call_async(req)
        if not self._wait_future(future, timeout_sec):
            return False, "timeout"
        res = future.result()
        if res is None:
            return False, "no response"
        return bool(res.success), str(res.message)

    def on_request_power(self, request, response):
        """컨베이어 전원 요청 처리 및 Jetank 트리거 발행."""
        # top_cctv2 ROI guard 등에서 요청하는 컨베이어 ON/OFF 핸들러
        desired_on = bool(request.data)
        with self._lock:
            if self.power_on is not None and self.power_on == desired_on:
                response.success = True
                response.message = f"power already {'ON' if desired_on else 'OFF'}"
                return response

            # 실제 전원 서비스 호출
            ok, msg = self._call_setbool(self.power_client, desired_on, self.power_timeout)
            if not ok:
                response.success = False
                response.message = f"power service failed: {msg}"
                return response

            # 전원 상태에 맞는 Jetank 시작 트리거 호출
            self.power_on = desired_on
            notify_client = self.jetank1_client if desired_on else self.jetank2_client
            notify_name = "jetank1" if desired_on else "jetank2"
            n_ok, n_msg = self._call_trigger(notify_client, self.notify_timeout)
            response.success = True
            if n_ok:
                response.message = f"power {'ON' if desired_on else 'OFF'}; notified {notify_name}"
            else:
                self.get_logger().warn(f"notify failed for {notify_name}: {n_msg}")
                response.message = f"power set; {notify_name} notify failed: {n_msg}"
            return response


def main() -> None:
    """ROS2 노드 실행 진입점."""
    rclpy.init()
    node = ConveyorCoordinator()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
