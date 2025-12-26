#!/usr/bin/env python3
import time
import threading

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool, Trigger

import Jetson.GPIO as GPIO

# =========================
# GPIO 설정 (BOARD 모드 기준)
# =========================
PUL_PIN = 36
DIR_PIN = 35
ENA_PIN = 33

STEP_DELAY = 0.0001  # 펄스 주기(너무 빠르면 모터가 못 돔)

# 전역 상태 (스레드에서 사용)
motor_running = False
program_exit = False
motor_dir = GPIO.LOW  # 필요 시 방향 전환 기능 확장 가능


class ConveyorGPIOnode(Node):
    """
    Conveyor 역할:
      - /conveyor/set_power (SetBool) 로 ON/OFF 제어
      - ON 성공 시 jetank1에 /jetank1/conveyor_on_event (Trigger) 전송
      - OFF 성공 시 jetank2에 /jetank2/conveyor_off_event (Trigger) 전송
    """

    def __init__(self):
        super().__init__('conveyor_gpio_node')

        # 파라미터(원하면 launch/cli로 조정)
        self.declare_parameter('step_delay', STEP_DELAY)
        self.declare_parameter('notify_timeout_sec', 0.2)   # 이벤트 전송 시 서비스 준비 wait 시간
        self.declare_parameter('idempotent_success', True)  # 이미 ON/OFF 상태에서 같은 명령 오면 success 처리할지

        self.step_delay = float(self.get_parameter('step_delay').value)
        self.notify_timeout_sec = float(self.get_parameter('notify_timeout_sec').value)
        self.idempotent_success = bool(self.get_parameter('idempotent_success').value)

        # 내부 상태머신
        # OFF | TURNING_ON | ON | TURNING_OFF | ERROR
        self.state = 'OFF'
        self._lock = threading.Lock()

        # GPIO 초기화
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup([PUL_PIN, DIR_PIN, ENA_PIN], GPIO.OUT, initial=GPIO.LOW)

        # 모터 스레드 시작
        self.motor_thread = threading.Thread(target=self.motor_task, daemon=True)
        self.motor_thread.start()

        # 서비스 서버: top_cctv2가 호출
        self.srv = self.create_service(SetBool, '/conveyor/set_power', self.set_power_callback)

        # 이벤트 클라이언트: 컨베이어 → jetank
        self.cli_jetank1_on = self.create_client(Trigger, '/jetank1/conveyor_on_event')
        self.cli_jetank2_off = self.create_client(Trigger, '/jetank2/conveyor_off_event')

        self.get_logger().info('Conveyor GPIO Node started')
        self.get_logger().info('Service: /conveyor/set_power (std_srvs/SetBool)')
        self.get_logger().info('Event targets: /jetank1/conveyor_on_event, /jetank2/conveyor_off_event')

    # =========================
    # 모터 구동 스레드
    # =========================
    def motor_task(self):
        global motor_running, program_exit, motor_dir

        self.get_logger().info('Motor thread started')

        # ENA: LOW = Enable(잠금/힘 유지), HIGH = Disable(힘 풀림)
        GPIO.output(ENA_PIN, GPIO.LOW)

        while not program_exit:
            GPIO.output(DIR_PIN, motor_dir)

            if motor_running:
                GPIO.output(PUL_PIN, GPIO.HIGH)
                time.sleep(self.step_delay)
                GPIO.output(PUL_PIN, GPIO.LOW)
                time.sleep(self.step_delay)
            else:
                time.sleep(0.05)

        self.get_logger().info('Motor thread exiting')

    # =========================
    # /conveyor/set_power 서비스 콜백
    # =========================
    def set_power_callback(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        global motor_running

        target_on = bool(request.data)

        with self._lock:
            cur = self.state

            # 전환 중(busy)
            if cur in ('TURNING_ON', 'TURNING_OFF'):
                response.success = False
                response.message = f'busy (state={cur})'
                self.get_logger().warn(f'[Conveyor] Reject request: {response.message}')
                return response

            # 멱등 처리(이미 ON인데 ON 요청 / 이미 OFF인데 OFF 요청)
            if target_on and cur == 'ON':
                response.success = self.idempotent_success
                response.message = 'already on'
                self.get_logger().info(f'[Conveyor] {response.message}')
                return response

            if (not target_on) and cur == 'OFF':
                response.success = self.idempotent_success
                response.message = 'already off'
                self.get_logger().info(f'[Conveyor] {response.message}')
                return response

            # 상태 전이 시작
            self.state = 'TURNING_ON' if target_on else 'TURNING_OFF'

        # 실제 동작: 여기서는 motor_running 플래그만 변경(스레드가 펄스를 생성)
        # 실물에서 "ON 완료 확인"이 필요하면 센서/드라이버 피드백을 확인하는 로직을 추가해야 함.
        if target_on:
            motor_running = True
            msg = 'Conveyor ON'
            self.get_logger().info('[Conveyor] Received ON request -> starting motor')
        else:
            motor_running = False
            msg = 'Conveyor OFF'
            self.get_logger().info('[Conveyor] Received OFF request -> stopping motor')

        # 여기서 “정상 동작 시작”을 성공으로 간주
        with self._lock:
            self.state = 'ON' if target_on else 'OFF'

        response.success = True
        response.message = msg

        # 이벤트 통지 (성공 시)
        if target_on:
            self._notify_jetank(self.cli_jetank1_on, label='jetank1 ON event', topic='/jetank1/conveyor_on_event')
        else:
            self._notify_jetank(self.cli_jetank2_off, label='jetank2 OFF event', topic='/jetank2/conveyor_off_event')

        self.get_logger().info(f'[Conveyor] Responding: success={response.success}, message="{response.message}"')
        return response

    # =========================
    # jetank 이벤트 통지
    # =========================
    def _notify_jetank(self, client, label: str, topic: str):
        """
        jetank 이벤트 통지 (DDS discovery 지연 대응)
        """
        max_retry = 10          # 총 재시도 횟수
        retry_interval = 0.3    # 재시도 간격 (초)

        for attempt in range(1, max_retry + 1):
            if client.service_is_ready():
                req = Trigger.Request()
                future = client.call_async(req)
                future.add_done_callback(lambda f: self._on_notify_done(f, label))

                self.get_logger().info(
                    f'[Conveyor] Sent {label} -> {topic} (attempt {attempt})'
                )
                return

            self.get_logger().warn(
                f'[Conveyor] Waiting for {topic} (attempt {attempt}/{max_retry})'
            )
            time.sleep(retry_interval)

        # 여기까지 오면 실패
        self.get_logger().error(
            f'[Conveyor] FAILED to notify {label}: service not available after retries'
        )


    def _on_notify_done(self, future, label: str):
        try:
            resp = future.result()
            self.get_logger().info(f'[Conveyor] Notify result ({label}): success={resp.success}, msg="{resp.message}"')
        except Exception as e:
            self.get_logger().warn(f'[Conveyor] Notify failed ({label}): {e!r}')

    # =========================
    # 종료 처리
    # =========================
    def destroy_node(self):
        global program_exit
        program_exit = True

        # 발열/안전: ENA HIGH로 Disable
        try:
            GPIO.output(ENA_PIN, GPIO.HIGH)
        except Exception:
            pass

        try:
            self.motor_thread.join(timeout=1.0)
        except Exception:
            pass

        try:
            GPIO.cleanup()
        except Exception:
            pass

        self.get_logger().info('Conveyor GPIO cleaned up')
        super().destroy_node()


def main():
    rclpy.init()
    node = ConveyorGPIOnode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
