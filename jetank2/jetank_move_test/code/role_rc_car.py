#!/usr/bin/env python3
"""RC car role placeholder for palletize notifications."""

# rc_car 역할은 팔레타이징 완료 알림을 받아 후속 동작을 트리거하는 자리.
# 사용 예:
#   python3 src/jetank_move_test/code/role_rc_car.py
#   ros2 service call /rc_car/palletize_done std_srvs/srv/Trigger "{}"

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


class RcCarRole(Node):
    """팔레타이징 완료 알림을 받는 rc_car 역할 노드."""
    def __init__(self):
        """완료 알림 서비스 서버를 생성."""
        super().__init__("rc_car_role")
        self.declare_parameter("notify_service", "/rc_car/palletize_done")
        service_name = self.get_parameter("notify_service").value
        # Jetank2 완료 알림 서비스
        self.srv = self.create_service(Trigger, service_name, self.on_notify)
        self.get_logger().info(f"RC car notify service: {service_name}")

    def on_notify(self, request, response):
        """알림 수신 시 후속 동작을 트리거."""
        # 현재는 로그만 출력하는 간단한 placeholder
        self.get_logger().info("RC car notified: palletize done")
        response.success = True
        response.message = "ack"
        return response


def main() -> None:
    """rc_car 역할 노드 실행 진입점."""
    rclpy.init()
    node = RcCarRole()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
