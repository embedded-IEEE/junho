import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool


class ConveyorController(Node):
    def __init__(self):
        super().__init__('conveyor_controller')

        # Gazebo /clock(시뮬 시간) 사용
        self.set_parameters([
            rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)
        ])

        self.cli = self.create_client(SetBool, '/conveyor/power')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('conveyor service 대기 중...')

    def set_power(self, on: bool):
        req = SetBool.Request()
        req.data = on
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is None:
            self.get_logger().error(f'서비스 호출 실패: {future.exception()}')
            return False

        res = future.result()
        self.get_logger().info(f'응답 결과: success={res.success}, message="{res.message}"') # 이 줄 추가
        if not res.success:
            self.get_logger().warn(f'컨베이어 power 응답 실패: {res.message}')

    def wait_sim_seconds(self, seconds: float):
        """
        /clock(시뮬 시간) 기준으로 seconds 만큼 경과할 때까지 대기
        """
        start = self.get_clock().now()
        target_ns = int(seconds * 1e9)

        # /clock이 아직 안 들어오면 now()가 0 근처일 수 있음
        # spin_once를 돌리며 /clock 업데이트를 받는다.
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            now = self.get_clock().now()
            elapsed_ns = (now - start).nanoseconds

            if elapsed_ns >= target_ns:
                break


def main():
    rclpy.init()
    node = ConveyorController()

    duration = 12.6  # "시뮬레이션 시간" 기준 초

    node.get_logger().info('컨베이어 ON (sim time 기준)')
    node.set_power(True)

    node.get_logger().info(f'시뮬 시간 {duration:.1f}s 대기...')
    node.wait_sim_seconds(duration)

    node.get_logger().info('컨베이어 OFF (sim time 기준)')
    node.set_power(False)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()