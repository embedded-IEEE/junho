#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

def main():
    rclpy.init()
    node = Node('zero_check')
    
    # 토픽 이름 확인 필수 (/jetank/arm_controller/joint_trajectory)
    pub = node.create_publisher(JointTrajectory, '/jetank/arm_controller/joint_trajectory', 10)
    
    # 사용자 로봇의 관절 이름 (순서대로)
    joint_names = [
        'Revolute_BEARING',      # 1번
        'Revolute_ARM_LOW',      # 2번
        'Revolute_SERVO_UPPER',  # 3번
        'Revolute_MAGNETIC_BAR', # 4번
        'Revolute_SERVO_TOP'     # 5번
    ]

    msg = JointTrajectory()
    msg.joint_names = joint_names
    point = JointTrajectoryPoint()
    
    # [핵심] 전부 0도로 보내기
    point.positions = [0.0, 0.0, 0.0, 0.0, 0.0]
    point.time_from_start = Duration(sec=2, nanosec=0)
    msg.points = [point]

    print("📡 모든 관절을 0.0(Rad)으로 보냅니다...")
    import time
    time.sleep(1) # 연결 대기
    pub.publish(msg)
    time.sleep(1)
    print("완료. 로봇의 자세를 보고 오프셋을 결정하세요.")

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
