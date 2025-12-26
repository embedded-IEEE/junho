#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

def main():
    rclpy.init()
    node = Node('zero_check')
    pub = node.create_publisher(JointTrajectory, '/jetank/arm_controller/joint_trajectory', 10)
    
    # 관절 이름 (사용자 환경에 맞게)
    joint_names = [
        'Revolute_BEARING', 'Revolute_ARM_LOW', 'Revolute_SERVO_UPPER', 
        'Revolute_MAGNETIC_BAR', 'Revolute_SERVO_TOP'
    ]

    msg = JointTrajectory()
    msg.joint_names = joint_names
    point = JointTrajectoryPoint()
    
    # [핵심] 모든 관절에 "0.0" (0도) 명령 전송
    point.positions = [0.0, 0.0, 0.0, 0.0, 0.0]
    point.time_from_start = Duration(sec=2, nanosec=0)
    msg.points = [point]

    print("📡 모든 관절을 0도로 이동합니다...")
    # 퍼블리셔 연결 대기 후 전송
    import time
    time.sleep(1)
    pub.publish(msg)
    time.sleep(1)
    print("완료. 로봇의 자세를 확인하세요.")

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
