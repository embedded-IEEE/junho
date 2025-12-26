#!/usr/bin/env python3

import time
import sys
import numpy as np
import os
import platform
import math

# ================= ROS2 =================
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.time import Time
from std_msgs.msg import Empty
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

# ================= TF =================
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped

# ================= 환경 감지 =================
IS_REAL_ROBOT = (platform.machine() == 'aarch64')

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'SCSCtrl'))

if IS_REAL_ROBOT:
    print("[System] Jetson 감지됨 → REAL ROBOT MODE")
    from magnet_driver import Electromagnet
    from SCSCtrl.scservo_sdk import *
    IN1, IN2, PULSE_TIME = 37, 38, 0.2
else:
    print("[System] PC 감지됨 → SIMULATION MODE")
    IN1, IN2, PULSE_TIME = 0, 0, 0.2


# ================= 설정 =================
class Config:
    DEVICE_NAME = '/dev/ttyTHS1'
    BAUDRATE = 1000000
    ID_BASE = 1
    ID_SHOULDER = 2
    ID_ELBOW = 3
    ID_WRIST_ROLL = 4
    ID_WRIST_PITCH = 5

    LINK_1 = 95.0
    LINK_2 = 142.0
    LINK_3 = 123.0

    SERVO_INIT_POS = {1: 478, 2: 959, 3: 936, 4: 512, 5: 531}
    INPUT_RANGE = 850
    ADDR_GOAL_POSITION = 42
    ADDR_GOAL_SPEED = 46
    ADDR_PRESENT_POSITION = 56


# ================= Jetank Controller =================
class JetankController(Node):
    def __init__(self):
        super().__init__('jetank1_controller')

        self.get_logger().info("Jetank1 Controller Initializing...")

        # -------- ROS 기본 --------
        qos = QoSProfile(depth=10)
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            '/jetank1/arm_controller/joint_trajectory',
            qos
        )

        # -------- Jenga attach/detach (Sim) --------
        self.jenga_pubs = {}
        for i in range(1, 11):
            self.jenga_pubs[i] = {
                'attach': self.create_publisher(Empty, f'/jetank1/jenga{i}/attach', qos),
                'detach': self.create_publisher(Empty, f'/jetank1/jenga{i}/detach', qos)
            }

        # -------- TF --------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        if not IS_REAL_ROBOT:
            self.tf_static_broadcaster = StaticTransformBroadcaster(self)
            self.publish_world_bridge()

        self.MAGNET_FRAME = 'jetank1/MAGNETIC_BAR_1'
        self.current_attached_id = None

        # -------- Servo --------
        self.target_ids = [1, 2, 3, 4, 5]
        self.current_servo_pos = Config.SERVO_INIT_POS.copy()
        self.servo_manager = None
        self.magnet = None

        if IS_REAL_ROBOT:
            self.init_hardware()

        # =====================================================
        # ⭐ Conveyor → Jetank1 이벤트 서비스 ⭐
        # =====================================================
        self.conveyor_event_srv = self.create_service(
            Trigger,
            '/jetank1/conveyor_on_event',
            self.on_conveyor_on_event
        )

        self.get_logger().info("[Jetank1] Ready to receive conveyor ON event")

    # ================= Conveyor Event =================
    def on_conveyor_on_event(self, request, response):
        self.get_logger().info("==============================================")
        self.get_logger().info("[Jetank1] Conveyor ON event RECEIVED")
        self.get_logger().info("[Jetank1] Next step: request jenga coords from top_cctv1")
        self.get_logger().info("==============================================")

        # TODO:
        # 1) top_cctv1 좌표 요청
        # 2) move_to_xyz 자동 실행

        response.success = True
        response.message = "Jetank1 received conveyor ON event"
        return response

    # ================= TF Bridge (Sim) =================
    def publish_world_bridge(self):
        t = TransformStamped()
        t.header.frame_id = 'world'
        t.child_frame_id = 'empty_world'
        t.transform.rotation.w = 1.0
        self.tf_static_broadcaster.sendTransform(t)

    # ================= Hardware Init =================
    def init_hardware(self):
        try:
            self.port_handler = PortHandler(Config.DEVICE_NAME)
            self.packet_handler = PacketHandler(1)
            self.port_handler.openPort()
            self.port_handler.setBaudRate(Config.BAUDRATE)

            self.group_sync_write_pos = GroupSyncWrite(
                self.port_handler, self.packet_handler,
                Config.ADDR_GOAL_POSITION, 2
            )
            self.group_sync_write_spd = GroupSyncWrite(
                self.port_handler, self.packet_handler,
                Config.ADDR_GOAL_SPEED, 2
            )

            self.magnet = Electromagnet(IN1, IN2, PULSE_TIME)
            self.servo_manager = True
            self.get_logger().info("[Hardware] Servo & Magnet Ready")

        except Exception as e:
            self.get_logger().error(f"[Hardware Init Error] {e}")

    # ================= Magnet =================
    def control_magnet(self, cmd):
        if IS_REAL_ROBOT:
            if cmd == "ON":
                self.magnet.grab()
            else:
                self.magnet.release()
            return

        msg = Empty()
        if cmd == "ON":
            for i in range(1, 11):
                self.jenga_pubs[i]['attach'].publish(msg)
                self.current_attached_id = i
                break
        else:
            if self.current_attached_id:
                self.jenga_pubs[self.current_attached_id]['detach'].publish(msg)
                self.current_attached_id = None

    # ================= IK & Motion =================
    def solve_ik(self, r, z, phi_deg):
        phi = np.radians(phi_deg)
        w_r = r - Config.LINK_3 * np.cos(phi)
        w_z = z - Config.LINK_3 * np.sin(phi)

        L1, L2 = Config.LINK_1, Config.LINK_2
        cos_t2 = (w_r**2 + w_z**2 - L1**2 - L2**2) / (2*L1*L2)
        cos_t2 = np.clip(cos_t2, -1.0, 1.0)

        t2 = np.arccos(cos_t2)
        t1 = np.arctan2(w_z, w_r) - np.arctan2(L2*np.sin(t2), L1 + L2*np.cos(t2))
        t3 = phi - (t1 + t2)

        return np.degrees(t1), -np.degrees(t2), np.degrees(t3)

    def move_to_xyz(self, x, y, z, phi=-90, roll=0, move_time=2.0):
        r = np.sqrt(x*x + y*y)
        base = np.degrees(np.arctan2(y, x))
        t1, t2, t3 = self.solve_ik(r, z, phi)

        angles = {
            Config.ID_BASE: base,
            Config.ID_SHOULDER: t1,
            Config.ID_ELBOW: t2,
            Config.ID_WRIST_ROLL: roll,
            Config.ID_WRIST_PITCH: t3
        }

        msg = JointTrajectory()
        msg.joint_names = [
            'Revolute_BEARING',
            'Revolute_ARM_LOW',
            'Revolute_SERVO_UPPER',
            'Revolute_MAGNETIC_BAR',
            'Revolute_SERVO_TOP'
        ]

        point = JointTrajectoryPoint()
        point.positions = [np.radians(v) for v in angles.values()]
        point.time_from_start = Duration(sec=int(move_time))
        msg.points.append(point)

        self.traj_pub.publish(msg)

    # ================= Close =================
    def close(self):
        self.get_logger().info("[Jetank1] Shutting down")
        self.destroy_node()


# ================= Main =================
def main():
    rclpy.init()
    node = JetankController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
