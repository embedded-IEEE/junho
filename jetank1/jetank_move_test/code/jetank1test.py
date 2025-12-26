#!/usr/bin/env python3
"""
Jetank Move-Only Controller (REAL/SIM)
- 입력: x y z [phi] [roll] [move_time]
  - 기본 단위: mm, mm, mm / deg / deg / sec
  - 예) 150 0 50
  - 예) 120 80 20 -90 0 2.0
- 동작: 입력한 목표 위치로 "한 번만" 이동하고 끝(자석/픽업/드롭 없음)
"""

import math
import os
import platform
import sys
import time
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from builtin_interfaces.msg import Duration
except ImportError:
    print("[Error] ROS2 라이브러리를 찾을 수 없습니다. (PC라면 ros-humble-rclpy 등을 확인하세요)")
    sys.exit(1)

# ---------------------------------------------------------
# 환경 감지
# ---------------------------------------------------------
IS_REAL_ROBOT = (platform.machine() == "aarch64")

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append("/home/jetson/SCSCtrl")

if IS_REAL_ROBOT:
    print(f"[System] Jetson({platform.machine()}) 감지됨 -> 하드웨어 모드 활성화")
    try:
        from SCSCtrl.scservo_sdk import *  # noqa: F403
    except ImportError as e:
        print(f"[Error] 하드웨어 라이브러리 로드 실패: {e}")
        IS_REAL_ROBOT = False


class Config:
    DEVICE_NAME = "/dev/ttyTHS1"
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


class JetankController(Node):
    def __init__(self, robot_name: str = "jetank1", use_sim_time: bool = True):
        super().__init__(f"{robot_name}_move_only")
        self.robot_name = robot_name

        # 실기라면 보통 False 권장(/clock 없으면 시간 멈춤)
        self.set_parameters([rclpy.parameter.Parameter("use_sim_time", rclpy.Parameter.Type.BOOL, use_sim_time)])

        qos_profile = QoSProfile(depth=10)

        # Gazebo 팔 제어용 Publisher (시뮬에서도 같이 쏴줌)
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            f"/{robot_name}/arm_controller/joint_trajectory",
            qos_profile,
        )

        self.joint_names = [
            "Revolute_BEARING",
            "Revolute_ARM_LOW",
            "Revolute_SERVO_UPPER",
            "Revolute_MAGNETIC_BAR",
            "Revolute_SERVO_TOP",
        ]

        self.SIM_CORRECTION = {
            Config.ID_BASE: {"dir": 1, "offset": 0.0},
            Config.ID_SHOULDER: {"dir": -1, "offset": 90.0},
            Config.ID_ELBOW: {"dir": -1, "offset": 0.0},
            Config.ID_WRIST_ROLL: {"dir": 1, "offset": 0.0},
            Config.ID_WRIST_PITCH: {"dir": 1, "offset": 90.0},
        }

        # 실기 서보 방향(필요 시 조정)
        self.dirs = {
            Config.ID_BASE: 1,
            Config.ID_SHOULDER: -1,
            Config.ID_ELBOW: 1,
            Config.ID_WRIST_ROLL: 1,
            Config.ID_WRIST_PITCH: 1,
        }
        self.target_ids = [1, 2, 3, 4, 5]
        self.current_servo_pos = Config.SERVO_INIT_POS.copy()

        self.servo_manager = False
        if IS_REAL_ROBOT:
            self.init_hardware()

    # --------------------------
    # 하드웨어 초기화
    # --------------------------
    def init_hardware(self) -> None:
        try:
            self.port_handler = PortHandler(Config.DEVICE_NAME)  # noqa: F405
            self.packet_handler = PacketHandler(1)  # noqa: F405
            ok = self.port_handler.openPort() and self.port_handler.setBaudRate(Config.BAUDRATE)
            print("[Hardware] Serial Port Opened." if ok else "[Error] Failed to open port!")

            self.group_sync_write_pos = GroupSyncWrite(  # noqa: F405
                self.port_handler, self.packet_handler, Config.ADDR_GOAL_POSITION, 2
            )
            self.group_sync_write_spd = GroupSyncWrite(  # noqa: F405
                self.port_handler, self.packet_handler, Config.ADDR_GOAL_SPEED, 2
            )
            self.servo_manager = True
        except Exception as e:
            print(f"[Error] Hardware Init Failed: {e}")
            self.servo_manager = False

    # --------------------------
    # IK
    # --------------------------
    def solve_ik_3dof_planar(self, r: float, z: float, phi_deg: float) -> Optional[Tuple[float, float, float]]:
        phi = np.radians(phi_deg)
        w_r = r - Config.LINK_3 * np.cos(phi)
        w_z = z - Config.LINK_3 * np.sin(phi)
        L1, L2 = Config.LINK_1, Config.LINK_2
        if np.sqrt(w_r**2 + w_z**2) > (L1 + L2):
            return None
        cos_angle = (w_r**2 + w_z**2 - L1**2 - L2**2) / (2 * L1 * L2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        theta2 = np.arccos(cos_angle)
        k1 = L1 + L2 * np.cos(theta2)
        k2 = L2 * np.sin(theta2)
        theta1 = np.arctan2(w_z, w_r) + np.arctan2(k2, k1)
        theta3 = phi - (theta1 - theta2)
        return np.degrees(theta1), -np.degrees(theta2), np.degrees(theta3)

    # --------------------------
    # 이동(성공 True / 실패 False)
    # --------------------------
    def move_to_xyz(
        self,
        x: float,
        y: float,
        z: float,
        phi: float = -90.0,
        roll: float = 0.0,
        move_time: float = 2.0,
    ) -> bool:
        rad_base = np.arctan2(y, x)
        deg_base = np.degrees(rad_base)
        r_dist = np.sqrt(x**2 + y**2)

        ik_result = self.solve_ik_3dof_planar(r_dist, z, phi_deg=phi)
        if ik_result is None:
            print(f"[IK] Unreachable: x={x} y={y} z={z} (r={r_dist:.2f}) phi={phi}")
            return False

        deg_shoulder, deg_elbow, deg_wrist_p = ik_result

        target_angles = {
            Config.ID_BASE: deg_base,
            Config.ID_SHOULDER: deg_shoulder,
            Config.ID_ELBOW: deg_elbow,
            Config.ID_WRIST_ROLL: roll,
            Config.ID_WRIST_PITCH: deg_wrist_p,
        }

        # 시뮬(Gazebo)
        self.publish_gazebo_command(target_angles, move_time)

        # 실기
        if IS_REAL_ROBOT and self.servo_manager:
            self.send_hardware_command(target_angles, move_time)

        return True

    def publish_gazebo_command(self, angles_deg: Dict[int, float], move_time: float) -> None:
        msg = JointTrajectory()
        msg.header.stamp.sec = 0
        msg.header.stamp.nanosec = 0
        msg.joint_names = self.joint_names
        point = JointTrajectoryPoint()

        def normalize_angle(angle_deg: float) -> float:
            angle_deg = angle_deg % 360.0
            if angle_deg > 180.0:
                angle_deg -= 360.0
            elif angle_deg < -180.0:
                angle_deg += 360.0
            return angle_deg

        def get_sim_rad(srv_id: int) -> float:
            cfg = self.SIM_CORRECTION[srv_id]
            input_deg = angles_deg[srv_id]
            raw_target = (input_deg * cfg["dir"]) + cfg["offset"]
            final_deg = normalize_angle(raw_target)
            return np.radians(final_deg)

        point.positions = [
            get_sim_rad(Config.ID_BASE),
            get_sim_rad(Config.ID_SHOULDER),
            get_sim_rad(Config.ID_ELBOW),
            get_sim_rad(Config.ID_WRIST_ROLL),
            get_sim_rad(Config.ID_WRIST_PITCH),
        ]

        sec = int(move_time)
        nanosec = int((move_time - sec) * 1e9)
        point.time_from_start = Duration(sec=sec, nanosec=nanosec)
        msg.points = [point]
        self.traj_pub.publish(msg)

    def send_hardware_command(self, angles_deg: Dict[int, float], move_time: float) -> None:
        goals, speeds, delta_pos_list = [], [], []

        for sid in self.target_ids:
            angle = angles_deg[sid]
            direction = self.dirs[sid]
            pos = Config.SERVO_INIT_POS[sid] + int((Config.INPUT_RANGE / 180.0) * angle * direction)
            pos = max(0, min(1023, pos))
            goals.append(pos)

            current = self.current_servo_pos.get(sid, Config.SERVO_INIT_POS[sid])
            delta_pos_list.append(abs(pos - current))
            self.current_servo_pos[sid] = pos

        scaling_factor = 1.0 / max(0.001, move_time)
        for delta in delta_pos_list:
            calc_speed = int((delta * scaling_factor) * 1.5)
            calc_speed = max(40, min(1000, calc_speed))
            speeds.append(calc_speed)

        for i, sid in enumerate(self.target_ids):
            param_spd = [SCS_LOBYTE(speeds[i]), SCS_HIBYTE(speeds[i])]  # noqa: F405
            self.group_sync_write_spd.addParam(sid, param_spd)
            param_pos = [SCS_LOBYTE(goals[i]), SCS_HIBYTE(goals[i])]  # noqa: F405
            self.group_sync_write_pos.addParam(sid, param_pos)

        self.group_sync_write_spd.txPacket()
        self.group_sync_write_spd.clearParam()
        self.group_sync_write_pos.txPacket()
        self.group_sync_write_pos.clearParam()
        print(f"[REAL] Goals: {goals}")

    def close(self) -> None:
        if IS_REAL_ROBOT and hasattr(self, "port_handler"):
            self.port_handler.closePort()
        self.destroy_node()


def parse_cmd(line: str) -> Tuple[float, float, float, float, float, float]:
    """
    x y z [phi] [roll] [move_time]
    기본값: phi=-90, roll=0, move_time=2
    """
    parts = line.replace(",", " ").split()
    if len(parts) < 3:
        raise ValueError("최소 3개 필요: x y z  (예: 150 0 50)")

    x = float(parts[0])
    y = float(parts[1])
    z = float(parts[2])

    phi = float(parts[3]) if len(parts) >= 4 else -90.0
    roll = float(parts[4]) if len(parts) >= 5 else 0.0
    move_time = float(parts[5]) if len(parts) >= 6 else 2.0
    return x, y, z, phi, roll, move_time


def main() -> None:
    rclpy.init()

    # 실기면 use_sim_time=False 권장
    use_sim_time = False if IS_REAL_ROBOT else True
    robot = JetankController(robot_name="jetank1", use_sim_time=use_sim_time)

    print("=========================================================")
    print(" [Move Only] 원하는 위치로만 이동 (그 뒤 아무 동작 안 함)")
    print(" Input: x y z [phi] [roll] [move_time]")
    print("  - 예) 150 0 50")
    print("  - 예) 120 80 20 -90 0 2.0")
    print(" Exit: q")
    print("=========================================================")

    try:
        while rclpy.ok():
            line = input("\nTarget >> ").strip().lower()
            if line in ("q", "quit", "exit"):
                break
            if not line:
                continue

            try:
                x, y, z, phi, roll, mv = parse_cmd(line)
            except ValueError as e:
                print(f"[Error] {e}")
                continue

            ok = robot.move_to_xyz(x, y, z, phi=phi, roll=roll, move_time=mv)
            if ok:
                print(f">> 이동 명령 완료: x={x} y={y} z={z} phi={phi} roll={roll} t={mv}")
            else:
                print(">> 이동 실패(도달 불가). 좌표/스케일/링크길이 확인 필요.")

            # ✅ 여기서 끝. 추가 동작(자석/픽업/복귀) 없음.

    except KeyboardInterrupt:
        pass
    finally:
        robot.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
