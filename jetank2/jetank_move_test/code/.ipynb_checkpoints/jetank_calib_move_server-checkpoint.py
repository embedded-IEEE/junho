#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[Jetank(로봇)에서 실행]
픽셀 캘리브레이션을 위해 "로봇을 움직이는 부분"만 분리한 서비스 서버.

서비스:
- /<robot_name>/calib/move_to_xyz  (MoveToXYZ.srv)

요구:
- MoveToXYZ.srv 인터페이스가 빌드되어 있어야 함 (아래 srv 파일 참고)
- Gazebo면 /<robot_name>/arm_controller/joint_trajectory 퍼블리시 가능해야 함
- 실기면 SCSCtrl 설치/연결되어 있으면 SyncWrite로 서보 구동(가능할 때만)

주의:
- 이 코드는 cam_world_calib.py(단일 파일)에서 "MoveOnlyArm" 부분만 떼어낸 형태임.
"""

import argparse
import platform
import sys
from typing import Dict, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from top_cctv_interfaces.srv import MoveToXYZ

IS_REAL_ROBOT = (platform.machine() == "aarch64")


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


def _normalize_angle_deg(angle_deg: float) -> float:
    a = angle_deg % 360.0
    if a > 180.0:
        a -= 360.0
    elif a < -180.0:
        a += 360.0
    return a


class MoveServer(Node):
    def __init__(self, robot_name: str, use_sim_time: Optional[bool]):
        super().__init__(f"{robot_name}_calib_move_server")
        self.robot_name = robot_name

        # 실기라면 보통 False 권장(/clock 없으면 시간 멈춤)
        if use_sim_time is None:
            use_sim_time = (False if IS_REAL_ROBOT else True)
        self.set_parameters([rclpy.parameter.Parameter("use_sim_time", rclpy.Parameter.Type.BOOL, bool(use_sim_time))])

        qos = QoSProfile(depth=10)
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            f"/{robot_name}/arm_controller/joint_trajectory",
            qos,
        )

        self.joint_names = [
            "Revolute_BEARING",
            "Revolute_ARM_LOW",
            "Revolute_SERVO_UPPER",
            "Revolute_MAGNETIC_BAR",
            "Revolute_SERVO_TOP",
        ]

        # cam_world_calib.py와 동일한 시뮬 보정
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
            self._init_hardware_best_effort()

        srv_name = f"/{robot_name}/calib/move_to_xyz"
        self.srv = self.create_service(MoveToXYZ, srv_name, self._on_move)
        self.get_logger().info(f"[MoveServer] Ready: {srv_name}")

    # ---------------- IK ----------------
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

    def move_to_xyz(
        self,
        x: float,
        y: float,
        z: float,
        phi: float = -90.0,
        roll: float = 0.0,
        move_time: float = 2.0,
    ) -> Tuple[bool, str]:
        rad_base = np.arctan2(y, x)
        deg_base = np.degrees(rad_base)
        r_dist = float(np.sqrt(x**2 + y**2))

        ik_result = self.solve_ik_3dof_planar(r_dist, z, phi_deg=phi)
        if ik_result is None:
            return False, f"unreachable (x={x} y={y} z={z} r={r_dist:.2f} phi={phi})"

        deg_shoulder, deg_elbow, deg_wrist_p = ik_result

        target_angles = {
            Config.ID_BASE: float(deg_base),
            Config.ID_SHOULDER: float(deg_shoulder),
            Config.ID_ELBOW: float(deg_elbow),
            Config.ID_WRIST_ROLL: float(roll),
            Config.ID_WRIST_PITCH: float(deg_wrist_p),
        }

        self._publish_gazebo_command(target_angles, move_time)

        if IS_REAL_ROBOT and self.servo_manager:
            self._send_hardware_command(target_angles, move_time)

        return True, "ok"

    # --------------- Gazebo publish ---------------
    def _publish_gazebo_command(self, angles_deg: Dict[int, float], move_time: float) -> None:
        msg = JointTrajectory()
        msg.header.stamp.sec = 0
        msg.header.stamp.nanosec = 0
        msg.joint_names = self.joint_names
        point = JointTrajectoryPoint()

        def get_sim_rad(srv_id: int) -> float:
            cfg = self.SIM_CORRECTION[srv_id]
            raw_target = (angles_deg[srv_id] * cfg["dir"]) + cfg["offset"]
            final_deg = _normalize_angle_deg(float(raw_target))
            return float(np.radians(final_deg))

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

    # --------------- Real robot best-effort ---------------
    def _init_hardware_best_effort(self) -> None:
        try:
            if "/home/jetson/SCSCtrl" not in sys.path:
                sys.path.append("/home/jetson/SCSCtrl")
            from SCSCtrl.scservo_sdk import PortHandler, PacketHandler, GroupSyncWrite, SCS_LOBYTE, SCS_HIBYTE  # type: ignore

            self._PortHandler = PortHandler
            self._PacketHandler = PacketHandler
            self._GroupSyncWrite = GroupSyncWrite
            self._SCS_LOBYTE = SCS_LOBYTE
            self._SCS_HIBYTE = SCS_HIBYTE

            self.port_handler = self._PortHandler(Config.DEVICE_NAME)
            self.packet_handler = self._PacketHandler(1)
            ok = self.port_handler.openPort() and self.port_handler.setBaudRate(Config.BAUDRATE)
            self.get_logger().info("[Hardware] Serial Port Opened." if ok else "[Hardware] Port open failed.")
            self.group_sync_write_pos = self._GroupSyncWrite(self.port_handler, self.packet_handler, Config.ADDR_GOAL_POSITION, 2)
            self.group_sync_write_spd = self._GroupSyncWrite(self.port_handler, self.packet_handler, Config.ADDR_GOAL_SPEED, 2)
            self.servo_manager = True
        except Exception as e:
            self.get_logger().warn(f"[Hardware] init skipped: {e}")
            self.servo_manager = False

    def _send_hardware_command(self, angles_deg: Dict[int, float], move_time: float) -> None:
        try:
            goals, speeds, delta_pos_list = [], [], []
            for sid in self.target_ids:
                angle = float(angles_deg[sid])
                direction = float(self.dirs[sid])
                pos = int(Config.SERVO_INIT_POS[sid] + int((Config.INPUT_RANGE / 180.0) * angle * direction))
                pos = max(0, min(1023, pos))
                goals.append(pos)

                current = int(self.current_servo_pos.get(sid, Config.SERVO_INIT_POS[sid]))
                delta_pos_list.append(abs(pos - current))
                self.current_servo_pos[sid] = pos

            scaling_factor = 1.0 / max(0.001, float(move_time))
            for delta in delta_pos_list:
                calc_speed = int((delta * scaling_factor) * 1.5)
                calc_speed = max(40, min(1000, calc_speed))
                speeds.append(calc_speed)

            for i, sid in enumerate(self.target_ids):
                param_spd = [self._SCS_LOBYTE(speeds[i]), self._SCS_HIBYTE(speeds[i])]
                self.group_sync_write_spd.addParam(sid, param_spd)
                param_pos = [self._SCS_LOBYTE(goals[i]), self._SCS_HIBYTE(goals[i])]
                self.group_sync_write_pos.addParam(sid, param_pos)

            self.group_sync_write_spd.txPacket()
            self.group_sync_write_spd.clearParam()
            self.group_sync_write_pos.txPacket()
            self.group_sync_write_pos.clearParam()
        except Exception as e:
            self.get_logger().warn(f"[Hardware] send skipped: {e}")

    # --------------- Service callback ---------------
    def _on_move(self, request: MoveToXYZ.Request, response: MoveToXYZ.Response):
        ok, msg = self.move_to_xyz(
            float(request.x),
            float(request.y),
            float(request.z),
            float(request.phi),
            float(request.roll),
            float(request.move_time),
        )
        response.success = bool(ok)
        response.message = str(msg)
        return response


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-name", default="jetank1")
    ap.add_argument("--use-sim-time", type=int, default=None, help="1/0 강제. 미지정이면 실기False/시뮬True 자동")
    args = ap.parse_args()

    rclpy.init()
    node = MoveServer(args.robot_name, None if args.use_sim_time is None else bool(args.use_sim_time))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
