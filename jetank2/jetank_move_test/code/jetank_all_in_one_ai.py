#!/usr/bin/env python3
"""
All-in-one controller with Top-CCTV AI target input.

This script mirrors jetank_all_in_one.py but replaces manual x/y input with
Top-CCTV inference results when configured.
"""

# -----------------------------------------------------------
# 이 파일은 다음 기능을 한 번에 묶은 통합 제어 스크립트입니다.
# - Top-CCTV AI 서비스에서 픽셀 좌표 수신
# - 픽셀 → 로봇 좌표 매핑(호모그래피/스케일)
# - 팔(서보) 제어 및 전자석 ON/OFF
# - 컨베이어 제어 및 ROI 기반 자동 정지(선택)
# - 사이클 반복 실행(픽/플레이스 시퀀스)
# -----------------------------------------------------------

import argparse
import math
import os
import platform
import sys
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSProfile
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from builtin_interfaces.msg import Duration
    from std_msgs.msg import Empty
    from std_srvs.srv import SetBool
    from tf2_ros import TransformException
    from tf2_ros.buffer import Buffer
    from tf2_ros.transform_listener import TransformListener
    from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
    from geometry_msgs.msg import TransformStamped
except ImportError:
    print("[Error] ROS2 라이브러리를 찾을 수 없습니다. (PC라면 ros-humble-rclpy 등을 확인하세요)")
    raise SystemExit(1)

from top_cctv_interfaces.srv import GetClosestPose


# Jetson 여부로 하드웨어 모드/시뮬레이션 모드 구분
IS_REAL_ROBOT = (platform.machine() == "aarch64")

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append("/home/jetson/SCSCtrl")

if IS_REAL_ROBOT:
    print(f"[System] Jetson({platform.machine()}) 감지됨 -> 하드웨어 모드 활성화")
    try:
        from magnet_driver import Electromagnet
        from SCSCtrl.scservo_sdk import *  # noqa: F403
        IN1, IN2, PULSE_TIME = 37, 38, 0.2
    except ImportError as exc:
        print(f"[Error] 하드웨어 라이브러리 로드 실패: {exc}")
        IS_REAL_ROBOT = False
else:
    print(f"[System] PC({platform.machine()}) 감지됨 -> 시뮬레이션(Gazebo) 모드 활성화")
    IN1, IN2, PULSE_TIME = 0, 0, 0.2


class Config:
    # 하드웨어/기구 파라미터(서보 ID, 링크 길이, 통신 주소 등)
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
    SERVO_INIT_POS = {1: 510, 2: 545, 3: 524, 4: 512, 5: 561}
    INPUT_RANGE = 850
    ANGLE_RANGE = 180.0
    ADDR_GOAL_POSITION = 42
    ADDR_GOAL_SPEED = 46
    ADDR_PRESENT_POSITION = 56


class JetankController(Node):
    # 단일 Jetank 팔 제어(시뮬레이션/실기 공용)
    def __init__(self, robot_name: str, enable_tf_bridge: bool = True):
        super().__init__(f"{robot_name}_controller")
        # 시뮬레이션에서는 /clock 사용(실기에서는 False가 필요할 수 있음)
        self.set_parameters([rclpy.parameter.Parameter("use_sim_time", rclpy.Parameter.Type.BOOL, True)])
        qos_profile = QoSProfile(depth=10)

        self.robot_name = robot_name
        # Gazebo의 JointTrajectory 컨트롤러로 팔 관절 목표 발행
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            f"/{robot_name}/arm_controller/joint_trajectory",
            qos_profile,
        )

        # Gazebo용 jenga attach/detach 토픽 퍼블리셔
        self.jenga_pubs = {}
        for i in range(1, 5):
            self.jenga_pubs[i] = {
                "attach": self.create_publisher(Empty, f"/{robot_name}/jenga{i}/attach", qos_profile),
                "detach": self.create_publisher(Empty, f"/{robot_name}/jenga{i}/detach", qos_profile),
            }

        # TF 조회를 위한 버퍼/리스너
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        if enable_tf_bridge:
            self.publish_world_bridge()

        # 전자석 기준 프레임/월드 프레임 정의
        self.current_attached_id = None
        self.MAGNET_FRAME = f"{robot_name}/MAGNETIC_BAR_1"
        self.WORLD_FRAME = "world"

        # Gazebo 컨트롤러의 관절 이름 순서(trajectory 메시지와 일치)
        self.joint_names = [
            "Revolute_BEARING",
            "Revolute_ARM_LOW",
            "Revolute_SERVO_UPPER",
            "Revolute_MAGNETIC_BAR",
            "Revolute_SERVO_TOP",
        ]

        # Gazebo 관절 방향/오프셋 보정 값
        self.SIM_CORRECTION = {
            Config.ID_BASE: {"dir": 1, "offset": 0.0},
            Config.ID_SHOULDER: {"dir": -1, "offset": 90.0},
            Config.ID_ELBOW: {"dir": -1, "offset": 0.0},
            Config.ID_WRIST_ROLL: {"dir": 1, "offset": 0.0},
            Config.ID_WRIST_PITCH: {"dir": 1, "offset": 90.0},
        }

        self.servo_manager = None
        self.magnet = None
        if IS_REAL_ROBOT:
            # 실기일 때만 하드웨어 초기화(시리얼, 전자석)
            self.init_hardware()

        # 실기 서보 방향(보드/기구에 따라 축 반전 필요)
        self.dirs = {
            Config.ID_BASE: 1,
            Config.ID_SHOULDER: -1,
            Config.ID_ELBOW: 1,
            Config.ID_WRIST_ROLL: 1,
            Config.ID_WRIST_PITCH: 1,
        }
        self.target_ids = [1, 2, 3, 4, 5]
        self.current_servo_pos = Config.SERVO_INIT_POS.copy()

        if IS_REAL_ROBOT and self.servo_manager:
            # 실기에서 현재 서보 위치 읽어 초기 위치 보정
            for sid in self.target_ids:
                pos = self.read_hardware_pos(sid)
                if pos != -1:
                    self.current_servo_pos[sid] = pos

    def publish_world_bridge(self) -> None:
        # world ↔ empty_world 고정 TF 생성(트리 오류 회피용)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "world"
        t.child_frame_id = "empty_world"
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self.tf_static_broadcaster.sendTransform(t)
        print(">> [TF Bridge] Linked 'world' <-> 'empty_world' to fix tree error.")

    def init_hardware(self) -> None:
        # 서보 통신 포트 오픈 및 동기식 쓰기 초기화
        try:
            self.port_handler = PortHandler(Config.DEVICE_NAME)  # noqa: F405
            self.packet_handler = PacketHandler(1)  # noqa: F405
            if self.port_handler.openPort() and self.port_handler.setBaudRate(Config.BAUDRATE):
                print("[Hardware] Serial Port Opened.")
            else:
                print("[Error] Failed to open port!")
            self.group_sync_write_pos = GroupSyncWrite(  # noqa: F405
                self.port_handler, self.packet_handler, Config.ADDR_GOAL_POSITION, 2
            )
            self.group_sync_write_spd = GroupSyncWrite(  # noqa: F405
                self.port_handler, self.packet_handler, Config.ADDR_GOAL_SPEED, 2
            )
            self.magnet = Electromagnet(in1_pin=IN1, in2_pin=IN2, demag_duration=PULSE_TIME)
            self.servo_manager = True
        except Exception as exc:
            print(f"[Error] Hardware Init Failed: {exc}")

    def read_hardware_pos(self, servo_id: int) -> int:
        # 실기 서보 현재 위치 읽기(통신 실패 시 -1)
        if not IS_REAL_ROBOT:
            return -1
        pos, res, err = self.packet_handler.read2ByteTxRx(  # noqa: F405
            self.port_handler, servo_id, Config.ADDR_PRESENT_POSITION
        )
        return pos if res == COMM_SUCCESS else -1  # noqa: F405

    def find_closest_jenga(self, threshold: float = 0.15) -> Optional[int]:
        # TF로 전자석과 가장 가까운 jenga를 선택
        min_dist = float("inf")
        closest_id = None
        base_frame = self.MAGNET_FRAME
        world_frame = "empty_world"

        from rclpy.duration import Duration as RclpyDuration

        tf_timeout = RclpyDuration(seconds=0.5)

        duration = 2.0
        start_time = self.get_clock().now()
        duration_ns = int(duration * 1e9)
        print(f"\n>> [TF] Gathering TF data for {duration} seconds...")
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            elapsed_ns = (self.get_clock().now() - start_time).nanoseconds
            if elapsed_ns >= duration_ns:
                break

        print(">> [TF] Calculating Coordinates & Distances...")
        print("=" * 100)
        print(f"{'Target':<10} | {'World Coord':<30} | {'Dist from Magnet':<20} | {'Note'}")
        print("-" * 100)

        dist_eps = 1e-4
        for i in range(1, 5):
            target_frame = f"jenga{i}"
            world_pose_str = "Unknown"
            dist_str = "Fail"
            note = ""

            try:
                # world 좌표계에서의 위치 (로그용)
                t_world = self.tf_buffer.lookup_transform(
                    world_frame,
                    target_frame,
                    rclpy.time.Time(),
                    timeout=tf_timeout,
                )
                wx = t_world.transform.translation.x
                wy = t_world.transform.translation.y
                wz = t_world.transform.translation.z
                world_pose_str = f"({wx:.2f}, {wy:.2f}, {wz:.2f})"
            except TransformException:
                pass

            try:
                # 전자석 기준 상대 좌표 → 거리 계산
                t_rel = self.tf_buffer.lookup_transform(
                    base_frame,
                    target_frame,
                    rclpy.time.Time(),
                    timeout=tf_timeout,
                )
                dx = t_rel.transform.translation.x
                dy = t_rel.transform.translation.y
                dz = t_rel.transform.translation.z
                dist_val = math.sqrt(dx**2 + dy**2 + dz**2)
                dist_str = f"{dist_val:.4f} m"

                if dist_val < (min_dist - dist_eps) or (
                    abs(dist_val - min_dist) <= dist_eps and (closest_id is None or i < closest_id)
                ):
                    min_dist = dist_val
                    closest_id = i
            except TransformException as exc:
                note = str(exc).split(".")[0]

            print(f"📦 {target_frame:<7} | {world_pose_str:<30} | {dist_str:<20} | {note}")

        print("=" * 100)

        if closest_id is not None and min_dist <= threshold:
            print(f">> [TF] ✅ Selected: jenga{closest_id} (Closest, Dist: {min_dist:.4f}m)")
            return closest_id

        print(f">> [TF] ❌ None found within {threshold}m (Min dist: {min_dist:.4f}m)")
        return None

    def detach_all(self) -> None:
        # Gazebo에서 jenga 모두 분리(시작 시 초기화 용도)
        print(">> [Init] Detaching ALL jengas (1~4)...")
        msg = Empty()
        for i in range(1, 5):
            if i in self.jenga_pubs:
                self.jenga_pubs[i]["detach"].publish(msg)
        self.current_attached_id = None
        print(">> [Init] Complete.")

    def control_magnet(self, command: str, target_id: Optional[int] = None) -> None:
        # 전자석 ON/OFF 및 Gazebo attach/detach 동기화
        msg = Empty()
        if command == "ON":
            if target_id is None:
                print(">> [Magnet] Scanning for nearest jenga...")
                found_id = self.find_closest_jenga(threshold=0.15)
                if found_id:
                    target_id = found_id
                else:
                    print(">> [Magnet] FAILED: No jenga nearby to attach.")
                    return

            if target_id in self.jenga_pubs:
                self.jenga_pubs[target_id]["attach"].publish(msg)
                self.current_attached_id = target_id
                print(f">> [ROS] 🧲 Attached jenga{target_id} (Topic: /jenga{target_id}/attach)")

            if IS_REAL_ROBOT and self.magnet:
                self.magnet.grab()

        elif command == "OFF":
            # 대상 미지정 시 현재 붙어있는 jenga를 분리
            target_detach = target_id if target_id is not None else self.current_attached_id
            if target_detach is None:
                print(">> [Magnet] Unknown target. Detaching ALL for safety.")
                self.detach_all()
                return

            if target_detach in self.jenga_pubs:
                self.jenga_pubs[target_detach]["detach"].publish(msg)
                print(f">> [ROS] 👋 Detached jenga{target_detach}")

            if target_detach == self.current_attached_id:
                self.current_attached_id = None

            if IS_REAL_ROBOT and self.magnet:
                self.magnet.release()

    def solve_ik_3dof_planar(self, r: float, z: float, phi_deg: float) -> Optional[Tuple[float, float, float]]:
        # r-z 평면 3자유도 IK (베이스 회전은 별도로 계산)
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
        move_time: float = 1.0,
    ) -> None:
        # x,y,z를 각 관절 각도로 변환하여 시뮬/실기 명령 발행
        rad_base = np.arctan2(y, x)
        deg_base = np.degrees(rad_base)
        r_dist = np.sqrt(x**2 + y**2)
        ik_result = self.solve_ik_3dof_planar(r_dist, z, phi_deg=phi)

        if ik_result is None:
            print(f"Unreachable: {x},{y},{z}")
            return

        deg_shoulder, deg_elbow, deg_wrist_p = ik_result
        target_angles = {
            Config.ID_BASE: deg_base,
            Config.ID_SHOULDER: deg_shoulder,
            Config.ID_ELBOW: deg_elbow,
            Config.ID_WRIST_ROLL: roll,
            Config.ID_WRIST_PITCH: deg_wrist_p,
        }
        # Gazebo 및 실기 양쪽으로 동일 명령 전송
        self.publish_gazebo_command(target_angles, move_time)
        if IS_REAL_ROBOT and self.servo_manager:
            self.send_hardware_command(target_angles, move_time)

    def publish_gazebo_command(self, angles_deg: Dict[int, float], move_time: float) -> None:
        # Gazebo용 JointTrajectory 메시지 생성
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

        # 관절 순서에 맞춘 라디안 변환
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
        # 실기 서보 제어값(0~1023) 및 속도 계산 후 동기식 전송
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

        scaling_factor = 1.0 / move_time
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
        # 노드 정리 및 시리얼 포트 닫기
        if IS_REAL_ROBOT and hasattr(self, "port_handler"):
            self.port_handler.closePort()
        self.destroy_node()


class ConveyorController(Node):
    # 컨베이어 ON/OFF를 서비스로 제어
    def __init__(self):
        super().__init__("conveyor_controller")
        # 시뮬레이션에서는 /clock 기반으로 타이밍 제어
        self.set_parameters([rclpy.parameter.Parameter("use_sim_time", rclpy.Parameter.Type.BOOL, True)])
        self.cli = self.create_client(SetBool, "/conveyor/power")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("conveyor service 대기 중...")

    def set_power(self, on: bool) -> bool:
        # /conveyor/power 서비스 호출
        req = SetBool.Request()
        req.data = on
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is None:
            self.get_logger().error(f"서비스 호출 실패: {future.exception()}")
            return False

        res = future.result()
        if not res.success:
            self.get_logger().warn(f"컨베이어 power 응답 실패: {res.message}")
        return res.success

    def wait_sim_seconds(self, seconds: float) -> None:
        # /clock 기반 대기(시뮬레이션 시간)
        start = self.get_clock().now()
        target_ns = int(seconds * 1e9)
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            now = self.get_clock().now()
            elapsed_ns = (now - start).nanoseconds
            if elapsed_ns >= target_ns:
                break

    def wait_wall_seconds(self, seconds: float) -> None:
        # 현재 구현은 sim time과 동일(필요 시 wall time으로 분리 가능)
        self.wait_sim_seconds(seconds)

    def wait_sim_seconds_checked(self, seconds: float, max_wait_wall: float = 2.0) -> None:
        # 타임아웃 감시용 래퍼(현재는 단순 호출)
        self.wait_sim_seconds(seconds)


def _wait_after_move(node: Node, move_time: float, extra: float = 10.0) -> None:
    # 이동 후 안정화를 위한 추가 대기
    _sleep_sim(node, move_time + extra)


def run_jetank1_sequence(
    robot: JetankController,
    x: float,
    y: float,
    roll: float = 0.0,
    hover_z: float = 0.0,
    pick_z: float = -71.0,
    drop_pose: Tuple[float, float, float, float] = (5.0, -150.0, -60.0, 0.0),
    phi: float = -90.0,
    move_time: float = 2.0,
    post_grab_wait: float = 3.0,
    post_release_wait: float = 3.0,
    on_detach: Optional[callable] = None,
) -> None:
    # Jetank1 기본 픽앤플레이스 시퀀스 (고정 시퀀스)
    print(f">> Sequence Start: ({x}, {y}, {pick_z}) Roll={roll}")

    # 1) 접근 → 2) 내려가서 집기 → 3) 상승
    robot.move_to_xyz(x, y, hover_z, phi=phi, roll=0.0, move_time=move_time)
    _wait_after_move(robot, 5.0)
    robot.move_to_xyz(x, y, pick_z, phi=phi, roll=0.0, move_time=move_time)
    _wait_after_move(robot, 5.0)
    robot.control_magnet("ON")
    _sleep_sim(robot, post_grab_wait)
    robot.move_to_xyz(x, y, hover_z, phi=phi, roll=0.0, move_time=move_time)
    _wait_after_move(robot, 5.0)

    drop_x, drop_y, drop_z, drop_roll = drop_pose
    if drop_roll == 0.0:
        drop_roll = roll

    # 4) 드롭 위치로 이동 → 5) 내려놓기 → 6) 복귀
    robot.move_to_xyz(drop_x, drop_y, hover_z, phi=phi, roll=drop_roll, move_time=move_time)
    _wait_after_move(robot, move_time)
    robot.move_to_xyz(drop_x, drop_y, drop_z, phi=phi, roll=drop_roll, move_time=move_time)
    _wait_after_move(robot, move_time)
    robot.control_magnet("OFF")
    if on_detach:
        on_detach()
    _sleep_sim(robot, post_release_wait)
    robot.move_to_xyz(drop_x, drop_y, hover_z, phi=phi, roll=drop_roll, move_time=move_time)
    _wait_after_move(robot, move_time)
    robot.move_to_xyz(150.0, 0.0, 50.0, phi=phi, roll=0.0, move_time=move_time)
    _wait_after_move(robot, move_time)


def run_jetank2_sequence(
    robot: JetankController,
    x: float,
    y: float,
    roll: float = 0.0,
    hover_z: float = 20.0,
    pick_z: float = -71.0,
    drop_pose: Tuple[float, float, float, float] = (0.0, -150.0, -20.0, 0.0),
    pre_drop_pose: Tuple[float, float, float, float] = (0.0, -150.0, 50.0, 0.0),
    phi: float = -90.0,
    move_time: float = 2.0,
    post_move_wait: float = 3.0,
    post_grab_wait: float = 1.0,
    post_release_wait: float = 0.0,
    on_pick_lifted: Optional[callable] = None,
    on_target_reached: Optional[callable] = None,
) -> None:
    # Jetank2 시퀀스: 컨베이어에서 픽업 후 지정 위치에 배치
    print(f">> Sequence Start: ({x}, {y}, {pick_z}) Roll={roll}")

    def _sleep_after_move(extra_time) -> None:
        _sleep_sim(robot, post_move_wait + extra_time)

    # 1) 접근 → 2) 내려가서 집기 → 3) 상승
    robot.move_to_xyz(x, y, hover_z, phi=phi, roll=0.0, move_time=move_time)
    _sleep_after_move(0.0)
    robot.move_to_xyz(x, y, pick_z, phi=phi, roll=0.0, move_time=move_time)
    _sleep_after_move(4.0)
    robot.control_magnet("ON")
    _sleep_sim(robot, post_grab_wait)
    robot.move_to_xyz(x, y, hover_z, phi=phi, roll=0.0, move_time=move_time)
    _sleep_after_move(0.0)
    if on_pick_lifted:
        on_pick_lifted()

    # 4) 프리드롭 위치(높이 확보) → 5) 드롭 위치 → 6) 놓기
    pre_x, pre_y, pre_z, pre_roll = pre_drop_pose
    robot.move_to_xyz(pre_x, pre_y, pre_z, phi=phi, roll=pre_roll, move_time=move_time)
    _sleep_after_move(0.0)

    drop_x, drop_y, drop_z, drop_roll = drop_pose

    robot.move_to_xyz(drop_x, drop_y, drop_z, phi=phi, roll=drop_roll, move_time=move_time)
    _sleep_after_move(4.0)
    if on_target_reached:
        on_target_reached()
    robot.control_magnet("OFF")
    _sleep_sim(robot, post_release_wait)
    robot.move_to_xyz(drop_x, drop_y, 0.0, phi=phi, roll=drop_roll, move_time=move_time)
    _sleep_after_move(0.0)
    robot.move_to_xyz(150.0, 0.0, 50.0, phi=phi, roll=0.0, move_time=move_time)
    _sleep_after_move(0.0)


def parse_command(cmd: str) -> Optional[Tuple[float, float, float]]:
    # "x y roll" 문자열 파싱
    try:
        parts = cmd.replace(",", " ").split()
        if len(parts) < 3:
            return None
        x, y, r = float(parts[0]), float(parts[1]), float(parts[2])
        return x, y, r
    except Exception:
        return None


def prompt_for_command(label: str, default_cmd: Tuple[float, float, float]) -> Tuple[float, float, float]:
    # 사용자 입력으로 목표 좌표를 받는 인터랙티브 입력
    default_str = f"{default_cmd[0]} {default_cmd[1]} {default_cmd[2]}"
    while True:
        user_input = input(f"{label} 입력 (x y roll) [default: {default_str}] >> ").strip()
        if not user_input:
            return default_cmd
        parsed = parse_command(user_input)
        if parsed:
            return parsed
        print("[Error] 형식이 올바르지 않습니다. 예: 150 0 0")


def parse_drop_pose(cmd: str) -> Optional[Tuple[float, float, float, float]]:
    # "x y z roll" 드롭 포즈 파싱
    try:
        parts = cmd.replace(",", " ").split()
        if len(parts) < 4:
            return None
        x, y, z, r = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
        return x, y, z, r
    except Exception:
        return None


def interactive_loop(robot: JetankController, use_jetank2: bool = False) -> None:
    # 수동 테스트용 인터랙티브 모드
    print("=========================================================")
    print(" [Interactive Pick & Place] ")
    print(" Input: x y roll  (e.g., 150 0 0)")
    print(" Exit:  q")
    print("=========================================================")

    while True:
        try:
            user_input = input("\nCommand (x y roll) >> ").strip().lower()
            if user_input in ["q", "quit", "exit"]:
                print("Exiting...")
                break

            if not user_input:
                continue
            parts = user_input.replace(",", " ").split()
            vals = [float(v) for v in parts]
            if len(vals) < 2:
                print("[Error] 최소 x, y 좌표가 필요합니다.")
                continue

            x, y = vals[0], vals[1]
            roll = vals[2] if len(vals) >= 3 else 0.0

            if use_jetank2:
                run_jetank2_sequence(robot, x, y, roll=roll)
            else:
                run_jetank1_sequence(robot, x, y, roll=roll)

        except ValueError:
            print("[Error] 숫자를 입력해주세요.")
        except Exception as exc:
            print(f"[Error] {exc}")

# 기본 캘리브레이션(픽셀 → 로봇 좌표 mm). CLI에서 따로 주지 않으면 이 값 사용.
DEFAULT_PX_POINTS = "123,253;157,253;189,256;223,255"
DEFAULT_WORLD_POINTS = "11,151;10,170;10,190;10,210"
DEFAULT_PX_POINTS_JETANK1 = "44,20;490,20;44,455;490,455"
DEFAULT_WORLD_POINTS_JETANK1 = "-136.8,97.9;-136.8,382.9;143.2,97.9;143.2,382.9"
DEFAULT_PX_POINTS_JETANK2 = "386,177;571,179;385,428;569,425"
DEFAULT_WORLD_POINTS_JETANK2 = "-67.315869,272.923090;-67.315869,414.910919;212.013740,272.923090;212.013740,414.910919"
DEFAULT_J1_MOVE_TIME = 2.0
DEFAULT_J1_EXTRA_WAIT = 2.5
DEFAULT_J1_PRE_GRAB_WAIT = 2.0
DEFAULT_J1_POST_GRAB_WAIT = 1.0
DEFAULT_J1_POST_RELEASE_WAIT = 1.0

def _sleep_sim(node: Node, seconds: float) -> None:
    # /clock(시뮬레이션 시간) 기반 sleep
    start = node.get_clock().now()
    target_ns = int(seconds * 1e9)
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
        now = node.get_clock().now()
        if (now - start).nanoseconds >= target_ns:
            break


def _wait_future_sim(node: Node, future, timeout_sec: Optional[float]) -> bool:
    # /clock 기준으로 future 완료까지 대기
    if timeout_sec is None:
        while rclpy.ok() and not future.done():
            rclpy.spin_once(node, timeout_sec=0.1)
        return future.done()
    start = node.get_clock().now()
    target_ns = int(timeout_sec * 1e9)
    while rclpy.ok() and not future.done():
        rclpy.spin_once(node, timeout_sec=0.1)
        now = node.get_clock().now()
        if (now - start).nanoseconds >= target_ns:
            return False
    return future.done()


def _wait_after_move_sim(node: Node, extra: float) -> None:
    # move 이후 추가 대기(시뮬레이션 시간)
    _sleep_sim(node, extra)


def _start_background_spin(node: Node) -> Tuple[SingleThreadedExecutor, threading.Thread]:
    # 별도 스레드로 ROS spin 돌려 타이머/콜백 동작 유지
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    return executor, thread


@dataclass
class AiPose:
    # Top-CCTV 서비스 응답을 Python 객체로 정리한 형태
    found: bool
    x: float
    y: float
    theta: float
    conf: float


class TopCctvClient(Node):
    # Top-CCTV AI 서비스 클라이언트(픽셀 좌표/회전/신뢰도 획득)
    def __init__(self, name: str = "top_cctv_ai_client", service_name: str = "/top_cctv1/get_closest_pose"):
        super().__init__(name)
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        self.service_name = service_name
        self.cli = self.create_client(GetClosestPose, self.service_name)
        # 서비스가 올라올 때까지 대기
        while rclpy.ok() and not self.cli.wait_for_service(timeout_sec=0.0):
            self.get_logger().info(f"waiting for {self.service_name} service...")
            _sleep_sim(self, 1.0)

    def get_pose(self, target_class: int, timeout_sec: float) -> Optional[AiPose]:
        # 비동기 서비스 호출 후 timeout까지 대기
        req = GetClosestPose.Request()
        req.target_class = int(target_class)
        future = self.cli.call_async(req)
        done = _wait_future_sim(self, future, timeout_sec)
        if not done:
            self.get_logger().warn("Top-CCTV service timeout")
            return None
        try:
            res = future.result()
        except Exception as exc:
            self.get_logger().error(f"Top-CCTV service error: {exc}")
            return None
        # 응답을 AiPose로 변환하여 반환
        return AiPose(
            found=bool(res.found),
            x=float(res.x),
            y=float(res.y),
            theta=float(res.theta),
            conf=float(res.conf),
        )


class ConveyorRoiGuard(Node):
    # ROI(관심영역) 내 물체 감지 시 컨베이어를 자동으로 정지/재시작
    def __init__(
        self,
        image_topic: str = "/jetank/top_cctv2",
        weights: Optional[str] = None,
        conf: float = 0.5,
        device: str = "cuda:0",
        roi_xmin_ratio: float = 0.22,
        roi_xmax_ratio: float = 0.40,
        roi_ymin_ratio: float = 0.42,
        roi_ymax_ratio: float = 0.58,
        target_class: int = -1,
        min_area: int = 0,
        stop_consecutive: int = 1,
        start_consecutive: int = 10,
        stop_delay_sec: float = 0.3,
        roi_debug: bool = True,
        roi_debug_topic: str = "/jetank/top_cctv2/roi_debug",
        infer_every_n: int = 6,
    ):
        super().__init__("conveyor_roi_guard")
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

        # ROI guard에 필요한 라이브러리 동적 로딩
        try:
            import cv2
            from cv_bridge import CvBridge
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import Image
            from std_srvs.srv import SetBool
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(f"ROI guard dependencies missing: {exc}") from exc

        # 가중치 경로 없으면 패키지 공유 폴더에서 로딩
        if weights is None:
            try:
                from ament_index_python.packages import get_package_share_directory
            except ImportError as exc:
                raise RuntimeError("ament_index_python is required to resolve roi weights") from exc
            weights = os.path.join(get_package_share_directory("top_cctv_infer"), "best.pt")

        # ROS 파라미터 선언(외부에서 --ros-args로 튜닝 가능)
        self.declare_parameter("infer_every_n", infer_every_n)
        self.declare_parameter("image_topic", image_topic)
        self.declare_parameter("weights", weights)
        self.declare_parameter("conf", conf)
        self.declare_parameter("device", device)
        self.declare_parameter("roi_xmin_ratio", roi_xmin_ratio)
        self.declare_parameter("roi_xmax_ratio", roi_xmax_ratio)
        self.declare_parameter("roi_ymin_ratio", roi_ymin_ratio)
        self.declare_parameter("roi_ymax_ratio", roi_ymax_ratio)
        self.declare_parameter("target_class", target_class)
        self.declare_parameter("min_area", min_area)
        self.declare_parameter("stop_consecutive", stop_consecutive)
        self.declare_parameter("start_consecutive", start_consecutive)
        self.declare_parameter("roi_stop_delay_sec", stop_delay_sec)
        self.declare_parameter("roi_debug", roi_debug)
        self.declare_parameter("roi_debug_topic", roi_debug_topic)

        # 파라미터 값 캐싱
        self.frame_count = 0
        self.image_topic = self.get_parameter("image_topic").value
        weights = self.get_parameter("weights").value
        self.conf = float(self.get_parameter("conf").value)
        self.device = self.get_parameter("device").value

        # YOLO 모델 및 영상 변환 객체 준비
        self.cv2 = cv2
        self.bridge = CvBridge()
        self.model = YOLO(weights)

        # 이미지 구독
        self.sub = self.create_subscription(
            Image, self.image_topic, self.on_image, qos_profile_sensor_data
        )

        # 디버그 영상 퍼블리셔(ROI 박스/검출 표시)
        self.debug_pub = None
        if bool(self.get_parameter("roi_debug").value):
            self.debug_topic = str(self.get_parameter("roi_debug_topic").value)
            self.debug_pub = self.create_publisher(Image, self.debug_topic, 10)

        # 컨베이어 전원 서비스 클라이언트
        self._setbool_type = SetBool
        self.cli = self.create_client(self._setbool_type, "/conveyor/power")

        # ROI 히트/카운터/상태 변수
        self.cache_roi_hit = False
        self.stop_count = 0
        self.start_count = 0
        self.power_on = None
        self.last_call_t = 0.0
        self.stop_pending_until = None
        self.stop_event = threading.Event()
        self.auto_start_blocked = False
        self.desired_power = True
        self.ensure_timer = self.create_timer(0.5, self._ensure_power_state)

        self.get_logger().info(f"ROI guard subscribed: {self.image_topic}")
        self.get_logger().info(f"ROI guard weights: {weights} / device={self.device} / conf={self.conf}")
        self.get_logger().info("ROI guard default: conveyor ON until ROI hit.")
        if self.debug_pub is not None:
            self.get_logger().info(f"ROI debug topic: {self.debug_topic}")

    def _roi_bounds(self, frame_w: int, frame_h: int) -> Tuple[float, float, float, float]:
        # 프레임 크기 기준 ROI 영역(비율)을 픽셀 좌표로 계산
        xmin_r = float(self.get_parameter("roi_xmin_ratio").value)
        xmax_r = float(self.get_parameter("roi_xmax_ratio").value)
        ymin_r = float(self.get_parameter("roi_ymin_ratio").value)
        ymax_r = float(self.get_parameter("roi_ymax_ratio").value)
        x_min = frame_w * xmin_r
        x_max = frame_w * xmax_r
        y_min = frame_h * ymin_r
        y_max = frame_h * ymax_r
        if x_min > x_max:
            x_min, x_max = x_max, x_min
        if y_min > y_max:
            y_min, y_max = y_max, y_min
        return x_min, x_max, y_min, y_max

    def _roi_hit_any(self, frame_w: int, frame_h: int, centers_xy: np.ndarray) -> bool:
        # 중심점들이 ROI에 하나라도 포함되는지 체크
        x_min, x_max, y_min, y_max = self._roi_bounds(frame_w, frame_h)

        if centers_xy.size == 0:
            return False

        cx = centers_xy[:, 0]
        cy = centers_xy[:, 1]
        hit = (cx >= x_min) & (cx <= x_max) & (cy >= y_min) & (cy <= y_max)
        return bool(np.any(hit))

    def _now_sec(self) -> float:
        # 현재 시간(초) - 시뮬레이션 시간 기반
        return self.get_clock().now().nanoseconds * 1e-9

    def _call_conveyor_power(self, on: bool) -> None:
        # 컨베이어 ON/OFF 서비스 호출 (짧은 간격 호출은 디바운스)
        if self.power_on is not None and self.power_on == on:
            return

        now = self._now_sec()
        if now - self.last_call_t < 0.2:
            return

        if not self.cli.service_is_ready():
            return

        req = self._setbool_type.Request()
        req.data = bool(on)

        fut = self.cli.call_async(req)
        self.last_call_t = now

        def _done_cb(f):
            try:
                resp = f.result()
                if resp is not None and resp.success:
                    self.power_on = on
                    if on:
                        self.stop_event.clear()
                    else:
                        self.stop_event.set()
                    self.get_logger().info(f"/conveyor/power -> {on} (ok) msg={resp.message}")
                else:
                    self.get_logger().warn(
                        f"/conveyor/power -> {on} (fail) msg={resp.message if resp else 'None'}"
                    )
            except Exception as exc:
                self.get_logger().error(f"service call exception: {exc}")

        fut.add_done_callback(_done_cb)

    def block_auto_start(self) -> None:
        # 자동 재시작을 막아야 할 때 호출(예: 픽업 중)
        self.auto_start_blocked = True
        self.start_count = 0
        self.get_logger().info("ROI guard: auto-start blocked")

    def unblock_auto_start(self) -> None:
        # 자동 재시작 허용
        if not self.auto_start_blocked:
            return
        self.auto_start_blocked = False
        self.start_count = 0
        self.get_logger().info("ROI guard: auto-start unblocked")

    def wait_for_stop(self, timeout_sec: Optional[float], clock_node: Optional[Node] = None) -> bool:
        # 컨베이어가 정지할 때까지 대기(옵션으로 timeout)
        if timeout_sec is None:
            return self.stop_event.wait()
        if clock_node is None:
            clock_node = self
        start = clock_node.get_clock().now()
        timeout_ns = int(timeout_sec * 1e9)
        while rclpy.ok():
            if self.stop_event.is_set():
                return True
            elapsed_ns = (clock_node.get_clock().now() - start).nanoseconds
            if elapsed_ns >= timeout_ns:
                return False
            rclpy.spin_once(clock_node, timeout_sec=0.1)
        return False

    def _ensure_power_state(self) -> None:
        # 주기적으로 desired_power와 실제 상태를 동기화
        if self.desired_power is None:
            return
        if self.power_on is not None and self.power_on == self.desired_power:
            return
        self._call_conveyor_power(self.desired_power)

    def on_image(self, msg) -> None:
        # 이미지 수신 시 주기적으로 YOLO 추론 후 ROI 히트 여부 판단
        self.frame_count += 1
        n = int(self.get_parameter("infer_every_n").value)
        if n > 1 and (self.frame_count % n) != 0:
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return

        h, w = frame.shape[:2]
        result = self.model.predict(frame, conf=self.conf, device=self.device, verbose=False)[0]
        obb = getattr(result, "obb", None)

        roi_hit = False
        centers_to_draw = None
        if obb is not None and obb.xywhr is not None and len(obb.xywhr) > 0:
            # OBB 결과에서 중심점 추출
            xywhr = obb.xywhr.cpu().numpy()
            cls = obb.cls.cpu().numpy().astype(int) if obb.cls is not None else None

            target_class = int(self.get_parameter("target_class").value)
            min_area = int(self.get_parameter("min_area").value)

            idxs = np.arange(xywhr.shape[0])
            if cls is not None and target_class >= 0:
                idxs = idxs[cls == target_class]

            if idxs.size > 0:
                ww = xywhr[idxs, 2]
                hh = xywhr[idxs, 3]
                area = ww * hh
                if min_area > 0:
                    idxs = idxs[area >= float(min_area)]

            if idxs.size > 0:
                centers = xywhr[idxs, 0:2]
                centers_to_draw = centers
                roi_hit = self._roi_hit_any(w, h, centers)

        self.cache_roi_hit = roi_hit

        # 연속 히트/미히트 카운터로 컨베이어 stop/start 판단
        stop_n = int(self.get_parameter("stop_consecutive").value)
        start_n = int(self.get_parameter("start_consecutive").value)

        if roi_hit:
            self.stop_count += 1
            self.start_count = 0
        else:
            if self.auto_start_blocked:
                self.start_count = 0
            else:
                self.start_count += 1
            self.stop_count = 0
            self.stop_pending_until = None

        if self.stop_count >= stop_n:
            delay = float(self.get_parameter("roi_stop_delay_sec").value)
            if delay > 0.0:
                now = self._now_sec()
                if self.stop_pending_until is None:
                    self.stop_pending_until = now + delay
                if now >= self.stop_pending_until:
                    self.desired_power = False
                    self._call_conveyor_power(False)
            else:
                self.desired_power = False
                self._call_conveyor_power(False)

        if self.start_count >= start_n and not self.auto_start_blocked:
            self.desired_power = True
            self._call_conveyor_power(True)

        if self.debug_pub is not None:
            # 디버그: ROI 박스와 중심점 표시
            annotated = frame.copy()
            x_min, x_max, y_min, y_max = self._roi_bounds(w, h)
            box_color = (0, 0, 255) if roi_hit else (0, 255, 255)
            self.cv2.rectangle(
                annotated,
                (int(x_min), int(y_min)),
                (int(x_max), int(y_max)),
                box_color,
                2,
            )
            if centers_to_draw is not None and centers_to_draw.size > 0:
                in_roi = (
                    (centers_to_draw[:, 0] >= x_min)
                    & (centers_to_draw[:, 0] <= x_max)
                    & (centers_to_draw[:, 1] >= y_min)
                    & (centers_to_draw[:, 1] <= y_max)
                )
                for idx, (cx, cy) in enumerate(centers_to_draw):
                    color = (0, 0, 255) if in_roi[idx] else (255, 0, 0)
                    self.cv2.circle(annotated, (int(cx), int(cy)), 4, color, -1)

            out = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            out.header = msg.header
            self.debug_pub.publish(out)


class BaseMapper:
    # 픽셀 → 월드 좌표 변환 인터페이스
    def map_point(self, px: float, py: float) -> Optional[Tuple[float, float]]:
        raise NotImplementedError


class PixelToWorldMapper(BaseMapper):
    # 호모그래피 또는 스케일(원점/축 반전 포함) 기반 변환
    def __init__(
        self,
        homography: Optional[np.ndarray],
        px_origin: Tuple[float, float],
        world_origin: Tuple[float, float],
        mm_per_px: Tuple[float, float],
        swap_xy: bool,
        invert_x: bool,
        invert_y: bool,
    ):
        self.homography = homography
        self.px_origin = px_origin
        self.world_origin = world_origin
        self.mm_per_px = mm_per_px
        self.swap_xy = swap_xy
        self.invert_x = invert_x
        self.invert_y = invert_y

    def map_point(self, px: float, py: float) -> Optional[Tuple[float, float]]:
        if self.homography is not None:
            # 호모그래피 행렬로 픽셀 좌표를 월드 좌표로 변환
            vec = np.array([px, py, 1.0], dtype=np.float64)
            out = self.homography @ vec
            if abs(out[2]) < 1e-9:
                return None
            return float(out[0] / out[2]), float(out[1] / out[2])

        # 단순 스케일 변환(원점/축 스왑/반전 포함)
        dx = px - self.px_origin[0]
        dy = py - self.px_origin[1]
        if self.swap_xy:
            dx, dy = dy, dx
        if self.invert_x:
            dx = -dx
        if self.invert_y:
            dy = -dy
        x = self.world_origin[0] + dx * self.mm_per_px[0]
        y = self.world_origin[1] + dy * self.mm_per_px[1]
        return x, y


class LinearAxisMapper(BaseMapper):
    # 점들이 거의 직선상일 때 x축 기준 1D 보간으로 매핑
    def __init__(self, px_points: List[Tuple[float, float]], world_points: List[Tuple[float, float]]):
        px = np.asarray(px_points, dtype=np.float64)
        world = np.asarray(world_points, dtype=np.float64)
        order = np.argsort(px[:, 0])
        self.px_x = px[order, 0]
        self.world_x = world[order, 0]
        self.world_y = world[order, 1]

    def map_point(self, px: float, py: float) -> Optional[Tuple[float, float]]:
        # 픽셀 x만 사용해 월드 x/y를 각각 보간
        x = float(np.interp(px, self.px_x, self.world_x))
        y = float(np.interp(px, self.px_x, self.world_y))
        return x, y


@dataclass
class AiConfig:
    # AI 서비스 호출/매핑/필터 설정을 묶은 구성체
    client: TopCctvClient
    mapper: BaseMapper
    target_class: int
    min_conf: float
    theta_unit: str
    use_theta_roll: bool
    roll_scale: float
    roll_offset: float
    retries: int
    retry_wait: float
    timeout_sec: float

    def request_command(self, default_roll: float) -> Optional[Tuple[float, float, float]]:
        # AI 응답을 여러 번 시도하고, 조건 만족 시 (x,y,roll) 반환
        for attempt in range(1, self.retries + 1):
            pose = self.client.get_pose(self.target_class, self.timeout_sec)
            if pose is None or not pose.found:
                self.client.get_logger().warn(f"[AI] attempt {attempt}: no detection")
                _sleep_sim(self.client, self.retry_wait)
                continue
            if pose.conf < self.min_conf:
                self.client.get_logger().warn(
                    f"[AI] attempt {attempt}: conf {pose.conf:.2f} < {self.min_conf:.2f}"
                )
                _sleep_sim(self.client, self.retry_wait)
                continue
            mapped = self.mapper.map_point(pose.x, pose.y)
            if mapped is None:
                self.client.get_logger().warn(f"[AI] attempt {attempt}: mapping failed")
                _sleep_sim(self.client, self.retry_wait)
                continue

            roll = default_roll
            if self.use_theta_roll:
                # theta를 roll에 반영(단위/스케일/오프셋 적용)
                theta_deg = pose.theta if self.theta_unit == "deg" else pose.theta * 180.0 / math.pi
                roll = self.roll_offset + (self.roll_scale * theta_deg)
            return mapped[0], mapped[1], roll
        return None


def _parse_pair(text: str, label: str) -> Tuple[float, float]:
    # "x,y" 문자열 파싱
    parts = text.replace(",", " ").split()
    if len(parts) != 2:
        raise ValueError(f"{label} must be 'x,y' (got: {text})")
    return float(parts[0]), float(parts[1])


def _parse_points(text: str, label: str) -> List[Tuple[float, float]]:
    # "x1,y1;x2,y2;..." 문자열 파싱
    pts = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        pts.append(_parse_pair(item, label))
    return pts


def _is_collinear(points: List[Tuple[float, float]], tol: float = 5e-2) -> bool:
    # 점들이 거의 일직선인지 판별(호모그래피 적용 여부 결정)
    if len(points) < 3:
        return True
    pts = np.asarray(points, dtype=np.float64)
    span = pts.max(axis=0) - pts.min(axis=0)
    if span.max() > 0 and (span.min() / span.max()) < tol:
        return True
    pts = pts - pts.mean(axis=0)
    _, s, _ = np.linalg.svd(pts, full_matrices=False)
    if s[0] == 0:
        return True
    return (s[1] / s[0]) < tol


def _compute_homography(
    px_points: List[Tuple[float, float]],
    world_points: List[Tuple[float, float]],
) -> np.ndarray:
    # DLT 방식으로 호모그래피 계산
    if len(px_points) < 4 or len(px_points) != len(world_points):
        raise ValueError("homography requires 4+ matching point pairs")

    a_rows = []
    for (x, y), (X, Y) in zip(px_points, world_points):
        a_rows.append([-x, -y, -1, 0, 0, 0, x * X, y * X, X])
        a_rows.append([0, 0, 0, -x, -y, -1, x * Y, y * Y, Y])
    a = np.asarray(a_rows, dtype=np.float64)
    _, _, vt = np.linalg.svd(a)
    h = vt[-1, :].reshape(3, 3)
    if abs(h[2, 2]) < 1e-9:
        return h
    return h / h[2, 2]


def _build_mapper(
    args: argparse.Namespace,
    logger: Node,
    px_points_text_override: Optional[str] = None,
    world_points_text_override: Optional[str] = None,
    default_px_points: str = DEFAULT_PX_POINTS,
    default_world_points: str = DEFAULT_WORLD_POINTS,
) -> Optional[BaseMapper]:
    # CLI/기본값을 조합해 매퍼를 구성
    if px_points_text_override is None and world_points_text_override is None:
        if args.px_points or args.world_points:
            px_points_text = args.px_points
            world_points_text = args.world_points
        else:
            px_points_text = default_px_points
            world_points_text = default_world_points
    else:
        if px_points_text_override is None:
            px_points_text = args.px_points or default_px_points
        else:
            px_points_text = px_points_text_override
        if world_points_text_override is None:
            world_points_text = args.world_points or default_world_points
        else:
            world_points_text = world_points_text_override

    px_points = _parse_points(px_points_text, "px_points") if px_points_text else []
    world_points = _parse_points(world_points_text, "world_points") if world_points_text else []

    homography = None
    if args.map_mode in ("auto", "homography") and px_points and world_points:
        if _is_collinear(px_points) or _is_collinear(world_points):
            logger.get_logger().warn("[AI] px points are nearly collinear; using 1D interpolation")
            return LinearAxisMapper(px_points, world_points)
        try:
            homography = _compute_homography(px_points, world_points)
            logger.get_logger().info(f"[AI] Homography ready ({len(px_points)} points)")
        except Exception as exc:
            logger.get_logger().error(f"[AI] Homography build failed: {exc}")
            homography = None

    if homography is None:
        if args.map_mode == "homography":
            logger.get_logger().error("[AI] map_mode=homography requires --px-points and --world-points")
            return None
        if args.map_mode in ("auto", "scale", "pixel"):
            if args.map_mode == "pixel":
                # 픽셀 좌표를 그대로 사용(디버그/실험용)
                mm_per_px = (1.0, 1.0)
            else:
                if args.mm_per_px_x is None or args.mm_per_px_y is None:
                    logger.get_logger().warn("[AI] mm-per-px not set; mapping disabled")
                    return None
                mm_per_px = (args.mm_per_px_x, args.mm_per_px_y)
        else:
            return None
    else:
        mm_per_px = (1.0, 1.0)

    px_origin = _parse_pair(args.px_origin, "px_origin") if args.px_origin else (0.0, 0.0)
    world_origin = _parse_pair(args.world_origin, "world_origin") if args.world_origin else (0.0, 0.0)

    return PixelToWorldMapper(
        homography=homography,
        px_origin=px_origin,
        world_origin=world_origin,
        mm_per_px=mm_per_px,
        swap_xy=args.swap_xy,
        invert_x=args.invert_x,
        invert_y=args.invert_y,
    )


def _ai_enabled_for(ai_for: str, robot_key: str) -> bool:
    # ai_for 옵션에 따라 특정 로봇만 AI 적용
    return ai_for in ("both", robot_key)


def _resolve_command(
    label: str,
    default_cmd: Tuple[float, float, float],
    ai: Optional[AiConfig],
    allow_manual: bool,
) -> Tuple[float, float, float]:
    # AI 결과 우선 사용, 실패 시 수동 입력(옵션)
    if ai is not None:
        cmd = ai.request_command(default_roll=default_cmd[2])
        if cmd:
            print(f">> [AI] {label} -> x={cmd[0]:.2f}, y={cmd[1]:.2f}, roll={cmd[2]:.1f}")
            return cmd
        if not allow_manual:
            raise RuntimeError(f"[AI] {label} target not found")
        print(f">> [AI] {label} fallback to manual input")
    return prompt_for_command(label, default_cmd)


def run_jetank1_sequence_ai(
    robot: JetankController,
    x: float,
    y: float,
    roll: float = 0.0,
    hover_z: float = 0.0,
    pick_z: float = -71.0,
    drop_pose: Tuple[float, float, float, float] = (5.0, -150.0, -60.0, 0.0),
    phi: float = -90.0,
    move_time: float = DEFAULT_J1_MOVE_TIME,
    extra_wait: float = DEFAULT_J1_EXTRA_WAIT,
    pre_grab_wait: float = DEFAULT_J1_PRE_GRAB_WAIT,
    post_grab_wait: float = DEFAULT_J1_POST_GRAB_WAIT,
    post_release_wait: float = DEFAULT_J1_POST_RELEASE_WAIT,
    on_detach: Optional[callable] = None,
) -> None:
    # AI 결과를 사용한 Jetank1 픽앤플레이스 시퀀스
    print(f">> Sequence Start: ({x}, {y}, {pick_z}) Roll={roll}")

    # 접근 → 집기 → 상승
    robot.move_to_xyz(x, y, hover_z, phi=phi, roll=0.0, move_time=move_time)
    _wait_after_move_sim(robot, extra_wait)
    robot.move_to_xyz(x, y, pick_z, phi=phi, roll=0.0, move_time=move_time)
    _wait_after_move_sim(robot, extra_wait)
    _sleep_sim(robot, pre_grab_wait)
    robot.control_magnet("ON")
    _sleep_sim(robot, post_grab_wait)
    robot.move_to_xyz(x, y, hover_z, phi=phi, roll=0.0, move_time=move_time)
    _wait_after_move_sim(robot, extra_wait)

    # 드롭 위치로 이동 → 내려놓기 → 복귀
    drop_x, drop_y, drop_z, drop_roll = drop_pose
    if drop_roll == 0.0:
        drop_roll = roll

    robot.move_to_xyz(drop_x, drop_y, hover_z, phi=phi, roll=drop_roll, move_time=move_time)
    _wait_after_move_sim(robot, extra_wait)
    robot.move_to_xyz(drop_x, drop_y, drop_z, phi=phi, roll=drop_roll, move_time=move_time)
    _wait_after_move_sim(robot, extra_wait)
    robot.control_magnet("OFF")
    if on_detach:
        on_detach()
    _sleep_sim(robot, post_release_wait)
    robot.move_to_xyz(drop_x, drop_y, hover_z, phi=phi, roll=drop_roll, move_time=move_time)
    _wait_after_move_sim(robot, extra_wait)
    robot.move_to_xyz(150.0, 0.0, 50.0, phi=phi, roll=0.0, move_time=move_time)
    _wait_after_move_sim(robot, extra_wait)


def run_cycle_with_ai(
    jetank1: JetankController,
    jetank2: JetankController,
    conveyor: ConveyorController,
    roi_guard: Optional[ConveyorRoiGuard],
    repeat_count: int,
    jetank1_cmd: str,
    jetank2_cmd: str,
    jetank1_y_increment: float,
    conveyor_duration: float,
    jetank2_drop_override: Optional[str],
    ai_for: str,
    ai_config_j1: Optional[AiConfig],
    ai_config_j2: Optional[AiConfig],
    manual_fallback: bool,
    roi_control_enabled: bool,
    roi_wait_stop_sec: float,
    roi_post_stop_delay_sec: float,
) -> None:
    # 전체 사이클(픽업 → 컨베이어 → 배치)을 반복 실행
    base_xyz = parse_command(jetank1_cmd) or (11.0, 151.0, 0.0)
    jetank2_xyz = parse_command(jetank2_cmd) or (0.0, 149.0, 0.0)

    # 고정 팔레타이징 드롭 포즈 시퀀스
    drop_sequence = [
        (0.0, -200.0, -47.0, 0.0),
        (-2.0, -160.0, -47.0, 0.0),
        (-11.0, -175.0, -35.0, 90.0),
        (10.0, -175.0, -35.0, 90.0),
    ]

    drop_override = parse_drop_pose(jetank2_drop_override) if jetank2_drop_override else None

    if roi_control_enabled:
        conveyor.get_logger().info("ROI guard active: conveyor control handled automatically.")
    else:
        conveyor.get_logger().info("컨베이어 초기 상태: ON (계속 회전)")
        conveyor.set_power(True)

    for cycle in range(1, repeat_count + 1):
        print(f"\n=== Cycle {cycle}/{repeat_count} ===")

        # Jetank1 목표 좌표 결정(기본값/AI)
        base_x, base_y, base_r = base_xyz
        default_x = base_x if cycle == 1 else 10.0
        if cycle == 1:
            default_y = base_y
        elif cycle == 2:
            default_y = 170.0
        else:
            default_y = 170.0 + (cycle - 2) * jetank1_y_increment
        default_j1 = (default_x, default_y, base_r)

        ai_j1 = ai_config_j1 if _ai_enabled_for(ai_for, "jetank1") else None
        j1x, j1y, j1r = _resolve_command("Jetank1", default_j1, ai_j1, manual_fallback)

        def stop_conveyor_after_detach() -> None:
            if roi_control_enabled:
                conveyor.get_logger().info("ROI guard active: skip manual conveyor stop.")
                return
            conveyor.get_logger().info(f"Jetank1 Detach 이후 {conveyor_duration:.1f}s 대기...")
            conveyor.wait_sim_seconds_checked(conveyor_duration)
            conveyor.get_logger().info("컨베이어 OFF (/clock 기준)")
            conveyor.set_power(False)

        # Jetank1 픽업 시퀀스 실행
        run_jetank1_sequence_ai(
            jetank1,
            j1x,
            j1y,
            roll=j1r,
            on_detach=stop_conveyor_after_detach,
        )

        conveyor.get_logger().info("컨베이어 완료. Jetank2 입력 대기...")
        if roi_control_enabled and roi_guard is not None:
            stopped = True
            if roi_wait_stop_sec <= 0.0:
                conveyor.get_logger().info("ROI guard: conveyor stop 대기...")
                roi_guard.wait_for_stop(None)
            else:
                conveyor.get_logger().info(f"ROI guard: conveyor stop 최대 {roi_wait_stop_sec:.1f}s 대기...")
                if not roi_guard.wait_for_stop(roi_wait_stop_sec, clock_node=conveyor):
                    conveyor.get_logger().warn("ROI guard stop timeout; continuing to Jetank2.")
                    stopped = False
            if stopped and roi_post_stop_delay_sec > 0.0:
                conveyor.get_logger().info(
                    f"ROI guard: stop 후 {roi_post_stop_delay_sec:.1f}s 대기 후 Jetank2 추론"
                )
                conveyor.wait_sim_seconds_checked(roi_post_stop_delay_sec)
        ai_j2 = ai_config_j2 if _ai_enabled_for(ai_for, "jetank2") else None
        j2x, j2y, j2r = _resolve_command("Jetank2", jetank2_xyz, ai_j2, manual_fallback)

        # 사이클 인덱스에 맞는 드롭 포즈 선택
        drop_pose = drop_override or drop_sequence[(cycle - 1) % len(drop_sequence)]

        def start_conveyor_after_jetank2_target() -> None:
            if roi_control_enabled:
                conveyor.get_logger().info("ROI guard active: skip manual conveyor start.")
                return
            conveyor.get_logger().info("Jetank2 목표 지점 도착. 컨베이어 ON")
            conveyor.set_power(True)

        def allow_conveyor_after_pick() -> None:
            if roi_control_enabled and roi_guard is not None:
                roi_guard.unblock_auto_start()

        if roi_control_enabled and roi_guard is not None:
            roi_guard.block_auto_start()

        # Jetank2 픽업/배치 시퀀스 실행
        run_jetank2_sequence(
            jetank2,
            j2x,
            j2y,
            roll=j2r,
            drop_pose=drop_pose,
            on_pick_lifted=allow_conveyor_after_pick,
            on_target_reached=start_conveyor_after_jetank2_target,
        )


def main() -> None:
    # CLI 인자 파싱 및 전체 실행 흐름 진입점
    parser = argparse.ArgumentParser(description="Jetank all-in-one controller (Top-CCTV AI)")
    parser.add_argument("--mode", choices=["cycle", "jetank1", "jetank2", "conveyor"], default="cycle")
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument("--jetank1-cmd", default=os.environ.get("JETANK1_CMD", "11 151 0"))
    parser.add_argument("--jetank2-cmd", default=os.environ.get("JETANK2_CMD", "0 149 0"))
    parser.add_argument("--jetank1-y-increment", type=float, default=20.0)
    parser.add_argument("--jetank2-drop", default=None)
    parser.add_argument("--conveyor-duration", type=float, default=12.8)

    parser.add_argument("--no-ai", action="store_true", help="disable Top-CCTV AI")
    parser.add_argument("--ai-for", choices=["jetank1", "jetank2", "both"], default="both")
    parser.add_argument("--ai-target-class", type=int, default=-1)
    parser.add_argument("--ai-min-conf", type=float, default=0.5)
    parser.add_argument("--ai-timeout", type=float, default=1.0)
    parser.add_argument("--ai-retries", type=int, default=5)
    parser.add_argument("--ai-retry-wait", type=float, default=0.2)
    parser.add_argument("--ai-no-manual-fallback", action="store_true")
    parser.add_argument("--ai-service-jetank1", default="/top_cctv1/get_closest_pose")
    parser.add_argument("--ai-service-jetank2", default="/top_cctv2/get_closest_pose")

    parser.add_argument(
        "--roi-guard",
        dest="roi_guard",
        action="store_true",
        default=True,
        help="enable conveyor ROI guard (camera2, default on)",
    )
    parser.add_argument(
        "--no-roi-guard",
        dest="roi_guard",
        action="store_false",
        help="disable conveyor ROI guard",
    )
    parser.add_argument("--roi-image-topic", default="/jetank/top_cctv2")
    parser.add_argument("--roi-weights", default=None)
    parser.add_argument("--roi-conf", type=float, default=0.5)
    parser.add_argument("--roi-device", default="cuda:0")
    parser.add_argument("--roi-infer-every-n", type=int, default=6)
    parser.add_argument("--roi-xmin-ratio", type=float, default=0.22)
    parser.add_argument("--roi-xmax-ratio", type=float, default=0.40)
    parser.add_argument("--roi-ymin-ratio", type=float, default=0.42)
    parser.add_argument("--roi-ymax-ratio", type=float, default=0.58)
    parser.add_argument("--roi-target-class", type=int, default=-1)
    parser.add_argument("--roi-min-area", type=int, default=0)
    parser.add_argument("--roi-stop-consecutive", type=int, default=1)
    parser.add_argument("--roi-start-consecutive", type=int, default=10)
    parser.add_argument("--roi-stop-delay-sec", type=float, default=0.3)
    parser.add_argument("--roi-wait-stop-sec", type=float, default=0.0)
    parser.add_argument("--roi-post-stop-delay-sec", type=float, default=3.0)
    parser.add_argument(
        "--roi-debug",
        dest="roi_debug",
        action="store_true",
        default=True,
        help="publish ROI debug overlay image",
    )
    parser.add_argument(
        "--no-roi-debug",
        dest="roi_debug",
        action="store_false",
        help="disable ROI debug overlay image",
    )
    parser.add_argument("--roi-debug-topic", default="/jetank/top_cctv2/roi_debug")

    parser.add_argument("--theta-unit", choices=["rad", "deg"], default="rad")
    parser.add_argument("--use-theta-roll", action="store_true")
    parser.add_argument("--roll-scale", type=float, default=1.0)
    parser.add_argument("--roll-offset", type=float, default=0.0)

    parser.add_argument("--map-mode", choices=["auto", "homography", "scale", "pixel", "none"], default="homography")
    parser.add_argument("--px-points", type=str, default=None)
    parser.add_argument("--world-points", type=str, default=None)
    parser.add_argument("--mm-per-px-x", type=float, default=None)
    parser.add_argument("--mm-per-px-y", type=float, default=None)
    parser.add_argument("--px-origin", type=str, default=None)
    parser.add_argument("--world-origin", type=str, default="0.5,1.0")
    parser.add_argument("--swap-xy", action="store_true")
    parser.add_argument("--invert-x", action="store_true")
    parser.add_argument("--invert-y", action="store_true")
    parser.add_argument("--j1-px-points", type=str, default=None)
    parser.add_argument("--j1-world-points", type=str, default=None)
    parser.add_argument("--j2-px-points", type=str, default=None)
    parser.add_argument("--j2-world-points", type=str, default=None)

    args = parser.parse_args()

    rclpy.init()
    ai_client_j1 = None
    ai_client_j2 = None
    ai_config_j1 = None
    ai_config_j2 = None
    roi_guard = None
    roi_executor = None
    roi_thread = None
    try:
        if args.roi_guard:
            # ROI guard는 별도 스레드로 spin
            try:
                roi_guard = ConveyorRoiGuard(
                    image_topic=args.roi_image_topic,
                    weights=args.roi_weights,
                    conf=args.roi_conf,
                    device=args.roi_device,
                    roi_xmin_ratio=args.roi_xmin_ratio,
                    roi_xmax_ratio=args.roi_xmax_ratio,
                    roi_ymin_ratio=args.roi_ymin_ratio,
                    roi_ymax_ratio=args.roi_ymax_ratio,
                    target_class=args.roi_target_class,
                    min_area=args.roi_min_area,
                    stop_consecutive=args.roi_stop_consecutive,
                    start_consecutive=args.roi_start_consecutive,
                    stop_delay_sec=args.roi_stop_delay_sec,
                    roi_debug=args.roi_debug,
                    roi_debug_topic=args.roi_debug_topic,
                    infer_every_n=args.roi_infer_every_n,
                )
                roi_executor, roi_thread = _start_background_spin(roi_guard)
            except Exception as exc:
                print(f"[ROI] guard disabled: {exc}")
                roi_guard = None
                roi_executor = None
                roi_thread = None

        if not args.no_ai and args.map_mode != "none":
            # AI 클라이언트 및 매퍼 구성
            if _ai_enabled_for(args.ai_for, "jetank1"):
                ai_client_j1 = TopCctvClient(
                    name="top_cctv_ai_client_1",
                    service_name=args.ai_service_jetank1,
                )
                mapper_j1 = _build_mapper(
                    args,
                    ai_client_j1,
                    px_points_text_override=args.j1_px_points,
                    world_points_text_override=args.j1_world_points,
                    default_px_points=DEFAULT_PX_POINTS_JETANK1,
                    default_world_points=DEFAULT_WORLD_POINTS_JETANK1,
                )
                if mapper_j1 is None:
                    ai_client_j1.get_logger().warn("[AI] mapper not configured; Jetank1 AI disabled")
                else:
                    ai_config_j1 = AiConfig(
                        client=ai_client_j1,
                        mapper=mapper_j1,
                        target_class=args.ai_target_class,
                        min_conf=args.ai_min_conf,
                        theta_unit=args.theta_unit,
                        use_theta_roll=args.use_theta_roll,
                        roll_scale=args.roll_scale,
                        roll_offset=args.roll_offset,
                        retries=args.ai_retries,
                        retry_wait=args.ai_retry_wait,
                        timeout_sec=args.ai_timeout,
                    )

            if _ai_enabled_for(args.ai_for, "jetank2"):
                ai_client_j2 = TopCctvClient(
                    name="top_cctv_ai_client_2",
                    service_name=args.ai_service_jetank2,
                )
                mapper_j2 = _build_mapper(
                    args,
                    ai_client_j2,
                    px_points_text_override=args.j2_px_points,
                    world_points_text_override=args.j2_world_points,
                    default_px_points=DEFAULT_PX_POINTS_JETANK2,
                    default_world_points=DEFAULT_WORLD_POINTS_JETANK2,
                )
                if mapper_j2 is None:
                    ai_client_j2.get_logger().warn("[AI] mapper not configured; Jetank2 AI disabled")
                else:
                    ai_config_j2 = AiConfig(
                        client=ai_client_j2,
                        mapper=mapper_j2,
                        target_class=args.ai_target_class,
                        min_conf=args.ai_min_conf,
                        theta_unit=args.theta_unit,
                        use_theta_roll=args.use_theta_roll,
                        roll_scale=args.roll_scale,
                        roll_offset=args.roll_offset,
                        retries=args.ai_retries,
                        retry_wait=args.ai_retry_wait,
                        timeout_sec=args.ai_timeout,
                    )

        manual_fallback = not args.ai_no_manual_fallback

        roi_wait_stop_sec = float(args.roi_wait_stop_sec)
        roi_post_stop_delay_sec = float(args.roi_post_stop_delay_sec)

        if args.mode == "cycle":
            roi_control_enabled = roi_guard is not None
            jetank1 = JetankController("jetank1", enable_tf_bridge=True)
            jetank2 = JetankController("jetank2", enable_tf_bridge=False)
            conveyor = ConveyorController()
            # 초기 자세/상태 정리
            print(">> Robot Ready. Initializing connection...")
            _sleep_sim(jetank1, 2.0)
            jetank1.detach_all()
            jetank2.detach_all()
            _sleep_sim(jetank1, 2.0)
            jetank1.move_to_xyz(150.0, 0.0, 50.0, phi=-90.0, roll=0.0, move_time=2.0)
            jetank2.move_to_xyz(150.0, 0.0, 50.0, phi=-90.0, roll=0.0, move_time=2.0)
            _sleep_sim(jetank1, 4.0)
            # 반복 사이클 실행
            run_cycle_with_ai(
                jetank1,
                jetank2,
                conveyor,
                roi_guard,
                repeat_count=args.repeat,
                jetank1_cmd=args.jetank1_cmd,
                jetank2_cmd=args.jetank2_cmd,
                jetank1_y_increment=args.jetank1_y_increment,
                conveyor_duration=args.conveyor_duration,
                jetank2_drop_override=args.jetank2_drop,
                ai_for=args.ai_for,
                ai_config_j1=ai_config_j1,
                ai_config_j2=ai_config_j2,
                manual_fallback=manual_fallback,
                roi_control_enabled=roi_control_enabled,
                roi_wait_stop_sec=roi_wait_stop_sec,
                roi_post_stop_delay_sec=roi_post_stop_delay_sec,
            )
            jetank1.close()
            jetank2.close()
            conveyor.destroy_node()
        elif args.mode == "jetank1":
            jetank1 = JetankController("jetank1", enable_tf_bridge=True)
            print(">> Robot Ready. Initializing connection...")
            _sleep_sim(jetank1, 2.0)
            jetank1.detach_all()
            _sleep_sim(jetank1, 1.0)
            jetank1.move_to_xyz(150.0, 0.0, 50.0, phi=-90.0, roll=0.0, move_time=2.0)
            _sleep_sim(jetank1, 4.0)
            # Jetank1 단독 모드
            if ai_config_j1:
                cmd = _resolve_command(
                    "Jetank1",
                    parse_command(args.jetank1_cmd) or (11.0, 151.0, 0.0),
                    ai_config_j1,
                    manual_fallback,
                )
                run_jetank1_sequence_ai(
                    jetank1,
                    cmd[0],
                    cmd[1],
                    roll=cmd[2],
                )
            else:
                print("[AI] disabled; switching to manual input")
                interactive_loop(jetank1, use_jetank2=False)
            jetank1.close()
        elif args.mode == "jetank2":
            jetank2 = JetankController("jetank2", enable_tf_bridge=True)
            print(">> Robot Ready. Initializing connection...")
            _sleep_sim(jetank2, 2.0)
            jetank2.detach_all()
            _sleep_sim(jetank2, 1.0)
            jetank2.move_to_xyz(150.0, 0.0, 50.0, phi=-90.0, roll=0.0, move_time=2.0)
            _sleep_sim(jetank2, 3.0)
            # Jetank2 단독 모드
            if ai_config_j2:
                cmd = _resolve_command(
                    "Jetank2",
                    parse_command(args.jetank2_cmd) or (0.0, 149.0, 0.0),
                    ai_config_j2,
                    manual_fallback,
                )
                run_jetank2_sequence(jetank2, cmd[0], cmd[1], roll=cmd[2])
            else:
                print("[AI] disabled; switching to manual input")
                interactive_loop(jetank2, use_jetank2=True)
            jetank2.close()
        else:
            conveyor = ConveyorController()
            conveyor.get_logger().info("컨베이어 ON (sim time 기준)")
            conveyor.set_power(True)
            conveyor.get_logger().info(f"시뮬 시간 {args.conveyor_duration:.1f}s 대기...")
            conveyor.wait_sim_seconds(args.conveyor_duration)
            conveyor.get_logger().info("컨베이어 OFF (sim time 기준)")
            conveyor.set_power(False)
            conveyor.destroy_node()
    finally:
        # 생성한 노드/스레드 정리
        if roi_executor is not None and roi_guard is not None:
            roi_executor.remove_node(roi_guard)
            roi_executor.shutdown()
        if roi_guard is not None:
            roi_guard.destroy_node()
        if ai_client_j1 is not None:
            ai_client_j1.destroy_node()
        if ai_client_j2 is not None:
            ai_client_j2.destroy_node()
        rclpy.shutdown()
        if roi_thread is not None:
            roi_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
