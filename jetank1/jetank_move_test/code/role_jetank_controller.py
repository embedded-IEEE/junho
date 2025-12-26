#!/usr/bin/env python3
"""Jetank controller shared by role nodes."""

# Jetank1/2에서 공통으로 쓰는 하드웨어/시뮬 제어 로직을 분리한 모듈.

import math
import os
import platform
import sys
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from builtin_interfaces.msg import Duration
    from std_msgs.msg import Empty
    from tf2_ros import TransformException    
    from tf2_ros.buffer import Buffer
    from tf2_ros.transform_listener import TransformListener
    from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
    from geometry_msgs.msg import TransformStamped
except ImportError:
    print("[Error] ROS2 라이브러리를 찾을 수 없습니다. (PC라면 ros-humble-rclpy 등을 확인하세요)")
    raise SystemExit(1)


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
    SERVO_INIT_POS = {1: 478, 2: 959, 3: 936, 4: 512, 5: 531}
    INPUT_RANGE = 850
    ANGLE_RANGE = 180.0
    ADDR_GOAL_POSITION = 42
    ADDR_GOAL_SPEED = 46
    ADDR_PRESENT_POSITION = 56


class JetankController(Node):
    """Jetank 팔 제어(시뮬/실기 공용) 핵심 클래스."""

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
