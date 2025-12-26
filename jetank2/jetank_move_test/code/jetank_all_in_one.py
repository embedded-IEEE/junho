#!/usr/bin/env python3
"""
All-in-one control for jetank1, jetank2, and conveyor in a single process.

Modes:
  - cycle: run jetank1 -> conveyor -> jetank2 sequence repeatedly
  - jetank1: interactive pick/place for jetank1
  - jetank2: interactive pick/place for jetank2
  - conveyor: turn conveyor on/off with sim-time wait
"""

import argparse
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
    from std_msgs.msg import Empty
    from std_srvs.srv import SetBool
    from tf2_ros import TransformException
    from tf2_ros.buffer import Buffer
    from tf2_ros.transform_listener import TransformListener
    from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
    from geometry_msgs.msg import TransformStamped
except ImportError:
    print("[Error] ROS2 라이브러리를 찾을 수 없습니다. (PC라면 ros-humble-rclpy 등을 확인하세요)")
    sys.exit(1)


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
    def __init__(self, robot_name: str, enable_tf_bridge: bool = True):
        super().__init__(f"{robot_name}_controller")
        self.set_parameters([rclpy.parameter.Parameter("use_sim_time", rclpy.Parameter.Type.BOOL, True)])
        qos_profile = QoSProfile(depth=10)

        self.robot_name = robot_name
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            f"/{robot_name}/arm_controller/joint_trajectory",
            qos_profile,
        )

        self.jenga_pubs = {}
        for i in range(1, 11):
            self.jenga_pubs[i] = {
                "attach": self.create_publisher(Empty, f"/{robot_name}/jenga{i}/attach", qos_profile),
                "detach": self.create_publisher(Empty, f"/{robot_name}/jenga{i}/detach", qos_profile),
            }

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        if enable_tf_bridge:
            self.publish_world_bridge()

        self.current_attached_id = None
        self.MAGNET_FRAME = f"{robot_name}/MAGNETIC_BAR_1"
        self.WORLD_FRAME = "world"

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

        self.servo_manager = None
        self.magnet = None
        if IS_REAL_ROBOT:
            self.init_hardware()

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
            for sid in self.target_ids:
                pos = self.read_hardware_pos(sid)
                if pos != -1:
                    self.current_servo_pos[sid] = pos

    def publish_world_bridge(self) -> None:
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
        if not IS_REAL_ROBOT:
            return -1
        pos, res, err = self.packet_handler.read2ByteTxRx(  # noqa: F405
            self.port_handler, servo_id, Config.ADDR_PRESENT_POSITION
        )
        return pos if res == COMM_SUCCESS else -1  # noqa: F405

    def find_closest_jenga(self, threshold: float = 0.15) -> Optional[int]:
        min_dist = float("inf")
        closest_id = None
        base_frame = self.MAGNET_FRAME
        world_frame = "empty_world"

        from rclpy.duration import Duration as RclpyDuration

        tf_timeout = RclpyDuration(seconds=0.5)

        duration = 2.0
        end_time = time.time() + duration
        print(f"\n>> [TF] Gathering TF data for {duration} seconds...")
        while time.time() < end_time:
            rclpy.spin_once(self, timeout_sec=0.1)

        print(">> [TF] Calculating Coordinates & Distances...")
        print("=" * 100)
        print(f"{'Target':<10} | {'World Coord':<30} | {'Dist from Magnet':<20} | {'Note'}")
        print("-" * 100)

        for i in range(1, 11):
            target_frame = f"jenga{i}"
            world_pose_str = "Unknown"
            dist_str = "Fail"
            note = ""

            try:
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

                if dist_val < min_dist:
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
        print(">> [Init] Detaching ALL jengas (1~10)...")
        msg = Empty()
        for i in range(1, 11):
            if i in self.jenga_pubs:
                self.jenga_pubs[i]["detach"].publish(msg)
        self.current_attached_id = None
        print(">> [Init] Complete.")

    def control_magnet(self, command: str, target_id: Optional[int] = None) -> None:
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
        self.publish_gazebo_command(target_angles, move_time)
        if IS_REAL_ROBOT and self.servo_manager:
            self.send_hardware_command(target_angles, move_time)

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
        if IS_REAL_ROBOT and hasattr(self, "port_handler"):
            self.port_handler.closePort()
        self.destroy_node()


class ConveyorController(Node):
    def __init__(self):
        super().__init__("conveyor_controller")
        self.set_parameters([rclpy.parameter.Parameter("use_sim_time", rclpy.Parameter.Type.BOOL, True)])
        self.cli = self.create_client(SetBool, "/conveyor/power")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("conveyor service 대기 중...")

    def set_power(self, on: bool) -> bool:
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
        start = self.get_clock().now()
        target_ns = int(seconds * 1e9)
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            now = self.get_clock().now()
            elapsed_ns = (now - start).nanoseconds
            if elapsed_ns >= target_ns:
                break

    def wait_wall_seconds(self, seconds: float) -> None:
        end_time = time.monotonic() + seconds
        while rclpy.ok() and time.monotonic() < end_time:
            rclpy.spin_once(self, timeout_sec=0.1)

    def wait_sim_seconds_checked(self, seconds: float, max_wait_wall: float = 2.0) -> None:
        start = self.get_clock().now()
        end_time = time.monotonic() + max_wait_wall
        while rclpy.ok() and time.monotonic() < end_time:
            rclpy.spin_once(self, timeout_sec=0.1)
            now = self.get_clock().now()
            if (now - start).nanoseconds > 0:
                self.wait_sim_seconds(seconds)
                return
        self.get_logger().warn("/clock 업데이트가 없어 wall time으로 대기합니다.")
        self.wait_wall_seconds(seconds)


def _wait_after_move(move_time: float, extra: float = 5.0) -> None:
    time.sleep(move_time + extra)


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
    post_grab_wait: float = 6.0,
    post_release_wait: float = 5.0,
    on_detach: Optional[callable] = None,
) -> None:
    print(f">> Sequence Start: ({x}, {y}, {pick_z}) Roll={roll}")

    robot.move_to_xyz(x, y, hover_z, phi=phi, roll=0.0, move_time=move_time)
    _wait_after_move(move_time)
    robot.move_to_xyz(x, y, pick_z, phi=phi, roll=0.0, move_time=move_time)
    _wait_after_move(move_time)
    robot.control_magnet("ON")
    time.sleep(post_grab_wait)
    robot.move_to_xyz(x, y, hover_z, phi=phi, roll=0.0, move_time=move_time)
    _wait_after_move(move_time)

    drop_x, drop_y, drop_z, drop_roll = drop_pose
    if drop_roll == 0.0:
        drop_roll = roll

    robot.move_to_xyz(drop_x, drop_y, hover_z, phi=phi, roll=drop_roll, move_time=move_time)
    _wait_after_move(move_time)
    robot.move_to_xyz(drop_x, drop_y, drop_z, phi=phi, roll=drop_roll, move_time=move_time)
    _wait_after_move(move_time)
    robot.control_magnet("OFF")
    if on_detach:
        on_detach()
    time.sleep(post_release_wait)
    robot.move_to_xyz(drop_x, drop_y, hover_z, phi=phi, roll=drop_roll, move_time=move_time)
    _wait_after_move(move_time)
    robot.move_to_xyz(150.0, 0.0, 50.0, phi=phi, roll=0.0, move_time=move_time)
    _wait_after_move(move_time)


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
    post_grab_wait: float = 6.0,
    post_release_wait: float = 10.0,
    on_target_reached: Optional[callable] = None,
) -> None:
    print(f">> Sequence Start: ({x}, {y}, {pick_z}) Roll={roll}")

    robot.move_to_xyz(x, y, hover_z, phi=phi, roll=0.0, move_time=move_time)
    _wait_after_move(move_time)
    robot.move_to_xyz(x, y, pick_z, phi=phi, roll=0.0, move_time=move_time)
    _wait_after_move(move_time)
    robot.control_magnet("ON")
    time.sleep(post_grab_wait)
    robot.move_to_xyz(x, y, hover_z, phi=phi, roll=0.0, move_time=move_time)
    _wait_after_move(move_time)

    pre_x, pre_y, pre_z, pre_roll = pre_drop_pose
    robot.move_to_xyz(pre_x, pre_y, pre_z, phi=phi, roll=pre_roll, move_time=move_time)
    _wait_after_move(move_time)

    drop_x, drop_y, drop_z, drop_roll = drop_pose

    robot.move_to_xyz(drop_x, drop_y, drop_z, phi=phi, roll=drop_roll, move_time=move_time)
    _wait_after_move(move_time)
    if on_target_reached:
        on_target_reached()
    robot.control_magnet("OFF")
    time.sleep(post_release_wait)
    robot.move_to_xyz(drop_x, drop_y, 0.0, phi=phi, roll=drop_roll, move_time=move_time)
    _wait_after_move(move_time)
    robot.move_to_xyz(150.0, 0.0, 50.0, phi=phi, roll=0.0, move_time=move_time)
    _wait_after_move(move_time)


def parse_command(cmd: str) -> Optional[Tuple[float, float, float]]:
    try:
        parts = cmd.replace(",", " ").split()
        if len(parts) < 3:
            return None
        x, y, r = float(parts[0]), float(parts[1]), float(parts[2])
        return x, y, r
    except Exception:
        return None


def prompt_for_command(label: str, default_cmd: Tuple[float, float, float]) -> Tuple[float, float, float]:
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
    try:
        parts = cmd.replace(",", " ").split()
        if len(parts) < 4:
            return None
        x, y, z, r = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
        return x, y, z, r
    except Exception:
        return None


def interactive_loop(robot: JetankController, use_jetank2: bool = False) -> None:
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


def run_cycle(
    jetank1: JetankController,
    jetank2: JetankController,
    conveyor: ConveyorController,
    repeat_count: int,
    jetank1_cmd: str,
    jetank2_cmd: str,
    jetank1_y_increment: float,
    conveyor_duration: float,
    jetank2_drop_override: Optional[str],
) -> None:
    base_xyz = parse_command(jetank1_cmd) or (11.0, 151.0, 0.0)
    jetank2_xyz = parse_command(jetank2_cmd) or (0.0, 149.0, 0.0)

    drop_sequence = [
        (0.0, -200.0, -47.0, 0.0),
        (-2.0, -160.0, -47.0, 0.0),
        (-9.0, -175.0, -35.0, 90.0),
        (9.0, -175.0, -35.0, 90.0),
    ]

    drop_override = parse_drop_pose(jetank2_drop_override) if jetank2_drop_override else None

    conveyor.get_logger().info("컨베이어 초기 상태: ON (계속 회전)")
    conveyor.set_power(True)

    for cycle in range(1, repeat_count + 1):
        print(f"\n=== Cycle {cycle}/{repeat_count} ===")

        base_x, base_y, base_r = base_xyz
        default_x = base_x if cycle == 1 else 10.0
        if cycle == 1:
            default_y = base_y
        elif cycle == 2:
            default_y = 170.0
        else:
            default_y = 170.0 + (cycle - 2) * jetank1_y_increment
        default_j1 = (default_x, default_y, base_r)
        j1x, j1y, j1r = prompt_for_command("Jetank1", default_j1)
        def stop_conveyor_after_detach() -> None:
            conveyor.get_logger().info(f"Jetank1 Detach 이후 {conveyor_duration:.1f}s 대기...")
            conveyor.wait_sim_seconds_checked(conveyor_duration)
            conveyor.get_logger().info("컨베이어 OFF (/clock 기준)")
            conveyor.set_power(False)

        run_jetank1_sequence(jetank1, j1x, j1y, roll=j1r, on_detach=stop_conveyor_after_detach)

        conveyor.get_logger().info("컨베이어 완료. Jetank2 입력 대기...")
        j2x, j2y, j2r = prompt_for_command("Jetank2", jetank2_xyz)
        if drop_override:
            drop_pose = drop_override
        else:
            drop_pose = drop_sequence[(cycle - 1) % len(drop_sequence)]
        def start_conveyor_after_jetank2_target() -> None:
            conveyor.get_logger().info("Jetank2 목표 지점 도착. 컨베이어 ON")
            conveyor.set_power(True)

        run_jetank2_sequence(
            jetank2,
            j2x,
            j2y,
            roll=j2r,
            drop_pose=drop_pose,
            on_target_reached=start_conveyor_after_jetank2_target,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Jetank all-in-one controller")
    parser.add_argument("--mode", choices=["cycle", "jetank1", "jetank2", "conveyor"], default="cycle")
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument("--jetank1-cmd", default=os.environ.get("JETANK1_CMD", "11 151 0"))
    parser.add_argument("--jetank2-cmd", default=os.environ.get("JETANK2_CMD", "0 149 0"))
    parser.add_argument("--jetank1-y-increment", type=float, default=20.0)
    parser.add_argument("--jetank2-drop", default=os.environ.get("JETANK2_DROP"))
    parser.add_argument("--conveyor-duration", type=float, default=12.8)
    args = parser.parse_args()

    rclpy.init()
    try:
        if args.mode == "cycle":
            jetank1 = JetankController("jetank1", enable_tf_bridge=True)
            jetank2 = JetankController("jetank2", enable_tf_bridge=False)
            conveyor = ConveyorController()
            print(">> Robot Ready. Initializing connection...")
            time.sleep(2.0)
            jetank1.detach_all()
            jetank2.detach_all()
            time.sleep(1.0)
            jetank1.move_to_xyz(150.0, 0.0, 50.0, phi=-90.0, roll=0.0, move_time=2.0)
            jetank2.move_to_xyz(150.0, 0.0, 50.0, phi=-90.0, roll=0.0, move_time=2.0)
            time.sleep(3.0)
            run_cycle(
                jetank1,
                jetank2,
                conveyor,
                repeat_count=args.repeat,
                jetank1_cmd=args.jetank1_cmd,
                jetank2_cmd=args.jetank2_cmd,
                jetank1_y_increment=args.jetank1_y_increment,
                conveyor_duration=args.conveyor_duration,
                jetank2_drop_override=args.jetank2_drop,
            )
            jetank1.close()
            jetank2.close()
            conveyor.destroy_node()
        elif args.mode == "jetank1":
            jetank1 = JetankController("jetank1", enable_tf_bridge=True)
            print(">> Robot Ready. Initializing connection...")
            time.sleep(2.0)
            jetank1.detach_all()
            time.sleep(1.0)
            jetank1.move_to_xyz(150.0, 0.0, 50.0, phi=-90.0, roll=0.0, move_time=2.0)
            time.sleep(3.0)
            interactive_loop(jetank1, use_jetank2=False)
            jetank1.close()
        elif args.mode == "jetank2":
            jetank2 = JetankController("jetank2", enable_tf_bridge=True)
            print(">> Robot Ready. Initializing connection...")
            time.sleep(2.0)
            jetank2.detach_all()
            time.sleep(1.0)
            jetank2.move_to_xyz(150.0, 0.0, 50.0, phi=-90.0, roll=0.0, move_time=2.0)
            time.sleep(3.0)
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
        rclpy.shutdown()


if __name__ == "__main__":
    main()
