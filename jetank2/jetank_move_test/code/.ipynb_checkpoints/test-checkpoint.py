#!/usr/bin/env python3
import time
import sys
import numpy as np
import os
import math

# ===================== ROS 2 Imports =====================
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

try:
    from top_cctv_interfaces.srv import GetClosestPose
except ImportError:
    print("[ERROR] 'top_cctv_interfaces' 패키지를 찾을 수 없습니다.")
    sys.exit(1)

# ===================== 경로 설정 =====================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append("/home/jetson/SCSCtrl")

# ===================== 하드웨어 라이브러리 =====================
from magnet_driver import Electromagnet

IN1 = 37
IN2 = 38
PULSE_TIME = 0.2

try:
    from SCSCtrl.scservo_sdk import *
except ImportError:
    print("Warning: SCSCtrl 라이브러리가 없습니다.")
    sys.exit(1)

# =========================================================
#  (추가) CCTV2 픽셀(cx,cy) -> Jetank2 월드(mm) 변환(호모그래피)
# =========================================================
# 캘리브 툴 결과를 그대로 복붙:
# px_points: "x1,y1;x2,y2;x3,y3;x4,y4"
# world_points(mm): "X1,Y1;X2,Y2;X3,Y3;X4,Y4"
DEFAULT_PX_POINTS_JETANK2 = "113,321;123,423;334,334;295,404"
DEFAULT_WORLD_POINTS_JETANK2 = "100.0,-20.0;160.0,-30.0;100.0,-130.0;140.0,-120.0"

# theta -> roll 설정(필요 시 조정)
THETA_UNIT = "rad"        # "rad" 또는 "deg"
USE_THETA_ROLL = False     # theta를 roll로 쓸지
ROLL_SCALE = 1.0
ROLL_OFFSET = 0.0
DEFAULT_ROLL = 0.0


def _parse_points_text(s: str):
    pts = []
    for token in s.strip().split(";"):
        a, b = token.strip().split(",")
        pts.append((float(a), float(b)))
    if len(pts) != 4:
        raise ValueError("points는 반드시 4쌍이어야 함: 'x1,y1;x2,y2;x3,y3;x4,y4'")
    return pts


def _homography_from_4pts(px_pts, world_pts):
    """
    4점 호모그래피(픽셀->월드)를 numpy SVD로 계산 (OpenCV 없이 동작)
    H @ [u,v,1]^T -> [x,y,1]^T (scale 포함)
    """
    A = []
    for (u, v), (x, y) in zip(px_pts, world_pts):
        A.append([-u, -v, -1,  0,  0,  0, u*x, v*x, x])
        A.append([ 0,  0,  0, -u, -v, -1, u*y, v*y, y])
    A = np.array(A, dtype=np.float64)

    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1, :]
    H = h.reshape(3, 3)

    # 스케일 정규화(선택)
    if abs(H[2, 2]) > 1e-9:
        H = H / H[2, 2]
    return H


def _apply_homography(H, u, v):
    p = np.array([u, v, 1.0], dtype=np.float64)
    q = H @ p
    if abs(q[2]) < 1e-9:
        raise RuntimeError("homography scale invalid")
    x = q[0] / q[2]
    y = q[1] / q[2]
    return float(x), float(y)


class HomographyMapper:
    def __init__(self, px_points_text: str, world_points_text: str):
        self.px_pts = _parse_points_text(px_points_text)
        self.world_pts = _parse_points_text(world_points_text)
        self.H = _homography_from_4pts(self.px_pts, self.world_pts)
        self.H_inv = np.linalg.inv(self.H)

    def px_to_world_mm(self, cx: float, cy: float):
        return _apply_homography(self.H, cx, cy)

    def world_mm_to_px(self, x_mm: float, y_mm: float):
        return _apply_homography(self.H_inv, x_mm, y_mm)


def theta_to_roll_deg(theta: float):
    if not USE_THETA_ROLL:
        return float(DEFAULT_ROLL)
    if THETA_UNIT == "rad":
        base = math.degrees(float(theta))
    else:
        base = float(theta)
    return float(base) * float(ROLL_SCALE) + float(ROLL_OFFSET)


# ===================== 설정 (Config) =====================
class Config:
    DEVICE_NAME = "/dev/ttyTHS1"
    BAUDRATE = 1000000

    ID_BASE = 1
    ID_SHOULDER = 2
    ID_ELBOW = 3
    ID_WRIST_ROLL = 4
    ID_WRIST_PITCH = 5

    LINK_1 = 100.0
    LINK_2 = 150.0
    LINK_3 = 120.0

    SERVO_INIT_POS = {
        1: 480, 2: 955, 3: 948, 4: 516, 5: 563
    }
    INPUT_RANGE = 850
    ANGLE_RANGE = 180.0

    ADDR_GOAL_POSITION = 42
    ADDR_GOAL_SPEED = 46
    ADDR_PRESENT_POSITION = 56


# ===================== 서보 매니저 =====================
class SCSServoManager:
    def __init__(self, device_name, baudrate):
        self.port_handler = PortHandler(device_name)
        self.packet_handler = PacketHandler(1)
        self.group_sync_write_pos = GroupSyncWrite(
            self.port_handler, self.packet_handler, Config.ADDR_GOAL_POSITION, 2
        )
        self.group_sync_write_spd = GroupSyncWrite(
            self.port_handler, self.packet_handler, Config.ADDR_GOAL_SPEED, 2
        )
        if not self.open_port(baudrate):
            raise RuntimeError("Port Error!")

    def open_port(self, baudrate):
        return bool(self.port_handler.openPort() and self.port_handler.setBaudRate(baudrate))

    def close_port(self):
        self.port_handler.closePort()

    def read_pos(self, servo_id):
        pos, res, err = self.packet_handler.read2ByteTxRx(
            self.port_handler, servo_id, Config.ADDR_PRESENT_POSITION
        )
        return pos if res == COMM_SUCCESS else -1

    def sync_write_pos_speed(self, ids, positions, speeds):
        for i, sid in enumerate(ids):
            spd = int(speeds[i])
            self.group_sync_write_spd.addParam(sid, [SCS_LOBYTE(spd), SCS_HIBYTE(spd)])
        self.group_sync_write_spd.txPacket()
        self.group_sync_write_spd.clearParam()

        for i, sid in enumerate(ids):
            pos = int(positions[i])
            self.group_sync_write_pos.addParam(sid, [SCS_LOBYTE(pos), SCS_HIBYTE(pos)])
        self.group_sync_write_pos.txPacket()
        self.group_sync_write_pos.clearParam()


# ===================== 로봇 팔 (IK) =====================
class RobotArm:
    def __init__(self, servo_manager: SCSServoManager):
        self.servo = servo_manager
        self.dirs = {1: 1, 2: -1, 3: 1, 4: 1, 5: 1}
        self.target_ids = [1, 2, 3, 4, 5]
        self.current_servo_pos = {}
        for sid in self.target_ids:
            val = self.servo.read_pos(sid)
            self.current_servo_pos[sid] = val if val != -1 else Config.SERVO_INIT_POS[sid]

    def solve_ik_3dof_planar(self, r, z, phi_deg):
        phi = np.radians(phi_deg)
        w_r = r - Config.LINK_3 * np.cos(phi)
        w_z = z - Config.LINK_3 * np.sin(phi)
        L1, L2 = Config.LINK_1, Config.LINK_2

        if np.sqrt(w_r**2 + w_z**2) > (L1 + L2):
            return None

        cos_angle = np.clip(
            (w_r**2 + w_z**2 - L1**2 - L2**2) / (2 * L1 * L2),
            -1.0,
            1.0,
        )
        theta2 = np.arccos(cos_angle)
        theta1 = np.arctan2(w_z, w_r) + np.arctan2(L2 * np.sin(theta2), L1 + L2 * np.cos(theta2))
        theta3 = phi - (theta1 - theta2)

        return np.degrees(theta1), np.degrees(theta2) * -1, np.degrees(theta3)

    def move_to_xyz(self, x, y, z, phi=-90, roll=0, move_time=1.0, verbose=True):
        rad_base = np.arctan2(y, x)
        ik_result = self.solve_ik_3dof_planar(np.sqrt(x**2 + y**2), z, phi_deg=phi)

        if ik_result is None:
            if verbose:
                print(f"[IK] Unreachable: {x:.1f}, {y:.1f}, {z:.1f}")
            return False

        deg_map = {
            1: np.degrees(rad_base),
            2: ik_result[0],
            3: ik_result[1],
            4: roll,
            5: ik_result[2],
        }

        goals, speeds = [], []
        scaling = 1.0 / max(move_time, 0.05)

        for sid in self.target_ids:
            pos = Config.SERVO_INIT_POS[sid] + int((Config.INPUT_RANGE / 180.0) * deg_map[sid] * self.dirs[sid])
            pos = max(0, min(1023, pos))
            goals.append(pos)
            delta = abs(pos - self.current_servo_pos.get(sid, pos))
            self.current_servo_pos[sid] = pos
            speeds.append(max(40, min(1000, int(delta * scaling * 1.5))))

        if verbose:
            print(f"[MOVE] XYZ=({x:.0f},{y:.0f},{z:.0f}) goals={goals}")
        self.servo.sync_write_pos_speed(self.target_ids, goals, speeds)
        return True


# ===================== ROS 2 Node =====================
class PickPlaceNode(Node):
    def __init__(self):
        super().__init__("pick_place_node")
        self.srv_conveyor = self.create_service(Trigger, "/jetank2/conveyor_off_event", self.conveyor_callback)
        self.cli_vision = self.create_client(GetClosestPose, "/top_cctv2/get_closest_pose")

        # (추가) 픽셀->월드 변환기
        self.mapper = HomographyMapper(DEFAULT_PX_POINTS_JETANK2, DEFAULT_WORLD_POINTS_JETANK2)

        self.start_pick_sequence = False
        self.get_logger().info("Ready for Pick & Place Service Requests...")

    def conveyor_callback(self, request, response):
        self.get_logger().info("[EVENT] Conveyor Off Signal Received!")
        self.start_pick_sequence = True
        response.success = True
        response.message = "Robot triggered"
        return response

    def call_vision_service(self):
        if not self.cli_vision.wait_for_service(timeout_sec=1.0):
            return None
        req = GetClosestPose.Request()
        req.target_class = -1
        return self.cli_vision.call_async(req)


# ===================== 유틸 및 패턴 =====================
def sleep_safe(sec):
    time.sleep(max(0, sec))


DROP_PATTERN_MM = [
    {"dx": +15.0, "dy": 0.0,  "roll": -30.0},
    {"dx": -15.0, "dy": 0.0,  "roll": -40.0},
    {"dx": 0.0,   "dy": +16.0, "roll": 75.0},
    {"dx": 0.0,   "dy": -17.0, "roll": 75.0},
]


# ===================== 메인 실행 =====================
def main(args=None):
    rclpy.init(args=args)

    manager = SCSServoManager(Config.DEVICE_NAME, Config.BAUDRATE)
    arm = RobotArm(manager)
    magnet = Electromagnet(in1_pin=IN1, in2_pin=IN2, demag_duration=PULSE_TIME)
    node = PickPlaceNode()

    # 파라미터(mm)
    hover_z, pick_z, drop_z, phi, move_time = 30.0, -70.0, -23.0, -90.0, 3.5
    drop_base_x, drop_base_y = 0.0, 150.0
    drop_idx = 0

    try:
        print("\n[READY] Robot Waiting for Conveyor Signal...")
        arm.move_to_xyz(0, 150, hover_z, phi=phi, roll=0, move_time=2.0)
        sleep_safe(2.0)

        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)

            if node.start_pick_sequence:
                node.start_pick_sequence = False
                print("\n>>> Start Sequence Initiated")

                print("[1] Calling Vision Service...")
                future = node.call_vision_service()

                if future is None:
                    print("[ERROR] Vision service not available (Is the node running?)")
                    continue

                # ================= 타임아웃 처리 로직 =================
                start_wait = time.time()
                timed_out = False

                while not future.done():
                    rclpy.spin_once(node, timeout_sec=0.1)
                    if time.time() - start_wait > 5.0:
                        timed_out = True
                        break

                if timed_out:
                    print("[WARNING] Vision Service Timeout! (No response for 5s)")
                    continue
                # ====================================================

                try:
                    result = future.result()
                except Exception as e:
                    print(f"[ERROR] Service call exception: {e}")
                    continue

                if result and result.found:
                    # ======== (핵심) 카메라 픽셀 -> 월드(mm) 변환 ========
                    cx = float(result.x)
                    cy = float(result.y)
                    pick_x, pick_y = node.mapper.px_to_world_mm(cx, cy)

                    # theta -> roll(deg)
                    target_roll = theta_to_roll_deg(float(result.theta))

                    print(f"[VISION] px=({cx:.1f},{cy:.1f}) -> world_mm=({pick_x:.1f},{pick_y:.1f}) roll={target_roll:.1f} conf={float(result.conf):.2f}")

                    # Pick Sequence
                    if not arm.move_to_xyz(pick_x, pick_y, hover_z, phi, target_roll, move_time):
                        print("!! Pick Unreachable !!")
                        continue
                    sleep_safe(move_time + 0.2)

                    arm.move_to_xyz(pick_x, pick_y, pick_z, phi, target_roll, move_time)
                    sleep_safe(move_time + 0.2)

                    print(">> Magnet ON")
                    magnet.grab()
                    sleep_safe(0.8)

                    arm.move_to_xyz(pick_x, pick_y, hover_z, phi, target_roll, move_time)
                    sleep_safe(move_time + 0.2)

                    # Drop Sequence
                    pat = DROP_PATTERN_MM[drop_idx % 4]
                    d_x, d_y, d_roll = drop_base_x + pat["dx"], drop_base_y + pat["dy"], -pat["roll"]

                    print(f"[DROP] Moving to ({d_x:.0f}, {d_y:.0f})")
                    arm.move_to_xyz(d_x, d_y, hover_z, phi, d_roll, max(move_time, 2.0))
                    sleep_safe(max(move_time, 2.0) + 0.2)
                    
                    arm.move_to_xyz(d_x, d_y, drop_z, phi, d_roll, move_time)
                    sleep_safe(move_time + 0.5)

                    print(">> Magnet OFF")
                    magnet.release()
                    sleep_safe(0.5)
                    
                    arm.move_to_xyz(d_x, d_y, hover_z, phi, d_roll, move_time)
                    drop_idx += 1

                    sleep_safe(max(move_time, 2.0) + 0.2)
                    # 홈 위치 대기
                    arm.move_to_xyz(0, 150, hover_z, phi, 0, 0.3)

                    sleep_safe(max(move_time, 2.0) + 0.2)
                else:
                    print("[VISION] Object not found.")

    except KeyboardInterrupt:
        print("\n[STOP] KeyboardInterrupt")
    finally:
        manager.close_port()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
