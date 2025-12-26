#!/usr/bin/env python3
import time
import sys
import numpy as np
import os

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
sys.path.append('/home/jetson/SCSCtrl')

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


# ===================== 설정 (Config) =====================
class Config:
    DEVICE_NAME = '/dev/ttyTHS1'
    BAUDRATE = 1000000

    ID_BASE = 1
    ID_SHOULDER = 2
    ID_ELBOW = 3
    ID_WRIST_ROLL = 4
    ID_WRIST_PITCH = 5

    LINK_1 = 95.0
    LINK_2 = 152.0
    LINK_3 = 123.0

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
        self.group_sync_write_pos = GroupSyncWrite(self.port_handler, self.packet_handler, Config.ADDR_GOAL_POSITION, 2)
        self.group_sync_write_spd = GroupSyncWrite(self.port_handler, self.packet_handler, Config.ADDR_GOAL_SPEED, 2)
        if not self.open_port(baudrate):
            raise RuntimeError("Port Error!")

    def open_port(self, baudrate):
        return bool(self.port_handler.openPort() and self.port_handler.setBaudRate(baudrate))

    def close_port(self):
        self.port_handler.closePort()

    def read_pos(self, servo_id):
        pos, res, err = self.packet_handler.read2ByteTxRx(self.port_handler, servo_id, Config.ADDR_PRESENT_POSITION)
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

        if np.sqrt(w_r**2 + w_z**2) > (L1 + L2): return None

        cos_angle = np.clip((w_r**2 + w_z**2 - L1**2 - L2**2) / (2 * L1 * L2), -1.0, 1.0)
        theta2 = np.arccos(cos_angle)
        theta1 = np.arctan2(w_z, w_r) + np.arctan2(L2 * np.sin(theta2), L1 + L2 * np.cos(theta2))
        theta3 = phi - (theta1 - theta2)

        return np.degrees(theta1), np.degrees(theta2) * -1, np.degrees(theta3)

    def move_to_xyz(self, x, y, z, phi=-90, roll=0, move_time=1.0, verbose=True):
        rad_base = np.arctan2(y, x)
        ik_result = self.solve_ik_3dof_planar(np.sqrt(x**2 + y**2), z, phi_deg=phi)
        
        if ik_result is None:
            if verbose: print(f"[IK] Unreachable: {x:.1f}, {y:.1f}, {z:.1f}")
            return False

        deg_map = {
            1: np.degrees(rad_base), 2: ik_result[0], 3: ik_result[1], 4: roll, 5: ik_result[2]
        }
        
        goals, speeds = [], []
        scaling = 1.0 / max(move_time, 0.05)
        
        for sid in self.target_ids:
            pos = Config.SERVO_INIT_POS[sid] + int((Config.INPUT_RANGE/180.0) * deg_map[sid] * self.dirs[sid])
            pos = max(0, min(1023, pos))
            goals.append(pos)
            delta = abs(pos - self.current_servo_pos.get(sid, pos))
            self.current_servo_pos[sid] = pos
            speeds.append(max(40, min(1000, int(delta * scaling * 1.5))))

        if verbose: print(f"[MOVE] XYZ=({x:.0f},{y:.0f},{z:.0f}) goals={goals}")
        self.servo.sync_write_pos_speed(self.target_ids, goals, speeds)
        return True


# ===================== ROS 2 Node =====================
class PickPlaceNode(Node):
    def __init__(self):
        super().__init__('pick_place_node')
        self.srv_conveyor = self.create_service(Trigger, '/jetank2/conveyor_off_event', self.conveyor_callback)
        self.cli_vision = self.create_client(GetClosestPose, '/top_cctv2/get_closest_pose')
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
def sleep_safe(sec): time.sleep(max(0, sec))

DROP_PATTERN_MM = [
    {'dx': +15.0, 'dy': 0.0,  'roll': 0.0},
    {'dx': -15.0, 'dy': 0.0,  'roll': 0.0},
    {'dx': 0.0,   'dy': +12.0, 'roll': 90.0},
    {'dx': 0.0,   'dy': -17.0, 'roll': 90.0},
]


# ===================== 메인 실행 =====================
def main(args=None):
    rclpy.init(args=args)
    
    manager = SCSServoManager(Config.DEVICE_NAME, Config.BAUDRATE)
    arm = RobotArm(manager)
    magnet = Electromagnet(in1_pin=IN1, in2_pin=IN2, demag_duration=PULSE_TIME)
    node = PickPlaceNode()

    # 파라미터
    hover_z, pick_z, drop_z, phi, move_time = 50.0, -66.0, -23.0, -90.0, 1.5
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
                
                # future가 완료될 때까지 spin하면서 대기 (최대 5초)
                while not future.done():
                    rclpy.spin_once(node, timeout_sec=0.1)
                    if time.time() - start_wait > 5.0:
                        timed_out = True
                        break
                
                if timed_out:
                    print("[WARNING] Vision Service Timeout! (No response for 5s)")
                    # 타임아웃 시 이번 사이클 포기하고 다시 대기
                    continue
                # ====================================================

                try:
                    result = future.result()
                except Exception as e:
                    print(f"[ERROR] Service call exception: {e}")
                    continue

                if result and result.found:
                    # m -> mm 변환
                    pick_x = result.x * 1000.0
                    pick_y = result.y * 1000.0
                    target_roll = np.degrees(result.theta)
                    
                    print(f"[VISION] Found! X={pick_x:.1f} Y={pick_y:.1f} Roll={target_roll:.1f}")

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
                    d_x, d_y, d_roll = drop_base_x + pat['dx'], drop_base_y + pat['dy'], pat['roll']
                    
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
                    
                    # 홈 위치 대기
                    arm.move_to_xyz(0, 150, hover_z, phi, 0, 0.3)
                    
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