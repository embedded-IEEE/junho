#!/usr/bin/env python3

import time
import sys
import numpy as np
import os
import platform # 환경 감지용

# --- ROS2 라이브러리 (필수) ---
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from builtin_interfaces.msg import Duration
    # [추가] Empty 메시지 타입 임포트
    from std_msgs.msg import Empty
except ImportError:
    print("[Error] ROS2 라이브러리를 찾을 수 없습니다. (PC라면 ros-humble-rclpy 등을 확인하세요)")
    sys.exit(1)

# ---------------------------------------------------------
# 1. 환경 감지 및 라이브러리 로드 설정
# ---------------------------------------------------------
IS_REAL_ROBOT = (platform.machine() == 'aarch64')

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append('/home/jetson/SCSCtrl') 

if IS_REAL_ROBOT:
    print(f"[System] Jetson({platform.machine()}) 감지됨 -> 하드웨어 모드 활성화")
    try:
        from magnet_driver import Electromagnet
        from SCSCtrl.scservo_sdk import *
        IN1, IN2, PULSE_TIME = 37, 38, 0.2
    except ImportError as e:
        print(f"[Error] 하드웨어 라이브러리 로드 실패: {e}")
        IS_REAL_ROBOT = False 
else:
    print(f"[System] PC({platform.machine()}) 감지됨 -> 시뮬레이션(Gazebo) 모드 활성화")
    IN1, IN2, PULSE_TIME = 0, 0, 0.2

# ---------------------------------------------------------
# 2. 설정 값 (공통)
# ---------------------------------------------------------
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
    SERVO_INIT_POS = {1: 510, 2: 545, 3: 524, 4: 512, 5: 561}
    INPUT_RANGE = 850     
    ANGLE_RANGE = 180.0
    ADDR_GOAL_POSITION = 42
    ADDR_GOAL_SPEED = 46
    ADDR_PRESENT_POSITION = 56

# ---------------------------------------------------------
# 3. 로봇 제어 클래스 (ROS2 Node 상속)
# ---------------------------------------------------------
class JetankController(Node):
    def __init__(self):
        super().__init__('jetank_controller')
        
        qos_profile = QoSProfile(depth=10)
        
        # 1. Gazebo 팔 제어용 Publisher
        self.traj_pub = self.create_publisher(
            JointTrajectory, 
            '/jetank/arm_controller/joint_trajectory', 
            qos_profile
        )

        # [추가] 2. Magnet Attach/Detach 제어용 Publisher
        self.attach_pub = self.create_publisher(Empty, '/jenga/attach', qos_profile)
        self.detach_pub = self.create_publisher(Empty, '/jenga/detach', qos_profile)
        
        self.joint_names = [
            'Revolute_BEARING', 'Revolute_ARM_LOW', 'Revolute_SERVO_UPPER',
            'Revolute_MAGNETIC_BAR', 'Revolute_SERVO_TOP'
        ]

        self.SIM_CORRECTION = {
            Config.ID_BASE:        {'dir': 1,  'offset': 0.0},
            Config.ID_SHOULDER:    {'dir': -1, 'offset': 90.0},
            Config.ID_ELBOW:       {'dir': -1, 'offset': 0.0},
            Config.ID_WRIST_ROLL:  {'dir': 1,  'offset': 0.0},
            Config.ID_WRIST_PITCH: {'dir': 1, 'offset': 90.0}
        }

        self.servo_manager = None
        self.magnet = None
        
        if IS_REAL_ROBOT:
            self.init_hardware()

        self.dirs = {
            Config.ID_BASE: 1, Config.ID_SHOULDER: -1, Config.ID_ELBOW: 1, 
            Config.ID_WRIST_ROLL: 1, Config.ID_WRIST_PITCH: 1 
        }
        self.target_ids = [1, 2, 3, 4, 5]
        self.current_servo_pos = Config.SERVO_INIT_POS.copy()
        
        if IS_REAL_ROBOT and self.servo_manager:
            for sid in self.target_ids:
                pos = self.read_hardware_pos(sid)
                if pos != -1: self.current_servo_pos[sid] = pos

    def init_hardware(self):
        try:
            self.port_handler = PortHandler(Config.DEVICE_NAME)
            self.packet_handler = PacketHandler(1)
            if self.port_handler.openPort() and self.port_handler.setBaudRate(Config.BAUDRATE):
                print("[Hardware] Serial Port Opened.")
            else:
                print("[Error] Failed to open port!")
            self.group_sync_write_pos = GroupSyncWrite(self.port_handler, self.packet_handler, Config.ADDR_GOAL_POSITION, 2)
            self.group_sync_write_spd = GroupSyncWrite(self.port_handler, self.packet_handler, Config.ADDR_GOAL_SPEED, 2)
            self.magnet = Electromagnet(in1_pin=IN1, in2_pin=IN2, demag_duration=PULSE_TIME)
            self.servo_manager = True 
        except Exception as e:
            print(f"[Error] Hardware Init Failed: {e}")

    def read_hardware_pos(self, servo_id):
        if not IS_REAL_ROBOT: return -1
        pos, res, err = self.packet_handler.read2ByteTxRx(self.port_handler, servo_id, Config.ADDR_PRESENT_POSITION)
        return pos if res == COMM_SUCCESS else -1

    def control_magnet(self, command):
        """자석 제어 (하드웨어 + ROS2 Topic 발행)"""
        # 공통: ROS2 메시지 생성
        msg = Empty()

        if command == "ON":
            # [추가] ROS2 Topic 발행 (/jenga/attach)
            self.attach_pub.publish(msg)
            print(">> [ROS] Published /jenga/attach")
            
            # 하드웨어 제어
            if IS_REAL_ROBOT and self.magnet:
                return self.magnet.grab()
            else:
                return "[Sim] Magnet ON (Virtual)"

        elif command == "OFF":
            # [추가] ROS2 Topic 발행 (/jenga/detach)
            self.detach_pub.publish(msg)
            print(">> [ROS] Published /jenga/detach")

            # 하드웨어 제어
            if IS_REAL_ROBOT and self.magnet:
                return self.magnet.release()
            else:
                return "[Sim] Magnet OFF (Virtual)"

    # --- IK 및 이동 로직 ---
    def solve_ik_3dof_planar(self, r, z, phi_deg):
        phi = np.radians(phi_deg)
        w_r = r - Config.LINK_3 * np.cos(phi)
        w_z = z - Config.LINK_3 * np.sin(phi)
        L1, L2 = Config.LINK_1, Config.LINK_2
        
        if np.sqrt(w_r**2 + w_z**2) > (L1 + L2): return None
            
        cos_angle = (w_r**2 + w_z**2 - L1**2 - L2**2) / (2 * L1 * L2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        theta2 = np.arccos(cos_angle)

        k1 = L1 + L2 * np.cos(theta2)
        k2 = L2 * np.sin(theta2)
        theta1 = np.arctan2(w_z, w_r) + np.arctan2(k2, k1) 
        theta3 = phi - (theta1 - theta2) 

        return np.degrees(theta1), -np.degrees(theta2), np.degrees(theta3)

    def move_to_xyz(self, x, y, z, phi=-90, roll=0, move_time=1.0):
        rad_base = np.arctan2(y, x)
        deg_base = np.degrees(rad_base)
        r_dist = np.sqrt(x**2 + y**2)
        ik_result = self.solve_ik_3dof_planar(r_dist, z, phi_deg=phi)
        
        if ik_result is None: 
            print(f"Unreachable: {x},{y},{z}")
            return

        deg_shoulder, deg_elbow, deg_wrist_p = ik_result
        target_angles = {
            Config.ID_BASE: deg_base, Config.ID_SHOULDER: deg_shoulder,
            Config.ID_ELBOW: deg_elbow, Config.ID_WRIST_ROLL: roll,
            Config.ID_WRIST_PITCH: deg_wrist_p
        }
        self.publish_gazebo_command(target_angles, move_time)
        if IS_REAL_ROBOT and self.servo_manager:
            self.send_hardware_command(target_angles, move_time)

    def publish_gazebo_command(self, angles_deg, move_time):
        msg = JointTrajectory()
        msg.header.stamp.sec = 0
        msg.header.stamp.nanosec = 0
        msg.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        def normalize_angle(angle_deg):
            angle_deg = angle_deg % 360.0 
            if angle_deg > 180.0: angle_deg -= 360.0
            elif angle_deg < -180.0: angle_deg += 360.0
            return angle_deg
            
        def get_sim_rad(srv_id):
            cfg = self.SIM_CORRECTION[srv_id]
            input_deg = angles_deg[srv_id]
            raw_target = (input_deg * cfg['dir']) + cfg['offset']
            final_deg = normalize_angle(raw_target)
            return np.radians(final_deg)

        point.positions = [
            get_sim_rad(Config.ID_BASE), get_sim_rad(Config.ID_SHOULDER),
            get_sim_rad(Config.ID_ELBOW), get_sim_rad(Config.ID_WRIST_ROLL),
            get_sim_rad(Config.ID_WRIST_PITCH)
        ]
        sec = int(move_time)
        nanosec = int((move_time - sec) * 1e9)
        point.time_from_start = Duration(sec=sec, nanosec=nanosec)
        msg.points = [point]
        self.traj_pub.publish(msg)

    def send_hardware_command(self, angles_deg, move_time):
        goals, speeds, delta_pos_list = [], [], []
        for sid in self.target_ids:
            angle = angles_deg[sid]
            direction = self.dirs[sid]
            pos = Config.SERVO_INIT_POS[sid] + int((Config.INPUT_RANGE/180.0) * angle * direction)
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
            param_spd = [SCS_LOBYTE(speeds[i]), SCS_HIBYTE(speeds[i])]
            self.group_sync_write_spd.addParam(sid, param_spd)
        self.group_sync_write_spd.txPacket()
        self.group_sync_write_spd.clearParam()

        for i, sid in enumerate(self.target_ids):
            param_pos = [SCS_LOBYTE(goals[i]), SCS_HIBYTE(goals[i])]
            self.group_sync_write_pos.addParam(sid, param_pos)
        self.group_sync_write_pos.txPacket()
        self.group_sync_write_pos.clearParam()
        print(f"[REAL] Goals: {goals}")

    def close(self):
        if IS_REAL_ROBOT and self.port_handler:
            self.port_handler.closePort()
        self.destroy_node()

# ---------------------------------------------------------
# 4. 메인 실행부
# ---------------------------------------------------------
def main():
    rclpy.init()
    
    robot = JetankController()
    
    try:
        print(">> Robot Ready. Initializing connection...")
        time.sleep(2.0) 
        
        print(">> Starting Sequence...")
        robot.move_to_xyz(150, 0, 50, phi=-90, roll=0, move_time=2.0)
        time.sleep(3.0)
        
        robot.move_to_xyz(0, 148, 20, phi=-90, roll=0, move_time=2.0)
        time.sleep(3.0)
        
        robot.move_to_xyz(0, 148, -73, phi=-90, roll=0, move_time=2.0)
        time.sleep(5.0)
        
        # [수정됨] 여기를 호출하면 내부적으로 ROS 토픽(/jenga/attach)이 발행됨
        print(">> Magnet ON (Attach)")
        robot.control_magnet("ON")  
        time.sleep(5.0)
        
        robot.move_to_xyz(0, 150, 50, phi=-90, roll=0, move_time=2.0)
        time.sleep(3.0)
        
        robot.move_to_xyz(0, -150, 50, phi=-90, roll=90, move_time=6.0)
        time.sleep(7.0)
        
        # [수정됨] 여기를 호출하면 내부적으로 ROS 토픽(/jenga/detach)이 발행됨
        print(">> Magnet OFF (Detach)")
        robot.control_magnet("OFF") 
        time.sleep(3.0)
        
        robot.move_to_xyz(150, 0, 50, phi=-90, roll=0, move_time=4.0)
        time.sleep(4.0)

    except KeyboardInterrupt:
        pass
    finally:
        robot.close()
        rclpy.shutdown()

if __name__ == "__main__":
    main()