#!/usr/bin/env python3

import time
import sys
import numpy as np
import os

# ... (라이브러리 import 부분은 위와 동일) ...
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append('/home/jetson/SCSCtrl') 

try:
    import rclpy
    from rclpy.node import Node
except ImportError:
    pass 

# 핀 번호 및 펄스 시간 설정
from magnet_driver import Electromagnet
IN1 = 37
IN2 = 38
PULSE_TIME = 0.2

try:
    from SCSCtrl.scservo_sdk import *
except ImportError:
    print("Warning: SCSCtrl 라이브러리가 없습니다.")

class Config:
    DEVICE_NAME = '/dev/ttyTHS1'
    BAUDRATE = 1000000

    ID_BASE = 1
    ID_SHOULDER = 2
    ID_ELBOW = 3
    ID_WRIST_ROLL = 4 # 전자석 달려있는 곳
    ID_WRIST_PITCH = 5

    LINK_1 = 100.0 
    LINK_2 = 150.0 
    LINK_3 = 120.0

    # [중요] 5번 모터(Pitch)의 512값이 정확히 '팔과 일직선'이 되는지 확인하세요.
    # 만약 512일 때 손목이 꺾여 있다면 이 값을 조절해야 합니다.
    SERVO_INIT_POS = {1: 496, 2: 530, 3: 519, 4: 522, 5: 952}
    
    INPUT_RANGE = 850    
    ANGLE_RANGE = 180.0  

    ADDR_GOAL_POSITION = 42
    ADDR_GOAL_SPEED = 46
    ADDR_PRESENT_POSITION = 56

# ... (SCSServoManager 클래스는 위와 동일하므로 생략) ...
class SCSServoManager:
    def __init__(self, device_name, baudrate):
        self.port_handler = PortHandler(device_name)
        self.packet_handler = PacketHandler(1)
        self.group_sync_write_pos = GroupSyncWrite(self.port_handler, self.packet_handler, Config.ADDR_GOAL_POSITION, 2)
        self.group_sync_write_spd = GroupSyncWrite(self.port_handler, self.packet_handler, Config.ADDR_GOAL_SPEED, 2)
        if not self.open_port(baudrate): print("Port Error!")
        
    def open_port(self, baudrate):
        if self.port_handler.openPort() and self.port_handler.setBaudRate(baudrate): return True
        return False
    def close_port(self): self.port_handler.closePort()
    def read_pos(self, servo_id):
        pos, res, err = self.packet_handler.read2ByteTxRx(self.port_handler, servo_id, Config.ADDR_PRESENT_POSITION)
        if res != COMM_SUCCESS: return -1
        return pos
    def sync_write_pos_speed(self, ids, positions, speeds):
        for i, sid in enumerate(ids):
            param = [SCS_LOBYTE(int(speeds[i])), SCS_HIBYTE(int(speeds[i]))]
            self.group_sync_write_spd.addParam(sid, param)
        self.group_sync_write_spd.txPacket()
        self.group_sync_write_spd.clearParam()
        for i, sid in enumerate(ids):
            param = [SCS_LOBYTE(int(positions[i])), SCS_HIBYTE(int(positions[i]))]
            self.group_sync_write_pos.addParam(sid, param)
        self.group_sync_write_pos.txPacket()
        self.group_sync_write_pos.clearParam()

class RobotArm:
    def __init__(self, servo_manager):
        self.servo = servo_manager
        
        self.dirs = {
            Config.ID_BASE: 1, 
            Config.ID_SHOULDER: -1, 
            Config.ID_ELBOW: 1, 
            Config.ID_WRIST_ROLL: 1,
            Config.ID_WRIST_PITCH: 1 
        } 
        
        self.target_ids = [1, 2, 3, 4, 5]
        self.current_servo_pos = {}
        
        # 초기 위치 읽기
        for sid in self.target_ids:
            val = self.servo.read_pos(sid)
            self.current_servo_pos[sid] = val if val != -1 else Config.SERVO_INIT_POS[sid]

    def solve_ik_3dof_planar(self, r, z, phi_deg):
        phi = np.radians(phi_deg)
        
        # 1. 손목 위치 계산
        w_r = r - Config.LINK_3 * np.cos(phi)
        w_z = z - Config.LINK_3 * np.sin(phi)

        L1 = Config.LINK_1
        L2 = Config.LINK_2
        
        # 도달 가능 여부 확인
        if np.sqrt(w_r**2 + w_z**2) > (L1 + L2):
            return None
            
        # 2. 제 2 코사인 법칙으로 Elbow 각도(theta2) 계산
        # cos_angle = (a^2 + b^2 - c^2) / 2ab
        cos_angle = (w_r**2 + w_z**2 - L1**2 - L2**2) / (2 * L1 * L2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        theta2 = np.arccos(cos_angle) # 항상 양수(0 ~ 180) 반환

        # 3. Shoulder 각도(theta1) 계산
        k1 = L1 + L2 * np.cos(theta2)
        k2 = L2 * np.sin(theta2)
        theta1 = np.arctan2(w_z, w_r) - np.arctan2(k2, k1) 
        # 주의: 'Elbow Up' 형상이면 위 식, 'Elbow Down'이면 부호가 다름
        # 일반적으로 위 식은 Elbow Down(아래로 꺾임) 형태가 나올 수 있음.
        # 일반적인 Robot Arm(Elbow Up)을 위해 수정:
        theta1 = np.arctan2(w_z, w_r) + np.arctan2(k2, k1) 
        
        # [핵심 수정] 손목 각도(theta3) 계산 공식 변경
        # 손목이 유지해야 할 각도 = 목표각도(phi) - (어깨각도 + 팔꿈치각도)
        # 하지만 theta2가 '내각'이므로, 팔이 꺾인 방향에 따라 부호를 결정해야 함.
        
        # CASE 1: 팔을 뻗을 때 손목이 하늘을 본다면 -> 부호를 반대로 바꿔보세요 (+theta2 또는 -theta2)
        # CASE 2: 팔을 뻗을 때 손목이 땅으로 처진다면 -> 위와 반대로.
        
        # 일반적으로 'Elbow Up' 형상일 때:
        # 글로벌 각도 합: theta1 - theta2 + theta3 = phi
        # 따라서 theta3 = phi - theta1 + theta2
        
        theta3 = phi - (theta1 - theta2) 

        # 각도 변환
        deg_1 = np.degrees(theta1)
        deg_2 = np.degrees(theta2) * -1 # 모터 방향에 맞게 반전
        deg_3 = np.degrees(theta3)
        
        return deg_1, deg_2, deg_3

    def move_to_xyz(self, x, y, z, phi=-90, roll=0, move_time=1.0):
        # ... (IK 계산 및 이동 로직은 동일) ...
        rad_base = np.arctan2(y, x)
        deg_base = np.degrees(rad_base)
        
        r_dist = np.sqrt(x**2 + y**2)
        ik_result = self.solve_ik_3dof_planar(r_dist, z, phi_deg=phi)
        
        if ik_result is None: 
            print("Unreachable!")
            return

        deg_shoulder, deg_elbow, deg_wrist_p = ik_result
        
        target_angles_map = {
            Config.ID_BASE: deg_base,
            Config.ID_SHOULDER: deg_shoulder,
            Config.ID_ELBOW: deg_elbow,
            Config.ID_WRIST_ROLL: roll,
            Config.ID_WRIST_PITCH: deg_wrist_p
        }
        
        goals = []
        delta_pos_list = []
        
        for sid in self.target_ids:
            angle = target_angles_map[sid]
            direction = self.dirs[sid]
            pos = Config.SERVO_INIT_POS[sid] + int((Config.INPUT_RANGE/180.0) * angle * direction)
            pos = max(0, min(1023, pos))
            goals.append(pos)
            current = self.current_servo_pos.get(sid, Config.SERVO_INIT_POS[sid])
            delta_pos_list.append(abs(pos - current))
            self.current_servo_pos[sid] = pos

        max_delta = max(delta_pos_list)
        if max_delta == 0: return

        speeds = []
        scaling_factor = 1.0 / move_time 
        for delta in delta_pos_list:
            calc_speed = int((delta * scaling_factor) * 1.5)
            if calc_speed < 40: calc_speed = 40
            if calc_speed > 1000: calc_speed = 1000
            speeds.append(calc_speed)

        print(f"XYZ:({x:.0f},{y:.0f},{z:.0f}) P:{phi} | Goals:{goals}")
        self.servo.sync_write_pos_speed(self.target_ids, goals, speeds)

def main():
    manager = SCSServoManager(Config.DEVICE_NAME, Config.BAUDRATE)
    arm = RobotArm(manager)
    magnet = Electromagnet(in1_pin=IN1, in2_pin=IN2, demag_duration=PULSE_TIME)
    try:
        # 테스트: 팔을 앞으로 뻗었다가 당겨봅니다.
        # 이때 손목이 계속 바닥(-90)을 유지하는지 보세요.
        print(">> idle")
        arm.move_to_xyz(150, 0, 50, phi=-90, roll=0, move_time=2.0)
        time.sleep(4.0)
        
        arm.move_to_xyz(0, 100, 50, phi=-90, roll=0, move_time=2.0)
        time.sleep(4.0)
        arm.move_to_xyz(0, 100, -65, phi=-90, roll=0, move_time=2.0)
        time.sleep(4.0)
        print(">>magnet on")
        result = magnet.grab()
        print(f"[Result] {result}")
        time.sleep(4.0)
        
        arm.move_to_xyz(0, 150, 50, phi=-90, roll=0, move_time=2.0)
        time.sleep(4.0)
        
        
        
        
        arm.move_to_xyz(0, -150, 50, phi=-90, roll=90, move_time=6.0)
        time.sleep(10.0)
        arm.move_to_xyz(0, -150, -65, phi=-90, roll=90, move_time=2.0)
        time.sleep(4.0)
        
        print(">>magnet off")
        result = magnet.release()
        print(f"[Result] {result}")
        time.sleep(4.0)
        
        arm.move_to_xyz(0, -150, 50, phi=-90, roll=90, move_time=2.0)
        time.sleep(4.0)

        
        arm.move_to_xyz(0, 170, 50, phi=-90, roll=0, move_time=5.0)
        time.sleep(7.0)
        arm.move_to_xyz(0, 170, -65, phi=-90, roll=0, move_time=2.0)
        time.sleep(4.0)
        print(">>magnet on")
        result = magnet.grab()
        print(f"[Result] {result}")
        time.sleep(4.0)
        
        arm.move_to_xyz(0, 170, 50, phi=-90, roll=0, move_time=2.0)
        time.sleep(4.0)
        
        
        
        
        arm.move_to_xyz(0, -150, 50, phi=-90, roll=0, move_time=6.0)
        time.sleep(10.0)
        arm.move_to_xyz(0, -150, -23, phi=-90, roll=0, move_time=2.0)
        time.sleep(4.0)
        
        print(">>magnet off")
        result = magnet.release()
        print(f"[Result] {result}")
        time.sleep(4.0)
        
        arm.move_to_xyz(0, -150, 50, phi=-90, roll=90, move_time=2.0)
        time.sleep(4.0)
        
    except KeyboardInterrupt: pass
    finally: manager.close_port()

if __name__ == "__main__":
    main()