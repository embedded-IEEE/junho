#!/usr/bin/env python3

import time
import sys
import numpy as np
import os

# --- 라이브러리 경로 설정 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'SCSCtrl')) 


# 핀 번호 및 펄스 시간 설정
from magnet_driver import Electromagnet
IN1 = 37
IN2 = 38
PULSE_TIME = 0.2

try:
    import rclpy
    from rclpy.node import Node
except ImportError:
    pass 

try:
    from SCSCtrl.scservo_sdk import *
except ImportError:
    print("Warning: SCSCtrl 라이브러리가 없습니다.")

# --- 설정 상수 ---
class Config:
    DEVICE_NAME = '/dev/ttyTHS1'
    BAUDRATE = 1000000

    # ID 4번(Wrist Yaw) 제외
    ID_BASE = 1
    ID_SHOULDER = 2
    ID_ELBOW = 3
    ID_WRIST_P = 5

    # 링크 길이 (mm)
    LINK_1 = 100.0 
    LINK_2 = 150.0 
    LINK_3 = 50.0 

    # [최신 캘리브레이션 값 적용]
    SERVO_INIT_POS = {
        1: 510, 
        2: 545, 
        3: 524, 
        5: 561
    }
    
    INPUT_RANGE = 850    
    ANGLE_RANGE = 180.0  

    ADDR_GOAL_POSITION = 42
    ADDR_GOAL_SPEED = 46
    ADDR_PRESENT_POSITION = 56

# --- 하드웨어 통신 클래스 ---
class SCSServoManager:
    def __init__(self, device_name, baudrate):
        self.port_handler = PortHandler(device_name)
        self.packet_handler = PacketHandler(1)
        self.group_sync_write_pos = GroupSyncWrite(self.port_handler, self.packet_handler, Config.ADDR_GOAL_POSITION, 2)
        self.group_sync_write_spd = GroupSyncWrite(self.port_handler, self.packet_handler, Config.ADDR_GOAL_SPEED, 2)
        
        if not self.open_port(baudrate): 
            print("Port Error! (권한 확인: sudo chmod 666 /dev/ttyTHS1)")
        
    def open_port(self, baudrate):
        if self.port_handler.openPort() and self.port_handler.setBaudRate(baudrate): return True
        return False
        
    def close_port(self): self.port_handler.closePort()

    def read_pos(self, servo_id):
        # [수정] 실제 모터 위치 읽기 기능 강화
        pos, res, err = self.packet_handler.read2ByteTxRx(self.port_handler, servo_id, Config.ADDR_PRESENT_POSITION)
        if res != COMM_SUCCESS: 
            # 읽기 실패 시 에러 출력하지 않고 -1 반환 (초기화 로직에서 처리)
            return -1
        return pos
    
    
    def sync_write_pos_speed(self, ids, positions, speeds):
        # 1. 속도 먼저 설정
        for i, sid in enumerate(ids):
            param = [SCS_LOBYTE(int(speeds[i])), SCS_HIBYTE(int(speeds[i]))]
            self.group_sync_write_spd.addParam(sid, param)
        self.group_sync_write_spd.txPacket()
        self.group_sync_write_spd.clearParam()
        
        # 2. 위치 이동 명령
        for i, sid in enumerate(ids):
            param = [SCS_LOBYTE(int(positions[i])), SCS_HIBYTE(int(positions[i]))]
            self.group_sync_write_pos.addParam(sid, param)
        self.group_sync_write_pos.txPacket()
        self.group_sync_write_pos.clearParam()
        
        
# --- 로봇 팔 제어 클래스 ---
class RobotArm:
    def __init__(self, servo_manager):
        self.servo = servo_manager
        
        self.dirs = {1: 1, 2: -1, 3: 1, 5: 1} 
        
        # [핵심 수정] 초기화 시 실제 모터 위치를 읽어옵니다.
        print(">>> 실제 모터 위치를 읽어오는 중...")
        self.current_servo_pos = {}
        
        target_ids = [Config.ID_BASE, Config.ID_SHOULDER, Config.ID_ELBOW, Config.ID_WRIST_P]
        for sid in target_ids:
            read_val = self.servo.read_pos(sid)
            if read_val != -1:
                self.current_servo_pos[sid] = read_val
                print(f"ID {sid}: Current Pos {read_val}")
            else:
                # 읽기 실패 시 Config의 초기값 사용
                self.current_servo_pos[sid] = Config.SERVO_INIT_POS[sid]
                print(f"ID {sid}: Read Fail -> Use Init Pos {self.current_servo_pos[sid]}")
                
                
    def solve_ik_3dof_planar(self, r, z, phi_deg=-90):
        # phi_deg=-90 : 항상 수직 아래를 보게 함
        phi = np.radians(phi_deg)
        w_r = r - Config.LINK_3 * np.cos(phi)
        w_z = z - Config.LINK_3 * np.sin(phi)

        L1 = Config.LINK_1
        L2 = Config.LINK_2
        L_sq = w_r**2 + w_z**2
        L = np.sqrt(L_sq)
        
        if L > (L1 + L2):
            print(f"Target unreachable! Dist: {L:.1f}")
            return None
            
        alpha = np.arctan2(w_z, w_r)
        cos_angle_elbow = (L_sq - L1**2 - L2**2) / (2 * L1 * L2)
        cos_angle_elbow = np.clip(cos_angle_elbow, -1.0, 1.0)
        
        theta2 = np.arccos(cos_angle_elbow)
        k1 = L1 + L2 * np.cos(theta2)
        k2 = L2 * np.sin(theta2)
        beta = np.arctan2(k2, k1)
        theta1 = alpha + beta
        theta3 = phi - (theta1 - theta2)

        deg_1 = np.degrees(theta1)
        deg_2 = np.degrees(theta2) * -1 
        deg_3 = np.degrees(theta3)
        
        return deg_1, deg_2, deg_3

    def move_to_xyz(self, x, y, z, move_time=1.0):
        """
        move_time: 이동에 걸리는 시간(초). 길수록 부드럽게 움직임.
        """
        # 1. Base (Yaw) 계산
        rad_base = np.arctan2(y, x)
        deg_base = np.degrees(rad_base)
        
        # 2. IK 계산 (항상 phi=-90도 유지)
        r_dist = np.sqrt(x**2 + y**2)
        ik_result = self.solve_ik_3dof_planar(r_dist, z, phi_deg=-90)
        
        if ik_result is None: return

        deg_shoulder, deg_elbow, deg_wrist = ik_result
        
        target_angles = [deg_base, deg_shoulder, deg_elbow, deg_wrist]
        target_ids = [Config.ID_BASE, Config.ID_SHOULDER, Config.ID_ELBOW, Config.ID_WRIST_P]
        
        goals = []
        delta_pos_list = []
        
        # 3. 목표 위치 계산
        for i, sid in enumerate(target_ids):
            angle = target_angles[i]
            direction = self.dirs[sid]
            pos = Config.SERVO_INIT_POS[sid] + int((Config.INPUT_RANGE/180.0) * angle * direction)
            
            if pos < 0: pos = 0
            if pos > 1023: pos = 1023
            
            goals.append(pos)
            
            # [수정] self.current_servo_pos에 실제 값이 들어있으므로
            # 실제 거리(delta)가 정확하게 계산됨 -> 속도 계산 정상화
            current = self.current_servo_pos.get(sid, Config.SERVO_INIT_POS[sid])
            delta_pos_list.append(abs(pos - current))
            
            self.current_servo_pos[sid] = pos

        # 4. [핵심] 속도 동기화 (Time-based Coordination)
        # 가장 많이 움직여야 하는 모터 기준으로 속도를 배분함
        # SCS Servo Speed: 대략 steps/sec 단위라고 가정하거나, 비례제어
        # 여기서는 단순히 거리 비례로 속도를 줍니다.
        
        # 기준 속도 (Base Speed) = 최대 이동 거리 / 시간
        max_delta = max(delta_pos_list)
        if max_delta == 0: return

        speeds = []
        scaling_factor = 1.0 / move_time 
        
        for delta in delta_pos_list:
            # 최소 속도 보정 (20 -> 50)
            # 너무 느리면 초기 기동이 안될 수 있으므로 최소값을 약간 올립니다.
            calc_speed = int((delta * scaling_factor) * 1.5)
            if calc_speed < 50: calc_speed = 50  # 최소 속도 상향 조정
            if calc_speed > 1000: calc_speed = 1000
            speeds.append(calc_speed)

        print(f"XYZ:({x},{y},{z}) | Goals:{goals} | Speeds:{speeds}")
            
        # 5. 명령 전송
        self.servo.sync_write_pos_speed(target_ids, goals, speeds)

    def control_magnet(self, on_off):
        print(f"--- Magnet {'ON' if on_off else 'OFF'} ---")

# --- 메인 실행 ---
def main():
    manager = SCSServoManager(Config.DEVICE_NAME, Config.BAUDRATE)
    arm = RobotArm(manager)
    
    try:
        print("=== Custom Arm : Vertical Wrist Control ===")
        magnet = Electromagnet(in1_pin=IN1, in2_pin=IN2, demag_duration=PULSE_TIME)
    
   
        print(">> idle")
        # 초기 위치 잡기 (이동시간 2초)
        arm.move_to_xyz(150, 0, 50, move_time=1.0)
        time.sleep(5.0)
        
        # 위로 들기 (수평 유지 확인)
        print(">> turn")
        arm.move_to_xyz(0, -150, 80, move_time=3.0)
        time.sleep(5.0)
        print(">> go down")
        arm.move_to_xyz(0, -150, 0, move_time=2.0)
        time.sleep(4.0)
        printf(">>magnet on")
        result = magnet.grab()
        print(f"[Result] {result}")
        
        time.sleep(3.0)
        print(">> go up")
        arm.move_to_xyz(0, -150, 80, move_time=2.0)
        time.sleep(4.0)
        
        # 다시 내리기
        print(">> turn")
        arm.move_to_xyz(0, 150, 50, move_time=3.0)
        time.sleep(5.0)
        print(">> go down")
        arm.move_to_xyz(0, 150, 0, move_time=2.0)
        time.sleep(4.0)
        
        printf(">>magnet off")
        result = magnet.release()
        print(f"[Result] {result}")
        time.sleep(4.0)
        
        print(">> go up")
        arm.move_to_xyz(0, 150, 50, move_time=2.0)
        time.sleep(4.0)

        # 멀리 뻗기
        print(">> idle")
        arm.move_to_xyz(150, 0, 50, move_time=2.0)
        time.sleep(2.5)
        magnet.cleanup()
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        manager.close_port()

if __name__ == "__main__":
    main()