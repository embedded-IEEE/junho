#!/usr/bin/env python3

import time
import sys
import numpy as np

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
# --- 라이브러리 임포트 (기존과 동일) ---
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
except ImportError:
    print("Error: 'rclpy' 라이브러리가 없습니다.")
    sys.exit(1)

# SCS 서보 모터 라이브러리 경로 (사용자 환경에 맞게 수정)
sys.path.append('/home/jetson/SCSCtrl') 
try:
    from SCSCtrl.scservo_sdk import *
except ImportError:
    print("Warning: SCSCtrl 라이브러리가 없습니다. (테스트 모드)")

# --- [수정됨] 설정 상수 (커스텀 로봇 스펙 반영) ---
class Config:
    DEVICE_NAME = '/dev/ttyTHS1'
    BAUDRATE = 1000000

    # 서보 ID 설정 (5축)
    ID_BASE = 1
    ID_SHOULDER = 2
    ID_ELBOW = 3
    ID_WRIST_P = 5
    ID_WRIST_Y = 4
    
    SERVO_IDS = [1, 2, 3, 5, 4]

    # [중요] 링크 길이 (mm) - 알려주신 값 반영
    # L1: 어깨~팔꿈치, L2: 팔꿈치~손목, L3: 손목~전자석끝
    LINK_1 = 100.0  # 10cm
    LINK_2 = 100.0  # 10cm
    LINK_3 = 80.0   # 5cm(Wrist) + 3cm(Yaw+Magnet) = 8cm

    # 초기 위치 (Calibration 값) - 조립 상태에 따라 수정 필요
    # 예: 512가 똑바른 0도라고 가정
    SERVO_INIT_POS = {1: 514, 2: 961, 3: 940, 4: 512, 5: 548}
    
    INPUT_RANGE = 850    # 모터 입력 범위 (0~1023 기준 움직임 폭)
    ANGLE_RANGE = 180.0  # 위 입력값에 해당하는 각도 범위

    # 통신 주소
    ADDR_GOAL_POSITION = 42
    ADDR_GOAL_SPEED = 46
    ADDR_PRESENT_POSITION = 56

# --- 하드웨어 통신 클래스 (기존과 동일) ---
class SCSServoManager:
    def __init__(self, device_name, baudrate):
        self.port_handler = PortHandler(device_name)
        self.packet_handler = PacketHandler(1)
        self.group_sync_write = GroupSyncWrite(self.port_handler, self.packet_handler, Config.ADDR_GOAL_POSITION, 2)
        if not self.open_port(baudrate): print("Port Error")
        
    def open_port(self, baudrate):
        if self.port_handler.openPort() and self.port_handler.setBaudRate(baudrate): return True
        return False
        
    def close_port(self): self.port_handler.closePort()
    
    def write_speed(self, servo_id, speed):
        self.packet_handler.write2ByteTxRx(self.port_handler, servo_id, Config.ADDR_GOAL_SPEED, int(speed))

    def sync_write_pos(self, servo_ids, goals):
        for i, sid in enumerate(servo_ids):
            param = [SCS_LOBYTE(int(goals[i])), SCS_HIBYTE(int(goals[i]))]
            self.group_sync_write.addParam(sid, param)
        self.group_sync_write.txPacket()
        self.group_sync_write.clearParam()

# --- [핵심] 커스텀 로봇 팔 제어 클래스 ---
class RobotArm:
    def __init__(self, servo_manager):
        self.servo = servo_manager
        
        # 방향 보정 (모터가 거꾸로 달려있으면 -1로 변경)
        # 순서: [Base, Shoulder, Elbow, Wrist_P, Wrist_Y]
        self.dirs = [1, 1, -1, -1, 1] 

    def solve_ik_3dof_planar(self, r, z, phi_deg=-90):
        """
        3축 평면 IK (어깨, 팔꿈치, 손목)
        r: 몸통 중심에서의 수평 거리 (Horizontal Distance)
        z: 높이 (Height)
        phi_deg: 지면 기준 그리퍼 각도 (기본값 -90도 = 수직 아래)
        """
        phi = np.radians(phi_deg)
        
        # 1. 손목 관절(Wrist Joint)의 위치 계산 (전자석 끝 위치에서 역추적)
        # 전자석 끝(r, z)에서 링크3 길이만큼 뒤로 물러남
        w_r = r - Config.LINK_3 * np.cos(phi)
        w_z = z - Config.LINK_3 * np.sin(phi)

        # 2. 제 2코사인 법칙으로 Shoulder(theta1), Elbow(theta2) 계산
        L1 = Config.LINK_1
        L2 = Config.LINK_2
        
        # 원점에서 손목 관절까지의 거리
        L_sq = w_r**2 + w_z**2
        L = np.sqrt(L_sq)
        
        # 도달 불가능 영역 체크
        if L > (L1 + L2):
            print("Target too far!")
            return None
            
        # 알파(alpha): 어깨-손목 직각삼각형 각도
        alpha = np.arctan2(w_z, w_r)
        
        # 코사인 법칙 적용
        cos_angle_elbow = (L_sq - L1**2 - L2**2) / (2 * L1 * L2)
        cos_angle_elbow = np.clip(cos_angle_elbow, -1.0, 1.0) # 안전장치
        
        # 팔꿈치 각도 (theta2)
        theta2 = np.arccos(cos_angle_elbow) # 라디안 (0 ~ 180)
        # 로봇 구조상 팔꿈치는 아래로 꺾이는지 위로 꺾이는지 결정 (보통 위로 꺾임: -theta2)
        # 여기서는 양수로 두고 나중에 방향 보정
        
        # 어깨 보정 각도 (beta)
        k1 = L1 + L2 * np.cos(theta2)
        k2 = L2 * np.sin(theta2)
        beta = np.arctan2(k2, k1)
        
        theta1 = alpha + beta  # 어깨 각도 (theta1)

        # 3. 손목 각도 (theta3) 계산
        # 공식: phi(목표각도) = theta1 + theta2 + theta3
        # 따라서 theta3 = phi - theta1 - theta2
        theta3 = phi - (theta1 - theta2) # 팔꿈치 꺾이는 방향에 따라 부호 주의

        # 결과 (Degree 변환)
        # 일반적인 서보 모터 기준 90도가 0점인 경우 보정 필요할 수 있음
        deg_1 = np.degrees(theta1)
        deg_2 = np.degrees(theta2) * -1 # 팔꿈치는 반대로 꺾임
        deg_3 = np.degrees(theta3)
        
        return deg_1, deg_2, deg_3

    def move_to_xyz(self, x, y, z, duration=1.0):
        """
        통합 좌표 이동 함수
        (x, y, z)로 이동하되, 전자석은 항상 바닥을 봄 (-90도)
        """
        # 1. Base (Yaw) 계산
        # atan2(y, x) -> 라디안 -> 도
        rad_base = np.arctan2(y, x)
        deg_base = np.degrees(rad_base)
        
        # 2. 평면 거리(r) 계산
        r_dist = np.sqrt(x**2 + y**2)
        
        # 3. 3축 IK 계산 (Target Angle = -90도: 수직)
        ik_result = self.solve_ik_3dof_planar(r_dist, z, phi_deg=-90)
        
        if ik_result is None:
            print("이동 불가능한 위치입니다.")
            return

        deg_shoulder, deg_elbow, deg_wrist = ik_result
        
        # 4. 모터 값 변환 및 이동
        # 각도 -> 서보 값(0~1023) 변환 로직
        target_angles = [deg_base, deg_shoulder, deg_elbow, deg_wrist, 0] # 마지막은 Wrist Yaw(0도 유지)
        ids = [Config.ID_BASE, Config.ID_SHOULDER, Config.ID_ELBOW, Config.ID_WRIST_P, Config.ID_WRIST_Y]
        
        goals = []
        for i, angle in enumerate(target_angles):
            # 초기값(512) + (입력범위/180도 * 각도 * 방향보정)
            pos = Config.SERVO_INIT_POS[ids[i]] + int((Config.INPUT_RANGE/180.0) * angle * self.dirs[i])
            goals.append(pos)
            
        # 속도 설정 (간단히 고정)
        for sid in ids:
            self.servo.write_speed(sid, 200) # 속도 200
            
        # 동시 이동 명령
        self.servo.sync_write_pos(ids, goals)
        print(f"Moved to: X={x}, Y={y}, Z={z} | Angles: {list(map(int, target_angles))}")

    def control_magnet(self, on_off):
        """
        전자석 제어 함수
        on_off: True(켜기), False(끄기)
        """
        state = "ON" if on_off else "OFF"
        print(f"--- Magnet {state} ---")
        # 실제 구현: GPIO 제어 코드 또는 서보 포트 제어
        # 예: 서보 포트를 스위치로 쓰는 경우 (PWM 2000 vs 1000)
        # self.servo.write_speed(6, 1000) ...
        # 현재는 로그만 출력

# --- 메인 실행 ---
def main():
    rclpy.init()
    manager = SCSServoManager(Config.DEVICE_NAME, Config.BAUDRATE)
    arm = RobotArm(manager)
    
    try:
        print("=== Custom 5-DOF Arm Start ===")
        
        # 1. 홈 포지션 (모두 0도 - "ㄱ"자 혹은 "1"자 대기)
        # 로봇 팔을 안전하게 들기
        arm.move_to_xyz(220, 0, 0, 2.0)
        time.sleep(2)
        arm.move_to_xyz(100, 0, 50, 2.0)
        time.sleep(2)
        
        # 2. 물건 잡으러 가기 (X=200, Y=0, Z=50)
        # 바닥에 가까운 위치
        print(">> Reaching Target...")
        arm.move_to_xyz(200, 0, 50, 1.0) 
        time.sleep(2)
        
        # 3. 전자석 켜기
        arm.control_magnet(True)
        time.sleep(1)
        
        # 4. 들어 올리기 (Z=200)
        print(">> Lifting...")
        arm.move_to_xyz(200, 0, 200, 1.0)
        time.sleep(2)
        
        # 5. 옆으로 옮기기 (X=100, Y=150) -> 몸통 회전 발생
        print(">> Moving Sideways...")
        arm.move_to_xyz(100, 150, 200, 1.0)
        time.sleep(2)
        
        # 6. 내려놓기
        arm.control_magnet(False)
        
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        manager.close_port()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
