#!/usr/bin/env python3
import time
import sys
import os

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
# SCS 라이브러리 경로 (기존 코드와 동일하게)
sys.path.append('/home/jetson/SCSCtrl')
try:
    from SCSCtrl.scservo_sdk import *
except ImportError:
    print("SCSCtrl 라이브러리가 필요합니다.")
    sys.exit(1)

# 설정
DEVICE_NAME = '/dev/ttyTHS1'
BAUDRATE = 1000000
SERVO_IDS = [1, 2, 3, 4, 5]  # 사용 중인 모터 ID

# 통신 준비
portHandler = PortHandler(DEVICE_NAME)
packetHandler = PacketHandler(1)

if not portHandler.openPort():
    print("포트 열기 실패")
    sys.exit(1)
if not portHandler.setBaudRate(BAUDRATE):
    print("보드레이트 설정 실패")
    sys.exit(1)

# 토크 끄기 함수 (손으로 움직일 수 있게)
def disable_torque(sid):
    packetHandler.write1ByteTxRx(portHandler, sid, 40, 0) # 40=Torque Enable 주소

print("=== 서보 모터 캘리브레이션 도구 ===")
print("1. 로봇의 힘(토크)을 끕니다.")
print("2. 손으로 로봇을 '초기 자세(0도)'로 만드세요.")
print("3. 그때의 숫자(Position)를 기록하세요.")
print("------------------------------------")

# 모든 모터 토크 끄기
for sid in SERVO_IDS:
    disable_torque(sid)
    print(f"ID {sid}: Torque OFF")

try:
    while True:
        status_line = ""
        for sid in SERVO_IDS:
            # 현재 위치 읽기 (주소 56)
            pos, result, error = packetHandler.read2ByteTxRx(portHandler, sid, 56)
            if result != COMM_SUCCESS:
                val = "Err"
            else:
                val = f"{pos:04d}" # 4자리 숫자로 표시
            status_line += f"[ID {sid}: {val}]  "
        
        print(f"\r{status_line}", end="") # 한 줄에서 계속 업데이트
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n종료합니다.")
    portHandler.closePort()