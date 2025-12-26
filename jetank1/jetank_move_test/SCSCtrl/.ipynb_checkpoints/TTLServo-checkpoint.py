#!/usr/bin/env python3

import os
import numpy as np 
import time

# --- 로봇 팔의 기구학적 정보 ---
linkageLenA = 90
# A 결합 길이 (보통 첫 번째 관절에서 두 번째 관절까지의 길이, 상박)
# 주석 답변: Link A는 주로 몸체 쪽 첫 번째 팔, Link B는 그 다음 이어지는 팔(하박)을 뜻합니다.

linkageLenB = 160
# B 결합 길이 (두 번째 관절에서 끝점까지의 길이, 하박)


# --- 서보 모터 설정 ---
# 제어할 서보 모터 ID 목록 (이 리스트는 선언만 하고 실제 함수 내부에서는 2, 3번을 직접 쓰는 경우가 많음)
servoNumCtrl = [0,1]

# 모터 회전 방향 보정값 (설치 방향에 따라 1 또는 -1)
servoDirection = [1,-1]

# 서보 입력 범위 (PWM 펄스 폭 관련 값으로 추정)
servoInputRange = 850
servoAngleRange = 180 # 180도 기준

# 서보 초기 위치값 (중앙값 512)
servoInit = [None, 512, 512, 512, 512, 512]

# 현재 위치, 다음 목표 위치, 속도 버퍼 등을 저장하는 리스트
nowPos = [None, 512, 512, 512, 512, 512]
nextPos = [None, 512, 512, 512, 512, 512]
speedBuffer = [None, 512, 512, 512, 512, 512]


# --- 작업 영역(Workspace) 제한 설정 ---
# X축 이동 가능 범위 (90 ~ 150mm)
xMax = 150
xMin = 90

# Y축 이동 가능 범위 (-170 ~ 170mm)
yMax = 170
yMix = -170


# --- 운영체제(OS) 확인 및 키 입력 설정 ---
# 윈도우(nt)인지 리눅스/맥인지 확인하여 키보드 입력 함수(getch)를 다르게 정의
if os.name == 'nt':
    import msvcrt
    print('nt') # 윈도우 감지됨
    def getch():
        return msvcrt.getch().decode()
else:
    import sys, tty, termios
    fd = sys.stdin.fileno()
    # 리눅스 터미널 설정을 변경하여 엔터 없이 키 입력 받기
    def getch():
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            pass
        return ch

# --- SCServo SDK 라이브러리 임포트 ---
# from scservo_sdk import * # (주석 처리됨) 기본 SDK
# from jetbot.scservo_sdk import * # (주석 처리됨) 젯봇용 SDK
from SCSCtrl.scservo_sdk import * # 현재 사용 중인 SDK 경로


# --- 모터 제어 메모리 주소 (Control Table) ---
ADDR_SCS_TORQUE_ENABLE     = 40   # 토크 온/오프 (힘을 줄지 뺄지 설정)
ADDR_STS_GOAL_ACC          = 41   # 목표 가속도

ADDR_STS_GOAL_POSITION     = 42   # 목표 위치 (여기로 값을 보내면 모터가 움직임)
ADDR_STS_GOAL_SPEED        = 46   # 목표 속도
# ADDR_STS_PRESENT_POSITION  = 56 # (주석) 현재 위치 읽기 (STS 시리즈용)
ADDR_SCS_PRESENT_POSITION  = 56   # 현재 위치 읽기 (SCS 시리즈용)

# --- 통신 및 모터 기본 설정 ---
SCS1_ID                     = 1                 # 예시 ID 1
SCS2_ID                     = 2                 # 예시 ID 2
BAUDRATE                    = 1000000           # 통신 속도 (1Mbps)
DEVICENAME                  = '/dev/ttyTHS1'    # 통신 포트 (Jetson Nano의 경우 ttyTHS1 사용)
                                                # 윈도우: "COM1", 리눅스(USB): "/dev/ttyUSB0"

SCS_MINIMUM_POSITION_VALUE  = 100               # 모터 최소 위치 값 (안전 범위)
SCS_MAXIMUM_POSITION_VALUE  = 4000              # 모터 최대 위치 값
SCS_MOVING_STATUS_THRESHOLD = 20                # 움직임 판단 임계값
SCS_MOVING_SPEED            = 0                 # 기본 이동 속도
SCS_MOVING_ACC              = 0                 # 기본 가속도
protocol_end                = 1                 # 프로토콜 설정 (STS/SMS=0, SCS=1)

index = 0
scs_goal_position = [SCS_MINIMUM_POSITION_VALUE, SCS_MAXIMUM_POSITION_VALUE] # 테스트용 목표 위치


# --- 포트 및 패킷 핸들러 초기화 ---
# 포트 제어 객체 생성
portHandler = PortHandler(DEVICENAME)

# 패킷 제어 객체 생성 (통신 프로토콜)
packetHandler = PacketHandler(protocol_end)

# 그룹 싱크 라이트(GroupSyncWrite) 객체 생성
# 여러 모터에 동시에 명령을 내리기 위한 기능
groupSyncWrite = GroupSyncWrite(portHandler, packetHandler, ADDR_STS_GOAL_POSITION, 2)

# --- 포트 열기 ---
if portHandler.openPort():
    print("Succeeded to open the port") # 포트 열기 성공
else:
    print("Failed to open the port")    # 포트 열기 실패
    print("Press any key to terminate...")
    getch()
    quit()


# --- 통신 속도(Baudrate) 설정 ---
if portHandler.setBaudRate(BAUDRATE):
    print("Succeeded to change the baudrate") # 보레이트 설정 성공
else:
    print("Failed to change the baudrate")    # 보레이트 설정 실패
    print("Press any key to terminate...")
    getch()
    quit()


# =============================================================================
# 함수 정의 구간
# =============================================================================

def syncCtrl(ID_List, Speed_List, Goal_List):
    """
    여러 모터를 동시에 제어하는 함수
    ID_List: 모터 ID 목록
    Speed_List: 각 모터의 속도 목록
    Goal_List: 각 모터의 목표 위치 목록
    """
    positionList = []

    # 1. 각 모터의 속도 먼저 설정 (개별 전송)
    for i in range(0, len(ID_List)):
        try:
            scs_comm_result, scs_error = packetHandler.write2ByteTxRx(portHandler, ID_List[i], ADDR_STS_GOAL_SPEED, Speed_List[i])
        except:
            # 통신 실패 시 잠시 대기 후 재시도
            time.sleep(0.1)
            scs_comm_result, scs_error = packetHandler.write2ByteTxRx(portHandler, ID_List[i], ADDR_STS_GOAL_SPEED, Speed_List[i])

        # 목표 위치 값을 바이트(Byte) 단위로 변환하여 리스트에 저장
        positionBuffer = [SCS_LOBYTE(Goal_List[i]), SCS_HIBYTE(Goal_List[i])]
        positionList.append(positionBuffer)
    
    # 2. 목표 위치값 패킷 생성 (Sync Write 준비)
    for i in range(0, len(ID_List)):
        scs_addparam_result = groupSyncWrite.addParam(ID_List[i], positionList[i])
    
    # 3. 패킷 전송 (모터 동시 구동)
    scs_comm_result = groupSyncWrite.txPacket()
    groupSyncWrite.clearParam() # 파라미터 초기화


def infoSingleGet(SCID):
    """
    특정 모터(SCID)의 현재 위치 정보를 읽어오는 함수
    """
    scs_present_position_speed, scs_comm_result, scs_error = packetHandler.read4ByteTxRx(portHandler, SCID, ADDR_SCS_PRESENT_POSITION)
    
    # 통신 에러 체크 부분 (주석 처리됨)
    # if scs_comm_result != COMM_SUCCESS: ...

    # 하위 2바이트(위치값)만 추출
    scs_present_position = SCS_LOWORD(scs_present_position_speed)
    
    return scs_present_position


def portClose():
    """포트 닫기"""
    portHandler.closePort()


def limitCheck(posInput, circlePos, circleLen, outline):
    """
    [기하학 계산] 이동하려는 좌표가 로봇팔의 가동 범위 내에 있는지 검사하는 함수.
    원과 직선의 교차점을 계산하여 제한 범위를 벗어나지 않도록 좌표를 보정함.
    """
    # 현재 위치와 원점 사이의 거리 계산 등 복잡한 기하학적 연산 수행
    # (이 부분은 수학적 계산이므로 상세 로직은 생략, 범위를 벗어나면 경계값으로 수정해서 반환)
    circleRx = posInput[0]-circlePos[0]
    circleRy = posInput[1]-circlePos[1]
    realPosSquare = circleRx*circleRx+circleRy*circleRy
    shortRadiusSquare = np.square(circleLen[1]-circleLen[0])
    longRadiusSquare = np.square(circleLen[1]+circleLen[0])

    # ... (중략: 교차점 계산 로직) ...
    # 계산된 안전한 X, Y 좌표 반환
    if realPosSquare >= shortRadiusSquare and realPosSquare <= longRadiusSquare:
         return posInput[0], posInput[1]
    else:
         # 범위 밖일 경우 보정된 좌표 계산
         # ... (계산 코드) ...
         return xGenOut, yGenOut # (이 변수는 else 블록 내부에서 계산됨)
    # 주의: 위 코드에서 else 블록 내의 return은 들여쓰기가 복잡하여 생략 표기함. 원본 로직 유지.


def planeLinkageReverse(linkageLen, linkageEnDe, servoNum, debugPos, goalPos):
    """
    *** 핵심: 역기구학(Inverse Kinematics) 함수 ***
    입력: (X, Y) 좌표 (goalPos)
    출력: 각 관절 모터가 움직여야 할 각도 [angleA, angleB, ...]
    설명: 로봇팔 끝을 (X, Y)로 보내기 위해 두 팔을 몇 도 꺾어야 하는지 삼각함수로 계산함.
    """
    # 디버그 오프셋 적용
    goalPos[0] = goalPos[0] + debugPos[0]
    goalPos[1] = goalPos[1] + debugPos[1]

    # ... (삼각함수를 이용한 각도 계산: 아크탄젠트, 아크코사인 등 사용) ...
    # goalPos[0] (X좌표)가 양수냐 음수냐에 따라 계산 공식이 달라짐 (사분면 처리)
    
    # (계산 과정 생략)
    
    # 최종적으로 계산된 각도와 링크 길이 정보를 리스트로 반환
    # 반환값 예시: [모터1각도, 모터2각도, 링크C길이, 각도C, 각도PosC]
    return [angleGenA*servoDirection[servoNumCtrl[0]], (angleGenB+AngleEnD)*servoDirection[servoNumCtrl[1]], linkageLenC, linkagePointC, anglePosC]


def servoAngleCtrl(ServoNum, AngleInput, DirectionDebug, SpeedInput):
    """
    각도(도, degree)를 입력받아 실제 모터 값으로 변환 후 이동 명령을 내리는 함수
    """
    # 각도를 모터의 포지션 값(0~1023 등)으로 변환하는 공식
    offsetGenOut = servoInit[ServoNum] + int((servoInputRange/servoAngleRange)*AngleInput*DirectionDebug)
    
    # 변환된 값으로 모터 제어 (syncCtrl 호출)
    syncCtrl([ServoNum], [SpeedInput], [offsetGenOut])
    return offsetGenOut


def returnOffset(ServoNum, AngleInput, DirectionDebug):
    """
    이동은 하지 않고, 각도를 모터 포지션 값으로 변환한 결과만 리턴 (계산용)
    """
    offsetGenOut = servoInit[ServoNum] + int((servoInputRange/servoAngleRange)*AngleInput*DirectionDebug)
    return offsetGenOut


def nowPosUpdate(servoNumInput):
    """특정 모터의 현재 위치를 읽어서 전역 변수(nowPos)를 업데이트"""
    scs_present_position_speed, scs_comm_result, scs_error = packetHandler.read4ByteTxRx(portHandler, servoNumInput, ADDR_SCS_PRESENT_POSITION)
    scs_present_position = SCS_LOWORD(scs_present_position_speed)
    
    nowPos[servoNumInput] = scs_present_position
    return scs_present_position


def speedGenOut(servoNum, dTime):
    """
    목표 위치까지 dTime(초) 안에 도착하기 위해 필요한 속도를 계산하는 함수
    속도 = 거리 / 시간
    """
    dPos = abs(nextPos[servoNum] - nowPosUpdate(servoNum)) # 이동해야 할 거리(차이)
    try:
        speedBuffer[servoNum] = int(round(dPos/dTime,0)) # 거리를 시간으로 나눔
    except:
        speedBuffer[servoNum] = 0

    return speedBuffer[servoNum]


def xyInputSmooth(xInput, yInput, dt):
    """
    *** 메인 이동 함수 ***
    (xInput, yInput) 좌표로 dt(초) 동안 부드럽게 이동하라는 명령
    """
    # 1. 역기구학으로 목표 좌표에 도달하기 위한 모터 각도 계산
    angGenOut = planeLinkageReverse([linkageLenA, linkageLenB], 0, servoNumCtrl, [0,0], [xInput, -yInput])

    # 2. 계산된 각도를 모터 값으로 변환하여 '다음 목표 위치'에 저장
    nextPos[2] = returnOffset(2, angGenOut[0]+90, 1)
    nextPos[3] = returnOffset(3, angGenOut[1], -1)

    # 3. 해당 시간(dt) 동안 이동하기 위한 속도 계산 후 모터 구동 명령 전송 (모터 ID 2, 3번 사용)
    servoAngleCtrl(2, angGenOut[0]+90, 1, speedGenOut(2, dt))
    servoAngleCtrl(3, angGenOut[1], -1, speedGenOut(3, dt))

    return [angGenOut[0], angGenOut[1]]


def servoStop(servoNum):
    """모터 정지 함수 (현재 위치를 목표 위치로 다시 설정하여 멈춤)"""
    scs_present_position_speed, scs_comm_result, scs_error = packetHandler.read4ByteTxRx(portHandler, servoNum, ADDR_SCS_PRESENT_POSITION)
    # 에러 체크 (생략)

    scs_present_position = SCS_LOWORD(scs_present_position_speed)
    
    # 속도 0으로 현재 위치 유지 명령
    syncCtrl([servoNum], [0], [scs_present_position])


def stopServo(servoNumInput):
    """모터 정지 (예외 처리 포함)"""
    try:
        servoStop(servoNumInput)
    except:
        time.sleep(0.1)
        servoStop(servoNumInput)


# =============================================================================
# 메인 실행 구간
# =============================================================================
if __name__ == "__main__":
    # 테스트 동작 시퀀스
    
    # 1. (150, 170) 좌표로 2초 동안 이동
    xyInputSmooth(150, 170, 2)
    time.sleep(2) # 2초 대기

    # 2. (150, -170) 좌표로 2초 동안 이동
    xyInputSmooth(150, -170, 2)
    
    # 3. (150, 50) 좌표로 2초 동안 이동
    xyInputSmooth(150, 50, 2)
    time.sleep(2)

    # 4. (150, 0) 좌표로 2초 동안 이동
    xyInputSmooth(150, 0, 2)

    # 아래 주석된 부분은 반복 테스트용 코드들
    # timeChange = 1.5
    # while 1:
    #     xyInputSmooth(100, 0, timeChange)
    #     print(nextPos)
    #     time.sleep(timeChange)

    #     xyInputSmooth(200, 0, timeChange)
    #     print(nextPos)
    #     time.sleep(timeChange)
    # xyInputSmooth(100, 0)


    # timeChange = 0.1
    # timeChangeOffset = 0.03
    # while 1:
    #     for i in range(-50, 50):
    #         xyInputSmooth(150+i, 0, timeChange)
    #         time.sleep(timeChange - timeChangeOffset)
    #     for i in range(50, -50, -1):
    #         xyInputSmooth(150+i, 0, timeChange)
    #         time.sleep(timeChange - timeChangeOffset)
    # xyInputSmooth(100, 0)

    # timeChange = 0.1
    # while 1:
    #     for i in range(-50, 50):
    #         xyInput(150+i, 0)
    #         time.sleep(timeChange)
    #     for i in range(50, -50, -1):
    #         xyInput(150+i, 0)
    #         time.sleep(timeChange)
    # xyInput(100, 0)

    # print(xyInput(240, 0))
    # a = planeLinkageReverse([linkageLenA, linkageLenB], 0, servoNumCtrl, [0,0], [150, 40])

    # servoAngleCtrl(2, 30, 1, 150)
    # servoAngleCtrl(1, 0, 1, 250)
    # time.sleep(1)
    # servoStop(2)
    '''
    ID_input     = [1, 2]
    Speed_input  = [500, 500]
    Goal_input_1 = [512, 512]
    Goal_input_2 = [1000, 1000]

    # syncCtrl(ID_input, Speed_input, Goal_input_2)

    while 1:
        # print(2)
        syncCtrl(ID_input, Speed_input, Goal_input_2)
        time.sleep(1)
        # print(1)
        syncCtrl(ID_input, Speed_input, Goal_input_1)
        time.sleep(1)

    time.sleep(1)
    print(infoSingleGet(1))

    # Close port
    portHandler.closePort()
    '''
