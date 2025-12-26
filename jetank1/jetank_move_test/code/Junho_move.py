import threading
from time import sleep
from jetbot import Robot
from SCSCtrl import TTLServo

robot = Robot()
moveStartTor = 0.15    

fb_input = 0           # 전방 및 후방 방향의 속도 매개변수를 저장합니다.
lr_input = 0           # 조향에 대한 매개변수를 저장합니다.

# TTL 서보: TTL 전압 레벨 기반의 시리얼 통신을 사용하는 서보 모터
servoCtrlTime = 0.0001   # TTL 서보 통신 후 오류를 피하기 위한 지연입니다.
speedInput = 300       # 카메라의 틸트 및 팬 회전 속도입니다.

armXStatus = 0         # 로봇 팔의 X 축 이동 상태 (원거리 및 근거리 방향)입니다.
armYStatus = 0         # 로봇 팔의 Y 축 이동 상태 (수직 방향)입니다.
cameStatus = 0         # 카메라의 이동 상태입니다.
goalX = 130            # 로봇 팔의 그리퍼의 X축 좌표를 저장하세요.
goalY = 50             # 로봇 팔의 그리퍼의 Y축 좌표를 저장하세요.
goalC = 0              # 카메라의 위치를 저장하세요.

xMax = 210             # 로봇 팔의 X 축 최대값을 설정하세요.
xMin = 140             # 로봇 팔의 X 축 최소값을 설정하세요.

yMax = 120             # 로봇 팔의 Y 축 최대값을 설정하세요.
yMix = -120            # 로봇 팔의 Y 축 최소값을 설정하세요.


cMax = 25              # 카메라의 최대 위치를 설정하세요.
cMin = -45             # 카메라의 최소 위치를 설정하세요.
cDan = 0               # 카메라의 위험한 선을 설정하세요.

movingTime = 0.005 # 로봇 팔 역운동학 함수의 인접한 두 위치 간의 실행 시간을 설정하세요.
movingSpeed = 1        # 로봇 팔의 이동 속도를 설정하세요.
grabStatus = 0         # 그리퍼의 이동 상태입니다.
cameraPosCheckMode = 1 # 카메라 위치 확인 기능의 스위치, 1 - 켜기, 0 - 끄기입니다.
                       # 한 번 켜면, 로봇 팔과 카메라 간의 간섭을 피할 수 있습니다.
cameraPosDanger = 0    # 카메라가 위험 지역에 있는지 여부입니다.
cameraMoveSafe = 0     # 카메라가 안전 지역으로 이동했는지 여부입니다.
TTLServo.xyInputSmooth(goalX, goalY, 3)
TTLServo.servoAngleCtrl(1, 0, 1, 100)
TTLServo.servoAngleCtrl(4, 0, 1, 100)
TTLServo.servoAngleCtrl(5, 0, 1, 100)
sleep(3.5)
print('ready!')