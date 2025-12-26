import Jetson.GPIO as GPIO
import time

# 핀 번호 설정 (BOARD 물리 번호 기준 예시)
# 사용하시는 핀 번호에 맞춰 수정하세요
IN1_PIN = 37 
IN2_PIN = 38

def setup_gpio():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(IN1_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(IN2_PIN, GPIO.OUT, initial=GPIO.LOW)

def grab_object():
    """자석 켜기"""
    print("Magnet ON")
    GPIO.output(IN1_PIN, GPIO.HIGH)
    GPIO.output(IN2_PIN, GPIO.LOW)

def release_object():
    """자석 끄기 (역전압 펄스 포함)"""
    print("Magnet OFF with Demagnetization Pulse")
    
    # 1단계: 일단 끄기
    GPIO.output(IN1_PIN, GPIO.LOW)
    GPIO.output(IN2_PIN, GPIO.LOW)
    
    # 2단계: 역전압 쏘기 (잔류 자기 제거)
    # IN1, IN2 상태를 grab_object와 반대로 설정
    GPIO.output(IN1_PIN, GPIO.LOW)
    GPIO.output(IN2_PIN, GPIO.HIGH)
    
    # ★ 핵심: 시간 조절 (0.1초 = 100ms)
    # 너무 길면 반대로 붙고, 너무 짧으면 안 떨어짐. 
    # 0.05 ~ 0.2 사이에서 테스트 필요
    time.sleep(0.2) 
    
    # 3단계: 완전 차단
    GPIO.output(IN1_PIN, GPIO.LOW)
    GPIO.output(IN2_PIN, GPIO.LOW)

try:
    setup_gpio()
    
    while True:
        cmd = input("Press Enter to Grab, 'd' to Drop: ")
        if cmd == 'd':
            release_object()
        else:
            grab_object()

except KeyboardInterrupt:
    GPIO.cleanup()