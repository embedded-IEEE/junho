import Jetson.GPIO as GPIO
import time

# 1. 핀 번호 설정 (BOARD 모드 기준)
PUL_PIN = 38  # Pulse
DIR_PIN = 37  # Direction
ENA_PIN = 36  # Enable

# 2. 지연 시간 설정 (속도 조절)
# 아두이노의 delayMicroseconds(100) = 0.0001초
# 리눅스/파이썬 특성상 너무 빠르면 모터가 탈조(Stall)할 수 있으므로
# 안 돌아가면 이 값을 0.0005 또는 0.001로 늘려보세요.
STEP_DELAY = 0.0001 

def main():
    # --- setup() ---
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(PUL_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(DIR_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(ENA_PIN, GPIO.OUT, initial=GPIO.LOW)
    
    print("스텝 모터 제어 시작")
    print("Ctrl+C를 누르면 종료됩니다.")

    try:
        # --- loop() ---
        while True:
            # 1. 정방향 (Forward) 6400 스텝
            print("Forward moving...")
            GPIO.output(DIR_PIN, GPIO.LOW)   # 방향 설정
            GPIO.output(ENA_PIN, GPIO.HIGH)  # 모터 활성화
            
            for _ in range(6400):
                GPIO.output(PUL_PIN, GPIO.HIGH)
                time.sleep(STEP_DELAY)
                GPIO.output(PUL_PIN, GPIO.LOW)
                time.sleep(STEP_DELAY)

            time.sleep(0.5) # 방향 전환 전 잠시 대기 (기계적 충격 방지)

            # 2. 역방향 (Backward) 6400 스텝
            print("Backward moving...")
            GPIO.output(DIR_PIN, GPIO.HIGH)  # 방향 반대로 설정
            GPIO.output(ENA_PIN, GPIO.HIGH)  # 모터 활성화
            
            for _ in range(6400):
                GPIO.output(PUL_PIN, GPIO.HIGH)
                time.sleep(STEP_DELAY)
                GPIO.output(PUL_PIN, GPIO.LOW)
                time.sleep(STEP_DELAY)

            time.sleep(0.5) # 다시 정방향 가기 전 잠시 대기

    except KeyboardInterrupt:
        print("\n강제 종료됨")

    finally:
        # 프로그램 종료 시 GPIO 해제 및 모터 힘 풀기
        GPIO.output(ENA_PIN, GPIO.LOW) # 필요 시 HIGH/LOW 상태 확인 (드라이버마다 다름)
        GPIO.cleanup()
        print("GPIO cleanup 완료")

if __name__ == '__main__':
    main()