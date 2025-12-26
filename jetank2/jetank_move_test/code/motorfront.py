import Jetson.GPIO as GPIO
import time
import threading  # 동시에 작업을 하기 위한 라이브러리

# 1. 핀 설정
PUL_PIN = 38
DIR_PIN = 37
ENA_PIN = 36

# 2. 속도 설정 (너무 빠르면 모터가 멈출 수 있음)
STEP_DELAY = 0.0005 

# 3. 상태 제어를 위한 전역 변수
motor_running = True  # 처음에는 켜진 상태로 시작
program_exit = False  # 프로그램 종료 신호

def motor_task():
    """
    모터를 계속 돌리는 역할을 하는 별도의 스레드 함수입니다.
    """
    global motor_running, program_exit

    print("--> 모터 스레드 시작됨")
    
    # 방향 및 Enable 설정 (초기값: 순방향)
    GPIO.output(DIR_PIN, GPIO.LOW)   # 순방향 설정
    GPIO.output(ENA_PIN, GPIO.HIGH)  # 모터 활성화

    while not program_exit:
        if motor_running:
            # 모터가 '가동' 상태일 때만 펄스를 보냄
            GPIO.output(PUL_PIN, GPIO.HIGH)
            time.sleep(STEP_DELAY)
            GPIO.output(PUL_PIN, GPIO.LOW)
            time.sleep(STEP_DELAY)
        else:
            # 모터가 '정지' 상태면 CPU를 쉬게 해줌
            time.sleep(0.1)

def main():
    global motor_running, program_exit

    # GPIO 초기화
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup([PUL_PIN, DIR_PIN, ENA_PIN], GPIO.OUT, initial=GPIO.LOW)

    print("프로그램 시작: 모터가 순방향으로 회전합니다.")
    print("'1' 입력: 정지")
    print("'2' 입력: 다시 회전")
    print("'q' 입력: 종료")

    # 모터 제어 스레드 시작 (백그라운드에서 실행)
    t = threading.Thread(target=motor_task)
    t.start()

    try:
        while True:
            user_input = input("\n명령 입력 (1:정지, 2:재시작, q:종료): ")

            if user_input == '1':
                motor_running = False
                print(">>> 모터 정지 명령됨")

            elif user_input == '2':
                motor_running = True
                print(">>> 모터 재시작 (순방향)")

            elif user_input == 'q':
                print("프로그램을 종료합니다...")
                program_exit = True  # 스레드 종료 신호
                break
            
    except KeyboardInterrupt:
        program_exit = True

    finally:
        # 스레드가 완전히 끝날 때까지 기다림
        t.join()
        GPIO.cleanup()
        print("GPIO 정리 완료.")

if __name__ == '__main__':
    main()