import Jetson.GPIO as GPIO
import time
import threading

# 1. 핀 설정
PUL_PIN = 36 
DIR_PIN = 35 
ENA_PIN = 33 

# 2. 속도 설정
STEP_DELAY = 0.001

# 3. 전역 변수
motor_running = True 
program_exit = False 
motor_dir = GPIO.LOW  # 현재 방향 저장 변수 (초기값: LOW)

def motor_task():
    global motor_running, program_exit, motor_dir

    print("--> 모터 스레드 시작됨")
    
    GPIO.output(ENA_PIN, GPIO.LOW) # 모터 활성화

    while not program_exit:
        # 실시간으로 방향 변수(motor_dir)를 적용
        GPIO.output(DIR_PIN, motor_dir)

        if motor_running:
            GPIO.output(PUL_PIN, GPIO.HIGH)
            time.sleep(STEP_DELAY)
            GPIO.output(PUL_PIN, GPIO.LOW)
            time.sleep(STEP_DELAY)
        else:
            time.sleep(0.1)

def main():
    global motor_running, program_exit, motor_dir

    GPIO.setmode(GPIO.BOARD)
    GPIO.setup([PUL_PIN, DIR_PIN, ENA_PIN], GPIO.OUT, initial=GPIO.LOW)

    print("프로그램 시작")
    print("'1' 입력: 정지")
    print("'2' 입력: 다시 회전")
    print("'3' 입력: 방향 전환 (역방향/정방향)") # 기능 추가됨
    print("'q' 입력: 종료")

    t = threading.Thread(target=motor_task)
    t.start()

    try:
        while True:
            user_input = input("\n명령 입력 (1:정지, 2:재시작, 3:방향전환, q:종료): ")

            if user_input == '1':
                motor_running = False
                print(">>> 모터 정지")

            elif user_input == '2':
                motor_running = True
                print(">>> 모터 재시작")

            elif user_input == '3':
                # 방향 토글 (LOW <-> HIGH)
                if motor_dir == GPIO.LOW:
                    motor_dir = GPIO.HIGH
                    print(">>> 방향 전환: 역방향 (HIGH)")
                else:
                    motor_dir = GPIO.LOW
                    print(">>> 방향 전환: 정방향 (LOW)")

            elif user_input == 'q':
                print("종료합니다...")
                program_exit = True
                break
            
    except KeyboardInterrupt:
        program_exit = True

    finally:
        t.join()
        GPIO.cleanup()
        print("GPIO 정리 완료.")

if __name__ == '__main__':
    main()
