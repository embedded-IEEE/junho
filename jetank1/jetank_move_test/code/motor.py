import Jetson.GPIO as GPIO
import time
import threading

# 1. 핀 설정
PUL_PIN = 36 
DIR_PIN = 35 
ENA_PIN = 33 

# 2. 속도 설정 (0.0001은 너무 빨라서 모터가 '지잉'거리고 안 돕니다. 0.001로 늦춤)
STEP_DELAY = 0.0001 

# 3. 전역 변수
motor_running = True 
program_exit = False 
motor_dir = GPIO.LOW 

def motor_task():
    global motor_running, program_exit, motor_dir
    print("--> 모터 스레드 시작됨")
    
    # [수정됨] ENA를 LOW로 해야 모터가 잠깁(Enable)니다. HIGH면 힘이 풀립니다.
    GPIO.output(ENA_PIN, GPIO.LOW) 

    while not program_exit:
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
    # 초기값을 HIGH로 두면 시작하자마자 모터가 풀릴 수 있으므로 LOW 권장
    GPIO.setup([PUL_PIN, DIR_PIN, ENA_PIN], GPIO.OUT, initial=GPIO.LOW)

    print("프로그램 시작")
    print("'1' 입력: 정지")
    print("'2' 입력: 다시 회전")
    print("'3' 입력: 방향 전환")
    print("'q' 입력: 종료")

    t = threading.Thread(target=motor_task)
    t.start()

    try:
        while True:
            user_input = input("\n명령 입력 (1, 2, 3, q): ")

            if user_input == '1':
                motor_running = False
                print(">>> 모터 정지")

            elif user_input == '2':
                motor_running = True
                print(">>> 모터 재시작")

            elif user_input == '3':
                if motor_dir == GPIO.LOW:
                    motor_dir = GPIO.HIGH
                    print(">>> 방향 전환: 역방향")
                else:
                    motor_dir = GPIO.LOW
                    print(">>> 방향 전환: 정방향")

            elif user_input == 'q':
                # 프로그램 종료 시 ENA를 풀어주는 것이 안전합니다 (발열 방지)
                GPIO.output(ENA_PIN, GPIO.HIGH)
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
