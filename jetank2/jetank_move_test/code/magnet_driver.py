import Jetson.GPIO as GPIO
import time

class Electromagnet:
    def __init__(self, in1_pin: int, in2_pin: int, demag_duration: float = 0.2):
        """
        초기화 함수
        :param in1_pin: L298N IN1 핀 번호 (BOARD 기준)
        :param in2_pin: L298N IN2 핀 번호 (BOARD 기준)
        :param demag_duration: 역전압(잔류자기 제거) 펄스 지속 시간 (초)
        """
        self.in1 = in1_pin
        self.in2 = in2_pin
        self.demag_duration = demag_duration
        self.is_active = False

        # GPIO 설정
        self._setup_gpio()

    def _setup_gpio(self):
        """GPIO 초기 설정 (내부 호출용)"""
        mode = GPIO.getmode()
        if mode is None:
            GPIO.setmode(GPIO.BOARD)
        
        # 핀 설정 및 초기화 (꺼짐 상태)
        GPIO.setup(self.in1, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.in2, GPIO.OUT, initial=GPIO.LOW)

    def grab(self):
        """
        물체 집기 (자석 ON)
        :return: 성공 여부 및 상태 메시지
        """
        try:
            GPIO.output(self.in1, GPIO.HIGH)
            GPIO.output(self.in2, GPIO.LOW)
            self.is_active = True
            return {"status": "success", "message": "Magnet ON (Grab)"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def release(self):
        """
        물체 놓기 (자석 OFF + 역전압 펄스)
        :return: 성공 여부 및 상태 메시지
        """
        try:
            # 1단계: 일단 끄기
            GPIO.output(self.in1, GPIO.LOW)
            GPIO.output(self.in2, GPIO.LOW)

            # 2단계: 역전압 쏘기 (잔류 자기 제거)
            GPIO.output(self.in1, GPIO.LOW)
            GPIO.output(self.in2, GPIO.HIGH)
            
            # 펄스 시간 대기 (Blocking)
            time.sleep(self.demag_duration)

            # 3단계: 완전 차단
            GPIO.output(self.in1, GPIO.LOW)
            GPIO.output(self.in2, GPIO.LOW)
            
            self.is_active = False
            return {"status": "success", "message": "Magnet OFF (Release with Pulse)"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def set_demag_duration(self, duration: float):
        """역전압 펄스 시간 동적 조절"""
        self.demag_duration = duration
        return {"status": "success", "new_duration": self.demag_duration}

    def cleanup(self):
        """GPIO 정리"""
        GPIO.output(self.in1, GPIO.LOW)
        GPIO.output(self.in2, GPIO.LOW)
        GPIO.cleanup()

    def __del__(self):
        """객체 소멸 시 안전장치"""
        self.cleanup()