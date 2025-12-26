#!/usr/bin/env python3
import time
import Jetson.GPIO as GPIO

IN1 = 37
IN2 = 38

GPIO.setmode(GPIO.BOARD)   # 너 지금 쓰는 37/38은 보통 BOARD 기준임
GPIO.setup(IN1, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(IN2, GPIO.OUT, initial=GPIO.LOW)

print("ON")
GPIO.output(IN1, GPIO.HIGH)
GPIO.output(IN2, GPIO.HIGH)   # 드라이버에 따라 둘 다 HIGH 필요할 수도 있어서 일단 둘 다 올려봄
time.sleep(2.0)

print("OFF")
GPIO.output(IN1, GPIO.LOW)
GPIO.output(IN2, GPIO.LOW)
time.sleep(1.0)

GPIO.cleanup()
print("DONE")
