#!/usr/bin/env python3

import time
import sys

import numpy as np
import threading
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 시스템 경로에 부모 디렉토리 추가
sys.path.append(parent_dir)

# --- ROS 2 라이브러리 임포트 ---
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
except ImportError:
    print("Error: 'rclpy' 라이브러리가 없습니다. ROS 2 환경을 확인해주세요.")
    sys.exit(1)

# --- Jetbot 및 SCSCtrl 임포트 ---
try:
    from jetbot import Robot
except ImportError:
    print("Warning: 'jetbot' 라이브러리가 없습니다. 주행 기능은 비활성화됩니다.")
    Robot = None

try:
    # SCS 서보 모터 SDK 경로 설정
    sys.path.append('/home/jetson/SCSCtrl') 
    from SCSCtrl.scservo_sdk import *
except ImportError:
    print("Error: SCSCtrl 라이브러리를 찾을 수 없습니다. 경로를 확인해주세요.")
    # 실제 하드웨어 연결 시 주석 해제 권장
    # sys.exit(1)

# --- 설정 상수 (Configuration) ---
class Config:
    # Serial Port Settings
    DEVICE_NAME = '/dev/ttyTHS1'  # Jetson Nano UART Port
    BAUDRATE = 1000000
    
    # Robot Arm Dimensions (mm)
    LINK_A_LEN = 90
    LINK_B_LEN = 160
    
    # Servo Settings
    SERVO_IDS = [1, 2, 3, 4, 5]
    
    # 초기 위치 (ID: Position)
    SERVO_INIT_POS = {1: 512, 2: 512, 3: 512, 4: 512, 5: 512}
    
    INPUT_RANGE = 850    # 서보 입력 범위 상수
    ANGLE_RANGE = 180    # 서보 각도 범위 상수
    
    # Register Addresses (SCS Series)
    ADDR_GOAL_POSITION = 42
    ADDR_GOAL_SPEED = 46
    ADDR_PRESENT_POSITION = 56

# --- 하드웨어 통신 관리자 ---
class SCSServoManager:
    def __init__(self, device_name, baudrate):
        self.port_handler = PortHandler(device_name)
        self.packet_handler = PacketHandler(1) # Protocol end 1
        self.group_sync_write = GroupSyncWrite(self.port_handler, self.packet_handler, Config.ADDR_GOAL_POSITION, 2)
        
        if not self.open_port(baudrate):
            print("WARNING: Failed to open port. Check connection.")

    def open_port(self, baudrate):
        if self.port_handler.openPort():
            print("Succeeded to open the port")
        else:
            print("Failed to open the port")
            return False

        if self.port_handler.setBaudRate(baudrate):
            print("Succeeded to change the baudrate")
            return True
        else:
            print("Failed to change the baudrate")
            return False

    def close_port(self):
        self.port_handler.closePort()

    def read_position(self, servo_id):
        pos_speed, result, error = self.packet_handler.read4ByteTxRx(
            self.port_handler, servo_id, Config.ADDR_PRESENT_POSITION
        )
        if result != COMM_SUCCESS:
            return -1
        return SCS_LOWORD(pos_speed)

    def write_speed(self, servo_id, speed):
        try:
            self.packet_handler.write2ByteTxRx(self.port_handler, servo_id, Config.ADDR_GOAL_SPEED, int(speed))
        except Exception:
            pass 

    def sync_write_pos(self, servo_ids, goals):
        """여러 서보의 위치를 동시에 제어"""
        for i, sid in enumerate(servo_ids):
            param = [SCS_LOBYTE(int(goals[i])), SCS_HIBYTE(int(goals[i]))]
            self.group_sync_write.addParam(sid, param)
        
        self.group_sync_write.txPacket()
        self.group_sync_write.clearParam()

# --- 로봇 팔 제어 ---
class RobotArm:
    def __init__(self, servo_manager):
        self.servo = servo_manager
        self.link_a = Config.LINK_A_LEN
        self.link_b = Config.LINK_B_LEN
        self.dir_correction_ik = [1, -1]  

    def limit_check(self, posInput, circlePos, circleLen, outline):
        circleRx = posInput[0]-circlePos[0]
        circleRy = posInput[1]-circlePos[1]
        realPosSquare = circleRx*circleRx+circleRy*circleRy
        shortRadiusSquare = np.square(circleLen[1]-circleLen[0])
        longRadiusSquare = np.square(circleLen[1]+circleLen[0])

        if realPosSquare >= shortRadiusSquare and realPosSquare <= longRadiusSquare:
            return posInput[0], posInput[1]
        else:
            lineK = (posInput[1]-circlePos[1])/(posInput[0]-circlePos[0]) if (posInput[0]-circlePos[0]) != 0 else 0
            lineB = circlePos[1]-(lineK*circlePos[0])
            
            if realPosSquare < shortRadiusSquare:
                targetRadiusSq = shortRadiusSquare
            else:
                targetRadiusSq = longRadiusSquare

            aX = 1 + lineK*lineK
            bX = 2*lineK*(lineB - circlePos[1]) - 2*circlePos[0]
            cX = circlePos[0]*circlePos[0] + (lineB - circlePos[1])*(lineB - circlePos[1]) - targetRadiusSq

            resultX = bX*bX - 4*aX*cX
            if resultX < 0: resultX = 0 
            
            x1 = (-bX + np.sqrt(resultX))/(2*aX)
            x2 = (-bX - np.sqrt(resultX))/(2*aX)
            y1 = lineK*x1 + lineB
            y2 = lineK*x2 + lineB
            
            return x1, y1

    def solve_ik_legacy(self, x, y):
        goalPos = [float(x), -float(y)] 
        linkageLen = [self.link_a, self.link_b]
        linkageLenREAL = np.sqrt(goalPos[0]**2 + goalPos[1]**2)
        goalPos[0], goalPos[1] = self.limit_check(goalPos, [0,0], [linkageLen[0], linkageLenREAL], 0.00001)

        angleGenA = 0
        angleGenB = 0

        if goalPos[0] < 0:
            goalPos[0] = -goalPos[0]
            mGenOut = linkageLenREAL**2 - linkageLen[0]**2 - goalPos[0]**2 - goalPos[1]**2
            nGenOut = mGenOut/(2*linkageLen[0])
            val = nGenOut/np.sqrt(goalPos[0]**2 + goalPos[1]**2)
            val = np.clip(val, -1, 1)
            angleGenA = np.arctan(goalPos[1]/goalPos[0]) + np.arcsin(val)
            angleGenB = np.arcsin((goalPos[1] - linkageLen[0]*np.cos(angleGenA))/linkageLenREAL) - angleGenA
            angleGenA = 90 - np.degrees(angleGenA)
            angleGenB = np.degrees(angleGenB)

        elif goalPos[0] == 0:
            val_acos = (linkageLen[0]**2 + goalPos[1]**2 - linkageLenREAL**2)/(2*linkageLen[0]*goalPos[1])
            val_acos = np.clip(val_acos, -1, 1)
            angleGenA = np.arccos(val_acos)
            cGenOut = np.tan(angleGenA)*linkageLen[0]
            dGenOut = goalPos[1]-(linkageLen[0]/np.cos(angleGenA)) if np.cos(angleGenA) != 0 else 0
            val_acos2 = (cGenOut**2 + linkageLenREAL**2 - dGenOut**2)/(2*cGenOut*linkageLenREAL)
            val_acos2 = np.clip(val_acos2, -1, 1)
            angleGenB = np.arccos(val_acos2)
            angleGenA = -np.degrees(angleGenA) + 90
            angleGenB = -np.degrees(angleGenB)

        elif goalPos[0] > 0:
            sqrtGenOut = np.sqrt(goalPos[0]**2 + goalPos[1]**2)
            nGenOut = (linkageLen[0]**2 + goalPos[0]**2 + goalPos[1]**2 - linkageLenREAL**2)/(2*linkageLen[0]*sqrtGenOut)
            nGenOut = np.clip(nGenOut, -1, 1)
            angleA = np.degrees(np.arccos(nGenOut))
            AB = goalPos[1]/goalPos[0]
            angleB = np.degrees(np.arctan(AB))
            angleGenA = angleB - angleA
            mGenOut = (linkageLen[0]**2 + linkageLenREAL**2 - goalPos[0]**2 - goalPos[1]**2)/(2*linkageLen[0]*linkageLenREAL)
            mGenOut = np.clip(mGenOut, -1, 1)
            angleGenB = np.degrees(np.arccos(mGenOut)) - 90

        return angleGenA * 1, angleGenB * -1

    def move_to_xy_smooth(self, x, y, duration):
        deg_a, deg_b = self.solve_ik_legacy(x, y)
        target_deg_2 = deg_a + 90
        target_deg_3 = deg_b
        
        goal_pos_2 = Config.SERVO_INIT_POS[2] + int(round((Config.INPUT_RANGE / 180.0) * target_deg_2 * 1))
        goal_pos_3 = Config.SERVO_INIT_POS[3] + int(round((Config.INPUT_RANGE / 180.0) * target_deg_3 * -1))

        curr_pos_2 = self.servo.read_position(2)
        curr_pos_3 = self.servo.read_position(3)
        if curr_pos_2 == -1: curr_pos_2 = goal_pos_2
        if curr_pos_3 == -1: curr_pos_3 = goal_pos_3

        calc_speed_2 = int(abs(goal_pos_2 - curr_pos_2) / duration) if duration > 0 else 0
        calc_speed_3 = int(abs(goal_pos_3 - curr_pos_3) / duration) if duration > 0 else 0

        if calc_speed_2 < 10: calc_speed_2 = 20
        if calc_speed_3 < 10: calc_speed_3 = 20

        self.servo.write_speed(2, calc_speed_2)
        self.servo.write_speed(3, calc_speed_3)
        self.servo.sync_write_pos([2, 3], [goal_pos_2, goal_pos_3])

    def move_servo_raw(self, servo_id, angle, speed, direction=1):
        offset = Config.SERVO_INIT_POS[servo_id]
        goal_pos = offset + int((Config.INPUT_RANGE / Config.ANGLE_RANGE) * angle * direction)
        self.servo.write_speed(servo_id, speed)
        self.servo.sync_write_pos([servo_id], [goal_pos])

    def home_all(self):
        """모든 서보를 초기 위치로"""
        for sid in Config.SERVO_IDS:
            self.move_servo_raw(sid, 0, 150)
            time.sleep(0.05)

# --- ROS 2 Node with Subscriber (핵심 수정 부분) ---
class JetankSubscriberNode(Node):
    def __init__(self, arm_controller, jetbot_controller):
        super().__init__('jetank_subscriber_node')
        
        self.arm = arm_controller
        self.jetbot = jetbot_controller
        
        # 1. Subscriber 생성: '/jetank_cmd' 토픽을 듣습니다.
        self.subscription = self.create_subscription(
            String,
            '/jetank_cmd',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        # 2. Publisher 생성: 상태 보고용
        self.publisher_ = self.create_publisher(String, '/robot_state', 10)
        
        self.get_logger().info('Jetank Node Started. Waiting for commands on /jetank_cmd ...')
        self.publish_state("Ready: Waiting for commands")

    def publish_state(self, state_text):
        msg = String()
        msg.data = state_text
        self.publisher_.publish(msg)
        self.get_logger().info(f'State: "{state_text}"')

    def listener_callback(self, msg):
        """명령어가 들어오면 실행되는 함수"""
        command_str = msg.data.lower().strip()
        self.get_logger().info(f'Received Command: "{command_str}"')
        
        try:
            # --- 명령어 파싱 및 실행 ---
            
            # 1. Home Command (초기화)
            if command_str == 'home':
                self.publish_state("Action: Homing Arm")
                self.arm.home_all()
                self.publish_state("Finished: Homing")

            # 2. Move Command (형식: "move 100 50")
            elif command_str.startswith('move'):
                # 문자열 분리: "move", "100", "50"
                parts = command_str.split()
                if len(parts) >= 3:
                    x = float(parts[1])
                    y = float(parts[2])
                    self.publish_state(f"Action: Moving Arm to ({x}, {y})")
                    self.arm.move_to_xy_smooth(x, y, 1.0) # 1초 동안 이동
                else:
                    self.get_logger().warn("Invalid move command format. Use: 'move x y'")

            # 3. Jetbot Drive Forward
            elif command_str == 'forward':
                if self.jetbot:
                    self.publish_state("Action: Driving Forward")
                    self.jetbot.forward(0.3)
                else:
                    self.get_logger().warn("Jetbot hardware not initialized")

            # 4. Jetbot Drive Stop
            elif command_str == 'stop':
                if self.jetbot:
                    self.publish_state("Action: Stopping Jetbot")
                    self.jetbot.stop()

            # 그 외 알 수 없는 명령
            else:
                self.get_logger().warn(f"Unknown command: {command_str}")

        except Exception as e:
            self.get_logger().error(f"Error executing command: {e}")
            self.publish_state(f"Error: {e}")

# --- 메인 실행 ---
def main(args=None):
    rclpy.init(args=args)
    
    manager = None
    jetbot = None
    node = None

    try:
        # 1. 하드웨어 초기화
        print("Initializing Hardware...")
        manager = SCSServoManager(Config.DEVICE_NAME, Config.BAUDRATE)
        arm = RobotArm(manager)
        
        if Robot:
            jetbot = Robot()
            
        # 초기 서보 정렬
        arm.home_all()
        
        # 2. ROS 노드 생성 및 실행
        node = JetankSubscriberNode(arm, jetbot)
        
        # 3. 스핀 (명령 대기 무한 루프)
        # 이 함수가 실행되면서 Subscriber의 콜백이 동작합니다.
        rclpy.spin(node)

    except KeyboardInterrupt:
        print("\nForce Stopped by User")
    except Exception as e:
        print(f"Runtime Error: {e}")
    finally:
        # 종료 처리
        if manager:
            manager.close_port()
        if jetbot:
            jetbot.stop()
        if node:
            node.destroy_node()
        rclpy.shutdown()
        print("=== Shutdown Complete ===")

if __name__ == "__main__":
    main()