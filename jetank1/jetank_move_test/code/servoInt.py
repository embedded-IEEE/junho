#!/usr/bin/env python3

import time
import sys
import numpy as np
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
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
    sys.path.append(os.path.join(parent_dir, 'SCSCtrl')) 
    from SCSCtrl.scservo_sdk import *
except ImportError:
    print("Error: SCSCtrl 라이브러리를 찾을 수 없습니다. 경로를 확인해주세요.")
    # 실제 하드웨어 연결 시 주석 해제 권장
    # sys.exit(1)

# --- ROS 2 상태 퍼블리셔 노드 클래스 (NEW) ---
class RobotStateNode(Node):
    def __init__(self):
        # 노드 이름: jetank_state_publisher
        super().__init__('jetank_state_publisher')
        # 토픽 이름: /robot_state, 큐 사이즈: 10
        self.publisher_ = self.create_publisher(String, '/robot_state', 10)
        self.get_logger().info('Robot State Publisher Node Started')

    def publish_state(self, state_text):
        """String 메시지로 현재 상태 발행"""
        msg = String()
        msg.data = state_text
        self.publisher_.publish(msg)
        self.get_logger().info(f'State Published: "{state_text}"')


# --- 설정 상수 (Configuration) ---
class Config:
    # Serial Port Settings
    DEVICE_NAME = '/dev/ttyTHS1'  # Jetson Nano UART Port : 디바이스 파일 
    BAUDRATE = 1000000
    
    # Robot Arm Dimensions (mm)
    LINK_A_LEN = 90 # 1번 관절 ~ 2번 관절까지 팔 길이
    LINK_B_LEN = 160 # 2 ~ 그리퍼 길이 
    
    # Servo Settings
    SERVO_IDS = [1, 2, 3, 4, 5]
    
    # 초기 위치 (ID: Position) - 기존 코드의 512 값 유지
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

# --- 로봇 팔 제어 (Legacy Logic Ported) ---
class RobotArm:
    def __init__(self, servo_manager):
        self.servo = servo_manager
        self.link_a = Config.LINK_A_LEN
        self.link_b = Config.LINK_B_LEN
        
        # [중요] 기존 코드의 servoDirection = [1, -1] 로직을 반영하기 위한 상수
        self.dir_correction_ik = [1, -1]  

    def limit_check(self, posInput, circlePos, circleLen, outline):
        """기존 코드의 limitCheck 함수 포팅"""
        circleRx = posInput[0]-circlePos[0]
        circleRy = posInput[1]-circlePos[1]
        realPosSquare = circleRx*circleRx+circleRy*circleRy
        shortRadiusSquare = np.square(circleLen[1]-circleLen[0])
        longRadiusSquare = np.square(circleLen[1]+circleLen[0])

        if realPosSquare >= shortRadiusSquare and realPosSquare <= longRadiusSquare:
            return posInput[0], posInput[1]
        else:
            # 범위 밖일 경우 보정 로직 (복잡한 수학식 그대로 유지)
            lineK = (posInput[1]-circlePos[1])/(posInput[0]-circlePos[0]) if (posInput[0]-circlePos[0]) != 0 else 0
            lineB = circlePos[1]-(lineK*circlePos[0])
            
            if realPosSquare < shortRadiusSquare:
                targetRadiusSq = shortRadiusSquare
                offset_sign = -1 # 내부 로직 단순화
            else:
                targetRadiusSq = longRadiusSquare
                offset_sign = 1

            aX = 1 + lineK*lineK
            bX = 2*lineK*(lineB - circlePos[1]) - 2*circlePos[0]
            cX = circlePos[0]*circlePos[0] + (lineB - circlePos[1])*(lineB - circlePos[1]) - targetRadiusSq

            resultX = bX*bX - 4*aX*cX
            if resultX < 0: resultX = 0 # 안전장치
            
            x1 = (-bX + np.sqrt(resultX))/(2*aX)
            x2 = (-bX - np.sqrt(resultX))/(2*aX)
            y1 = lineK*x1 + lineB
            y2 = lineK*x2 + lineB
            
            return x1, y1 # 근사치 리턴

    def solve_ik_legacy(self, x, y):
        """
        [핵심] 기존 코드의 planeLinkageReverse 로직을 그대로 사용
        """
        # 1. Y축 반전 (기존 코드 xyInputSmooth에서 -yInput을 넘기는 것 반영)
        goalPos = [float(x), -float(y)] 
        
        linkageLen = [self.link_a, self.link_b]
        
        # 실제 거리 계산
        linkageLenREAL = np.sqrt(goalPos[0]**2 + goalPos[1]**2)

        # limitCheck (간소화 호출)
        goalPos[0], goalPos[1] = self.limit_check(goalPos, [0,0], [linkageLen[0], linkageLenREAL], 0.00001)

        # --- 기존 코드의 핵심 분기 로직 ---
        angleGenA = 0
        angleGenB = 0

        if goalPos[0] < 0:
            goalPos[0] = -goalPos[0]
            mGenOut = linkageLenREAL**2 - linkageLen[0]**2 - goalPos[0]**2 - goalPos[1]**2
            nGenOut = mGenOut/(2*linkageLen[0])
            
            # 수치해석 안전장치
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

        # 결과 리턴 (기존 코드의 방향 보정 적용)
        # servoDirection = [1, -1] 적용
        return angleGenA * 1, angleGenB * -1

    def move_to_xy_smooth(self, x, y, duration):
        """xyInputSmooth 로직의 Class 버전"""
        
        # 1. 역기구학 계산 (Legacy 로직 사용)
        deg_a, deg_b = self.solve_ik_legacy(x, y)
        
        # 2. 오프셋 및 방향 적용 (xyInputSmooth 로직 복제)
        target_deg_2 = deg_a + 90
        target_deg_3 = deg_b
        
        # 3. 목표값 계산 (int 변환 대신 round 사용으로 정밀도 향상)
        goal_pos_2 = Config.SERVO_INIT_POS[2] + int(round((Config.INPUT_RANGE / 180.0) * target_deg_2 * 1))
        goal_pos_3 = Config.SERVO_INIT_POS[3] + int(round((Config.INPUT_RANGE / 180.0) * target_deg_3 * -1))

        # 4. 현재 위치 읽기 및 속도 계산
        curr_pos_2 = self.servo.read_position(2)
        curr_pos_3 = self.servo.read_position(3)
        if curr_pos_2 == -1: curr_pos_2 = goal_pos_2
        if curr_pos_3 == -1: curr_pos_3 = goal_pos_3

        calc_speed_2 = int(abs(goal_pos_2 - curr_pos_2) / duration) if duration > 0 else 0
        calc_speed_3 = int(abs(goal_pos_3 - curr_pos_3) / duration) if duration > 0 else 0

        # 최소 속도 보정
        if calc_speed_2 < 10: calc_speed_2 = 20
        if calc_speed_3 < 10: calc_speed_3 = 20

        # 5. 실행
        self.servo.write_speed(2, calc_speed_2)
        self.servo.write_speed(3, calc_speed_3)
        self.servo.sync_write_pos([2, 3], [goal_pos_2, goal_pos_3])

    def move_servo_raw(self, servo_id, angle, speed, direction=1):
        """초기화를 위한 단일 서보 이동"""
        offset = Config.SERVO_INIT_POS[servo_id]
        goal_pos = offset + int((Config.INPUT_RANGE / Config.ANGLE_RANGE) * angle * direction)
        self.servo.write_speed(servo_id, speed)
        self.servo.sync_write_pos([servo_id], [goal_pos])

# --- 메인 실행 ---
def main():
    manager = None
    jetbot = None
    ros_node = None
    
    # 1. ROS 2 초기화
    rclpy.init()
    ros_node = RobotStateNode()
    
    try:
        ros_node.publish_state("System Initializing")
        
        # 2. 하드웨어 초기화
        manager = SCSServoManager(Config.DEVICE_NAME, Config.BAUDRATE)
        arm = RobotArm(manager)
        
        if Robot:
            jetbot = Robot()

        print("=== Robot Control Started with ROS 2 Publisher ===")
        ros_node.publish_state("Hardware Ready")
        
        # 3. 서보 초기화 동작
        ros_node.publish_state("Servo Homing")
        for sid in Config.SERVO_IDS:
            arm.move_servo_raw(sid, 0, 150)
            time.sleep(0.1)
        time.sleep(2)

        # 4. 이동 시퀀스 테스트
        print(">> Move 1: (100, 0)")
        ros_node.publish_state("Arm Moving: Pos (100, 0)")
        arm.move_to_xy_smooth(100, 0, 1.0)
        time.sleep(2.0)

        print(">> Move 2: (200, 0)")
        ros_node.publish_state("Arm Moving: Pos (200, 0)")
        arm.move_to_xy_smooth(200, 0, 1.0)
        time.sleep(2.0)
        
        print(">> Move 3: (100, 0) Slow")
        ros_node.publish_state("Arm Moving: Pos (100, 0) Slow")
        arm.move_to_xy_smooth(100, 0, 2.0)
        time.sleep(4)

        # 5. Jetbot 주행 테스트
        if jetbot:
            print(">> Jetbot Forward")
            ros_node.publish_state("Jetbot Driving Forward")
            jetbot.forward(0.4)
            time.sleep(0.5)
            
            jetbot.stop()
            print(">> Jetbot Stopped")
            ros_node.publish_state("Jetbot Stopped")

        ros_node.publish_state("Task Completed")

    except Exception as e:
        print(f"Runtime Error: {e}")
        if ros_node:
            ros_node.publish_state(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
    except KeyboardInterrupt:
        print("\nForce Stopped by User")
        if ros_node:
            ros_node.publish_state("Force Stopped")
        
    finally:
        if manager:
            manager.close_port()
        if jetbot:
            jetbot.stop()
            
        # ROS 2 종료 처리
        if ros_node:
            ros_node.destroy_node()
        rclpy.shutdown()
        print("=== Resource Released & ROS Shutdown ===")

if __name__ == "__main__":
    main()
