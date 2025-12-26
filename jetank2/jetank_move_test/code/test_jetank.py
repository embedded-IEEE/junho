#!/usr/bin/env python3

import time
import sys
import numpy as np
import os
import platform # 환경 감지용
import math

# --- ROS2 라이브러리 (필수) ---
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from builtin_interfaces.msg import Duration
    from std_msgs.msg import Empty
    from rclpy.time import Time 
    
    # [TF 관련 라이브러리]
    from tf2_ros import TransformException
    from tf2_ros.buffer import Buffer
    from tf2_ros.transform_listener import TransformListener
    # [추가됨] 끊어진 트리를 잇기 위한 Static Broadcaster
    from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
    from geometry_msgs.msg import TransformStamped

except ImportError:
    print("[Error] ROS2 라이브러리를 찾을 수 없습니다. (PC라면 ros-humble-rclpy 등을 확인하세요)")
    sys.exit(1)

# ---------------------------------------------------------
# 1. 환경 감지 및 라이브러리 로드 설정
# ---------------------------------------------------------
# Jetson Nano 등은 보통 aarch64 아키텍처를 사용합니다.
IS_REAL_ROBOT = (platform.machine() == 'aarch64')

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append('/home/jetson/SCSCtrl') 

if IS_REAL_ROBOT:
    print(f"[System] Jetson({platform.machine()}) 감지됨 -> 하드웨어 모드 활성화")
    try:
        from magnet_driver import Electromagnet
        from SCSCtrl.scservo_sdk import *
        IN1, IN2, PULSE_TIME = 37, 38, 0.2
    except ImportError as e:
        print(f"[Error] 하드웨어 라이브러리 로드 실패: {e}")
        IS_REAL_ROBOT = False 
else:
    print(f"[System] PC({platform.machine()}) 감지됨 -> 시뮬레이션(Gazebo) 모드 활성화")
    IN1, IN2, PULSE_TIME = 0, 0, 0.2

# ---------------------------------------------------------
# 2. 설정 값 (공통)
# ---------------------------------------------------------
class Config:
    DEVICE_NAME = '/dev/ttyTHS1'
    BAUDRATE = 1000000
    ID_BASE = 1
    ID_SHOULDER = 2
    ID_ELBOW = 3
    ID_WRIST_ROLL = 4 
    ID_WRIST_PITCH = 5
    LINK_1 = 95.0 
    LINK_2 = 142.0
    LINK_3 = 123.0
    SERVO_INIT_POS = {1: 457, 2: 949, 3: 961, 4: 516, 5: 540}
    INPUT_RANGE = 850      
    ANGLE_RANGE = 180.0
    ADDR_GOAL_POSITION = 42
    ADDR_GOAL_SPEED = 46
    ADDR_PRESENT_POSITION = 56

# ---------------------------------------------------------
# 3. 로봇 제어 클래스 (ROS2 Node 상속)
# ---------------------------------------------------------
class JetankController(Node):
    def __init__(self):
        super().__init__('jetank_controller')
        # 시뮬레이션일 때만 use_sim_time True
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, not IS_REAL_ROBOT)])
        qos_profile = QoSProfile(depth=10)
        
        # 1. Gazebo 팔 제어용 Publisher
        self.traj_pub = self.create_publisher(
            JointTrajectory, 
            '/jetank1/arm_controller/joint_trajectory', 
            qos_profile
        )

        # 2. Magnet Attach/Detach 제어용 Publisher (Jenga 1~10 미리 생성)
        self.jenga_pubs = {}
        for i in range(1, 11):
            self.jenga_pubs[i] = {
                'attach': self.create_publisher(Empty, f'/jetank1/jenga{i}/attach', qos_profile),
                'detach': self.create_publisher(Empty, f'/jetank1/jenga{i}/detach', qos_profile)
            }

        # 3. TF Listener 설정 (거리 계산용)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # [수정] TF Bridge는 시뮬레이션(Gazebo)에서만 필요함
        if not IS_REAL_ROBOT:
            self.tf_static_broadcaster = StaticTransformBroadcaster(self)
            self.publish_world_bridge()

        self.current_attached_id = None
        self.MAGNET_FRAME = 'jetank1/MAGNETIC_BAR_1' 
        self.WORLD_FRAME = 'world'

        self.joint_names = [
            'Revolute_BEARING', 'Revolute_ARM_LOW', 'Revolute_SERVO_UPPER',
            'Revolute_MAGNETIC_BAR', 'Revolute_SERVO_TOP'
        ]

        self.SIM_CORRECTION = {
            Config.ID_BASE:        {'dir': 1,  'offset': 0.0},
            Config.ID_SHOULDER:    {'dir': -1, 'offset': 90.0},
            Config.ID_ELBOW:       {'dir': -1, 'offset': 0.0},
            Config.ID_WRIST_ROLL:  {'dir': 1,  'offset': 0.0},
            Config.ID_WRIST_PITCH: {'dir': 1, 'offset': 90.0}
        }

        self.servo_manager = None
        self.magnet = None
        
        if IS_REAL_ROBOT:
            self.init_hardware()

        self.dirs = {
            Config.ID_BASE: 1, Config.ID_SHOULDER: -1, Config.ID_ELBOW: 1, 
            Config.ID_WRIST_ROLL: 1, Config.ID_WRIST_PITCH: 1 
        }
        self.target_ids = [1, 2, 3, 4, 5]
        self.current_servo_pos = Config.SERVO_INIT_POS.copy()
        
        if IS_REAL_ROBOT and self.servo_manager:
            for sid in self.target_ids:
                pos = self.read_hardware_pos(sid)
                if pos != -1: self.current_servo_pos[sid] = pos

    # [수정] 시뮬레이션 전용 함수
    def publish_world_bridge(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'         
        t.child_frame_id = 'empty_world'    
        
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.w = 1.0

        self.tf_static_broadcaster.sendTransform(t)
        print(">> [Sim Only] Linked 'world' <-> 'empty_world' for Gazebo.")

    def init_hardware(self):
        try:
            self.port_handler = PortHandler(Config.DEVICE_NAME)
            self.packet_handler = PacketHandler(1)
            if self.port_handler.openPort() and self.port_handler.setBaudRate(Config.BAUDRATE):
                print("[Hardware] Serial Port Opened.")
            else:
                print("[Error] Failed to open port!")
            self.group_sync_write_pos = GroupSyncWrite(self.port_handler, self.packet_handler, Config.ADDR_GOAL_POSITION, 2)
            self.group_sync_write_spd = GroupSyncWrite(self.port_handler, self.packet_handler, Config.ADDR_GOAL_SPEED, 2)
            self.magnet = Electromagnet(in1_pin=IN1, in2_pin=IN2, demag_duration=PULSE_TIME)
            self.servo_manager = True 
        except Exception as e:
            print(f"[Error] Hardware Init Failed: {e}")

    def read_hardware_pos(self, servo_id):
        if not IS_REAL_ROBOT: return -1
        pos, res, err = self.packet_handler.read2ByteTxRx(self.port_handler, servo_id, Config.ADDR_PRESENT_POSITION)
        return pos if res == COMM_SUCCESS else -1

    def find_closest_jenga(self, threshold=0.15):
        # [수정] 실물 로봇이면 TF 조회를 하지 않고 종료
        if IS_REAL_ROBOT:
            return None

        min_dist = float('inf')
        closest_id = None
        base_frame = self.MAGNET_FRAME 
        world_frame = 'empty_world'
        from rclpy.duration import Duration
        tf_timeout = Duration(seconds=0.5)

        print(f"\n>> [Sim] Finding closest jenga via TF...")
        
        # TF 버퍼가 찰 때까지 약간 대기 (spin_once 이용)
        start_wait = time.time()
        while time.time() - start_wait < 1.0:
            rclpy.spin_once(self, timeout_sec=0.1)

        print("=" * 60)
        for i in range(1, 11):
            target_frame = f'jenga{i}'
            try:
                t_rel = self.tf_buffer.lookup_transform(
                    base_frame,
                    target_frame,
                    rclpy.time.Time(),
                    timeout=tf_timeout
                )
                dx = t_rel.transform.translation.x
                dy = t_rel.transform.translation.y
                dz = t_rel.transform.translation.z
                dist_val = math.sqrt(dx**2 + dy**2 + dz**2)

                print(f" - {target_frame}: {dist_val:.4f} m")

                if dist_val < min_dist:
                    min_dist = dist_val
                    closest_id = i
            except TransformException:
                pass
        print("=" * 60)
        
        if closest_id is not None and min_dist <= threshold:
            print(f">> [Sim] Target Found: jenga{closest_id} ({min_dist:.3f}m)")
            return closest_id
        else:
            print(f">> [Sim] No jenga found within {threshold}m")
            return None

    def detach_all(self):
        # [수정] 시뮬레이션에서만 동작
        if IS_REAL_ROBOT:
            return 
        print(">> [Sim] Detaching ALL jengas...")
        msg = Empty()
        for i in range(1, 11):
            if i in self.jenga_pubs:
                self.jenga_pubs[i]['detach'].publish(msg)
        self.current_attached_id = None

    # [핵심 수정 부분]
    def control_magnet(self, command, target_id=None):
        # ==========================================
        # 1. 실물 로봇 (Real Robot) 로직
        # ==========================================
        if IS_REAL_ROBOT:
            if self.magnet is None:
                print("[Error] Hardware Magnet not initialized!")
                return

            if command == "ON":
                print(">> [REAL] Magnet ON (Grab)")
                self.magnet.grab()
            elif command == "OFF":
                print(">> [REAL] Magnet OFF (Release)")
                self.magnet.release()
            return  # 실물이면 여기서 함수 종료

        # ==========================================
        # 2. 시뮬레이션 (Gazebo) 로직
        # ==========================================
        msg = Empty()
        if command == "ON":
            if target_id is None:
                print(">> [Sim] Scanning for jenga to attach...")
                found_id = self.find_closest_jenga(threshold=0.15) 
                if found_id:
                    target_id = found_id
                else:
                    print(">> [Sim] FAILED: No jenga nearby.")
                    return 

            if target_id in self.jenga_pubs:
                self.jenga_pubs[target_id]['attach'].publish(msg)
                self.current_attached_id = target_id 
                print(f">> [Sim] Attached jenga{target_id}")
        
        elif command == "OFF":
            target_detach = target_id if target_id is not None else self.current_attached_id
            
            if target_detach is None:
                self.detach_all() # 안전책
                return

            if target_detach in self.jenga_pubs:
                self.jenga_pubs[target_detach]['detach'].publish(msg)
                print(f">> [Sim] Detached jenga{target_detach}")
                
            if target_detach == self.current_attached_id:
                self.current_attached_id = None

    def solve_ik_3dof_planar(self, r, z, phi_deg):
        phi = np.radians(phi_deg)
        w_r = r - Config.LINK_3 * np.cos(phi)
        w_z = z - Config.LINK_3 * np.sin(phi)
        L1, L2 = Config.LINK_1, Config.LINK_2
        if np.sqrt(w_r**2 + w_z**2) > (L1 + L2): return None
        cos_angle = (w_r**2 + w_z**2 - L1**2 - L2**2) / (2 * L1 * L2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        theta2 = np.arccos(cos_angle)
        k1 = L1 + L2 * np.cos(theta2)
        k2 = L2 * np.sin(theta2)
        theta1 = np.arctan2(w_z, w_r) + np.arctan2(k2, k1) 
        theta3 = phi - (theta1 - theta2) 
        return np.degrees(theta1), -np.degrees(theta2), np.degrees(theta3)

    def move_to_xyz(self, x, y, z, phi=-90, roll=0, move_time=1.0):
        rad_base = np.arctan2(y, x)
        deg_base = np.degrees(rad_base)
        r_dist = np.sqrt(x**2 + y**2)
        ik_result = self.solve_ik_3dof_planar(r_dist, z, phi_deg=phi)
        
        if ik_result is None: 
            print(f"Unreachable: {x},{y},{z}")
            return

        deg_shoulder, deg_elbow, deg_wrist_p = ik_result
        target_angles = {
            Config.ID_BASE: deg_base, Config.ID_SHOULDER: deg_shoulder,
            Config.ID_ELBOW: deg_elbow, Config.ID_WRIST_ROLL: roll,
            Config.ID_WRIST_PITCH: deg_wrist_p
        }
        
        # 시뮬레이션 명령 (항상 보냄 - 실물 연결 시에도 RViz 확인용으로 좋음)
        self.publish_gazebo_command(target_angles, move_time)
        
        # 실물 명령 (연결되어 있을 때만)
        if IS_REAL_ROBOT and self.servo_manager:
            self.send_hardware_command(target_angles, move_time)

    def publish_gazebo_command(self, angles_deg, move_time):
        msg = JointTrajectory()
        msg.header.stamp.sec = 0
        msg.header.stamp.nanosec = 0
        msg.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        
        def normalize_angle(angle_deg):
            angle_deg = angle_deg % 360.0 
            if angle_deg > 180.0: angle_deg -= 360.0
            elif angle_deg < -180.0: angle_deg += 360.0
            return angle_deg
            
        def get_sim_rad(srv_id):
            cfg = self.SIM_CORRECTION[srv_id]
            input_deg = angles_deg[srv_id]
            raw_target = (input_deg * cfg['dir']) + cfg['offset']
            final_deg = normalize_angle(raw_target)
            return np.radians(final_deg)

        point.positions = [
            get_sim_rad(Config.ID_BASE), get_sim_rad(Config.ID_SHOULDER),
            get_sim_rad(Config.ID_ELBOW), get_sim_rad(Config.ID_WRIST_ROLL),
            get_sim_rad(Config.ID_WRIST_PITCH)
        ]
        sec = int(move_time)
        nanosec = int((move_time - sec) * 1e9)
        point.time_from_start = Duration(sec=sec, nanosec=nanosec)
        msg.points = [point]
        self.traj_pub.publish(msg)

    def send_hardware_command(self, angles_deg, move_time):
        goals, speeds, delta_pos_list = [], [], []
        for sid in self.target_ids:
            angle = angles_deg[sid]
            direction = self.dirs[sid]
            pos = Config.SERVO_INIT_POS[sid] + int((Config.INPUT_RANGE/180.0) * angle * direction)
            pos = max(0, min(1023, pos))
            goals.append(pos)
            current = self.current_servo_pos.get(sid, Config.SERVO_INIT_POS[sid])
            delta_pos_list.append(abs(pos - current))
            self.current_servo_pos[sid] = pos

        scaling_factor = 1.0 / move_time
        for delta in delta_pos_list:
            calc_speed = int((delta * scaling_factor) * 1.5)
            calc_speed = max(40, min(1000, calc_speed))
            speeds.append(calc_speed)

        for i, sid in enumerate(self.target_ids):
            param_spd = [SCS_LOBYTE(speeds[i]), SCS_HIBYTE(speeds[i])]
            self.group_sync_write_spd.addParam(sid, param_spd)
            param_pos = [SCS_LOBYTE(goals[i]), SCS_HIBYTE(goals[i])]
            self.group_sync_write_pos.addParam(sid, param_pos)

        self.group_sync_write_spd.txPacket()
        self.group_sync_write_spd.clearParam()
        self.group_sync_write_pos.txPacket()
        self.group_sync_write_pos.clearParam()
        # print(f"[REAL] Goals: {goals}")

    def close(self):
        if IS_REAL_ROBOT and hasattr(self, 'port_handler'):
            self.port_handler.closePort()
        self.destroy_node()

def main():
    rclpy.init()
    robot = JetankController()
    try:
        print(">> Robot Ready. Initializing connection...")
        time.sleep(2.0) 
        
        # 시작 시 초기화
        robot.detach_all()
        robot.control_magnet("OFF") 
        
        time.sleep(1.0)
        robot.move_to_xyz(150, 0, 50, phi=-90, roll=0, move_time=2.0)
        time.sleep(3.0)
        
        print("=========================================================")
        print(" [Interactive Pick & Place] ")
        print(" Input: x y roll  (e.g., 150 0 0)")
        print(" Exit:  q")
        print("=========================================================")

        while True:
            try:
                user_input = input("\nCommand (x y roll) >> ").strip().lower()
                if user_input in ['q', 'quit', 'exit']:
                    print("Exiting...")
                    break
                
                if not user_input: continue
                parts = user_input.replace(',', ' ').split()
                vals = [float(v) for v in parts]
                if len(vals) < 2:
                    print("[Error] 최소 x, y 좌표가 필요합니다.")
                    continue
                
                x, y = vals[0], vals[1]
                roll = vals[2] if len(vals) >= 3 else 0.0
                
                hover_z = 20.0
                pick_z = -80.0 
                phi = -90.0
                mv_time = 2.0
                
                print(f">> Sequence Start: ({x}, {y}, {pick_z}) Roll={roll}")

                # 1. 이동
                robot.move_to_xyz(x, y, hover_z, phi=phi, roll=0, move_time=mv_time)
                time.sleep(mv_time + 1.0)
                robot.move_to_xyz(x, y, pick_z, phi=phi, roll=0, move_time=mv_time)
                time.sleep(mv_time + 1.0)
                
                # 2. 잡기 (여기서 실물/시뮬 자동 분기)
                robot.control_magnet("ON")
                time.sleep(1.5) 
                
                # 3. 들어 올리기
                robot.move_to_xyz(x, y, hover_z, phi=phi, roll=0, move_time=mv_time)
                time.sleep(mv_time + 1.0)

                # 4. 놓을 위치로 이동
                drop_x, drop_y = 0, -150
                robot.move_to_xyz(drop_x, drop_y, hover_z, phi=phi, roll=roll, move_time=mv_time)
                time.sleep(mv_time + 1.0)
                robot.move_to_xyz(drop_x, drop_y, -85, phi=phi, roll=roll, move_time=mv_time)
                time.sleep(mv_time + 1.0)
                
                # 5. 놓기
                robot.control_magnet("OFF")
                time.sleep(1.0)
                
                # 6. 복귀
                robot.move_to_xyz(drop_x, drop_y, hover_z, phi=phi, roll=roll, move_time=mv_time)
                time.sleep(mv_time + 1.0)
                robot.move_to_xyz(150, 0, 50, phi=phi, roll=0, move_time=mv_time)
                time.sleep(mv_time + 1.0)

            except ValueError:
                print("[Error] 숫자를 입력해주세요.")
            except Exception as e:
                print(f"[Error] {e}")

    except KeyboardInterrupt:
        pass
    finally:
        robot.close()
        rclpy.shutdown()

if __name__ == "__main__":
    main()