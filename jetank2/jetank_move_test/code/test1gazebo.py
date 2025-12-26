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
    SERVO_INIT_POS = {1: 510, 2: 545, 3: 524, 4: 512, 5: 561}
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
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
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
        
        # [핵심 수정] Static TF Broadcaster 생성 (world <-> empty_world 연결용)
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.publish_world_bridge()

        self.current_attached_id = None
        self.MAGNET_FRAME = 'jetank1/MAGNETIC_BAR_1' 
        self.WORLD_FRAME = 'world' # 이제 표준 world 프레임을 씁니다.

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

    # [추가된 함수] world와 empty_world를 이어주는 다리 놓기
    def publish_world_bridge(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'         # ROS 표준 (로봇이 있는 곳)
        t.child_frame_id = 'empty_world'    # Gazebo (젠가가 있는 곳)
        
        # 두 좌표계는 같다고 가정 (0,0,0)
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self.tf_static_broadcaster.sendTransform(t)
        print(">> [TF Bridge] Linked 'world' <-> 'empty_world' to fix tree error.")

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
        min_dist = float('inf')
        closest_id = None
        
        # 기준 프레임 정의
        base_frame = self.MAGNET_FRAME 
        world_frame = 'empty_world'

        # [중요] 타임아웃 설정을 위한 Duration 객체 생성 (0.5초 대기)
        from rclpy.duration import Duration
        tf_timeout = Duration(seconds=0.5)

        # 1. TF 버퍼 채우기 (2.0초 대기)
        duration = 2.0
        end_time = time.time() + duration
        print(f"\n>> [TF] Gathering TF data for {duration} seconds...")
        
        while time.time() < end_time:
            rclpy.spin_once(self, timeout_sec=0.1)
            # time.sleep 제거 -> spin_once가 충분히 대기함

        print(f">> [TF] Calculating Coordinates & Distances...")
        print("=" * 100)
        print(f"{'Target':<10} | {'World Coord':<30} | {'Dist from Magnet':<20} | {'Note'}")
        print("-" * 100)

        for i in range(1, 11):
            target_frame = f'jenga{i}'
            world_pose_str = "Unknown"
            dist_str = "Fail"
            note = ""

            # (A) World 좌표 확인 (디버깅용)
            try:
                # timeout 옵션 추가: 데이터가 없으면 0.5초까지 기다림
                t_world = self.tf_buffer.lookup_transform(
                    world_frame,
                    target_frame,
                    rclpy.time.Time(),
                    timeout=tf_timeout
                )
                wx = t_world.transform.translation.x
                wy = t_world.transform.translation.y
                wz = t_world.transform.translation.z
                world_pose_str = f"({wx:.2f}, {wy:.2f}, {wz:.2f})"
            except TransformException:
                # World 좌표조차 모르면 젠가가 스폰되지 않았거나 이름이 틀린 것
                pass

            # (B) 상대 거리 계산 (핵심)
            try:
                # timeout 옵션 추가: Magnet 변환 정보가 올 때까지 기다림
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
                dist_str = f"{dist_val:.4f} m"

                if dist_val < min_dist:
                    min_dist = dist_val
                    closest_id = i
            
            except TransformException as ex:
                # 에러 원인을 note에 간략히 저장
                note = str(ex).split('.')[0] # 메시지가 너무 기니까 첫 문장만

            # 출력
            print(f"📦 {target_frame:<7} | {world_pose_str:<30} | {dist_str:<20} | {note}")

        print("=" * 100)
        
        if closest_id is not None and min_dist <= threshold:
            print(f">> [TF] ✅ Selected: jenga{closest_id} (Closest, Dist: {min_dist:.4f}m)")
            return closest_id
        else:
            print(f">> [TF] ❌ None found within {threshold}m (Min dist: {min_dist:.4f}m)")
            return None

    def detach_all(self):
        print(">> [Init] Detaching ALL jengas (1~10)...")
        msg = Empty()
        for i in range(1, 11):
            if i in self.jenga_pubs:
                self.jenga_pubs[i]['detach'].publish(msg)
        self.current_attached_id = None
        print(">> [Init] Complete.")

    def control_magnet(self, command, target_id=None):
        msg = Empty()
        if command == "ON":
            if target_id is None:
                print(">> [Magnet] Scanning for nearest jenga...")
                found_id = self.find_closest_jenga(threshold=0.15) 
                if found_id:
                    target_id = found_id
                else:
                    print(">> [Magnet] FAILED: No jenga nearby to attach.")
                    return 

            if target_id in self.jenga_pubs:
                self.jenga_pubs[target_id]['attach'].publish(msg)
                self.current_attached_id = target_id 
                print(f">> [ROS] 🧲 Attached jenga{target_id} (Topic: /jenga{target_id}/attach)")
            
            if IS_REAL_ROBOT and self.magnet:
                return self.magnet.grab()

        elif command == "OFF":
            target_detach = target_id if target_id is not None else self.current_attached_id
            if target_detach is None:
                print(">> [Magnet] Unknown target. Detaching ALL for safety.")
                self.detach_all()
                return

            if target_detach in self.jenga_pubs:
                self.jenga_pubs[target_detach]['detach'].publish(msg)
                print(f">> [ROS] 👋 Detached jenga{target_detach}")
                
            if target_detach == self.current_attached_id:
                self.current_attached_id = None

            if IS_REAL_ROBOT and self.magnet:
                return self.magnet.release()

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
        self.publish_gazebo_command(target_angles, move_time)
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
        print(f"[REAL] Goals: {goals}")

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
        robot.detach_all()
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
                user_input = input("\nCommand (x y z | on/off) >> ").strip().lower()
                
                # 1. 종료 커맨드
                if user_input in ['q', 'quit', 'exit']:
                    print("Exiting...")
                    break
                
                # 2. 빈 입력 무시
                if not user_input: continue

                # 3. [추가됨] Magnet 제어 커맨드 (숫자 변환 전에 확인)
                if user_input == 'on':
                    robot.control_magnet("ON")
                    continue  # 좌표 이동 로직 건너뛰고 다시 입력 대기
                elif user_input == 'off':
                    robot.control_magnet("OFF")
                    continue

                # 4. 좌표 입력 처리 (숫자가 아니면 여기서 에러 발생 -> except로 이동)
                parts = user_input.replace(',', ' ').split()
                vals = [float(v) for v in parts]
                
                # x, y, z 입력 확인 (최소 3개 값 필요)
                if len(vals) < 3:
                    print("[Error] x, y, z 좌표 3개가 필요합니다.")
                    continue
                
                x, y, z = vals[0], vals[1], vals[2]
                roll = vals[3] if len(vals) >= 4 else 0.0
                
                phi = -90.0
                mv_time = 2.0
                
                print(f">> Moving to: ({x}, {y}, {z}) Roll={roll}")
                robot.move_to_xyz(x, y, z, phi=phi, roll=roll, move_time=mv_time)

            except ValueError:
                # 숫자가 아닌데 on/off/q 도 아닌 경우
                print("[Error] 정확한 좌표(숫자) 또는 명령어(on/off)를 입력해주세요.")
            except Exception as e:
                print(f"[Error] {e}")

    except KeyboardInterrupt:
        pass
    finally:
        robot.close()
        rclpy.shutdown()

if __name__ == "__main__":
    main()