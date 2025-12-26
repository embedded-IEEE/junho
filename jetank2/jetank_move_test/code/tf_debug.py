#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException
import math

class TFDebugger(Node):
    def __init__(self):
        super().__init__('tf_debugger')
        
        # [중요] 시뮬레이션 시간 사용 설정
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # 확인하고 싶은 프레임 이름들
        self.WORLD_FRAME = 'empty_world'
        self.ROBOT_FRAME = 'jetank/MAGNETIC_BAR_1'
        # self.TARGET_JENGA는 반복문에서 처리하므로 제거

        # 1초마다 출력
        self.timer = self.create_timer(1.0, self.print_poses)

    def print_poses(self):
        print("\n" + "="*80)
        current_time = self.get_clock().now().nanoseconds / 1e9
        print(f"[Time: {current_time:.2f}s] TF Data Analysis")
        print("-" * 80)

        # ---------------------------------------------------------
        # 1. 로봇 팔 끝 (Magnet) 위치 확인 (World 기준)
        # ---------------------------------------------------------
        rx, ry, rz = None, None, None
        try:
            t_robot = self.tf_buffer.lookup_transform(
                self.WORLD_FRAME,
                self.ROBOT_FRAME,
                rclpy.time.Time()
            )
            rx = t_robot.transform.translation.x
            ry = t_robot.transform.translation.y
            rz = t_robot.transform.translation.z
            print(f"📍 [Robot Magnet] World Position: (X={rx:.4f}, Y={ry:.4f}, Z={rz:.4f})")
        except TransformException as ex:
            print(f"❌ [Robot] Failed to find robot frame '{self.ROBOT_FRAME}': {ex}")

        print("-" * 80)
        print(f"{'Target':<10} | {'World Coord (x,y,z)':<30} | {'Dist to Magnet (m)':<20} | {'Status'}")
        print("-" * 80)

        # ---------------------------------------------------------
        # 2. Jenga 1 ~ 10 반복 측정
        # ---------------------------------------------------------
        for i in range(1, 11):
            target_jenga = f"jenga{i}"
            world_str = "Unknown"
            dist_str = "Unknown"
            status = "❌ TF Missing"
            
            # (A) 젠가 절대 좌표 (World 기준) 확인
            try:
                t_jenga = self.tf_buffer.lookup_transform(
                    self.WORLD_FRAME,
                    target_jenga,
                    rclpy.time.Time()
                )
                jx = t_jenga.transform.translation.x
                jy = t_jenga.transform.translation.y
                jz = t_jenga.transform.translation.z
                world_str = f"({jx:.2f}, {jy:.2f}, {jz:.2f})"
                status = "⚠️ No Robot TF" # 로봇 좌표가 없으면 거리 계산 불가하므로 일단 경고
            except TransformException:
                pass # world_str은 이미 Unknown

            # (B) 로봇(Magnet) <-> 젠가 거리 계산
            if rx is not None and world_str != "Unknown":
                try:
                    t_rel = self.tf_buffer.lookup_transform(
                        self.ROBOT_FRAME,
                        target_jenga,
                        rclpy.time.Time()
                    )
                    dx = t_rel.transform.translation.x
                    dy = t_rel.transform.translation.y
                    dz = t_rel.transform.translation.z
                    dist = math.sqrt(dx**2 + dy**2 + dz**2)
                    dist_str = f"{dist:.4f} m"
                    
                    # 15cm 이내면 잡을 수 있는 거리라고 표시 (예시)
                    if dist < 0.15:
                        status = "✅ Catchable"
                    else:
                        status = "👀 Visible"
                except TransformException:
                    dist_str = "Calc Error"

            # 한 줄 출력
            print(f"{target_jenga:<10} | {world_str:<30} | {dist_str:<20} | {status}")

def main():
    rclpy.init()
    node = TFDebugger()
    try:
        print(">> TF Debugger Started (Monitoring jenga1 ~ jenga10)...")
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()