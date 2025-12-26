#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage

class GzTfFixer(Node):
    def __init__(self):
        super().__init__('gz_tf_fixer')
        # 1. 불량 데이터(기준 없음)가 들어오는 토픽 구독
        self.sub = self.create_subscription(
            TFMessage, 
            '/gz_tf_raw', 
            self.tf_callback, 
            10
        )
        # 2. 정상 데이터(world 기준)를 내보낼 토픽
        self.pub = self.create_publisher(TFMessage, '/tf', 10)
        print(">> TF Fixer Running: Relaying /gz_tf_raw -> /tf with frame_id='world'")

    def tf_callback(self, msg):
        # 들어온 모든 변환 정보(Transform)를 검사
        for t in msg.transforms:
            # 만약 기준(frame_id)이 비어있으면 'world'로 채워줌
            if not t.header.frame_id:
                t.header.frame_id = 'emty_world'
            # (선택) 만약 로봇 이름이 네임스페이스 없이 들어오면 수정 가능
            # if t.child_frame_id == 'jetank':
            #     t.child_frame_id = 'jetank/base_link'
                
        # 수정된 메시지를 진짜 /tf 로 발송
        self.pub.publish(msg)

def main():
    rclpy.init()
    node = GzTfFixer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
