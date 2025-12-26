#!/usr/bin/env python3
"""Top CCTV 1 inference role (service server)."""

# top_cctv1 카메라 이미지로 AI 추론을 수행하고
# /top_cctv1/get_closest_pose 서비스를 제공하는 역할.
# 사용 예:

import rclpy

from top_cctv_infer.infer_service_server import InferServiceServer


def main() -> None:
    """top_cctv1 추론 서비스 노드를 실행."""
    rclpy.init()
    # top_cctv1 전용 서비스 노드 실행
    node = InferServiceServer(
        node_name="top_cctv1_infer_service",
        image_topic="/jetank/top_cctv1",
        service_name="/top_cctv1/get_closest_pose",
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
