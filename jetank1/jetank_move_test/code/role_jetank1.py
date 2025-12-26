#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Jetank1 role (service-driven):
- /jetank1/start_pick (Trigger) 서비스 서버 제공
- 호출되면 /top_cctv1/get_closest_pose (GetClosestPose)로 AI 좌표 요청
- mapper로 (cx,cy,theta) -> (world x,y,roll) 변환
- JetankController로 실제 pick/place 시퀀스 수행
"""

import os
import sys
import threading
import time
from typing import Tuple

from role_dds_config import ensure_cyclonedds_env

ensure_cyclonedds_env()

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.parameter import Parameter
from std_srvs.srv import Trigger

# Top-CCTV AI 노드가 제공하는 서비스의 요청/응답 메시지 구조를 불러온다 
from top_cctv_interfaces.srv import GetClosestPose

# 같은 폴더의 모듈 가져오기
    # JetankController: 실제 팔/전자석/TF/Gazebo/실기 제어
    # role_ai_utils: AI 결과(픽셀/각도)를 월드(x,y,roll)로 바꾸는 도구들
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from role_jetank_controller import JetankController, IS_REAL_ROBOT  # noqa: E402
from role_ai_utils import build_mapper, pose_to_command, request_ai_pose  # noqa: E402

# Homography: 한 평면 위의 점들을 다른 평면으로 투영 변환해서 대응
DEFAULT_PX_POINTS_JETANK1 = "288,430;274,317;155,455;130,329"
DEFAULT_WORLD_POINTS_JETANK1 = "-43.0,160.0;-43.0,100.0;30.0,100.0;30.0,170.0"


def _parse_xyzr(text: str) -> Tuple[float, float, float]:
    parts = text.replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError("must be 'x y roll'")
    return float(parts[0]), float(parts[1]), float(parts[2])


def _parse_pose4(text: str) -> Tuple[float, float, float, float]:
    parts = text.replace(",", " ").split()
    if len(parts) != 4:
        raise ValueError("must be 'x y z roll'")
    return float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])


def _parse_pose3(text: str) -> Tuple[float, float, float]:
    parts = text.replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError("must be 'x y z'")
    return float(parts[0]), float(parts[1]), float(parts[2])


# 공용 컨트롤러 기능을 그대로 쓰면서, "Jetank1" 역할만 추가
class Jetank1AiClient(JetankController):
    """
    Jetank1 역할(실기 구동):
    - Trigger 서비스(/jetank1/start_pick) 제공
    - AI 서비스(/top_cctv1/get_closest_pose) 호출하여 pose 수신
    - mapper로 world 변환
    - move_to_xyz + control_magnet로 pick/place 시퀀스 실행
    """

    def __init__(self):
        super().__init__("jetank1", enable_tf_bridge=True)

        self.cb_group = ReentrantCallbackGroup()
        
        # 서비스가 연속으로 들어와도 시퀀스가 중복 실행되지 않게 막는다.
        self._busy = False
        self._lock = threading.Lock()

        # ===== 파라미터(기존 Jetank1Role argparse 기본값 최대한 동일) =====
        self.declare_parameter("start_service_name", "/jetank1/conveyor_on_event")

        self.declare_parameter("ai_service_name", "/top_cctv1/get_closest_pose")
        self.declare_parameter("ai_target_class", -1)
        self.declare_parameter("ai_min_conf", 0.5)
        self.declare_parameter("ai_timeout", 1.0)
        self.declare_parameter("ai_retries", 5)
        self.declare_parameter("ai_retry_wait", 0.2)

        self.declare_parameter("roll_scale", 1.0)
        self.declare_parameter("use_theta_roll", False)
        self.declare_parameter("theta_unit", "rad")   # rad|deg
        self.declare_parameter("roll_offset", 0.0)
        self.declare_parameter("default_roll", 0.0)


        self.declare_parameter("map_mode", "homography")  # auto|homography|scale|pixel|none
        self.declare_parameter("px_points", DEFAULT_PX_POINTS_JETANK1)
        self.declare_parameter("world_points", DEFAULT_WORLD_POINTS_JETANK1)
        self.declare_parameter("mm_per_px_x", None)
        self.declare_parameter("mm_per_px_y", None)
        self.declare_parameter("px_origin", None)
        self.declare_parameter("world_origin", "0.5,1.0")
        self.declare_parameter("swap_xy", False)
        self.declare_parameter("invert_x", False)
        self.declare_parameter("invert_y", False)

        self.declare_parameter("hover_z", 0.0)
        self.declare_parameter("pick_z", -71.0)
        self.declare_parameter("drop_pose", "0.0,-150.0,-75.0,0.0")  # x,y,z,roll
        self.declare_parameter("phi", -90.0)
        self.declare_parameter("move_time", 4.0)
        self.declare_parameter("post_move_wait", 2.5)
        self.declare_parameter("pre_grab_wait", 3.0)
        self.declare_parameter("post_grab_wait", 1.0)
        self.declare_parameter("post_release_wait", 1.0)
        self.declare_parameter("home_pose", "150.0,0.0,50.0")

        self.declare_parameter("fallback_cmd", None)  # "x y roll"
        self.declare_parameter("init_detach", True)
        self.declare_parameter("init_home", True)
        self.declare_parameter("init_wait", 2.0)
        # self.declare_parameter("use_sim_time", False)
        self.declare_parameter("use_sim_time_override", False)

        # 실기/시뮬 시간
        use_sim_time = bool(self.get_parameter("use_sim_time").value)
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, use_sim_time)])

        # ===== AI 서비스 클라이언트 =====
        # /top_cctv1/get_closest_pose 서비스에 요청해서 "가장 가까운 젠가"의 (cx, cy, theta, conf)를 받는 구조
        ai_service = str(self.get_parameter("ai_service_name").value)
        self.ai_client = self.create_client(GetClosestPose, ai_service, callback_group=self.cb_group)

        # ===== mapper 구성 =====
        # 픽셀 -> 월드 변환기를 만든다 
        self.mapper = build_mapper(
            map_mode=str(self.get_parameter("map_mode").value),
            px_points_text=str(self.get_parameter("px_points").value),
            world_points_text=str(self.get_parameter("world_points").value),
            mm_per_px_x=self.get_parameter("mm_per_px_x").value,
            mm_per_px_y=self.get_parameter("mm_per_px_y").value,
            px_origin=self.get_parameter("px_origin").value,
            world_origin=str(self.get_parameter("world_origin").value),
            swap_xy=bool(self.get_parameter("swap_xy").value),
            invert_x=bool(self.get_parameter("invert_x").value),
            invert_y=bool(self.get_parameter("invert_y").value),
            logger=self,
        )

        # ===== start 서비스 서버 =====
        svc = str(self.get_parameter("start_service_name").value)
        self.conveyor_event_srv = self.create_service(
            Trigger, svc, self.on_conveyor_on_event, callback_group=self.cb_group
        )
        self.get_logger().info(f"[Jetank1] Ready service: {svc}")
        self.get_logger().info(f"[Jetank1] AI service client: {ai_service}")

        # ===== 초기 안전 상태 =====
        if bool(self.get_parameter("init_detach").value):
            self.detach_all()
            time.sleep(float(self.get_parameter("init_wait").value))

        if bool(self.get_parameter("init_home").value):
            home_x, home_y, home_z = _parse_pose3(str(self.get_parameter("home_pose").value))
            self.move_to_xyz(
                home_x,
                home_y,
                home_z,
                phi=float(self.get_parameter("phi").value),
                roll=0.0,
                move_time=float(self.get_parameter("move_time").value),
            )
            time.sleep(float(self.get_parameter("init_wait").value))

    # ====== 서비스 콜백 ======
    def on_conveyor_on_event(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        with self._lock:
            if self._busy:
                response.success = False
                response.message = "busy"
                return response
            self._busy = True

        threading.Thread(target=self._run_sequence, daemon=True).start()
        response.success = True
        response.message = "started"
        return response

    def magnet_hw_only(self, on: bool) -> None:
        """
        실물용: TF/attach 없이 전자석만 제어.
        JetankController는 이미 self.magnet(Electromagnet)를 가지고 있으므로
        가능하면 그걸로 제어한다.
        """
        # 0) 실기인데 magnet 객체가 있으면 그걸로 끝
        if IS_REAL_ROBOT and getattr(self, "magnet", None) is not None:
            try:
                if on:
                    self.magnet.grab()
                    self.get_logger().info("[MagnetHW] ON via Electromagnet.grab()")
                else:
                    self.magnet.release()
                    self.get_logger().info("[MagnetHW] OFF via Electromagnet.release()")
                return
            except Exception as e:
                raise RuntimeError(f"Electromagnet control failed: {e}")

        # 1) (옵션) 시뮬/PC에서는 그냥 로그만 (또는 무시)
        self.get_logger().warn("[MagnetHW] magnet object not available (not real robot?)")


    # ====== 실제 시퀀스 ======

    def _run_sequence(self):
        try:
            if self.mapper is None:
                self.get_logger().error("mapper not configured; aborting")
                return

            pose = request_ai_pose(
                node=self,
                client=self.ai_client,
                target_class=int(self.get_parameter("ai_target_class").value),
                timeout_sec=float(self.get_parameter("ai_timeout").value),
                retries=int(self.get_parameter("ai_retries").value),
                retry_wait=float(self.get_parameter("ai_retry_wait").value),
                min_conf=float(self.get_parameter("ai_min_conf").value),
            )

            cmd = None
            if pose is not None:
                cmd = pose_to_command(
                    pose=pose,
                    mapper=self.mapper,
                    default_roll=float(self.get_parameter("default_roll").value),
                    use_theta_roll=bool(self.get_parameter("use_theta_roll").value),
                    theta_unit=str(self.get_parameter("theta_unit").value),
                    roll_scale=float(self.get_parameter("roll_scale").value),
                    roll_offset=float(self.get_parameter("roll_offset").value),
                )

            if cmd is None:
                fb = self.get_parameter("fallback_cmd").value
                if fb is None:
                    self.get_logger().error("AI pose unavailable; no fallback configured")
                    return
                cmd = _parse_xyzr(str(fb))
                self.get_logger().warn("AI fallback: using fallback_cmd")

            x, y, roll = float(cmd[0]), float(cmd[1]), float(cmd[2])
            self.get_logger().info(f"[Jetank1] execute cmd: x={x:.2f}, y={y:.2f}, roll={roll:.2f}")
            self._execute_sequence(x, y, roll)

        except Exception as e:
            self.get_logger().error(f"sequence failed: {e}")
        finally:
            with self._lock:
                self._busy = False

    def _execute_sequence(self, x: float, y: float, roll: float) -> None:
        drop_pose = _parse_pose4(str(self.get_parameter("drop_pose").value))
        hover_z = float(self.get_parameter("hover_z").value)
        pick_z = float(self.get_parameter("pick_z").value)
        phi = float(self.get_parameter("phi").value)
        move_time = float(self.get_parameter("move_time").value)
        wait_after_move = float(self.get_parameter("post_move_wait").value)

        # 1) 접근
        self.move_to_xyz(x, y, hover_z, phi=phi, roll=0.0, move_time=move_time)
        time.sleep(wait_after_move)
        
        time.sleep(0.7)

        # 2) 내려가서 집기
        self.move_to_xyz(x, y, pick_z, phi=phi, roll=0.0, move_time=move_time)
        time.sleep(wait_after_move)
        
        time.sleep(0.7)

        time.sleep(float(self.get_parameter("pre_grab_wait").value))
        self.magnet_hw_only(True)
        time.sleep(float(self.get_parameter("post_grab_wait").value))

        # 3) 들어올리기
        self.move_to_xyz(x, y, hover_z, phi=phi, roll=0.0, move_time=move_time)
        time.sleep(wait_after_move)
        
        time.sleep(0.7)

        drop_x, drop_y, drop_z, drop_roll = drop_pose
        if drop_roll == 0.0:
            drop_roll = roll

        # 4) 드롭 위치로 이동
        self.move_to_xyz(drop_x, drop_y, hover_z, phi=phi, roll=drop_roll, move_time=move_time)
        time.sleep(wait_after_move)
        
        time.sleep(0.7)

        # 5) 내려놓기
        self.move_to_xyz(drop_x, drop_y, drop_z, phi=phi, roll=drop_roll, move_time=move_time)
        time.sleep(wait_after_move)

        self.magnet_hw_only(False)
        time.sleep(float(self.get_parameter("post_release_wait").value))

        # 6) 복귀
        self.move_to_xyz(drop_x, drop_y, hover_z, phi=phi, roll=drop_roll, move_time=move_time)
        time.sleep(wait_after_move)

        # 홈 자세
        home_x, home_y, home_z = _parse_pose3(str(self.get_parameter("home_pose").value))
        self.move_to_xyz(home_x, home_y, home_z, phi=phi, roll=0.0, move_time=move_time)
        time.sleep(wait_after_move)


def main():
    rclpy.init()
    node = Jetank1AiClient()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
