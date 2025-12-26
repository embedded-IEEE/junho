#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Jetank1 role (service-driven, fixed targets):
- /jetank1/conveyor_on_event (Trigger) 서비스 서버 제공
- 서비스 신호를 받으면 "고정 좌표 리스트"에서 순서대로 1개를 선택해 이동/집기/드롭 수행
- 드롭(내려놓기) 위치는 고정: (20, -150, -65)
- 나머지(초기 detach/home, busy-lock, move_to_xyz, 자석 제어 등) 기능은 그대로 유지

고정 픽 좌표(순환):
  1) (50, 100, -69)
  2) (100, 130, -60)
  3) (80, 180, -58)
  4) (0, 180, -65)

고정 드롭 좌표:
  (20, -150, -65)
"""

import os
import sys
import threading
import time
from typing import Tuple, List

from role_dds_config import ensure_cyclonedds_env

ensure_cyclonedds_env()

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.parameter import Parameter
from std_srvs.srv import Trigger

# (유지) Top-CCTV AI 서비스 인터페이스 (코드/의존성 유지용)
from top_cctv_interfaces.srv import GetClosestPose

# 같은 폴더의 모듈 가져오기
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from role_jetank_controller import JetankController, IS_REAL_ROBOT  # noqa: E402
from role_ai_utils import build_mapper, pose_to_command, request_ai_pose  # noqa: E402

# (유지) Homography 기본값 (지금은 고정좌표 모드라 사용 안 하지만 기능 유지)
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


class Jetank1AiClient(JetankController):
    """
    Jetank1 역할:
    - /jetank1/conveyor_on_event Trigger 서비스 제공
    - 신호 들어오면 고정 좌표(픽)로 이동해서 집고, 고정 좌표(드롭)로 내려놓고, 홈으로 복귀
    """

    # ===== 사용자 요청 고정 좌표 =====
    FIXED_PICK_TARGETS: List[Tuple[float, float, float]] = [
        (0.0, 80.0, -69.0),
        (0.0, 150.0, -71.0),
        (-50.0, 120.0, -70.0),
        (0.0, 180.0, -68.0),  # <-- 여기만 -3 더 내려줌
    ]
    FIXED_DROP: Tuple[float, float, float] = (0.0, -150.0, -65.0)

    def __init__(self):
        super().__init__("jetank1", enable_tf_bridge=True)

        self.cb_group = ReentrantCallbackGroup()

        # 서비스가 연속으로 들어와도 시퀀스가 중복 실행되지 않게 막는다.
        self._busy = False
        self._lock = threading.Lock()

        # 고정 타겟 순환 인덱스
        self._target_idx = 0

        # ===== 파라미터(기존 기능 최대한 유지) =====
        self.declare_parameter("start_service_name", "/jetank1/conveyor_on_event")

        # (유지) AI 관련 파라미터/클라이언트: 고정 좌표 모드에서는 호출하지 않지만, 기능은 남김
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

        # ===== 동작 파라미터 =====
        # hover_z가 0이면 자동으로 (pick_z + hover_offset)로 계산해서 사용 (너무 위로 뜨는 것 방지)
        self.declare_parameter("hover_z", 0.0)
        self.declare_parameter("hover_offset", 30.0)  # pick_z 기준 위로 얼마나 띄울지
        self.declare_parameter("phi", -90.0)
        self.declare_parameter("move_time", 4.0)
        self.declare_parameter("post_move_wait", 2.5)
        self.declare_parameter("pre_grab_wait", 3.0)
        self.declare_parameter("post_grab_wait", 1.0)
        self.declare_parameter("post_release_wait", 1.0)
        self.declare_parameter("home_pose", "150.0,0.0,50.0")

        self.declare_parameter("init_detach", True)
        self.declare_parameter("init_home", True)
        self.declare_parameter("init_wait", 2.0)

        # 실기/시뮬 시간 (그대로 유지)
        use_sim_time = bool(self.get_parameter("use_sim_time").value)
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, use_sim_time)])

        # ===== AI 서비스 클라이언트 (유지) =====
        ai_service = str(self.get_parameter("ai_service_name").value)
        self.ai_client = self.create_client(GetClosestPose, ai_service, callback_group=self.cb_group)

        # ===== mapper 구성 (유지) =====
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
        self.get_logger().info(f"[Jetank1] (kept) AI service client: {ai_service}")
        self.get_logger().info(f"[Jetank1] Fixed targets enabled (round-robin): {len(self.FIXED_PICK_TARGETS)} targets")

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
        mag = getattr(self, "magnet", None)
        self.get_logger().info(
            f"[MagnetHW] request={'ON' if on else 'OFF'} | IS_REAL_ROBOT={IS_REAL_ROBOT} | magnet_obj={'OK' if mag else 'None'}"
        )

        if mag is None:
            self.get_logger().error("[MagnetHW] self.magnet is None -> JetankController에서 전자석 객체가 생성되지 않았음")
            return

        # IS_REAL_ROBOT 여부와 상관없이 'magnet 객체가 있으면' 무조건 시도 (도커에서 platform 감지 틀릴 때 대비)
        try:
            if on:
                mag.grab()
                self.get_logger().info("[MagnetHW] ON (grab) called")
            else:
                mag.release()
                self.get_logger().info("[MagnetHW] OFF (release) called")
        except Exception as e:
            self.get_logger().error(f"[MagnetHW] exception: {e}")
            raise


    # ====== 실제 시퀀스 ======
    def _get_next_fixed_target(self) -> Tuple[float, float, float]:
        with self._lock:
            idx = self._target_idx
            self._target_idx = (self._target_idx + 1) % len(self.FIXED_PICK_TARGETS)
        return self.FIXED_PICK_TARGETS[idx]

    def _run_sequence(self):
        try:
            # ---- 고정 좌표 모드: 신호마다 다음 타겟으로 pick & place ----
            x, y, pick_z = self._get_next_fixed_target()
            drop_x, drop_y, drop_z = self.FIXED_DROP

            roll = float(self.get_parameter("default_roll").value)  # 필요하면 파라미터로 조절
            self.get_logger().info(
                f"[Jetank1] FIXED target: x={x:.2f}, y={y:.2f}, z={pick_z:.2f} -> DROP ({drop_x:.2f},{drop_y:.2f},{drop_z:.2f})"
            )

            self._execute_fixed_sequence(
                pick_x=x,
                pick_y=y,
                pick_z=pick_z,
                pick_roll=roll,
                drop_x=drop_x,
                drop_y=drop_y,
                drop_z=drop_z,
                drop_roll=roll,
            )

        except Exception as e:
            self.get_logger().error(f"sequence failed: {e}")
        finally:
            with self._lock:
                self._busy = False

    def _compute_hover_z(self, pick_z: float) -> float:
        hover_z_param = float(self.get_parameter("hover_z").value)
        if abs(hover_z_param) > 1e-6:
            return hover_z_param
        hover_offset = float(self.get_parameter("hover_offset").value)
        return pick_z + hover_offset

    def _execute_fixed_sequence(
        self,
        pick_x: float,
        pick_y: float,
        pick_z: float,
        pick_roll: float,
        drop_x: float,
        drop_y: float,
        drop_z: float,
        drop_roll: float,
    ) -> None:
        phi = float(self.get_parameter("phi").value)
        move_time = float(self.get_parameter("move_time").value)
        wait_after_move = float(self.get_parameter("post_move_wait").value)

        hover_z_pick = self._compute_hover_z(pick_z)
        hover_z_drop = self._compute_hover_z(drop_z)

        # 1) 접근(픽 상공)
        self.move_to_xyz(pick_x, pick_y, hover_z_pick, phi=phi, roll=0.0, move_time=move_time)
        time.sleep(wait_after_move)
        time.sleep(0.7)

        # 2) 내려가서 집기
        self.move_to_xyz(pick_x, pick_y, pick_z, phi=phi, roll=0.0, move_time=move_time)
        time.sleep(wait_after_move)
        time.sleep(0.7)

        time.sleep(float(self.get_parameter("pre_grab_wait").value))
        self.magnet_hw_only(True)
        time.sleep(float(self.get_parameter("post_grab_wait").value))

        # 3) 들어올리기
        self.move_to_xyz(pick_x, pick_y, hover_z_pick, phi=phi, roll=0.0, move_time=move_time)
        time.sleep(wait_after_move)
        time.sleep(0.7)

        # 4) 드롭 상공으로 이동
        self.move_to_xyz(drop_x, drop_y, hover_z_drop, phi=phi, roll=drop_roll, move_time=move_time)
        time.sleep(wait_after_move)
        time.sleep(0.7)

        # 5) 내려놓기
        self.move_to_xyz(drop_x, drop_y, drop_z, phi=phi, roll=drop_roll, move_time=move_time)
        time.sleep(wait_after_move)

        self.magnet_hw_only(False)
        time.sleep(float(self.get_parameter("post_release_wait").value))

        # 6) 드롭 상공으로 복귀
        self.move_to_xyz(drop_x, drop_y, hover_z_drop, phi=phi, roll=drop_roll, move_time=move_time)
        time.sleep(wait_after_move)

        # 7) 홈 자세
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

