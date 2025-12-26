#!/usr/bin/env python3
"""Jetank1 role: request Top-CCTV1 pose and move to pick/place."""

# Jetank1은 top_cctv1 AI 좌표를 받아 젠가를 픽업하고
# 컨베이어로 옮기는 동작을 수행한다.


import argparse
import os
import sys
import threading
import time
from typing import Optional, Tuple

import rclpy
from rclpy.parameter import Parameter
from std_srvs.srv import Trigger
from top_cctv_interfaces.srv import GetClosestPose

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from jetank_all_in_one_ai import JetankController  # noqa: E402
from role_ai_utils import (  # noqa: E402
    build_mapper,
    pose_to_command,
    request_ai_pose,
)


DEFAULT_PX_POINTS_JETANK1 = "44,20;490,20;44,455;490,455"
DEFAULT_WORLD_POINTS_JETANK1 = "-136.8,97.9;-136.8,382.9;143.2,97.9;143.2,382.9"


def _parse_xyzr(text: str) -> Tuple[float, float, float]:
    """'x y roll' 문자열을 (x, y, roll)로 파싱."""
    parts = text.replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError("must be 'x y roll'")
    return float(parts[0]), float(parts[1]), float(parts[2])


def _parse_pose4(text: str) -> Tuple[float, float, float, float]:
    """'x y z roll' 문자열을 (x, y, z, roll)로 파싱."""
    parts = text.replace(",", " ").split()
    if len(parts) != 4:
        raise ValueError("must be 'x y z roll'")
    return float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])


def _parse_pose3(text: str) -> Tuple[float, float, float]:
    """'x y z' 문자열을 (x, y, z)로 파싱."""
    parts = text.replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError("must be 'x y z'")
    return float(parts[0]), float(parts[1]), float(parts[2])


class Jetank1Role(JetankController):
    """Jetank1 픽업/드롭 시퀀스를 수행하는 역할 노드."""
    def __init__(self, args: argparse.Namespace):
        """AI 매퍼/서비스 및 시작 트리거를 초기화."""
        super().__init__("jetank1", enable_tf_bridge=True)
        self.args = args
        # 실기/시뮬 시간 선택
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, self.args.use_sim_time)])
        self._busy = False
        self._lock = threading.Lock()

        # Top-CCTV1 서비스 클라이언트 및 매퍼 구성
        self.ai_client = self.create_client(GetClosestPose, self.args.ai_service)

        self.mapper = build_mapper(
            map_mode=self.args.map_mode,
            px_points_text=self.args.px_points,
            world_points_text=self.args.world_points,
            mm_per_px_x=self.args.mm_per_px_x,
            mm_per_px_y=self.args.mm_per_px_y,
            px_origin=self.args.px_origin,
            world_origin=self.args.world_origin,
            swap_xy=self.args.swap_xy,
            invert_x=self.args.invert_x,
            invert_y=self.args.invert_y,
            logger=self,
        )

        self.start_srv = self.create_service(Trigger, self.args.start_service, self.on_start)
        self.get_logger().info(f"Jetank1 start service: {self.args.start_service}")

    def on_start(self, request, response):
        """컨베이어 ON 후 호출되는 시작 트리거 처리."""
        # 컨베이어에서 전원 ON 후 호출되는 시작 트리거
        with self._lock:
            if self._busy:
                response.success = False
                response.message = "busy"
                return response
            self._busy = True

        thread = threading.Thread(target=self._run_sequence, daemon=True)
        thread.start()
        response.success = True
        response.message = "started"
        return response

    def _run_sequence(self):
        """AI 좌표 요청 → 픽앤플레이스 실행."""
        # AI 좌표 요청 후 픽앤플레이스 실행
        try:
            if self.mapper is None:
                self.get_logger().error("mappx_pointsper not configured; aborting")
                return

            pose = request_ai_pose(
                node=self,
                client=self.ai_client,
                target_class=self.args.ai_target_class,
                timeout_sec=self.args.ai_timeout,
                retries=self.args.ai_retries,
                retry_wait=self.args.ai_retry_wait,
                min_conf=self.args.ai_min_conf,
            )
            cmd = None
            if pose is not None:
                cmd = pose_to_command(
                    pose=pose,
                    mapper=self.mapper,
                    default_roll=self.args.default_roll,
                    use_theta_roll=self.args.use_theta_roll,
                    theta_unit=self.args.theta_unit,
                    roll_scale=self.args.roll_scale,
                    roll_offset=self.args.roll_offset,
                )

            if cmd is None:
                if self.args.fallback_cmd is None:
                    self.get_logger().error("AI pose unavailable; no fallback configured")
                    return
                cmd = _parse_xyzr(self.args.fallback_cmd)
                self.get_logger().warn("AI fallback: using fallback_cmd")

            self._execute_sequence(cmd[0], cmd[1], cmd[2])
        except Exception as exc:
            self.get_logger().error(f"sequence failed: {exc}")
        finally:
            with self._lock:
                self._busy = False

    def _execute_sequence(self, x: float, y: float, roll: float) -> None:
        """Jetank1 실제 픽업/드롭 동작 시퀀스."""
        # Jetank1 기본 픽업 → 드롭 시퀀스
        drop_pose = _parse_pose4(self.args.drop_pose)
        hover_z = self.args.hover_z
        pick_z = self.args.pick_z
        phi = self.args.phi
        move_time = self.args.move_time
        wait_after_move = self.args.post_move_wait

        # 1) 접근
        self.move_to_xyz(x, y, hover_z, phi=phi, roll=0.0, move_time=move_time)
        time.sleep(wait_after_move)
        # 2) 내려가서 집기
        self.move_to_xyz(x, y, pick_z, phi=phi, roll=0.0, move_time=move_time)
        time.sleep(wait_after_move)
        time.sleep(self.args.pre_grab_wait)
        self.control_magnet("ON")
        time.sleep(self.args.post_grab_wait)
        # 3) 들어올리기
        self.move_to_xyz(x, y, hover_z, phi=phi, roll=0.0, move_time=move_time)
        time.sleep(wait_after_move)

        drop_x, drop_y, drop_z, drop_roll = drop_pose
        if drop_roll == 0.0:
            drop_roll = roll

        # 4) 드롭 위치로 이동
        self.move_to_xyz(drop_x, drop_y, hover_z, phi=phi, roll=drop_roll, move_time=move_time)
        time.sleep(wait_after_move)
        # 5) 내려놓기
        self.move_to_xyz(drop_x, drop_y, drop_z, phi=phi, roll=drop_roll, move_time=move_time)
        time.sleep(wait_after_move)
        self.control_magnet("OFF")
        time.sleep(self.args.post_release_wait)
        # 6) 복귀
        self.move_to_xyz(drop_x, drop_y, hover_z, phi=phi, roll=drop_roll, move_time=move_time)
        time.sleep(wait_after_move)

        # 홈 자세로 복귀
        home_x, home_y, home_z = _parse_pose3(self.args.home_pose)
        self.move_to_xyz(home_x, home_y, home_z, phi=phi, roll=0.0, move_time=move_time)
        time.sleep(wait_after_move)


def _build_parser() -> argparse.ArgumentParser:
    """CLI 인자 파서를 구성."""
    parser = argparse.ArgumentParser(description="Jetank1 role (Top-CCTV1 pick/place)")
    parser.add_argument("--start-service", default="/jetank1/start_pick")

    parser.add_argument("--ai-service", default="/top_cctv1/get_closest_pose")
    parser.add_argument("--ai-target-class", type=int, default=-1)
    parser.add_argument("--ai-min-conf", type=float, default=0.5)
    parser.add_argument("--ai-timeout", type=float, default=1.0)
    parser.add_argument("--ai-retries", type=int, default=5)
    parser.add_argument("--ai-retry-wait", type=float, default=0.2)

    parser.add_argument("--use-theta-roll", action="store_true")
    parser.add_argument("--theta-unit", choices=["rad", "deg"], default="rad")
    parser.add_argument("--roll-scale", type=float, default=1.0)
    parser.add_argument("--roll-offset", type=float, default=0.0)
    parser.add_argument("--default-roll", type=float, default=0.0)

    parser.add_argument("--map-mode", choices=["auto", "homography", "scale", "pixel", "none"], default="homography")
    parser.add_argument("--px-points", type=str, default=DEFAULT_PX_POINTS_JETANK1)
    parser.add_argument("--world-points", type=str, default=DEFAULT_WORLD_POINTS_JETANK1)
    parser.add_argument("--mm-per-px-x", type=float, default=None)
    parser.add_argument("--mm-per-px-y", type=float, default=None)
    parser.add_argument("--px-origin", type=str, default=None)
    parser.add_argument("--world-origin", type=str, default="0.5,1.0")
    parser.add_argument("--swap-xy", action="store_true")
    parser.add_argument("--invert-x", action="store_true")
    parser.add_argument("--invert-y", action="store_true")

    parser.add_argument("--hover-z", type=float, default=0.0)
    parser.add_argument("--pick-z", type=float, default=-71.0)
    parser.add_argument("--drop-pose", default="5.0,-150.0,-60.0,0.0")
    parser.add_argument("--phi", type=float, default=-90.0)
    parser.add_argument("--move-time", type=float, default=2.0)
    parser.add_argument("--post-move-wait", type=float, default=2.5)
    parser.add_argument("--pre-grab-wait", type=float, default=2.0)
    parser.add_argument("--post-grab-wait", type=float, default=1.0)
    parser.add_argument("--post-release-wait", type=float, default=1.0)
    parser.add_argument("--home-pose", default="150.0,0.0,50.0")

    parser.add_argument("--fallback-cmd", default=None, help="use when AI fails (x y roll)")
    parser.add_argument("--init-detach", action="store_true", default=True)
    parser.add_argument("--no-init-detach", dest="init_detach", action="store_false")
    parser.add_argument("--init-home", action="store_true", default=True)
    parser.add_argument("--no-init-home", dest="init_home", action="store_false")
    parser.add_argument("--init-wait", type=float, default=2.0)
    parser.add_argument("--use-sim-time", action="store_true", default=False)

    return parser


def main() -> None:
    """Jetank1 역할 노드 실행 진입점."""
    parser = _build_parser()
    args = parser.parse_args()

    rclpy.init()
    node = Jetank1Role(args)
    # 초기 안전 상태: detach + 홈 자세
    if args.init_detach:
        node.detach_all()
        time.sleep(args.init_wait)
    if args.init_home:
        home_x, home_y, home_z = _parse_pose3(args.home_pose)
        node.move_to_xyz(home_x, home_y, home_z, phi=args.phi, roll=0.0, move_time=args.move_time)
        time.sleep(args.init_wait)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
