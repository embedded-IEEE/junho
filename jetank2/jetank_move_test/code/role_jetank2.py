#!/usr/bin/env python3
"""Jetank2 role: request Top-CCTV2 pose and palletize."""

# Jetank2는 컨베이어에서 젠가를 픽업해 팔레타이징 위치로 옮기고,
# ROI guard 재활성화 및 rc_car 완료 알림을 수행한다.
# 사용 예:
#   python3 src/jetank_move_test/code/role_jetank2.py
#   ros2 service call /jetank2/start_pick std_srvs/srv/Trigger "{}"
#   python3 src/jetank_move_test/code/role_jetank2.py --fallback-cmd "0 149 0"

import argparse
import os
import sys
import threading
import time
from typing import Optional, Tuple

import rclpy
from rclpy.parameter import Parameter
from std_srvs.srv import SetBool, Trigger
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


DEFAULT_PX_POINTS_JETANK2 = "386,177;571,179;385,428;569,425"
DEFAULT_WORLD_POINTS_JETANK2 = (
    "-67.315869,272.923090;-67.315869,414.910919;"
    "212.013740,272.923090;212.013740,414.910919"
)


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


class Jetank2Role(JetankController):
    """Jetank2 픽업/팔레타이징 시퀀스를 수행하는 역할 노드."""
    def __init__(self, args: argparse.Namespace):
        """AI 매퍼/서비스 및 ROI/rc_car 연동을 초기화."""
        super().__init__("jetank2", enable_tf_bridge=True)
        self.args = args
        # 실기/시뮬 시간 선택
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, self.args.use_sim_time)])
        self._busy = False
        self._lock = threading.Lock()

        # Top-CCTV2 서비스 및 ROI/RC 카 클라이언트
        self.ai_client = self.create_client(GetClosestPose, self.args.ai_service)
        self.roi_client = self.create_client(SetBool, self.args.roi_enable_service)
        self.rc_car_client = None
        if self.args.rc_car_done_service:
            self.rc_car_client = self.create_client(Trigger, self.args.rc_car_done_service)

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
        self.get_logger().info(f"Jetank2 start service: {self.args.start_service}")

    def on_start(self, request, response):
        """컨베이어 OFF 후 호출되는 시작 트리거 처리."""
        # 컨베이어 OFF 후 호출되는 시작 트리거
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
        """AI 좌표 요청 → 팔레타이징 시퀀스 실행."""
        # AI 좌표 요청 후 픽업/팔레타이징 실행
        try:
            if self.mapper is None:
                self.get_logger().error("mapper not configured; aborting")
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

    def _enable_roi_guard(self) -> None:
        """프리드롭 도달 시 ROI guard를 재활성화."""
        # Jetank2 목표 구간 도달 시 ROI guard 재활성화
        if not self.args.roi_enable_service:
            return
        if not self.roi_client.service_is_ready():
            self.roi_client.wait_for_service(timeout_sec=0.5)
        req = SetBool.Request()
        req.data = True
        self.roi_client.call_async(req)

    def _notify_rc_car_done(self) -> None:
        """팔레타이징 완료 신호를 rc_car에 전달."""
        # rc_car에 팔레타이징 완료 신호 전달
        if self.rc_car_client is None:
            return
        if not self.rc_car_client.service_is_ready():
            self.rc_car_client.wait_for_service(timeout_sec=0.5)
        req = Trigger.Request()
        self.rc_car_client.call_async(req)

    def _execute_sequence(self, x: float, y: float, roll: float) -> None:
        """Jetank2 실제 픽업/드롭 동작 시퀀스."""
        # Jetank2 픽업 → 프리드롭 → 드롭 → 복귀
        hover_z = self.args.hover_z
        pick_z = self.args.pick_z
        phi = self.args.phi
        move_time = self.args.move_time
        wait_after_move = self.args.post_move_wait

        pre_drop_pose = _parse_pose4(self.args.pre_drop_pose)
        drop_pose = _parse_pose4(self.args.drop_pose)

        # 1) 접근
        self.move_to_xyz(x, y, hover_z, phi=phi, roll=0.0, move_time=move_time)
        time.sleep(wait_after_move)
        # 2) 내려가서 집기
        self.move_to_xyz(x, y, pick_z, phi=phi, roll=0.0, move_time=move_time)
        time.sleep(wait_after_move)
        self.control_magnet("ON")
        time.sleep(self.args.post_grab_wait)
        # 3) 들어올리기
        self.move_to_xyz(x, y, hover_z, phi=phi, roll=0.0, move_time=move_time)
        time.sleep(wait_after_move)

        pre_x, pre_y, pre_z, pre_roll = pre_drop_pose
        # 4) 프리드롭 위치 도달 후 ROI guard 재활성화
        self.move_to_xyz(pre_x, pre_y, pre_z, phi=phi, roll=pre_roll, move_time=move_time)
        time.sleep(wait_after_move)
        self._enable_roi_guard()

        drop_x, drop_y, drop_z, drop_roll = drop_pose
        # 5) 드롭 위치에서 놓기
        self.move_to_xyz(drop_x, drop_y, drop_z, phi=phi, roll=drop_roll, move_time=move_time)
        time.sleep(wait_after_move)
        self.control_magnet("OFF")
        time.sleep(self.args.post_release_wait)
        self._notify_rc_car_done()

        # 6) 복귀
        self.move_to_xyz(drop_x, drop_y, hover_z, phi=phi, roll=drop_roll, move_time=move_time)
        time.sleep(wait_after_move)

        home_x, home_y, home_z = _parse_pose3(self.args.home_pose)
        self.move_to_xyz(home_x, home_y, home_z, phi=phi, roll=0.0, move_time=move_time)
        time.sleep(wait_after_move)


def _build_parser() -> argparse.ArgumentParser:
    """CLI 인자 파서를 구성."""
    parser = argparse.ArgumentParser(description="Jetank2 role (Top-CCTV2 palletize)")
    parser.add_argument("--start-service", default="/jetank2/start_pick")

    parser.add_argument("--ai-service", default="/top_cctv2/get_closest_pose")
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
    parser.add_argument("--px-points", type=str, default=DEFAULT_PX_POINTS_JETANK2)
    parser.add_argument("--world-points", type=str, default=DEFAULT_WORLD_POINTS_JETANK2)
    parser.add_argument("--mm-per-px-x", type=float, default=None)
    parser.add_argument("--mm-per-px-y", type=float, default=None)
    parser.add_argument("--px-origin", type=str, default=None)
    parser.add_argument("--world-origin", type=str, default="0.5,1.0")
    parser.add_argument("--swap-xy", action="store_true")
    parser.add_argument("--invert-x", action="store_true")
    parser.add_argument("--invert-y", action="store_true")

    parser.add_argument("--hover-z", type=float, default=20.0)
    parser.add_argument("--pick-z", type=float, default=-71.0)
    parser.add_argument("--drop-pose", default="0.0,-150.0,-20.0,0.0")
    parser.add_argument("--pre-drop-pose", default="0.0,-150.0,50.0,0.0")
    parser.add_argument("--phi", type=float, default=-90.0)
    parser.add_argument("--move-time", type=float, default=2.0)
    parser.add_argument("--post-move-wait", type=float, default=3.0)
    parser.add_argument("--post-grab-wait", type=float, default=1.0)
    parser.add_argument("--post-release-wait", type=float, default=0.0)
    parser.add_argument("--home-pose", default="150.0,0.0,50.0")

    parser.add_argument("--roi-enable-service", default="/top_cctv2/roi_guard_enable")
    parser.add_argument("--rc-car-done-service", default="/rc_car/palletize_done")

    parser.add_argument("--fallback-cmd", default=None, help="use when AI fails (x y roll)")
    parser.add_argument("--init-detach", action="store_true", default=True)
    parser.add_argument("--no-init-detach", dest="init_detach", action="store_false")
    parser.add_argument("--init-home", action="store_true", default=True)
    parser.add_argument("--no-init-home", dest="init_home", action="store_false")
    parser.add_argument("--init-wait", type=float, default=2.0)
    parser.add_argument("--use-sim-time", action="store_true", default=False)

    return parser


def main() -> None:
    """Jetank2 역할 노드 실행 진입점."""
    parser = _build_parser()
    args = parser.parse_args()

    rclpy.init()
    node = Jetank2Role(args)
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
