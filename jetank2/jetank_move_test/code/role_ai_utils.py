#!/usr/bin/env python3
"""Shared AI mapping and service helpers for jetank roles."""

# Jetank 역할들이 공통으로 사용하는 AI 추론 요청/좌표 매핑 유틸리티 모음.
# 사용 예:
#   mapper = build_mapper(
#       map_mode="homography",
#       px_points_text="44,20;490,20;44,455;490,455",
#       world_points_text="-136.8,97.9;-136.8,382.9;143.2,97.9;143.2,382.9",
#       mm_per_px_x=None,
#       mm_per_px_y=None,
#       px_origin=None,
#       world_origin="0,0",
#       swap_xy=False,
#       invert_x=False,
#       invert_y=False,
#       logger=node,
#   )
#   pose = request_ai_pose(
#       node=node,
#       client=client,
#       target_class=-1,
#       timeout_sec=1.0,
#       retries=3,
#       retry_wait=0.2,
#       min_conf=0.5,
#   )
#   cmd = pose_to_command(
#       pose=pose,
#       mapper=mapper,
#       default_roll=0.0,
#       use_theta_roll=False,
#       theta_unit="rad",
#       roll_scale=1.0,
#       roll_offset=0.0,
#   )

from dataclasses import dataclass
import time
from typing import List, Optional, Tuple

import numpy as np

from top_cctv_interfaces.srv import GetClosestPose


@dataclass
class AiPose:
    """AI 서비스 응답을 담는 데이터 구조."""
    # AI 서비스 응답을 로컬 객체로 정리
    found: bool
    x: float
    y: float
    theta: float
    conf: float


class BaseMapper:
    """픽셀 → 월드 좌표 매핑 인터페이스."""
    # 픽셀 → 월드 좌표 매핑 인터페이스
    def map_point(self, px: float, py: float) -> Optional[Tuple[float, float]]:
        """픽셀 좌표를 월드 좌표로 변환."""
        raise NotImplementedError


class PixelToWorldMapper(BaseMapper):
    """호모그래피 또는 스케일 기반 좌표 변환기."""
    # 호모그래피 또는 스케일 기반 좌표 변환
    def __init__(
        self,
        homography: Optional[np.ndarray],
        px_origin: Tuple[float, float],
        world_origin: Tuple[float, float],
        mm_per_px: Tuple[float, float],
        swap_xy: bool,
        invert_x: bool,
        invert_y: bool,
    ):
        """매핑에 필요한 파라미터를 저장."""
        self.homography = homography
        self.px_origin = px_origin
        self.world_origin = world_origin
        self.mm_per_px = mm_per_px
        self.swap_xy = swap_xy
        self.invert_x = invert_x
        self.invert_y = invert_y

    def map_point(self, px: float, py: float) -> Optional[Tuple[float, float]]:
        """픽셀 좌표를 월드 좌표로 변환."""
        if self.homography is not None:
            # 호모그래피 행렬 적용
            vec = np.array([px, py, 1.0], dtype=np.float64)
            out = self.homography @ vec
            if abs(out[2]) < 1e-9:
                return None
            return float(out[0] / out[2]), float(out[1] / out[2])

        # 원점/축 변환 포함한 스케일 매핑
        dx = px - self.px_origin[0]
        dy = py - self.px_origin[1]
        if self.swap_xy:
            dx, dy = dy, dx
        if self.invert_x:
            dx = -dx
        if self.invert_y:
            dy = -dy
        x = self.world_origin[0] + dx * self.mm_per_px[0]
        y = self.world_origin[1] + dy * self.mm_per_px[1]
        return x, y


class LinearAxisMapper(BaseMapper):
    """직선상 점 분포일 때 1D 보간 매핑."""
    # 점들이 거의 일직선일 때 1D 보간으로 매핑
    def __init__(self, px_points: List[Tuple[float, float]], world_points: List[Tuple[float, float]]):
        """보간에 사용할 기준점들을 정렬해 저장."""
        px = np.asarray(px_points, dtype=np.float64)
        world = np.asarray(world_points, dtype=np.float64)
        order = np.argsort(px[:, 0])
        self.px_x = px[order, 0]
        self.world_x = world[order, 0]
        self.world_y = world[order, 1]

    def map_point(self, px: float, py: float) -> Optional[Tuple[float, float]]:
        """픽셀 x값을 기준으로 월드 x/y를 보간."""
        x = float(np.interp(px, self.px_x, self.world_x))
        y = float(np.interp(px, self.px_x, self.world_y))
        return x, y


def _parse_pair(text: str, label: str) -> Tuple[float, float]:
    """'x,y' 문자열을 (float, float)로 파싱."""
    # "x,y" 문자열 파싱
    parts = text.replace(",", " ").split()
    if len(parts) != 2:
        raise ValueError(f"{label} must be 'x,y' (got: {text})")
    return float(parts[0]), float(parts[1])


def _parse_points(text: str, label: str) -> List[Tuple[float, float]]:
    """'x1,y1;x2,y2;...' 문자열을 좌표 리스트로 파싱."""
    # "x1,y1;x2,y2;..." 문자열 파싱
    pts = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        pts.append(_parse_pair(item, label))
    return pts


def _is_collinear(points: List[Tuple[float, float]], tol: float = 5e-2) -> bool:
    """점들이 거의 일직선인지 여부를 판단."""
    # 점들이 거의 일직선인지 판별
    if len(points) < 3:
        return True
    pts = np.asarray(points, dtype=np.float64)
    span = pts.max(axis=0) - pts.min(axis=0)
    if span.max() > 0 and (span.min() / span.max()) < tol:
        return True
    pts = pts - pts.mean(axis=0)
    _, s, _ = np.linalg.svd(pts, full_matrices=False)
    if s[0] == 0:
        return True
    return (s[1] / s[0]) < tol


def _compute_homography(
    px_points: List[Tuple[float, float]],
    world_points: List[Tuple[float, float]],
) -> np.ndarray:
    """DLT 방식으로 호모그래피 행렬을 계산."""
    # DLT 방식으로 호모그래피 계산
    if len(px_points) < 4 or len(px_points) != len(world_points):
        raise ValueError("homography requires 4+ matching point pairs")

    a_rows = []
    for (x, y), (X, Y) in zip(px_points, world_points):
        a_rows.append([-x, -y, -1, 0, 0, 0, x * X, y * X, X])
        a_rows.append([0, 0, 0, -x, -y, -1, x * Y, y * Y, Y])
    a = np.asarray(a_rows, dtype=np.float64)
    _, _, vt = np.linalg.svd(a)
    h = vt[-1, :].reshape(3, 3)
    if abs(h[2, 2]) < 1e-9:
        return h
    return h / h[2, 2]


def build_mapper(
    map_mode: str,
    px_points_text: Optional[str],
    world_points_text: Optional[str],
    mm_per_px_x: Optional[float],
    mm_per_px_y: Optional[float],
    px_origin: Optional[str],
    world_origin: Optional[str],
    swap_xy: bool,
    invert_x: bool,
    invert_y: bool,
    logger=None,
) -> Optional[BaseMapper]:
    """입력 파라미터에 따라 적절한 매퍼를 생성."""
    # 매핑 방식/포인트 구성으로 매퍼 생성
    if map_mode == "none":
        return None

    px_points = _parse_points(px_points_text, "px_points") if px_points_text else []
    world_points = _parse_points(world_points_text, "world_points") if world_points_text else []

    homography = None
    if map_mode in ("auto", "homography") and px_points and world_points:
        if _is_collinear(px_points) or _is_collinear(world_points):
            if logger is not None:
                logger.get_logger().warn("[AI] points nearly collinear; using 1D interpolation")
            return LinearAxisMapper(px_points, world_points)
        try:
            homography = _compute_homography(px_points, world_points)
            if logger is not None:
                logger.get_logger().info(f"[AI] homography ready ({len(px_points)} points)")
        except Exception as exc:
            if logger is not None:
                logger.get_logger().error(f"[AI] homography build failed: {exc}")
            homography = None

    if homography is None:
        if map_mode == "homography":
            if logger is not None:
                logger.get_logger().error("[AI] map_mode=homography requires px/world points")
            return None
        if map_mode in ("auto", "scale", "pixel"):
            if map_mode == "pixel":
                mm_per_px = (1.0, 1.0)
            else:
                if mm_per_px_x is None or mm_per_px_y is None:
                    if logger is not None:
                        logger.get_logger().warn("[AI] mm_per_px not set; mapping disabled")
                    return None
                mm_per_px = (mm_per_px_x, mm_per_px_y)
        else:
            return None
    else:
        mm_per_px = (1.0, 1.0)

    px_origin_pair = _parse_pair(px_origin, "px_origin") if px_origin else (0.0, 0.0)
    world_origin_pair = _parse_pair(world_origin, "world_origin") if world_origin else (0.0, 0.0)

    return PixelToWorldMapper(
        homography=homography,
        px_origin=px_origin_pair,
        world_origin=world_origin_pair,
        mm_per_px=mm_per_px,
        swap_xy=swap_xy,
        invert_x=invert_x,
        invert_y=invert_y,
    )


def _wait_future(future, timeout_sec: Optional[float]) -> bool:
    """future 완료를 기다리고 타임아웃을 적용."""
    # 간단한 future 대기 유틸
    if timeout_sec is None:
        while not future.done():
            time.sleep(0.01)
        return True
    deadline = time.monotonic() + float(timeout_sec)
    while time.monotonic() < deadline:
        if future.done():
            return True
        time.sleep(0.01)
    return future.done()


def request_ai_pose(
    node,
    client,
    target_class: int,
    timeout_sec: float,
    retries: int,
    retry_wait: float,
    min_conf: float,
) -> Optional[AiPose]:
    """AI 서비스 호출을 재시도/타임아웃 포함으로 수행."""
    # AI 서비스 요청을 재시도/타임아웃 포함으로 수행
    for attempt in range(1, retries + 1):
        if not client.service_is_ready():
            client.wait_for_service(timeout_sec=0.5)
        req = GetClosestPose.Request()
        req.target_class = int(target_class)
        future = client.call_async(req)
        if not _wait_future(future, timeout_sec):
            node.get_logger().warn(f"[AI] attempt {attempt}: timeout")
            time.sleep(retry_wait)
            continue
        res = future.result()
        if res is None or not res.found:
            node.get_logger().warn(f"[AI] attempt {attempt}: no detection")
            time.sleep(retry_wait)
            continue
        if float(res.conf) < min_conf:
            node.get_logger().warn(
                f"[AI] attempt {attempt}: conf {float(res.conf):.2f} < {min_conf:.2f}"
            )
            time.sleep(retry_wait)
            continue
        return AiPose(
            found=bool(res.found),
            x=float(res.x),
            y=float(res.y),
            theta=float(res.theta),
            conf=float(res.conf),
        )
    return None


def pose_to_command(
    pose: AiPose,
    mapper: BaseMapper,
    default_roll: float,
    use_theta_roll: bool,
    theta_unit: str,
    roll_scale: float,
    roll_offset: float,
) -> Optional[Tuple[float, float, float]]:
    """AI 포즈를 (x, y, roll) 제어 명령으로 변환."""
    # AI 포즈를 (x,y,roll) 명령으로 변환
    mapped = mapper.map_point(pose.x, pose.y)
    if mapped is None:
        return None

    roll = default_roll
    if use_theta_roll:
        # theta를 roll로 변환(단위/스케일/오프셋 적용)
        theta_deg = pose.theta if theta_unit == "deg" else pose.theta * 180.0 / np.pi
        roll = roll_offset + (roll_scale * theta_deg)
    return mapped[0], mapped[1], roll
