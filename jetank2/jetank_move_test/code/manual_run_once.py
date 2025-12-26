#!/usr/bin/env python3
"""
Manual single-run controller:
  1) Prompt jetank1 x y roll and run sequence
  2) Run conveyor for N seconds (wall time)
  3) Prompt jetank2 x y z roll and move once, then keep position
"""

import sys
import time
from typing import Tuple

import rclpy
from rclpy.executors import MultiThreadedExecutor

from jetank_all_in_one import ConveyorController, JetankController, run_jetank1_sequence


def prompt_xyz_roll(label: str, default: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    default_str = f"{default[0]} {default[1]} {default[2]} {default[3]}"
    while True:
        raw = input(f"{label} 입력 (x y z roll) [default: {default_str}] >> ").strip()
        if not raw:
            return default
        parts = raw.replace(",", " ").split()
        if len(parts) < 4:
            print("[Error] 형식이 올바르지 않습니다. 예: 150 0 50 0")
            continue
        try:
            x, y, z, r = [float(v) for v in parts[:4]]
            return x, y, z, r
        except ValueError:
            print("[Error] 숫자를 입력해주세요.")


def prompt_xy_roll(label: str, default: Tuple[float, float, float]) -> Tuple[float, float, float]:
    default_str = f"{default[0]} {default[1]} {default[2]}"
    while True:
        raw = input(f"{label} 입력 (x y roll) [default: {default_str}] >> ").strip()
        if not raw:
            return default
        parts = raw.replace(",", " ").split()
        if len(parts) < 3:
            print("[Error] 형식이 올바르지 않습니다. 예: 150 0 0")
            continue
        try:
            x, y, r = [float(v) for v in parts[:3]]
            return x, y, r
        except ValueError:
            print("[Error] 숫자를 입력해주세요.")


def prompt_seconds(label: str, default: float) -> float:
    while True:
        raw = input(f"{label} 초 [default: {default:.1f}] >> ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            return -1.0
        if not raw:
            return default
        try:
            val = float(raw)
            if val <= 0.0:
                print("[Error] 0보다 큰 값을 입력해주세요.")
                continue
            return val
        except ValueError:
            print("[Error] 숫자를 입력해주세요.")


def main() -> None:
    rclpy.init()
    executor = MultiThreadedExecutor()

    jetank1 = JetankController("jetank1", enable_tf_bridge=True)
    jetank2 = JetankController("jetank2", enable_tf_bridge=False)
    conveyor = ConveyorController()

    executor.add_node(jetank1)
    executor.add_node(jetank2)
    executor.add_node(conveyor)

    try:
        print(">> Robot Ready. Initializing connection...")
        time.sleep(2.0)
        jetank1.detach_all()
        jetank2.detach_all()
        time.sleep(1.0)

        j2x, j2y, j2z, j2r = 0.0, 149.0, -71.0, 0.0
        jetank2.move_to_xyz(j2x, j2y, j2z, phi=-90.0, roll=j2r, move_time=2.0)
        time.sleep(3.0)
        print(f">> Jetank2 목표 좌표 유지 중: {j2x} {j2y} {j2z} {j2r}")

        j1x, j1y, j1r = prompt_xy_roll("Jetank1", (10.0, 150.0, 0.0))
        run_jetank1_sequence(jetank1, j1x, j1y, roll=j1r)

        print(">> 컨베이어 반복 실행. 종료하려면 q 또는 Ctrl+C")
        while rclpy.ok():
            duration = prompt_seconds("컨베이어 동작 시간", 12.8)
            if duration <= 0.0:
                break
            conveyor.get_logger().info(f"컨베이어 ON (/clock 기준) - {duration:.1f}s")
            conveyor.set_power(True)
            conveyor.wait_sim_seconds(duration)
            conveyor.get_logger().info("컨베이어 OFF (/clock 기준)")
            conveyor.set_power(False)
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        jetank1.close()
        jetank2.close()
        conveyor.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
