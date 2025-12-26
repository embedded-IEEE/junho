#!/usr/bin/env python3
"""
Run a single “action” sequence in order:
  1) jetank1.py (attach/detach)
  2) conveyor_controller_test.py (conveyor on/off)
  3) jetank2.py (attach/place)

Each step blocks until the script finishes, then the next step starts.
The whole sequence repeats 4 times. If any step fails, the run stops.
jetank1/jetank2 get one auto command, then “q” is sent to exit cleanly.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict

BASE_DIR = Path(__file__).resolve().parent

# Default commands per script; can override via environment variables.
JETANK1_DEFAULT_CMD = "10 150 0"
JETANK2_DEFAULT_CMD = "0 149 -71"
JETANK1_BASE_CMD = os.environ.get("JETANK1_CMD", JETANK1_DEFAULT_CMD)
JETANK2_CMD = os.environ.get("JETANK2_CMD", JETANK2_DEFAULT_CMD)
JETANK1_Y_INCREMENT = 20.0  # add to Y each cycle for jetank1

STEPS = [
    ("jetank1.py", "Jetank1 attach/detach"),
    ("conveyor_controller_test.py", "Conveyor belt cycle"),
    ("jetank2.py", "Jetank2 attach/place"),
]
REPEAT_COUNT = 4

# Drop pose sequence for jetank2 (x y z roll)
JETANK2_DROP_SEQUENCE = [
    (0.0, -200.0, -50.0, 0.0),
    (0.0, -160.0, -47.0, 0.0),
    (-10.0, -190.0, -35.0, 90.0),
    (10.0, -185.0, -35.0, 90.0),
]


def parse_command(cmd: str) -> Optional[tuple[float, float, float]]:
    """Parse 'x y roll' string into floats; return None on failure."""
    try:
        parts = cmd.replace(",", " ").split()
        if len(parts) < 3:
            return None
        x, y, r = float(parts[0]), float(parts[1]), float(parts[2])
        return x, y, r
    except Exception:
        return None


def run_script(
    script_path: Path,
    cycle_idx: int,
    step_desc: str,
    command: Optional[str],
    extra_env: Optional[Dict[str, str]] = None,
) -> None:
    if not script_path.exists():
        raise FileNotFoundError(f"{script_path} not found")

    cmd = [sys.executable, str(script_path)]
    input_data = None
    if command is not None:
        # Send one command then quit so the script proceeds once and exits.
        input_data = f"{command}\nq\n"

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    print(f"[Cycle {cycle_idx}] {step_desc} -> {cmd} (cmd: {command or 'n/a'})")
    result = subprocess.run(cmd, cwd=BASE_DIR, input=input_data, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"{script_path.name} failed with exit code {result.returncode}")
    print(f"[Cycle {cycle_idx}] Done: {script_path.name} (exit {result.returncode})")


def main() -> None:
    base_xyz = parse_command(JETANK1_BASE_CMD) or parse_command(JETANK1_DEFAULT_CMD) or (10.0, 150.0, 0.0)

    try:
        for cycle in range(1, REPEAT_COUNT + 1):
            print(f"\n=== Cycle {cycle}/{REPEAT_COUNT} ===")
            for script_name, description in STEPS:
                command: Optional[str] = None
                env_extra: Dict[str, str] = {}
                if script_name == "jetank1.py":
                    x, y, r = base_xyz
                    command = f"{x} {y + (cycle - 1) * JETANK1_Y_INCREMENT} {r}"
                elif script_name == "jetank2.py":
                    command = JETANK2_CMD
                    drop = JETANK2_DROP_SEQUENCE[(cycle - 1) % len(JETANK2_DROP_SEQUENCE)]
                    env_extra["JETANK2_DROP"] = f"{drop[0]} {drop[1]} {drop[2]} {drop[3]}"
                run_script(BASE_DIR / script_name, cycle, description, command, env_extra)
        print("\nAll cycles completed successfully.")
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
    except Exception as exc:  # noqa: BLE001
        print(f"\nAborting due to error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
