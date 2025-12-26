#!/usr/bin/env python3
"""Shared CycloneDDS environment setup for role scripts."""

import os
from pathlib import Path


def ensure_cyclonedds_env() -> None:
    """Ensure CycloneDDS env vars point to src/cyclonedds.xml."""
    xml_path = Path(__file__).resolve().parents[2] / "cyclonedds.xml"

    rmw = os.environ.get("RMW_IMPLEMENTATION")
    if rmw != "rmw_cyclonedds_cpp":
        if rmw:
            print(f"[Warn] RMW_IMPLEMENTATION={rmw} -> rmw_cyclonedds_cpp")
        os.environ["RMW_IMPLEMENTATION"] = "rmw_cyclonedds_cpp"

    if xml_path.exists():
        uri = f"file://{xml_path}"
        if os.environ.get("CYCLONEDDS_URI") != uri:
            os.environ["CYCLONEDDS_URI"] = uri
            print(f"[Info] CYCLONEDDS_URI set to {uri}")
    else:
        print(f"[Warn] cyclonedds.xml not found: {xml_path}")
