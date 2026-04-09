#!/usr/bin/env python3
"""
Two-step calibration helper for subarm (SoFranka leader) + Franka follower.

All Franka command joints are **radians**. Leader in degree mode: first convert each joint to
radians once (radians(°)), then apply per-joint sign and offset (same as FrankaFERSubarmTeleoperator):

    leader_rad[i] = radians(leader_deg[i])   if use_degrees
                    = raw joint i             if --leader-normalized ([-100,100], not physical rad)
    offset_rad[i] = franka_rad[i] - joint_scale[i] * leader_rad[i]

Use --joint-scale for axis direction (±1) vs Franka; default is 1 for every joint.

Step 1 — joint alignment (same physical pose → same command radians):
  - Move subarm and Franka to the SAME pose (e.g. home).
  - Default SoFranka mode: **degrees** (matches Franka needing rad).
    Use --leader-normalized only if you intentionally use [-100,100] joint reads instead.
  - franka_rad: from prompt (radians), or use --franka-input-degrees for Franka in °.

Step 2 — gripper 0..1 over real stroke:
  - Fully close leader gripper, press Enter (records raw).
  - Fully open leader gripper, press Enter (records raw).
  - Maps: closed raw → 0.0, open raw → 1.0 (linear clip).

Copy the printed JSON into your code or load manually into
FrankaFERSubarmTeleoperatorConfig / FrankaFERGripperSubarmTeleoperatorConfig.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from lerobot.teleoperators.sofranka import SoFranka, SoFrankaConfig
from lerobot.utils.utils import init_logging

init_logging()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--leader-port", default="/dev/ttyACM0")
    p.add_argument(
        "--leader-normalized",
        action="store_true",
        help="SoFranka joints in [-100,100] (RANGE_M100_100), not °. Default is degree mode for Franka→rad.",
    )
    p.add_argument(
        "--franka-input-degrees",
        action="store_true",
        help="Interpret the 7 Franka numbers as degrees (converted to rad for offset).",
    )
    p.add_argument("--skip-arm", action="store_true", help="Only run gripper range step")
    p.add_argument("--skip-gripper", action="store_true", help="Only run arm offset step")
    p.add_argument(
        "--joint-scale",
        type=str,
        default="-1,1,-1,-1,-1,1,-1",
        help="Seven comma-separated floats (±1 typical): multiplies leader_rad before offset. "
        "Example for many subarms vs Franka: -1,1,-1,-1,1,1,1",
    )
    return p.parse_args()


def parse_joint_scale_csv(s: str) -> list[float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 7:
        raise ValueError(f"--joint-scale must have 7 comma-separated numbers, got {len(parts)}")
    return [float(x) for x in parts]


def prompt_franka_joint_values(*, input_in_degrees: bool) -> list[float]:
    unit = "degrees" if input_in_degrees else "radians"
    example = (
        "0,-45,0,-135,0,90,-51.5"
        if input_in_degrees
        else "0,-0.785,0,-2.356,0,1.571,-0.9"
    )
    s = input(
        f"Enter 7 Franka joint positions in {unit}, comma-separated\n"
        f"(same physical pose as subarm), e.g. {example}\n> "
    ).strip()
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 7:
        raise ValueError(f"Expected 7 values, got {len(parts)}")
    vals = [float(x) for x in parts]
    if input_in_degrees:
        return [math.radians(v) for v in vals]
    return vals


def leader_raw_to_radians_for_offset(leader_vals: list[float], *, leader_use_degrees: bool) -> list[float]:
    """Leader column in the same unit as FrankaFERSubarmTeleoperator (rad if degrees mode)."""
    if leader_use_degrees:
        return [math.radians(v) for v in leader_vals]
    return list(leader_vals)


def main():
    args = parse_args()
    joint_scale = parse_joint_scale_csv(args.joint_scale)
    leader_use_degrees = not args.leader_normalized
    leader = SoFranka(
        SoFrankaConfig(
            port=args.leader_port,
            use_degrees=leader_use_degrees,
        )
    )

    out: dict = {}

    try:
        print("Connecting leader (calibrate motors if prompted)...")
        leader.connect(calibrate=True)

        if not args.skip_arm:
            input("\n=== Step 1: arm pose matching ===\nMove SUBARM and FRANKA to the SAME pose, then press Enter...")
            raw = leader.get_action()
            leader_vals = [float(raw[f"joint_{i}.pos"]) for i in range(1, 8)]
            print("Subarm leader joints raw (joint_1..7), same units as get_action:", leader_vals)
            leader_rad = leader_raw_to_radians_for_offset(leader_vals, leader_use_degrees=leader_use_degrees)
            print(
                "Leader used for offset (rad):",
                leader_rad,
                "(from °)" if leader_use_degrees else "(normalized [-100,100] — not rad; offset pairs with that mode)",
            )
            franka_rad = prompt_franka_joint_values(input_in_degrees=args.franka_input_degrees)
            print("Franka target (rad):", franka_rad)
            offset = [
                franka_rad[i] - joint_scale[i] * leader_rad[i] for i in range(7)
            ]
            out["arm"] = {
                "joint_scale": joint_scale,
                "joint_offset_rad": offset,
                "note": (
                    "franka_cmd_rad[i] = joint_scale[i] * leader_rad[i] + joint_offset_rad[i]. "
                    "leader_rad: from ° via radians(°) when SoFranka use_degrees; else raw [-100,100] "
                    "(must match runtime). offset_rad[i] = franka_rad[i] - joint_scale[i] * leader_rad[i]."
                ),
            }
            print("joint_scale:", joint_scale)
            print("joint_offset_rad:", offset)

        if not args.skip_gripper:
            input("\n=== Step 2: gripper range ===\nClose leader gripper FULLY, then press Enter...")
            raw_closed = leader.get_action()
            closed = float(raw_closed["gripper.pos"])
            input("Open leader gripper FULLY, then press Enter...")
            raw_open = leader.get_action()
            open_ = float(raw_open["gripper.pos"])
            out["gripper"] = {
                "gripper_raw_at_closed": closed,
                "gripper_raw_at_open": open_,
                "note": "gripper.pos = clip((raw - closed) / (open - closed), 0, 1)",
            }
            print(f"Recorded gripper raw closed={closed}, open={open_}")

    finally:
        if leader.is_connected:
            try:
                leader.disconnect()
            except OSError:
                pass

    print("\n=== Paste into config or save as JSON ===\n")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
