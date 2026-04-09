#!/usr/bin/env python3
"""
Monitor subarm (SoFranka) only — does NOT connect to Franka.

Default: SoFranka joint reads are **degrees** (matches FrankaFERSubarmTeleoperatorConfig.use_degrees=True);
teleop does rad(°) once, then joint_scale * leader_rad + joint_offset_rad — Franka commands are rad.

Use --leader-normalized for [-100,100] RANGE_M100_100 (same as标定 --leader-normalized).

Shows 标定前 (leader→rad if °) and 标定后 (cmd rad). Gripper: raw + [0,1] (handles closed/open order).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from lerobot.teleoperators.sofranka import SoFranka, SoFrankaConfig
from lerobot.utils.utils import init_logging

init_logging()
logger = logging.getLogger(__name__)

_DEFAULT_CAL = Path(__file__).resolve().parent / "subarm_cal.json"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--leader-port", default="/dev/ttyACM0")
    p.add_argument(
        "--leader-normalized",
        action="store_true",
        help="SoFranka joints in [-100,100] (not °). Default is degree mode (°→rad like teleop).",
    )
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--calibrate", action="store_true", help="Run SoFranka motor calibration on connect")
    p.add_argument(
        "--calibration-json",
        type=Path,
        default=_DEFAULT_CAL,
        help=f"Path to JSON (default: {_DEFAULT_CAL})",
    )
    return p.parse_args()


def load_calibration(path: Path) -> tuple[list[float], list[float], float, float]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    arm = data.get("arm", {})
    g = data.get("gripper", {})
    scale = [float(x) for x in arm.get("joint_scale", [1.0] * 7)]
    offset = [float(x) for x in arm.get("joint_offset_rad", [0.0] * 7)]
    if len(scale) != 7 or len(offset) != 7:
        raise ValueError("arm.joint_scale and arm.joint_offset_rad must each have length 7")
    closed = float(g.get("gripper_raw_at_closed", 0.0))
    open_ = float(g.get("gripper_raw_at_open", 100.0))
    return scale, offset, closed, open_


def map_arm(
    raw: dict[str, float],
    scale: list[float],
    offset: list[float],
    *,
    leader_use_degrees: bool,
) -> list[float]:
    def q(i: int) -> float:
        v = float(raw[f"joint_{i + 1}.pos"])
        return math.radians(v) if leader_use_degrees else v

    return [math.degrees(scale[i] * q(i) + offset[i]) for i in range(7)]


def map_gripper(g_raw: float, closed: float, open_: float) -> float:
    """Match FrankaFERGripperSubarmTeleoperator.get_action gripper mapping."""
    if abs(open_ - closed) < 1e-9:
        g01 = g_raw / 100.0
    elif closed <= open_:
        g01 = (g_raw - closed) / (open_ - closed)
    else:
        g01 = (closed - g_raw) / (closed - open_)
    return float(np.clip(g01, 0.0, 1.0))


def format_row(label: str, values: list[float] | tuple[float, ...], width: int = 10) -> str:
    parts = [f"{v:{width}.4f}" for v in values]
    return f"{label:26s} " + " | ".join(parts)


def leader_before_calibration_rad(
    raw_joints: list[float], *, leader_use_degrees: bool
) -> tuple[list[float], str]:
    """Display 'before mapping' joint column: rad if leader reports degrees, else bus units."""
    if leader_use_degrees:
        return [math.radians(v) for v in raw_joints], "标定前 (rad，主臂为 ° 已换算)"
    return list(raw_joints), "标定前 (与 teleop 同单位，非物理弧度)"


def read_leader_action_with_retry(leader: SoFranka, *, retries: int = 3, delay_s: float = 0.05):
    last_err: BaseException | None = None
    for attempt in range(retries):
        try:
            action = leader.get_action()
            normalized = bool(leader.bus.calibration)
            return action, normalized
        except OSError as exc:
            last_err = exc
            logger.warning("Read attempt %s/%s failed: %s", attempt + 1, retries, exc)
            time.sleep(delay_s)
    raise last_err  # type: ignore[misc]


def main():
    args = parse_args()
    cal_path = args.calibration_json
    if not cal_path.is_file():
        logger.error("Calibration file not found: %s", cal_path)
        sys.exit(1)

    scale, offset, g_closed, g_open = load_calibration(cal_path)
    leader_use_degrees = not args.leader_normalized
    leader = SoFranka(
        SoFrankaConfig(
            port=args.leader_port,
            use_degrees=leader_use_degrees,
        )
    )

    joint_labels = [f"j{i}" for i in range(1, 8)]

    try:
        leader.connect(calibrate=args.calibrate)
        dt = 1.0 / args.fps
        logger.info("Subarm only — no Franka connection. Leader: %s", args.leader_port)
        logger.info("Calibration: %s", cal_path.resolve())
        logger.info("Press Ctrl+C to stop")

        while True:
            loop_start = time.perf_counter()
            try:
                action, normalized = read_leader_action_with_retry(leader)
            except OSError as exc:
                logger.error("Serial read failed repeatedly. Exiting: %s", exc)
                break

            raw_joints = [float(action[f"joint_{i}.pos"]) for i in range(1, 8)]
            before_joints, before_label = leader_before_calibration_rad(
                raw_joints, leader_use_degrees=leader_use_degrees
            )
            mapped_joints = map_arm(action, scale, offset, leader_use_degrees=leader_use_degrees)
            g_raw = float(action.get("gripper.pos", 0.0))
            g01 = map_gripper(g_raw, g_closed, g_open)

            print("\033[2J\033[H", end="")
            norm_txt = "normalized" if normalized else "RAW (no motor calibration — map may be meaningless)"
            print(f"Subarm monitor — leader bus: {norm_txt}")
            print(f"JSON: {cal_path}")
            print()
            print(before_label)
            print(format_row("  j1..j7", before_joints))
            print()
            print("标定后 (cmd，rad，将发给 Franka 的关节角)")
            print(format_row("  j1..j7", mapped_joints))
            print()
            print(f"gripper raw            {g_raw:12.4f}   (closed={g_closed:.4f}, open={g_open:.4f})")
            print(f"gripper [0,1]          {g01:12.4f}")
            print()
            print("列: " + " | ".join(joint_labels))

            elapsed = time.perf_counter() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    finally:
        if leader.is_connected:
            try:
                leader.disconnect()
            except OSError as exc:
                logger.warning("disconnect: %s", exc)


if __name__ == "__main__":
    main()
