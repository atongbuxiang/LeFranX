#!/usr/bin/env python3
"""Teleoperate Franka arm only with a SoFranka leader."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from common import build_robot_config

from lerobot.robots.franka_fer import FrankaFER
from lerobot.teleoperators.franka_fer_subarm import (
    FrankaFERSubarmTeleoperator,
    FrankaFERSubarmTeleoperatorConfig,
)
from lerobot.utils.utils import init_logging

init_logging()
logger = logging.getLogger(__name__)

_DEFAULT_CAL = Path(__file__).resolve().parents[1] / "franka_gripper" / "subarm_cal.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot-ip", default="172.16.0.1")
    parser.add_argument("--robot-port", type=int, default=5000)
    parser.add_argument("--leader-port", default="/dev/ttyACM0")
    parser.add_argument(
        "--leader-normalized",
        action="store_true",
        help="SoFranka joints [-100,100] (must match calibration). Default: degrees.",
    )
    parser.add_argument(
        "--calibrate-leader",
        action="store_true",
        help="Run SoFranka motor calibration when connecting the leader.",
    )
    parser.add_argument(
        "--calibration-json",
        type=Path,
        default=_DEFAULT_CAL,
        help=f"Arm calibration JSON (default: {_DEFAULT_CAL})",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera-name", default="realsense_wrist")
    parser.add_argument("--realsense-id", default="")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--use-depth", action="store_true", default=False)
    return parser.parse_args()


def build_subarm_teleop_from_calibration(
    cal_path: Path, *, leader_port: str, leader_normalized: bool
) -> FrankaFERSubarmTeleoperator:
    if not cal_path.is_file():
        logger.error("Calibration file not found: %s", cal_path)
        sys.exit(1)

    with cal_path.open(encoding="utf-8") as f:
        cal = json.load(f)

    arm_cal = cal.get("arm", cal)
    leader_use_degrees = not leader_normalized
    arm_kwargs: dict = {
        "port": leader_port,
        "use_degrees": leader_use_degrees,
    }
    if "joint_scale" in arm_cal:
        arm_kwargs["joint_scale"] = tuple(float(x) for x in arm_cal["joint_scale"])
    if "joint_offset_rad" in arm_cal:
        arm_kwargs["joint_offset_rad"] = tuple(float(x) for x in arm_cal["joint_offset_rad"])

    return FrankaFERSubarmTeleoperator(FrankaFERSubarmTeleoperatorConfig(**arm_kwargs))


def main():
    args = parse_args()
    cal_path = args.calibration_json

    robot = FrankaFER(build_robot_config(args))
    teleop = build_subarm_teleop_from_calibration(
        cal_path,
        leader_port=args.leader_port,
        leader_normalized=args.leader_normalized,
    )

    try:
        logger.info("Calibration: %s", cal_path.resolve())
        logger.info("Leader: degrees=%s (--leader-normalized if calibrated as [-100,100])", not args.leader_normalized)
        robot.connect(calibrate=False)
        teleop.connect(calibrate=args.calibrate_leader)

        dt = 1.0 / args.fps
        while True:
            loop_start = time.perf_counter()
            robot.send_action(teleop.get_action())
            elapsed = time.perf_counter() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    finally:
        if teleop.is_connected:
            teleop.disconnect()
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
