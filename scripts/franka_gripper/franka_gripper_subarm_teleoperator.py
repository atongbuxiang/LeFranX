#!/usr/bin/env python3
"""Teleoperate Franka + external gripper using subarm (SoFranka) leader + subarm_cal.json."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from common import build_robot_config

from lerobot.robots.franka_fer_gripper import FrankaFERGripper
from lerobot.teleoperators.franka_fer_gripper_subarm import (
    FrankaFERGripperSubarmTeleoperator,
    FrankaFERGripperSubarmTeleoperatorConfig,
)
from lerobot.teleoperators.franka_fer_subarm import FrankaFERSubarmTeleoperatorConfig
from lerobot.utils.utils import init_logging

init_logging()
logger = logging.getLogger(__name__)

_DEFAULT_CAL = Path(__file__).resolve().parent / "subarm_cal.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot-ip", default="172.16.0.1")
    parser.add_argument("--robot-port", type=int, default=5000)
    parser.add_argument("--gripper-port", default="/dev/ttyUSB0")
    parser.add_argument("--gripper-baud", type=int, default=115200)
    parser.add_argument("--gripper-home", type=float, default=1.0)
    parser.add_argument("--leader-port", default="/dev/ttyACM0")
    parser.add_argument(
        "--leader-normalized",
        action="store_true",
        help="SoFranka joints [-100,100] (must match标定). Default: degrees → rad (same as franka_gripper_subarm_monitor_calibrated.py).",
    )
    parser.add_argument(
        "--calibrate-leader",
        action="store_true",
        help="Run SoFranka motor calibration when connecting the leader (same as monitor --calibrate).",
    )
    parser.add_argument(
        "--calibration-json",
        type=Path,
        default=_DEFAULT_CAL,
        help=f"Arm+gripper JSON (default: {_DEFAULT_CAL})",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera-name", default="realsense")
    parser.add_argument("--realsense-id", default="241122305042")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--use-depth", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cal_path = args.calibration_json
    if not cal_path.is_file():
        logger.error("Calibration file not found: %s", cal_path)
        sys.exit(1)

    robot = FrankaFERGripper(build_robot_config(args))

    with cal_path.open(encoding="utf-8") as f:
        cal = json.load(f)
    arm_cal = cal.get("arm", {})
    grip_cal = cal.get("gripper", {})

    leader_use_degrees = not args.leader_normalized
    arm_kwargs: dict = {
        "port": args.leader_port,
        "use_degrees": leader_use_degrees,
    }
    if "joint_scale" in arm_cal:
        arm_kwargs["joint_scale"] = tuple(float(x) for x in arm_cal["joint_scale"])
    if "joint_offset_rad" in arm_cal:
        arm_kwargs["joint_offset_rad"] = tuple(float(x) for x in arm_cal["joint_offset_rad"])

    top_kwargs: dict = {"arm_config": FrankaFERSubarmTeleoperatorConfig(**arm_kwargs)}
    if "gripper_raw_at_closed" in grip_cal:
        top_kwargs["gripper_raw_at_closed"] = float(grip_cal["gripper_raw_at_closed"])
    if "gripper_raw_at_open" in grip_cal:
        top_kwargs["gripper_raw_at_open"] = float(grip_cal["gripper_raw_at_open"])

    teleop = FrankaFERGripperSubarmTeleoperator(
        FrankaFERGripperSubarmTeleoperatorConfig(**top_kwargs)
    )

    try:
        logger.info("Calibration: %s", cal_path.resolve())
        logger.info("Leader: degrees=%s (use --leader-normalized if标定为 [-100,100])", leader_use_degrees)
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
