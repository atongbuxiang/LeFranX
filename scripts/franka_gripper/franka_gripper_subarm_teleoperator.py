#!/usr/bin/env python3

import argparse
import logging
import time

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


def parse_args():
    parser = argparse.ArgumentParser(description="Teleoperate Franka + gripper with subarm")
    parser.add_argument("--robot-ip", default="192.168.18.1")
    parser.add_argument("--robot-port", type=int, default=5000)
    parser.add_argument("--gripper-port", default="/dev/ttyACM1")
    parser.add_argument("--gripper-baud", type=int, default=115200)
    parser.add_argument("--gripper-home", type=float, default=1.0)
    parser.add_argument("--leader-port", default="/dev/ttyACM0")
    parser.add_argument("--leader-use-degrees", action="store_true")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera-name", default="realsense")
    parser.add_argument("--realsense-id", default="")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--use-depth", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    robot = FrankaFERGripper(build_robot_config(args))
    teleop = FrankaFERGripperSubarmTeleoperator(
        FrankaFERGripperSubarmTeleoperatorConfig(
            arm_config=FrankaFERSubarmTeleoperatorConfig(
                port=args.leader_port,
                use_degrees=args.leader_use_degrees,
            )
        )
    )

    try:
        robot.connect(calibrate=False)
        teleop.connect(calibrate=False)

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
