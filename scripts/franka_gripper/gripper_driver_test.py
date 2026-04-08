#!/usr/bin/env python3

import argparse
import logging
import time

from lerobot.robots.gripper import Gripper, GripperConfig
from lerobot.utils.utils import init_logging

init_logging()
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone Zhixing gripper driver test")
    parser.add_argument("--gripper-port", default="/dev/ttyUSB0")
    parser.add_argument("--gripper-baud", type=int, default=115200)
    parser.add_argument("--gripper-id", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--force", type=int, default=30)
    parser.add_argument("--home", type=float, default=1.0)
    parser.add_argument(
        "--targets",
        type=float,
        nargs="+",
        default=[1.0, 0.0, 1.0],
        help="Normalized target positions in [0, 1], e.g. --targets 1.0 0.0 1.0",
    )
    parser.add_argument("--sleep-after-move", type=float, default=1.0)
    parser.add_argument("--read-count", type=int, default=5)
    parser.add_argument("--read-interval", type=float, default=0.2)
    return parser.parse_args()


def read_feedback(gripper: Gripper) -> tuple[int, float]:
    driver = gripper.driver
    if driver is None or driver.instrument is None:
        raise RuntimeError("Gripper driver is not connected")

    raw_pos = int(driver.instrument.read_long(driver.FEEDBACK_POS_HIGH_8_REG))
    normalized_pos = float(driver.read_pos())
    return raw_pos, normalized_pos


def main():
    args = parse_args()
    gripper = Gripper(
        GripperConfig(
            serial_port=args.gripper_port,
            baud_rate=args.gripper_baud,
            slave_id=args.gripper_id,
            timeout_s=args.timeout,
            default_force=args.force,
            home_position=args.home,
        )
    )

    try:
        gripper.connect(calibrate=False)
        raw_pos, normalized_pos = read_feedback(gripper)
        logger.info("Initial feedback: raw=%s normalized=%.6f", raw_pos, normalized_pos)

        for target in args.targets:
            logger.info("Sending target gripper.pos=%.6f", target)
            gripper.send_action({"gripper.pos": target})
            time.sleep(args.sleep_after_move)

            for index in range(args.read_count):
                raw_pos, normalized_pos = read_feedback(gripper)
                logger.info(
                    "Read %d/%d after target %.6f: raw=%s normalized=%.6f",
                    index + 1,
                    args.read_count,
                    target,
                    raw_pos,
                    normalized_pos,
                )
                time.sleep(args.read_interval)
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    finally:
        if gripper.is_connected:
            gripper.disconnect()


if __name__ == "__main__":
    main()
