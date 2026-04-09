#!/usr/bin/env python3

import argparse
import time

from common import build_robot_config, get_episode_actions, load_dataset

from lerobot.robots.franka_fer_gripper import FrankaFERGripper
from lerobot.utils.utils import init_logging

init_logging()


def parse_args():
    parser = argparse.ArgumentParser(description="Replay Franka + gripper subarm dataset")
    parser.add_argument("dataset_path")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--robot-ip", default="172.16.0.1")
    parser.add_argument("--robot-port", type=int, default=5000)
    parser.add_argument("--gripper-port", default="/dev/ttyACM1")
    parser.add_argument("--gripper-baud", type=int, default=115200)
    parser.add_argument("--gripper-home", type=float, default=1.0)
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
    dataset = load_dataset(args.dataset_path)
    action_keys = list(robot.action_features)
    actions = get_episode_actions(dataset, args.episode, action_keys)
    dt = 1.0 / dataset.fps / args.speed

    try:
        robot.connect(calibrate=False)
        for action in actions:
            start = time.perf_counter()
            robot.send_action(action)
            elapsed = time.perf_counter() - start
            if elapsed < dt:
                time.sleep(dt - elapsed)
    finally:
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
