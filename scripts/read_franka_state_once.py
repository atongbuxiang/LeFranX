#!/usr/bin/env python3
"""Read the current Franka arm state once and print it as JSON."""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

import numpy as np

from common import DEFAULT_HOME, ee_pose_flat_to_xyz_quat

from lerobot.robots.franka_fer import FrankaFER
from lerobot.robots.franka_fer.franka_fer_config import FrankaFERConfig
from lerobot.robots.franka_fer_gripper import FrankaFERGripper
from lerobot.robots.franka_fer_gripper.franka_fer_gripper_config import FrankaFERGripperConfig
from lerobot.robots.gripper.config_gripper import GripperConfig
from lerobot.utils.utils import init_logging

init_logging()
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", default="172.16.0.1")
    parser.add_argument("--robot-port", type=int, default=5000)
    parser.add_argument(
        "--with-gripper",
        action="store_true",
        help="Also connect to the external gripper and print gripper.pos.",
    )
    parser.add_argument("--gripper-port", default="/dev/ttyUSB0")
    parser.add_argument("--gripper-baud", type=int, default=115200)
    parser.add_argument("--gripper-home", type=float, default=1.0)
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Print compact one-line JSON instead of pretty JSON.",
    )
    return parser.parse_args()


def build_arm_config(args: argparse.Namespace) -> FrankaFERConfig:
    return FrankaFERConfig(
        server_ip=args.robot_ip,
        server_port=args.robot_port,
        home_position=list(DEFAULT_HOME),
        max_relative_target=None,
        cameras={},
    )


def build_robot(args: argparse.Namespace) -> FrankaFER | FrankaFERGripper:
    arm_config = build_arm_config(args)
    if not args.with_gripper:
        return FrankaFER(arm_config)

    gripper_config = GripperConfig(
        serial_port=args.gripper_port,
        baud_rate=args.gripper_baud,
        home_position=args.gripper_home,
    )
    return FrankaFERGripper(
        FrankaFERGripperConfig(
            arm_config=arm_config,
            gripper_config=gripper_config,
            cameras={},
            synchronize_actions=True,
            emergency_stop_both=True,
        )
    )


def _prefixed(obs: dict[str, Any], key: str) -> Any:
    if key in obs:
        return obs[key]
    arm_key = f"arm_{key}"
    if arm_key in obs:
        return obs[arm_key]
    raise KeyError(key)


def _as_float_list(values: list[Any] | np.ndarray) -> list[float]:
    return [float(value) for value in values]


def format_state(obs: dict[str, Any]) -> dict[str, Any]:
    ee_pose_flat = [_prefixed(obs, f"ee_pose.{i:02d}") for i in range(16)]
    ee_pose_matrix = np.asarray(ee_pose_flat, dtype=np.float64).reshape(4, 4).T
    ee_pose_xyz_quat = ee_pose_flat_to_xyz_quat(ee_pose_flat)

    state: dict[str, Any] = {
        "joint_positions_rad": _as_float_list([_prefixed(obs, f"joint_{i}.pos") for i in range(7)]),
        "joint_velocities_rad_s": _as_float_list([_prefixed(obs, f"joint_{i}.vel") for i in range(7)]),
        "ee_pose_matrix": [[float(value) for value in row] for row in ee_pose_matrix],
        "ee_pose_xyz_quat_xyzw": _as_float_list(ee_pose_xyz_quat),
    }

    if "gripper.pos" in obs:
        state["gripper_pos"] = float(obs["gripper.pos"])

    return state


def main() -> None:
    args = parse_args()
    robot = build_robot(args)

    try:
        robot.connect(calibrate=False)
        state = format_state(robot.get_observation())
        if args.compact:
            print(json.dumps(state, ensure_ascii=False, separators=(",", ":")))
        else:
            print(json.dumps(state, ensure_ascii=False, indent=2))
    finally:
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
