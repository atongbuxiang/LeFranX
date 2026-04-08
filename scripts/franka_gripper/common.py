#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.cameras.configs import ColorMode
from lerobot.robots.franka_fer.franka_fer_config import FrankaFERConfig
from lerobot.robots.franka_fer_gripper.franka_fer_gripper_config import FrankaFERGripperConfig
from lerobot.robots.gripper.config_gripper import GripperConfig

if TYPE_CHECKING:
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset


DEFAULT_HOME = [0, -0.785, 0, -2.356, 0, 1.571, -0.9]


def parse_camera_specs(camera_specs: Iterable[str] | None) -> dict[str, str]:
    cameras = {}
    if not camera_specs:
        return cameras

    for spec in camera_specs:
        if "=" not in spec:
            raise ValueError(
                f"Invalid camera specification '{spec}'. Expected format is 'camera_name=realsense_id'."
            )
        name, serial = spec.split("=", 1)
        name = name.strip()
        serial = serial.strip()
        if not name or not serial:
            raise ValueError(
                f"Invalid camera specification '{spec}'. Both camera name and realsense id are required."
            )
        cameras[name] = serial

    return cameras


def build_robot_config(args) -> FrankaFERGripperConfig:
    arm_config = FrankaFERConfig(
        server_ip=args.robot_ip,
        server_port=args.robot_port,
        home_position=list(DEFAULT_HOME),
        max_relative_target=None,
        cameras={},
    )
    gripper_config = GripperConfig(
        serial_port=args.gripper_port,
        baud_rate=args.gripper_baud,
        home_position=args.gripper_home,
    )

    cameras = {}
    camera_specs = parse_camera_specs(getattr(args, "camera", None))
    if not camera_specs and getattr(args, "realsense_id", None):
        camera_specs[args.camera_name] = args.realsense_id

    for camera_name, realsense_id in camera_specs.items():
        cameras[camera_name] = RealSenseCameraConfig(
            serial_number_or_name=realsense_id,
            fps=args.fps,
            width=args.camera_width,
            height=args.camera_height,
            color_mode=ColorMode.RGB,
            use_depth=getattr(args, "use_depth", True),
        )

    return FrankaFERGripperConfig(
        arm_config=arm_config,
        gripper_config=gripper_config,
        cameras=cameras,
        synchronize_actions=True,
        emergency_stop_both=True,
    )


def load_dataset(dataset_path: str) -> LeRobotDataset:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    return LeRobotDataset(str(dataset_path))


def get_episode_actions(dataset: LeRobotDataset, episode_idx: int, action_keys: list[str]) -> list[dict[str, float]]:
    import torch

    if hasattr(dataset, "episode_data_index") and "from" in dataset.episode_data_index:
        start_idx = dataset.episode_data_index["from"][episode_idx]
        end_idx = dataset.episode_data_index["to"][episode_idx]
    else:
        start_idx = 0
        end_idx = len(dataset)

    actions = []
    for frame_idx in range(start_idx, end_idx):
        frame_data = dataset[frame_idx]
        action = {}
        action_data = frame_data["action"]
        if isinstance(action_data, torch.Tensor):
            for i, action_key in enumerate(action_keys):
                if i < len(action_data):
                    action[action_key] = float(action_data[i].item())
        elif isinstance(action_data, dict):
            action.update({k: float(v) for k, v in action_data.items()})
        actions.append(action)
    return actions
