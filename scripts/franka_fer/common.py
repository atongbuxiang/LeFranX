#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.datasets.utils import DEFAULT_FEATURES, build_dataset_frame, hw_to_dataset_features
from lerobot.robots.franka_fer.franka_fer_config import FrankaFERConfig


DEFAULT_HOME = [0, -0.785, 0, -2.356, 0, 1.571, 0]
OBS_STATE_NAMES = [f"joint_{i}.pos" for i in range(7)]
OBS_VELOCITY_NAMES = [f"joint_{i}.vel" for i in range(7)]
OBS_EEPOSE_NAMES = ["x", "y", "z", "qx", "qy", "qz", "qw"]


def parse_camera_specs(camera_specs: Iterable[str] | None) -> dict[str, str]:
    cameras = {}
    if not camera_specs:
        return cameras

    for spec in camera_specs:
        if "=" not in spec:
            raise ValueError(f"Invalid camera specification {spec!r}. Expected 'camera_name=realsense_id'.")
        name, serial = spec.split("=", 1)
        name = name.strip()
        serial = serial.strip()
        if not name or not serial:
            raise ValueError(f"Invalid camera specification {spec!r}. Both camera name and id are required.")
        cameras[name] = serial

    return cameras


def build_robot_config(args) -> FrankaFERConfig:
    camera_specs = parse_camera_specs(getattr(args, "camera", None))
    if not camera_specs and getattr(args, "realsense_id", None):
        camera_specs[args.camera_name] = args.realsense_id

    cameras = {
        camera_name: RealSenseCameraConfig(
            serial_number_or_name=realsense_id,
            fps=args.fps,
            width=args.camera_width,
            height=args.camera_height,
            color_mode=ColorMode.RGB,
            use_depth=getattr(args, "use_depth", False),
        )
        for camera_name, realsense_id in camera_specs.items()
    }

    return FrankaFERConfig(
        server_ip=args.robot_ip,
        server_port=args.robot_port,
        home_position=list(DEFAULT_HOME),
        max_relative_target=None,
        cameras=cameras,
    )


def _rotation_matrix_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))

    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        qw = 0.25 * s
        qx = (rotation[2, 1] - rotation[1, 2]) / s
        qy = (rotation[0, 2] - rotation[2, 0]) / s
        qz = (rotation[1, 0] - rotation[0, 1]) / s
    else:
        diag = np.diag(rotation)
        idx = int(np.argmax(diag))
        if idx == 0:
            s = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
            qw = (rotation[2, 1] - rotation[1, 2]) / s
            qx = 0.25 * s
            qy = (rotation[0, 1] + rotation[1, 0]) / s
            qz = (rotation[0, 2] + rotation[2, 0]) / s
        elif idx == 1:
            s = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
            qw = (rotation[0, 2] - rotation[2, 0]) / s
            qx = (rotation[0, 1] + rotation[1, 0]) / s
            qy = 0.25 * s
            qz = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
            qw = (rotation[1, 0] - rotation[0, 1]) / s
            qx = (rotation[0, 2] + rotation[2, 0]) / s
            qy = (rotation[1, 2] + rotation[2, 1]) / s
            qz = 0.25 * s

    quaternion = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if norm <= 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (quaternion / norm).astype(np.float32)


def ee_pose_flat_to_xyz_quat(flat_ee_pose: Iterable[float]) -> np.ndarray:
    matrix_flat = np.asarray(list(flat_ee_pose), dtype=np.float64)
    if matrix_flat.shape != (16,):
        raise ValueError(f"Expected 16 ee_pose values, got shape {matrix_flat.shape}")

    transform = matrix_flat.reshape(4, 4).T
    position = transform[:3, 3]

    rotation = transform[:3, :3]
    u, _, vt = np.linalg.svd(rotation)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt

    quaternion = _rotation_matrix_to_quaternion_xyzw(rotation)
    return np.concatenate([position.astype(np.float32), quaternion], dtype=np.float32)


def build_recording_dataset_features(robot, *, use_videos: bool) -> dict[str, dict]:
    action_features = hw_to_dataset_features(robot.action_features, "action", use_video=use_videos)
    camera_features = {
        key: value for key, value in robot.observation_features.items() if isinstance(value, tuple)
    }
    observation_camera_features = hw_to_dataset_features(
        camera_features, "observation", use_video=use_videos
    )

    observation_features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(OBS_STATE_NAMES),),
            "names": OBS_STATE_NAMES,
        },
        "observation.velocity": {
            "dtype": "float32",
            "shape": (len(OBS_VELOCITY_NAMES),),
            "names": OBS_VELOCITY_NAMES,
        },
        "observation.eepose": {
            "dtype": "float32",
            "shape": (len(OBS_EEPOSE_NAMES),),
            "names": OBS_EEPOSE_NAMES,
        },
    }

    return {
        **action_features,
        **observation_features,
        **observation_camera_features,
    }


def build_recording_observation_frame(
    dataset_features: dict[str, dict],
    raw_observation: dict[str, object],
) -> dict[str, np.ndarray]:
    state = np.array([raw_observation[name] for name in OBS_STATE_NAMES], dtype=np.float32)
    velocity = np.array([raw_observation[name] for name in OBS_VELOCITY_NAMES], dtype=np.float32)
    ee_pose_flat = [raw_observation[f"ee_pose.{i:02d}"] for i in range(16)]
    eepose = ee_pose_flat_to_xyz_quat(ee_pose_flat)

    frame = {
        "observation.state": state,
        "observation.velocity": velocity,
        "observation.eepose": eepose,
    }

    camera_features = {
        key: value
        for key, value in dataset_features.items()
        if key.startswith("observation.images.")
    }
    frame.update(build_dataset_frame(camera_features, raw_observation, prefix="observation"))
    return frame


def ensure_episode_buffer(dataset) -> None:
    if getattr(dataset, "episode_buffer", None) is None:
        dataset.episode_buffer = dataset.create_episode_buffer()


def assert_dataset_features_compatible(dataset, expected_features: dict[str, dict]) -> None:
    default_feature_keys = set(DEFAULT_FEATURES)
    actual_features = {
        key: value for key, value in dataset.features.items() if key not in default_feature_keys
    }
    if actual_features != expected_features:
        raise ValueError(
            "Existing dataset feature schema does not match the current recorder format. "
            "Use a new dataset name, or recreate the dataset instead of resuming."
        )
