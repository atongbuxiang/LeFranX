#!/usr/bin/env python3
"""Run an OpenPI remote policy on the Franka + gripper robot.

The policy inference loop and robot control loop are intentionally decoupled:

- inference keeps sending fresh observations to the OpenPI websocket server;
- control runs at a fixed frequency and always executes the newest valid action;
- action rows whose scheduled timestamps are already in the past are dropped.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


logger = logging.getLogger(__name__)

DEFAULT_CAMERAS = {
    "realsense_wrist": "241122305042",
    "realsense_topdown": "241122300571",
}
DEFAULT_HOME = [0, -1.14, 0, -2.37, -0.12, 1.79, 0.164]
OBS_STATE_NAMES = [f"arm_joint_{i}.pos" for i in range(7)] + ["gripper.pos"]


@dataclass(frozen=True)
class ActionChunk:
    actions: np.ndarray
    execute_at_monotonic_s: np.ndarray
    observation_monotonic_s: float
    sequence_id: int


class LatestActionBuffer:
    """Thread-safe handoff for the newest action chunk."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._chunk: ActionChunk | None = None

    def update(self, chunk: ActionChunk) -> None:
        with self._condition:
            if self._chunk is None or chunk.sequence_id > self._chunk.sequence_id:
                self._chunk = chunk
                self._condition.notify_all()

    def get(self) -> ActionChunk | None:
        with self._condition:
            return self._chunk

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--remote-host", default="10.1.83.246", help="OpenPI policy server host.")
    parser.add_argument("--remote-port", type=int, default=8000, help="OpenPI policy server port.")
    parser.add_argument("--api-key", default=None, help="Optional API key for the OpenPI websocket server.")
    parser.add_argument(
        "--openpi-root",
        type=Path,
        default="/home/rognuc/openpi0.5",
        help="OpenPI repo root. Used only to import the lightweight openpi-client package if it is not installed.",
    )
    parser.add_argument("--prompt", required=True, help="Pick up the green apple and place it in the upper left corner of the table.")

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run inference and scheduling, but do not send actions to the robot.",
    )

    parser.add_argument("--robot-ip", default="172.16.0.1")
    parser.add_argument("--robot-port", type=int, default=5000)
    parser.add_argument("--gripper-port", default="/dev/ttyUSB0")
    parser.add_argument("--gripper-baud", type=int, default=115200)
    parser.add_argument("--gripper-home", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=10.0, help="Unified camera capture and robot control frequency.")
    parser.add_argument(
        "--camera",
        action="append",
        default=[],
        help="Repeatable RealSense config in the form 'camera_name=realsense_id'.",
    )
    parser.add_argument(
        "--no-default-cameras",
        action="store_true",
        help="Do not add the default realsense_wrist and realsense_topdown cameras.",
    )
    parser.add_argument("--camera-name", default="realsense_wrist")
    parser.add_argument("--realsense-id", default="")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--use-depth", action="store_true", default=False)
    parser.add_argument(
        "--base-camera",
        default="realsense_topdown",
        help="Robot observation camera key to send as OpenPI images/base.",
    )
    parser.add_argument(
        "--wrist-camera",
        default="realsense_wrist",
        help="Robot observation camera key to send as OpenPI images/wrist.",
    )

    return parser.parse_args()


def configure_openpi_client_import(openpi_root: Path) -> None:
    try:
        import openpi_client  # noqa: F401

        return
    except ImportError:
        client_src = openpi_root.expanduser().resolve() / "packages" / "openpi-client" / "src"
        if client_src.is_dir():
            sys.path.insert(0, str(client_src))
            return
        raise


def normalize_camera_args(args: argparse.Namespace) -> None:
    camera_specs = list(args.camera or [])
    if not args.no_default_cameras and not camera_specs and not args.realsense_id:
        camera_specs = [f"{name}={serial}" for name, serial in DEFAULT_CAMERAS.items()]
    args.camera = camera_specs


def parse_camera_specs(camera_specs: list[str] | None) -> dict[str, str]:
    cameras: dict[str, str] = {}
    if not camera_specs:
        return cameras

    for spec in camera_specs:
        if "=" not in spec:
            raise ValueError(f"Invalid camera spec {spec!r}. Expected 'camera_name=realsense_id'.")
        name, serial = spec.split("=", 1)
        name = name.strip()
        serial = serial.strip()
        if not name or not serial:
            raise ValueError(f"Invalid camera spec {spec!r}. Both camera name and id are required.")
        cameras[name] = serial
    return cameras


def build_robot_config(args: argparse.Namespace):
    from lerobot.cameras.configs import ColorMode
    from lerobot.cameras.realsense import RealSenseCameraConfig
    from lerobot.robots.franka_fer.franka_fer_config import FrankaFERConfig
    from lerobot.robots.franka_fer_gripper.franka_fer_gripper_config import FrankaFERGripperConfig
    from lerobot.robots.gripper.config_gripper import GripperConfig

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

    camera_specs = parse_camera_specs(args.camera)
    if not camera_specs and args.realsense_id:
        camera_specs[args.camera_name] = args.realsense_id

    cameras = {
        camera_name: RealSenseCameraConfig(
            serial_number_or_name=realsense_id,
            fps=args.fps,
            width=args.camera_width,
            height=args.camera_height,
            color_mode=ColorMode.RGB,
            use_depth=args.use_depth,
        )
        for camera_name, realsense_id in camera_specs.items()
    }

    return FrankaFERGripperConfig(
        arm_config=arm_config,
        gripper_config=gripper_config,
        cameras=cameras,
        synchronize_actions=True,
        emergency_stop_both=True,
    )


def read_policy_raw_observation(
    robot: Any,
    args: argparse.Namespace,
    robot_io_lock: threading.Lock,
) -> dict[str, Any]:
    """Read the fields needed by the policy without interleaving robot socket commands."""
    obs: dict[str, Any] = {}

    with robot_io_lock:
        arm_obs = robot.arm.get_observation()
        gripper_obs = robot.gripper.get_observation()

    for key, value in arm_obs.items():
        if not key.startswith(("camera", "cam")):
            obs[f"arm_{key}"] = value
    obs.update(gripper_obs)

    for camera_name in {args.base_camera, args.wrist_camera}:
        if camera_name not in robot.cameras:
            available = ", ".join(sorted(robot.cameras))
            raise KeyError(f"Camera {camera_name!r} is not configured. Available cameras: {available}")
        obs[camera_name] = robot.cameras[camera_name].read()

    return obs


def build_policy_observation(raw_observation: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    missing_state_keys = [key for key in OBS_STATE_NAMES if key not in raw_observation]
    if missing_state_keys:
        raise KeyError(f"Robot observation is missing state keys: {missing_state_keys}")
    if args.base_camera not in raw_observation:
        raise KeyError(f"Robot observation is missing base camera key: {args.base_camera!r}")
    if args.wrist_camera not in raw_observation:
        raise KeyError(f"Robot observation is missing wrist camera key: {args.wrist_camera!r}")

    state = np.asarray([raw_observation[key] for key in OBS_STATE_NAMES], dtype=np.float32)
    return {
        "images": {
            "base": np.asarray(raw_observation[args.base_camera]),
            "wrist": np.asarray(raw_observation[args.wrist_camera]),
        },
        "state": state,
        "prompt": args.prompt,
    }


def action_row_to_robot_action(action_row: np.ndarray, action_keys: list[str]) -> dict[str, float]:
    action = np.asarray(action_row, dtype=np.float32).reshape(-1)
    if action.shape[0] < len(action_keys):
        raise ValueError(f"Policy action has dim {action.shape[0]}, but robot expects {len(action_keys)} dims")

    robot_action = {key: float(action[i]) for i, key in enumerate(action_keys)}
    if "gripper.pos" in robot_action:
        robot_action["gripper.pos"] = float(np.clip(robot_action["gripper.pos"], 0.0, 1.0))
    return robot_action


def inference_loop(
    *,
    robot: Any,
    policy_client: Any,
    action_buffer: LatestActionBuffer,
    robot_io_lock: threading.Lock,
    stop_event: threading.Event,
    args: argparse.Namespace,
) -> None:
    sequence_id = 0
    control_dt_s = 1.0 / args.fps
    while not stop_event.is_set():
        try:
            observation_time_s = time.monotonic()
            raw_observation = read_policy_raw_observation(robot, args, robot_io_lock)
            request = build_policy_observation(raw_observation, args)
            result = policy_client.infer(request)
            received_time_s = time.monotonic()

            actions = np.asarray(result["actions"], dtype=np.float32)
            if actions.ndim != 2:
                raise ValueError(f"Expected policy actions with shape [T, D], got {actions.shape}")

            execute_at_s = observation_time_s + (np.arange(actions.shape[0], dtype=np.float64) + 1.0) * control_dt_s
            valid_mask = execute_at_s > received_time_s
            dropped = int(actions.shape[0] - np.count_nonzero(valid_mask))
            if dropped:
                logger.info(
                    "Dropped %s expired action(s) from seq=%s after %.1fms inference latency",
                    dropped,
                    sequence_id,
                    (received_time_s - observation_time_s) * 1000.0,
                )

            actions = actions[valid_mask]
            execute_at_s = execute_at_s[valid_mask]
            if actions.shape[0] == 0:
                logger.warning("Dropping policy result seq=%s because all actions are expired", sequence_id)
                sequence_id += 1
                continue

            action_buffer.update(
                ActionChunk(
                    actions=actions,
                    execute_at_monotonic_s=execute_at_s,
                    observation_monotonic_s=observation_time_s,
                    sequence_id=sequence_id,
                )
            )
            logger.debug(
                "Updated action chunk seq=%s shape=%s infer_age=%.1fms first_delay=%.1fms",
                sequence_id,
                actions.shape,
                (received_time_s - observation_time_s) * 1000.0,
                (execute_at_s[0] - received_time_s) * 1000.0,
            )
            sequence_id += 1
        except Exception:
            if stop_event.is_set():
                return
            logger.exception("Policy inference loop failed; retrying shortly")
            time.sleep(0.1)


def control_loop(
    *,
    robot: Any,
    action_buffer: LatestActionBuffer,
    robot_io_lock: threading.Lock,
    stop_event: threading.Event,
    action_keys: list[str],
    args: argparse.Namespace,
) -> None:
    dt_s = 1.0 / args.fps
    current_sequence_id: int | None = None
    action_cursor = 0

    next_tick_s = time.monotonic()
    while not stop_event.is_set():
        loop_start_s = time.monotonic()
        chunk = action_buffer.get()

        if chunk is not None and chunk.sequence_id != current_sequence_id:
            current_sequence_id = chunk.sequence_id
            action_cursor = 0

        if chunk is not None:
            due_count = int(np.searchsorted(chunk.execute_at_monotonic_s, loop_start_s, side="right"))
            if due_count > action_cursor:
                row_index = due_count - 1
                skipped = row_index - action_cursor
                if skipped > 0:
                    logger.warning(
                        "Skipping %s expired action(s) from seq=%s due to control-loop lag",
                        skipped,
                        chunk.sequence_id,
                    )

                action = action_row_to_robot_action(chunk.actions[row_index], action_keys)
                if args.dry_run:
                    logger.info("Dry-run action seq=%s row=%s: %s", chunk.sequence_id, row_index, action)
                else:
                    with robot_io_lock:
                        robot.send_action(action)
                action_cursor = row_index + 1

        next_tick_s += dt_s
        sleep_s = next_tick_s - time.monotonic()
        if sleep_s > 0:
            stop_event.wait(sleep_s)
        else:
            next_tick_s = time.monotonic()


def main() -> None:
    args = parse_args()
    from lerobot.robots.franka_fer_gripper import FrankaFERGripper
    from lerobot.utils.utils import init_logging

    init_logging()
    if args.fps <= 0:
        raise ValueError(f"--fps must be positive, got {args.fps}")

    normalize_camera_args(args)
    configure_openpi_client_import(args.openpi_root)
    from openpi_client import websocket_client_policy

    robot = FrankaFERGripper(build_robot_config(args))
    action_buffer = LatestActionBuffer()
    robot_io_lock = threading.Lock()
    stop_event = threading.Event()

    def request_stop(signum: int, frame: Any) -> None:
        del signum, frame
        stop_event.set()

    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, request_stop)

    inference_thread: threading.Thread | None = None
    try:
        logger.info("Connecting robot...")
        robot.connect(calibrate=False)
        action_keys = list(robot.action_features)
        logger.info("Robot action keys: %s", action_keys)

        logger.info("Connecting OpenPI server at %s:%s...", args.remote_host, args.remote_port)
        policy_client = websocket_client_policy.WebsocketClientPolicy(
            host=args.remote_host,
            port=args.remote_port,
            api_key=args.api_key,
        )
        metadata = policy_client.get_server_metadata()
        if metadata:
            logger.info("Policy metadata: %s", metadata)

        inference_thread = threading.Thread(
            target=inference_loop,
            name="openpi-inference",
            kwargs={
                "robot": robot,
                "policy_client": policy_client,
                "action_buffer": action_buffer,
                "robot_io_lock": robot_io_lock,
                "stop_event": stop_event,
                "args": args,
            },
            daemon=True,
        )
        inference_thread.start()
        control_loop(
            robot=robot,
            action_buffer=action_buffer,
            robot_io_lock=robot_io_lock,
            stop_event=stop_event,
            action_keys=action_keys,
            args=args,
        )
    finally:
        stop_event.set()
        if inference_thread is not None:
            inference_thread.join(timeout=2.0)
        signal.signal(signal.SIGINT, original_sigint)
        if robot.is_connected:
            with robot_io_lock:
                robot.stop()
                robot.disconnect()
        logger.info("OpenPI remote client stopped")


if __name__ == "__main__":
    main()
