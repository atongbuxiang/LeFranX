#!/usr/bin/env python3
"""Run an OpenPI remote policy on the Franka + gripper robot.

This variant runs synchronously:

- read one robot observation;
- infer one action chunk from the OpenPI server;
- execute actions 3 through 10 from that chunk (1-based, inclusive);
- then read a fresh observation and infer again.
"""

from __future__ import annotations

import argparse
import logging
import signal
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from openpi_remote_client import (
    action_row_to_robot_action,
    build_policy_observation,
    build_robot_config,
    configure_openpi_client_import,
    normalize_camera_args,
    read_policy_raw_observation,
)


logger = logging.getLogger(__name__)

EXECUTE_START_INDEX = 2
EXECUTE_END_INDEX_EXCLUSIVE = 10


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
    parser.add_argument("--prompt", required=True, help="Task prompt sent to the OpenPI policy server.")
    parser.add_argument("--dry-run", action="store_true", help="Infer actions but do not send them to the robot.")

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


def infer_action_chunk(
    robot: Any,
    policy_client: Any,
    args: argparse.Namespace,
    robot_io_lock: threading.Lock,
) -> np.ndarray:
    raw_observation = read_policy_raw_observation(robot, args, robot_io_lock)
    request = build_policy_observation(raw_observation, args)
    result = policy_client.infer(request)
    actions = np.asarray(result["actions"], dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"Expected policy actions with shape [T, D], got {actions.shape}")
    return actions


def execute_action_window(
    robot: Any,
    action_rows: np.ndarray,
    action_keys: list[str],
    args: argparse.Namespace,
    stop_event: threading.Event,
) -> None:
    dt_s = 1.0 / args.fps

    for row_index, action_row in action_rows:
        if stop_event.is_set():
            return

        started_at_s = time.monotonic()
        action = action_row_to_robot_action(action_row, action_keys)
        if args.dry_run:
            logger.info("Dry-run action row=%s: %s", row_index, action)
        else:
            robot.send_action(action)

        elapsed_s = time.monotonic() - started_at_s
        if elapsed_s < dt_s:
            stop_event.wait(dt_s - elapsed_s)


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
    robot_io_lock = threading.Lock()
    stop_event = threading.Event()

    def request_stop(signum: int, frame: Any) -> None:
        del signum, frame
        stop_event.set()

    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, request_stop)

    try:
        logger.info("Connecting robot...")
        robot.connect(calibrate=False)
        action_keys = list(robot.action_features)
        logger.info("Robot action keys: %s", action_keys)
        logger.info(
            "Chunk execution window: actions %s-%s (1-based)",
            EXECUTE_START_INDEX + 1,
            EXECUTE_END_INDEX_EXCLUSIVE,
        )

        logger.info("Connecting OpenPI server at %s:%s...", args.remote_host, args.remote_port)
        policy_client = websocket_client_policy.WebsocketClientPolicy(
            host=args.remote_host,
            port=args.remote_port,
            api_key=args.api_key,
        )
        metadata = policy_client.get_server_metadata()
        if metadata:
            logger.info("Policy metadata: %s", metadata)

        sequence_id = 0
        while not stop_event.is_set():
            inferred_at_s = time.monotonic()
            actions = infer_action_chunk(robot, policy_client, args, robot_io_lock)
            logger.info(
                "Received action chunk seq=%s with %s rows in %.1fms",
                sequence_id,
                actions.shape[0],
                (time.monotonic() - inferred_at_s) * 1000.0,
            )

            if actions.shape[0] <= EXECUTE_START_INDEX:
                logger.warning(
                    "Skipping chunk seq=%s because it has only %s row(s), fewer than the required first %s rows",
                    sequence_id,
                    actions.shape[0],
                    EXECUTE_START_INDEX + 1,
                )
                sequence_id += 1
                continue

            execute_stop = min(actions.shape[0], EXECUTE_END_INDEX_EXCLUSIVE)
            selected = list(enumerate(actions[EXECUTE_START_INDEX:execute_stop], start=EXECUTE_START_INDEX))
            logger.info(
                "Executing %s action(s) from seq=%s: rows %s-%s (0-based), actions %s-%s (1-based)",
                len(selected),
                sequence_id,
                selected[0][0],
                selected[-1][0],
                selected[0][0] + 1,
                selected[-1][0] + 1,
            )
            execute_action_window(robot, selected, action_keys, args, stop_event)
            sequence_id += 1
    finally:
        stop_event.set()
        signal.signal(signal.SIGINT, original_sigint)
        if robot.is_connected:
            with robot_io_lock:
                robot.stop()
                robot.disconnect()
        logger.info("OpenPI chunked remote client stopped")


if __name__ == "__main__":
    main()
