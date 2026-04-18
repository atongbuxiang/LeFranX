#!/usr/bin/env python3

import argparse
import logging
import shutil
import time
from pathlib import Path

from common import (
    assert_dataset_features_compatible,
    build_recording_dataset_features,
    build_recording_observation_frame,
    build_robot_config,
    ensure_episode_buffer,
)

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame
from lerobot.robots.franka_fer_gripper import FrankaFERGripper
from lerobot.teleoperators.franka_fer_gripper_spacemouse import (
    FrankaFERGripperSpaceMouseTeleoperator,
    FrankaFERGripperSpaceMouseTeleoperatorConfig,
)
from lerobot.utils.control_utils import is_headless
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import _init_rerun

DEFAULT_FPS = 30
DEFAULT_TASK_DESCRIPTION = "Teleoperate Franka + gripper with SpaceMouse."
DEFAULT_DATASET_NAME = "franka_gripper_spacemouse"

init_logging()
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Record data with Franka + gripper SpaceMouse teleoperation")
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--task", type=str, default=DEFAULT_TASK_DESCRIPTION)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--robot-ip", default="192.168.18.1")
    parser.add_argument("--robot-port", type=int, default=5000)
    parser.add_argument("--gripper-port", default="/dev/ttyACM1")
    parser.add_argument("--gripper-baud", type=int, default=115200)
    parser.add_argument("--gripper-home", type=float, default=1.0)
    parser.add_argument(
        "--camera",
        action="append",
        default=[],
        help="Repeatable RealSense camera config in the form 'camera_name=realsense_id'",
    )
    parser.add_argument("--camera-name", default="realsense")
    parser.add_argument("--realsense-id", default="241122305042")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--use-depth", action="store_true", default=False)
    return parser.parse_args()


def get_existing_episode_count(dataset_path):
    try:
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            return 0

        required_dirs = ["data", "meta"]
        if not all((dataset_path / dir_name).exists() for dir_name in required_dirs):
            logger.warning("Dataset directory %s exists but is missing required subdirectories", dataset_path)
            return 0

        data_dir = dataset_path / "data"
        episode_files = list(data_dir.glob("**/episode_*.parquet"))
        episode_count = len(episode_files)
        logger.info("Found %s existing episode files in %s", episode_count, data_dir)
        return episode_count
    except Exception as exc:
        logger.warning("Could not check existing episodes: %s", exc)
        return 0


def init_recording_keyboard_listener():
    events = {
        "recording": False,
        "toggle_recording": False,
        "save_episode": False,
        "clear_buffer": False,
        "stop_recording": False,
    }

    if is_headless():
        logger.warning("Headless environment detected. Keyboard controls are unavailable.")
        return None, events

    from pynput import keyboard

    pressed_keys = set()

    def normalize_key(key):
        if hasattr(key, "char") and key.char is not None:
            return key.char.lower()
        return key

    def on_press(key):
        normalized = normalize_key(key)
        if normalized in pressed_keys:
            return
        pressed_keys.add(normalized)

        try:
            if normalized == "r":
                events["toggle_recording"] = True
            elif normalized == "n":
                events["save_episode"] = True
            elif normalized == "x":
                events["clear_buffer"] = True
            elif key == keyboard.Key.esc:
                events["stop_recording"] = True
        except Exception as exc:
            print(f"Error handling key press: {exc}")

    def on_release(key):
        pressed_keys.discard(normalize_key(key))

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    return listener, events


def home_robot_and_gripper(robot):
    print("\n=== INITIAL HOMING ===")
    if hasattr(robot.arm, "reset_to_home"):
        print("Sending arm to home position...")
        success = robot.arm.reset_to_home()
        print("Arm homing initiated successfully" if success else "WARNING: Arm homing failed!")
        time.sleep(2.0)

    if hasattr(robot.gripper, "reset_to_home"):
        print("Sending gripper to home position...")
        success = robot.gripper.reset_to_home()
        print("Gripper homing initiated successfully" if success else "WARNING: Gripper homing failed!")
        time.sleep(1.0)

    print("Waiting for robot to stabilize...")
    time.sleep(1.0)


def handle_control_events(dataset, events):
    ensure_episode_buffer(dataset)

    if events["toggle_recording"]:
        events["toggle_recording"] = False
        events["recording"] = not events["recording"]
        state = "started" if events["recording"] else "stopped"
        buffer_size = dataset.episode_buffer["size"]
        log_say(f"Recording {state}. Current buffer has {buffer_size} frames.")

    if events["clear_buffer"]:
        events["clear_buffer"] = False
        if events["recording"]:
            events["recording"] = False
            log_say("Recording stopped before clearing buffer.")
        if dataset.episode_buffer["size"] == 0:
            log_say("Buffer is already empty.")
        else:
            dataset.clear_episode_buffer()
            log_say("Current buffer cleared.")

    if events["save_episode"]:
        events["save_episode"] = False
        if events["recording"]:
            events["recording"] = False
            log_say("Recording stopped before saving buffer.")
        if dataset.episode_buffer["size"] == 0:
            log_say("Buffer is empty, nothing to save.")
            return

        dataset.save_episode()
        log_say(f"Saved episode {dataset.num_episodes}. Ready for episode {dataset.num_episodes + 1}.")


def run_record_loop(robot, teleop, dataset, dataset_features, events, fps, task):
    dt = 1.0 / fps

    while not events["stop_recording"]:
        loop_start = time.perf_counter()
        handle_control_events(dataset, events)

        action = teleop.get_action()
        performed_action = robot.send_action(action)
        observation = robot.get_observation()

        if events["recording"]:
            frame = {}
            frame.update(build_dataset_frame(dataset_features, performed_action, prefix="action"))
            frame.update(build_recording_observation_frame(dataset_features, observation))
            dataset.add_frame(frame, task=task, timestamp=dataset.episode_buffer["size"] / fps)

        elapsed = time.perf_counter() - loop_start
        if elapsed < dt:
            time.sleep(dt - elapsed)


def main():
    args = parse_args()
    if not args.camera and not args.realsense_id:
        raise ValueError("Provide at least one camera via '--camera name=id' or '--realsense-id'.")

    logger.info("Setting up Franka gripper SpaceMouse recording test...")
    logger.info("Dataset: %s", args.dataset_name)
    logger.info("Task: %s", args.task)
    logger.info("Camera specs: %s", args.camera if args.camera else [f"{args.camera_name}={args.realsense_id}"])

    robot = FrankaFERGripper(build_robot_config(args))
    teleop = FrankaFERGripperSpaceMouseTeleoperator(FrankaFERGripperSpaceMouseTeleoperatorConfig())

    logger.info("Setting up dataset...")
    dataset_features = build_recording_dataset_features(robot, use_videos=True)

    logger.info("Robot action features: %s", list(robot.action_features.keys()))
    logger.info("Robot observation features: %s", list(robot.observation_features.keys()))
    logger.info("Dataset features: %s", list(dataset_features.keys()))

    dataset_path = Path.cwd() / "data" / args.dataset_name
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    existing_episodes = 0
    if args.resume:
        existing_episodes = get_existing_episode_count(dataset_path)
        if existing_episodes > 0:
            logger.info("Found %s existing episodes. Resuming from episode %s", existing_episodes, existing_episodes + 1)
        else:
            logger.info("No existing episodes found. Starting from episode 1")

    if existing_episodes > 0:
        logger.info("Loading existing dataset from %s with %s episodes", dataset_path, existing_episodes)
        dataset = LeRobotDataset(dataset_path)
        assert_dataset_features_compatible(dataset, dataset_features)
    else:
        logger.info("Creating new dataset at %s", dataset_path)
        if dataset_path.exists() and not args.resume:
            logger.warning("Removing existing dataset directory: %s", dataset_path)
            shutil.rmtree(dataset_path)
        elif dataset_path.exists() and args.resume:
            raise ValueError(f"Cannot resume from invalid dataset: {dataset_path}")

        dataset = LeRobotDataset.create(
            repo_id=str(dataset_path),
            fps=args.fps,
            features=dataset_features,
            robot_type=robot.name,
            use_videos=True,
            image_writer_threads=4,
        )
    ensure_episode_buffer(dataset)

    listener = None
    try:
        logger.info("Connecting robot...")
        robot.connect(calibrate=False)
        _init_rerun(session_name="franka_gripper_spacemouse_record")
        listener, events = init_recording_keyboard_listener()

        if not robot.is_connected:
            raise ValueError("Robot is not connected!")

        home_robot_and_gripper(robot)

        logger.info("Connecting teleoperation...")
        teleop.connect(calibrate=False)
        teleop.set_robot(robot)
        time.sleep(1.0)
        if hasattr(teleop, "reset_initial_pose"):
            teleop.reset_initial_pose()

        logger.info("Starting recording session...")
        print("\nKeyboard controls:")
        print("  r: start/stop writing frames into the current buffer")
        print("  n: save current buffer and move to the next episode")
        print("  x: clear current buffer")
        print("  Esc: stop recording session")
        log_say(f"Ready for episode {dataset.num_episodes + 1}. Press r to start recording.")

        run_record_loop(
            robot=robot,
            teleop=teleop,
            dataset=dataset,
            dataset_features=dataset_features,
            events=events,
            fps=args.fps,
            task=args.task,
        )
    finally:
        logger.info("Cleaning up...")
        if listener is not None:
            listener.stop()
        if teleop.is_connected:
            teleop.disconnect()
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
