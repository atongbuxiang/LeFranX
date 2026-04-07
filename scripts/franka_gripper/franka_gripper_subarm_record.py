#!/usr/bin/env python3

import argparse
import logging
import shutil
import time
from pathlib import Path

from common import build_robot_config

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.robots.franka_fer_gripper import FrankaFERGripper
from lerobot.teleoperators.franka_fer_gripper_subarm import (
    FrankaFERGripperSubarmTeleoperator,
    FrankaFERGripperSubarmTeleoperatorConfig,
)
from lerobot.teleoperators.franka_fer_subarm import FrankaFERSubarmTeleoperatorConfig
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import _init_rerun

DEFAULT_NUM_EPISODES = 100
DEFAULT_FPS = 30
DEFAULT_EPISODE_TIME_SEC = 60
DEFAULT_TASK_DESCRIPTION = "Teleoperate Franka + gripper with subarm."
DEFAULT_DATASET_NAME = "franka_gripper_subarm"

init_logging()
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Record data with Franka + gripper subarm teleoperation")
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--num-episodes", type=int, default=DEFAULT_NUM_EPISODES)
    parser.add_argument("--episode-time", type=float, default=DEFAULT_EPISODE_TIME_SEC)
    parser.add_argument("--task", type=str, default=DEFAULT_TASK_DESCRIPTION)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--robot-ip", default="192.168.18.1")
    parser.add_argument("--robot-port", type=int, default=5000)
    parser.add_argument("--gripper-port", default="/dev/ttyACM1")
    parser.add_argument("--gripper-baud", type=int, default=115200)
    parser.add_argument("--gripper-home", type=float, default=1.0)
    parser.add_argument("--leader-port", default="/dev/ttyACM0")
    parser.add_argument("--leader-use-degrees", action="store_true")
    parser.add_argument(
        "--camera",
        action="append",
        default=[],
        help="Repeatable RealSense camera config in the form 'camera_name=realsense_id'",
    )
    parser.add_argument("--camera-name", default="realsense")
    parser.add_argument("--realsense-id", default="")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--use-depth", action="store_true", default=True)
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


def run_record_loop(robot, teleop, dataset, dataset_features, events, fps, control_time_s, task):
    dt = 1.0 / fps
    start_time = time.perf_counter()
    frame_index = 0

    while time.perf_counter() - start_time < control_time_s and not events.get("exit_early", False):
        loop_start = time.perf_counter()

        action = teleop.get_action()
        performed_action = robot.send_action(action)
        observation = robot.get_observation()

        frame = {}
        frame.update(build_dataset_frame(dataset_features, performed_action, prefix="action"))
        frame.update(build_dataset_frame(dataset_features, observation, prefix="observation"))
        dataset.add_frame(frame, task=task, timestamp=frame_index / fps)

        frame_index += 1
        elapsed = time.perf_counter() - loop_start
        if elapsed < dt:
            time.sleep(dt - elapsed)


def main():
    args = parse_args()
    if not args.camera and not args.realsense_id:
        raise ValueError("Provide at least one camera via '--camera name=id' or '--realsense-id'.")

    logger.info("Setting up Franka gripper subarm recording test...")
    logger.info("Dataset: %s", args.dataset_name)
    logger.info("Episodes: %s", args.num_episodes)
    logger.info("Episode time: %ss", args.episode_time)
    logger.info("Task: %s", args.task)
    logger.info("Camera specs: %s", args.camera if args.camera else [f"{args.camera_name}={args.realsense_id}"])

    robot = FrankaFERGripper(build_robot_config(args))
    teleop = FrankaFERGripperSubarmTeleoperator(
        FrankaFERGripperSubarmTeleoperatorConfig(
            arm_config=FrankaFERSubarmTeleoperatorConfig(
                port=args.leader_port,
                use_degrees=args.leader_use_degrees,
            )
        )
    )

    logger.info("Setting up dataset...")
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    logger.info("Robot action features: %s", list(robot.action_features.keys()))
    logger.info("Robot observation features: %s", list(robot.observation_features.keys()))

    dataset_path = Path.cwd() / args.dataset_name
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

    listener = None
    try:
        logger.info("Connecting robot...")
        robot.connect(calibrate=False)
        _init_rerun(session_name="franka_gripper_subarm_record")
        listener, events = init_keyboard_listener()

        if not robot.is_connected:
            raise ValueError("Robot is not connected!")

        logger.info("Starting recording session...")
        total_episodes_to_record = args.num_episodes
        log_say(f"Recording {total_episodes_to_record} episodes (starting from episode {existing_episodes + 1})")
        log_say("Press ESC to stop recording, left arrow to re-record current episode")

        recorded_episodes = existing_episodes
        while recorded_episodes < total_episodes_to_record and not events.get("stop_recording", False):
            current_episode = recorded_episodes + 1
            print(f"\n=== EPISODE {current_episode} PREPARATION ===")
            print(f"1. Teleoperation status: {'connected' if teleop.is_connected else 'disconnected'}")
            if teleop.is_connected:
                print("   - Disconnecting teleoperation...")
                teleop.disconnect()

            print("2. Homing robot and gripper...")
            if hasattr(robot.arm, "reset_to_home"):
                print("   - Sending arm to home position...")
                success = robot.arm.reset_to_home()
                print("   - Arm homing initiated successfully" if success else "   - WARNING: Arm homing failed!")
                print("   - Waiting for arm to reach home position...")
                time.sleep(2.0)

            if hasattr(robot.gripper, "reset_to_home"):
                print("   - Sending gripper to home position...")
                success = robot.gripper.reset_to_home()
                print("   - Gripper homing initiated successfully" if success else "   - WARNING: Gripper homing failed!")
                time.sleep(1.0)

            print("   - Waiting for robot to stabilize at home position...")
            time.sleep(1.0)

            print("3. Ready for episode")
            print("=" * 60)
            print(f">>> Press ENTER to start recording episode {current_episode} <<<")
            print(">>> Or press Ctrl+C to stop recording <<<")
            print("=" * 60)
            input("Waiting for your confirmation: ")

            print("4. Connecting/reconnecting teleoperation...")
            if not teleop.is_connected:
                teleop.connect(calibrate=False)
                print("   - Waiting for subarm stream to stabilize...")
                time.sleep(1.0)

            log_say(f"Recording episode {current_episode} of {total_episodes_to_record}")
            events["exit_early"] = False
            run_record_loop(
                robot=robot,
                teleop=teleop,
                dataset=dataset,
                dataset_features=dataset_features,
                events=events,
                fps=args.fps,
                control_time_s=args.episode_time,
                task=args.task,
            )

            if events.get("rerecord_episode", False):
                log_say("Re-recording episode - will reset and restart")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                if teleop.is_connected:
                    teleop.disconnect()
                continue

            dataset.save_episode()
            recorded_episodes += 1
            logger.info("Episode %s saved successfully", recorded_episodes)

            if teleop.is_connected:
                teleop.disconnect()
                print(f"\n{'=' * 60}")
                print(f"Episode {recorded_episodes} COMPLETE - teleoperator disconnected")
                print(f"{'=' * 60}\n")

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
