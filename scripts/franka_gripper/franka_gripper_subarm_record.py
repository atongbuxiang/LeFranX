#!/usr/bin/env python3

'''
  r: start/stop writing frames into the current buffer
  n: save current buffer and move to the next episode
  x: clear current buffer
  Esc: stop recording session
  Live preview: enabled (OpenCV window)
'''
"""Record Franka + gripper demos with subarm (SoFranka) teleop; uses subarm_cal.json like franka_gripper_subarm_teleoperator.py."""

import argparse
import json
import logging
import queue
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np
from common import (
    assert_dataset_features_compatible,
    build_recording_dataset_features,
    build_recording_observation_frame,
    build_robot_config,
    ensure_episode_buffer,
)

from lerobot.datasets.compute_stats import compute_episode_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import (
    build_dataset_frame,
    check_timestamps_sync,
    get_episode_data_index,
    validate_episode_buffer,
)
from lerobot.robots.franka_fer_gripper import FrankaFERGripper
from lerobot.teleoperators.franka_fer_gripper_subarm import (
    FrankaFERGripperSubarmTeleoperator,
    FrankaFERGripperSubarmTeleoperatorConfig,
)
from lerobot.teleoperators.franka_fer_subarm import FrankaFERSubarmTeleoperatorConfig
from lerobot.utils.control_utils import is_headless
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import _init_rerun

DEFAULT_FPS = 30
DEFAULT_TASK_DESCRIPTION = "Teleoperate Franka + gripper with subarm."
DEFAULT_DATASET_NAME = "pick_banana"
DEFAULT_DATASET_ROOT = Path("/mnt/data")
DEFAULT_CAMERAS = {
    "realsense_wrist": "241122305042",
    "realsense_topdown": "241122300571",
}

init_logging()
logger = logging.getLogger(__name__)

_DEFAULT_CAL = Path(__file__).resolve().parent / "subarm_cal.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"Root directory to store dataset folder. Default: {DEFAULT_DATASET_ROOT}",
    )
    parser.add_argument("--task", type=str, default=DEFAULT_TASK_DESCRIPTION)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing dataset directory when not resuming.",
    )
    parser.add_argument("--robot-ip", default="172.16.0.1")
    parser.add_argument("--robot-port", type=int, default=5000)
    parser.add_argument("--gripper-port", default="/dev/ttyUSB0")
    parser.add_argument("--gripper-baud", type=int, default=115200)
    parser.add_argument("--gripper-home", type=float, default=1.0)
    parser.add_argument("--leader-port", default="/dev/ttyACM0")
    parser.add_argument(
        "--leader-normalized",
        action="store_true",
        help="SoFranka joints [-100,100] (must match标定). Default: degrees (same as teleoperator / monitor).",
    )
    parser.add_argument(
        "--calibrate-leader",
        action="store_true",
        help="Run SoFranka motor calibration when connecting the leader.",
    )
    parser.add_argument(
        "--calibration-json",
        type=Path,
        default=_DEFAULT_CAL,
        help=f"Arm+gripper JSON (default: {_DEFAULT_CAL})",
    )
    parser.add_argument(
        "--camera",
        action="append",
        default=[],
        help=(
            "Repeatable RealSense camera config in the form 'camera_name=realsense_id'. "
            "When omitted, defaults to realsense_wrist=241122305042 and realsense_topdown=241122300571."
        ),
    )
    parser.add_argument("--camera-name", default="realsense_wrist")
    parser.add_argument("--realsense-id", default="")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--use-depth", action="store_true", default=False)
    parser.add_argument(
        "--hide-preview",
        action="store_true",
        help="Disable OpenCV live camera preview window during recording.",
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="Do not use RealSense; dataset is proprioception + action only (no images).",
    )
    return parser.parse_args()


def detect_first_realsense_serial() -> str:
    """Return first connected RealSense serial, or empty string if not found."""
    try:
        import pyrealsense2 as rs
    except Exception as exc:
        logger.debug("pyrealsense2 import failed: %s", exc)
        return ""

    try:
        ctx = rs.context()
        for dev in ctx.query_devices():
            serial = dev.get_info(rs.camera_info.serial_number)
            if serial:
                return str(serial)
    except Exception as exc:
        logger.warning("Failed to query RealSense devices: %s", exc)
    return ""


def build_subarm_teleop_from_calibration(
    cal_path: Path, *, leader_port: str, leader_normalized: bool
) -> FrankaFERGripperSubarmTeleoperator:
    if not cal_path.is_file():
        logger.error("Calibration file not found: %s", cal_path)
        sys.exit(1)
    with cal_path.open(encoding="utf-8") as f:
        cal = json.load(f)
    arm_cal = cal.get("arm", {})
    grip_cal = cal.get("gripper", {})
    leader_use_degrees = not leader_normalized
    arm_kwargs: dict = {
        "port": leader_port,
        "use_degrees": leader_use_degrees,
    }
    if "joint_scale" in arm_cal:
        arm_kwargs["joint_scale"] = tuple(float(x) for x in arm_cal["joint_scale"])
    if "joint_offset_rad" in arm_cal:
        arm_kwargs["joint_offset_rad"] = tuple(float(x) for x in arm_cal["joint_offset_rad"])
    top_kwargs: dict = {"arm_config": FrankaFERSubarmTeleoperatorConfig(**arm_kwargs)}
    if "gripper_raw_at_closed" in grip_cal:
        top_kwargs["gripper_raw_at_closed"] = float(grip_cal["gripper_raw_at_closed"])
    if "gripper_raw_at_open" in grip_cal:
        top_kwargs["gripper_raw_at_open"] = float(grip_cal["gripper_raw_at_open"])
    return FrankaFERGripperSubarmTeleoperator(FrankaFERGripperSubarmTeleoperatorConfig(**top_kwargs))


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


def clear_current_episode_buffer(dataset) -> None:
    episode_index = dataset.episode_buffer["episode_index"]

    if dataset.image_writer is not None:
        for cam_key in dataset.meta.camera_keys:
            img_dir = dataset._get_image_file_path(
                episode_index=episode_index, image_key=cam_key, frame_index=0
            ).parent
            if img_dir.is_dir():
                shutil.rmtree(img_dir)

    dataset.episode_buffer = dataset.create_episode_buffer(episode_index=episode_index)


def wait_for_episode_images(dataset, episode_buffer: dict, timeout_s: float = 120.0) -> None:
    if dataset.image_writer is None:
        return

    image_paths = []
    for cam_key in dataset.meta.camera_keys:
        image_paths.extend(Path(path) for path in episode_buffer.get(cam_key, []))

    if not image_paths:
        return

    deadline = time.monotonic() + timeout_s
    pending = set(image_paths)
    while pending:
        pending = {path for path in pending if not path.exists()}
        if not pending:
            return
        if time.monotonic() >= deadline:
            missing = ", ".join(str(path) for path in list(sorted(pending))[:3])
            raise TimeoutError(f"Timed out waiting for episode images to be written. Missing: {missing}")
        time.sleep(0.01)


def save_episode_buffer(dataset, episode_buffer: dict) -> int:
    validate_episode_buffer(episode_buffer, dataset.meta.total_episodes, dataset.features)

    episode_length = episode_buffer.pop("size")
    tasks = episode_buffer.pop("task")
    episode_tasks = list(set(tasks))
    episode_index = episode_buffer["episode_index"]

    episode_buffer["index"] = np.arange(dataset.meta.total_frames, dataset.meta.total_frames + episode_length)
    episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

    for task_name in episode_tasks:
        if dataset.meta.get_task_index(task_name) is None:
            dataset.meta.add_task(task_name)

    episode_buffer["task_index"] = np.array([dataset.meta.get_task_index(task_name) for task_name in tasks])

    for key, ft in dataset.features.items():
        if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["image", "video"]:
            continue
        episode_buffer[key] = np.stack(episode_buffer[key])

    dataset._save_episode_table(episode_buffer, episode_index)
    ep_stats = compute_episode_stats(episode_buffer, dataset.features)

    has_video_keys = len(dataset.meta.video_keys) > 0
    use_batched_encoding = dataset.batch_encoding_size > 1

    if has_video_keys and not use_batched_encoding:
        dataset.encode_episode_videos(episode_index)

    dataset.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats)

    if has_video_keys and use_batched_encoding:
        dataset.episodes_since_last_encoding += 1
        if dataset.episodes_since_last_encoding == dataset.batch_encoding_size:
            start_ep = dataset.num_episodes - dataset.batch_encoding_size
            end_ep = dataset.num_episodes
            logger.info(
                "Batch encoding %s videos for episodes %s to %s",
                dataset.batch_encoding_size,
                start_ep,
                end_ep - 1,
            )
            dataset.batch_encode_videos(start_ep, end_ep)
            dataset.episodes_since_last_encoding = 0

    ep_data_index = get_episode_data_index(dataset.meta.episodes, [episode_index])
    ep_data_index_np = {k: t.numpy() for k, t in ep_data_index.items()}
    check_timestamps_sync(
        episode_buffer["timestamp"],
        episode_buffer["episode_index"],
        ep_data_index_np,
        dataset.fps,
        dataset.tolerance_s,
    )

    parquet_files = list(dataset.root.rglob("*.parquet"))
    assert len(parquet_files) == dataset.num_episodes
    video_files = list(dataset.root.rglob("*.mp4"))
    assert len(video_files) == (dataset.num_episodes - dataset.episodes_since_last_encoding) * len(
        dataset.meta.video_keys
    )

    return episode_index


class AsyncEpisodeSaver:
    def __init__(self, dataset, dataset_path: Path):
        self.dataset = dataset
        self.dataset_path = dataset_path
        self._queue: queue.Queue[dict | None] = queue.Queue()
        self._pending = 0
        self._lock = threading.Lock()
        self._failure: Exception | None = None
        self._thread = threading.Thread(target=self._worker_loop, name="episode-saver", daemon=True)
        self._thread.start()

    def _worker_loop(self):
        while True:
            episode_buffer = self._queue.get()
            if episode_buffer is None:
                self._queue.task_done()
                return

            try:
                wait_for_episode_images(self.dataset, episode_buffer)
                episode_index = save_episode_buffer(self.dataset, episode_buffer)
                logger.info("Dataset saved under: %s", self.dataset_path.resolve())
                logger.info("Saved episode %s in background thread.", episode_index + 1)
            except Exception as exc:
                logger.exception("Failed to save episode in background thread.")
                with self._lock:
                    if self._failure is None:
                        self._failure = exc
            finally:
                with self._lock:
                    self._pending -= 1
                self._queue.task_done()

    def enqueue(self, episode_buffer: dict) -> int:
        episode_index = int(episode_buffer["episode_index"])
        with self._lock:
            if self._failure is not None:
                raise RuntimeError("Background episode saver has failed.") from self._failure
            self._pending += 1
        self._queue.put(episode_buffer)
        return episode_index

    def pending_count(self) -> int:
        with self._lock:
            return self._pending

    def raise_if_failed(self) -> None:
        with self._lock:
            failure = self._failure
        if failure is not None:
            raise RuntimeError("Background episode saver has failed.") from failure

    def close(self) -> None:
        self._queue.join()
        self.raise_if_failed()
        self._queue.put(None)
        self._thread.join()
        self.raise_if_failed()


def handle_control_events(dataset, events, dataset_path: Path, episode_saver: AsyncEpisodeSaver, state: dict):
    ensure_episode_buffer(dataset)
    episode_saver.raise_if_failed()

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
            clear_current_episode_buffer(dataset)
            log_say("Current buffer cleared.")

    if events["save_episode"]:
        events["save_episode"] = False
        if events["recording"]:
            events["recording"] = False
            log_say("Recording stopped before saving buffer.")
        if dataset.episode_buffer["size"] == 0:
            log_say("Buffer is empty, nothing to save.")
            return

        episode_buffer = dataset.episode_buffer
        saved_episode_index = episode_saver.enqueue(episode_buffer)
        state["current_episode_index"] = saved_episode_index + 1
        dataset.episode_buffer = dataset.create_episode_buffer(episode_index=state["current_episode_index"])
        pending = episode_saver.pending_count()
        log_say(
            f"Queued episode {saved_episode_index + 1} for saving. "
            f"Ready for episode {state['current_episode_index'] + 1}. Pending saves: {pending}."
        )
        logger.info("Episode %s queued for background save.", saved_episode_index + 1)


def extract_preview_frame(observation: dict) -> tuple[str, np.ndarray] | None:
    """Pick the first RGB-like image from observation for live preview."""
    for key, value in observation.items():
        if key.endswith("_depth"):
            continue
        if not isinstance(value, np.ndarray):
            continue
        if value.ndim != 3:
            continue
        if value.shape[2] not in (3, 4):
            continue
        frame = value[..., :3]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return key, frame
    return None


def show_preview_if_available(observation: dict, *, enabled: bool):
    if not enabled:
        return
    preview = extract_preview_frame(observation)
    if preview is None:
        return
    cam_name, frame = preview
    try:
        import cv2

        # OpenCV expects BGR ordering for correct colors.
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow(f"subarm_record preview: {cam_name}", bgr)
        cv2.waitKey(1)
    except Exception:
        return


def run_record_loop(
    robot,
    teleop,
    dataset,
    dataset_features,
    events,
    fps,
    task,
    dataset_path: Path,
    show_preview: bool,
    episode_saver: AsyncEpisodeSaver,
    state: dict,
):
    dt = 1.0 / fps

    while not events["stop_recording"]:
        loop_start = time.perf_counter()
        handle_control_events(dataset, events, dataset_path, episode_saver, state)

        action = teleop.get_action()
        performed_action = robot.send_action(action)
        observation = robot.get_observation()
        show_preview_if_available(observation, enabled=show_preview)

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
    if not args.no_camera and not args.camera and not args.realsense_id:
        args.camera = [f"{name}={serial}" for name, serial in DEFAULT_CAMERAS.items()]
        logger.info("Using default RealSense cameras: %s", args.camera)

    cal_path = args.calibration_json
    logger.info("Setting up Franka gripper subarm recording test...")
    logger.info("Calibration: %s", cal_path.resolve())
    logger.info("Leader: degrees=%s (--leader-normalized if标定为 [-100,100])", not args.leader_normalized)
    logger.info("Dataset: %s", args.dataset_name)
    logger.info("Task: %s", args.task)
    if args.no_camera:
        logger.info("Cameras: disabled (--no-camera)")
    else:
        logger.info("Camera specs: %s", args.camera if args.camera else [f"{args.camera_name}={args.realsense_id}"])

    robot = FrankaFERGripper(build_robot_config(args))
    teleop = build_subarm_teleop_from_calibration(
        cal_path,
        leader_port=args.leader_port,
        leader_normalized=args.leader_normalized,
    )

    logger.info("Setting up dataset...")
    dataset_features = build_recording_dataset_features(robot, use_videos=not args.no_camera)

    logger.info("Robot action features: %s", list(robot.action_features.keys()))
    logger.info("Robot observation features: %s", list(robot.observation_features.keys()))
    logger.info("Dataset features: %s", list(dataset_features.keys()))

    dataset_root = args.dataset_root if args.dataset_root is not None else (Path.cwd() / "data")
    dataset_path = dataset_root / args.dataset_name
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Dataset path: %s", dataset_path.resolve())
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
        if dataset_path.exists() and args.resume:
            raise ValueError(f"Cannot resume from invalid dataset: {dataset_path}")
        if dataset_path.exists() and args.overwrite:
            logger.warning("Removing existing dataset directory (--overwrite): %s", dataset_path)
            shutil.rmtree(dataset_path)
        elif dataset_path.exists() and not args.overwrite:
            raise ValueError(
                f"Dataset already exists: {dataset_path}. "
                "Use --resume to append episodes, or --overwrite to recreate from scratch."
            )

        dataset = LeRobotDataset.create(
            repo_id=str(dataset_path),
            fps=args.fps,
            features=dataset_features,
            robot_type=robot.name,
            use_videos=not args.no_camera,
            image_writer_threads=0 if args.no_camera else 4,
        )
    ensure_episode_buffer(dataset)

    listener = None
    episode_saver = AsyncEpisodeSaver(dataset, dataset_path)
    state = {"current_episode_index": dataset.num_episodes}
    try:
        logger.info("Connecting robot...")
        robot.connect(calibrate=False)
        _init_rerun(session_name="franka_gripper_subarm_record")
        listener, events = init_recording_keyboard_listener()

        if not robot.is_connected:
            raise ValueError("Robot is not connected!")

        #home_robot_and_gripper(robot)

        logger.info("Connecting teleoperation...")
        teleop.connect(calibrate=args.calibrate_leader)
        time.sleep(1.0)

        logger.info("Starting recording session...")
        print("\nKeyboard controls:")
        print("  r: start/stop writing frames into the current buffer")
        print("  n: save current buffer and move to the next episode")
        print("  x: clear current buffer")
        print("  Esc: stop recording session")
        if not args.no_camera and not args.hide_preview:
            print("  Live preview: enabled (OpenCV window)")
        log_say(f"Ready for episode {dataset.num_episodes + 1}. Press r to start recording.")

        run_record_loop(
            robot=robot,
            teleop=teleop,
            dataset=dataset,
            dataset_features=dataset_features,
            events=events,
            fps=args.fps,
            task=args.task,
            dataset_path=dataset_path,
            show_preview=(not args.no_camera and not args.hide_preview),
            episode_saver=episode_saver,
            state=state,
        )
    finally:
        logger.info("Cleaning up...")
        save_error = None
        try:
            episode_saver.close()
        except Exception as exc:
            save_error = exc
        try:
            import cv2

            cv2.destroyAllWindows()
        except Exception:
            pass
        if listener is not None:
            listener.stop()
        if teleop.is_connected:
            teleop.disconnect()
        if robot.is_connected:
            robot.disconnect()
        if save_error is not None:
            raise save_error


if __name__ == "__main__":
    main()
