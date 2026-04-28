#!/usr/bin/env python3

'''
  r: start/stop writing frames into the current buffer
  s: start/stop automatic joint 7 scan
  n: save current buffer and move to the next episode
  x: clear current buffer
  Esc: stop recording session
  Live preview: enabled (OpenCV window)
'''
"""Record Franka arm-only RealSense wrist scans with SoFranka positioning."""

import argparse
import json
import logging
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from common import (
    assert_dataset_features_compatible,
    build_recording_dataset_features,
    build_recording_observation_frame,
    build_robot_config,
    ensure_episode_buffer,
)

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame
from lerobot.robots.franka_fer import FrankaFER
from lerobot.teleoperators.franka_fer_subarm import (
    FrankaFERSubarmTeleoperator,
    FrankaFERSubarmTeleoperatorConfig,
)
from lerobot.utils.control_utils import is_headless
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import _init_rerun

DEFAULT_FPS = 15
DEFAULT_TASK_DESCRIPTION = "Scan objects with a RealSense mounted on Franka joint 7."
DEFAULT_DATASET_NAME = "franka_fer_realsense_scan"
DEFAULT_DATASET_ROOT = Path("/mnt/data")
DEFAULT_CAMERAS = {
    "realsense_wrist": "241122305042",
}

# Conservative Franka/Panda joint 7 software limits. Override only if the robot setup permits it.
DEFAULT_JOINT7_MIN_DEG = -165.0
DEFAULT_JOINT7_MAX_DEG = 165.0

init_logging()
logger = logging.getLogger(__name__)

_DEFAULT_CAL = Path(__file__).resolve().parents[1] / "franka_gripper" / "subarm_cal.json"


@dataclass
class Joint7ScanController:
    scan_angle_rad: float
    scan_speed_rad_s: float
    joint_min_rad: float
    joint_max_rad: float

    running: bool = False
    has_latched_target: bool = False
    start_target_rad: float = 0.0
    target_rad: float = 0.0
    elapsed_s: float = 0.0
    last_update_s: float | None = None
    stop_message: str | None = None

    def toggle(self, current_joint7_rad: float | None) -> None:
        if self.running:
            self.running = False
            self.last_update_s = None
            self.has_latched_target = True
            self.stop_message = "Joint 7 scan stopped by user; holding current target."
            return

        if self.scan_angle_rad == 0.0 or self.scan_speed_rad_s == 0.0:
            self.stop_message = "Joint 7 scan not started because angle or speed is zero."
            return

        start = self.target_rad if self.has_latched_target else current_joint7_rad
        if start is None:
            start = 0.0
        start = self._clamp(float(start))

        self.running = True
        self.has_latched_target = True
        self.start_target_rad = start
        self.target_rad = start
        self.elapsed_s = 0.0
        self.last_update_s = None
        self.stop_message = (
            f"Joint 7 scan started at {math.degrees(start):.1f} deg "
            f"for {math.degrees(self.scan_angle_rad):.1f} deg."
        )

    def update(self, leader_joint7_rad: float, now_s: float) -> float:
        if not self.running:
            return self.target_rad if self.has_latched_target else float(leader_joint7_rad)

        if self.last_update_s is None:
            dt = 0.0
        else:
            dt = max(0.0, now_s - self.last_update_s)
        self.last_update_s = now_s

        direction = 1.0 if self.scan_angle_rad > 0.0 else -1.0
        self.elapsed_s += dt
        requested_delta = direction * abs(self.scan_speed_rad_s) * self.elapsed_s

        if abs(requested_delta) >= abs(self.scan_angle_rad):
            requested_delta = self.scan_angle_rad
            self.running = False
            self.last_update_s = None
            self.stop_message = "Joint 7 scan reached the requested angle; holding final target."

        requested_target = self.start_target_rad + requested_delta
        clamped_target = self._clamp(requested_target)
        if clamped_target != requested_target:
            self.running = False
            self.last_update_s = None
            self.stop_message = (
                "Joint 7 scan stopped at software limit "
                f"{math.degrees(clamped_target):.1f} deg; holding target."
            )

        self.target_rad = clamped_target
        return self.target_rad

    def consume_stop_message(self) -> str | None:
        message = self.stop_message
        self.stop_message = None
        return message

    def _clamp(self, value: float) -> float:
        return min(max(value, self.joint_min_rad), self.joint_max_rad)


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
    parser.add_argument("--leader-port", default="/dev/ttyACM0")
    parser.add_argument(
        "--leader-normalized",
        action="store_true",
        help="SoFranka joints [-100,100] (must match calibration). Default: degrees.",
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
        help=f"Arm calibration JSON (default: {_DEFAULT_CAL})",
    )
    parser.add_argument(
        "--camera",
        action="append",
        default=[],
        help=(
            "Repeatable RealSense camera config in the form 'camera_name=realsense_id'. "
            "When omitted, defaults to realsense_wrist=241122305042."
        ),
    )
    parser.add_argument("--camera-name", default="realsense_wrist")
    parser.add_argument("--realsense-id", default="")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--use-depth", action="store_true", default=False)
    parser.add_argument(
        "--scan-speed-deg-s",
        type=float,
        default=25.0,
        help="Automatic joint 7 scan speed in degrees/s. Sign is ignored; use --scan-angle-deg for direction.",
    )
    parser.add_argument(
        "--scan-angle-deg",
        type=float,
        default=360.0,
        help="Requested joint 7 scan angle in degrees. Use a negative value to scan in the opposite direction.",
    )
    parser.add_argument(
        "--joint7-min-deg",
        type=float,
        default=DEFAULT_JOINT7_MIN_DEG,
        help=f"Joint 7 software minimum in degrees. Default: {DEFAULT_JOINT7_MIN_DEG}",
    )
    parser.add_argument(
        "--joint7-max-deg",
        type=float,
        default=DEFAULT_JOINT7_MAX_DEG,
        help=f"Joint 7 software maximum in degrees. Default: {DEFAULT_JOINT7_MAX_DEG}",
    )
    return parser.parse_args()


def build_subarm_teleop_from_calibration(
    cal_path: Path, *, leader_port: str, leader_normalized: bool
) -> FrankaFERSubarmTeleoperator:
    if not cal_path.is_file():
        logger.error("Calibration file not found: %s", cal_path)
        sys.exit(1)

    with cal_path.open(encoding="utf-8") as f:
        cal = json.load(f)

    arm_cal = cal.get("arm", cal)
    leader_use_degrees = not leader_normalized
    arm_kwargs: dict = {
        "port": leader_port,
        "use_degrees": leader_use_degrees,
    }
    if "joint_scale" in arm_cal:
        arm_kwargs["joint_scale"] = tuple(float(x) for x in arm_cal["joint_scale"])
    if "joint_offset_rad" in arm_cal:
        arm_kwargs["joint_offset_rad"] = tuple(float(x) for x in arm_cal["joint_offset_rad"])

    return FrankaFERSubarmTeleoperator(FrankaFERSubarmTeleoperatorConfig(**arm_kwargs))


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
        "toggle_scan": False,
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
            elif normalized == "s":
                events["toggle_scan"] = True
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


def handle_control_events(dataset, events, dataset_path: Path):
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
        logger.info("Dataset saved under: %s", dataset_path.resolve())


def extract_preview_frame(observation: dict) -> tuple[str, np.ndarray] | None:
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


def show_preview_if_available(observation: dict):
    preview = extract_preview_frame(observation)
    if preview is None:
        return
    cam_name, frame = preview
    try:
        import cv2

        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow(f"franka_fer_realsense_scan preview: {cam_name}", bgr)
        cv2.waitKey(1)
    except Exception:
        return


def handle_scan_toggle(events, scan_controller: Joint7ScanController, current_joint7_rad: float | None) -> None:
    if not events["toggle_scan"]:
        return
    events["toggle_scan"] = False
    scan_controller.toggle(current_joint7_rad)
    message = scan_controller.consume_stop_message()
    if message:
        log_say(message)


def run_record_loop(
    robot,
    teleop,
    dataset,
    dataset_features,
    events,
    fps,
    task,
    dataset_path: Path,
    scan_controller: Joint7ScanController,
):
    dt = 1.0 / fps
    last_robot_joint7_rad = None

    try:
        initial_observation = robot.get_observation()
        last_robot_joint7_rad = float(initial_observation["joint_6.pos"])
        show_preview_if_available(initial_observation)
    except Exception as exc:
        logger.warning("Could not read initial joint 7 observation before loop: %s", exc)

    while not events["stop_recording"]:
        loop_start = time.perf_counter()
        handle_control_events(dataset, events, dataset_path)
        handle_scan_toggle(events, scan_controller, last_robot_joint7_rad)

        action = teleop.get_action()
        action["joint_6.pos"] = scan_controller.update(float(action["joint_6.pos"]), loop_start)
        message = scan_controller.consume_stop_message()
        if message:
            log_say(message)

        performed_action = robot.send_action(action)
        observation = robot.get_observation()
        if "joint_6.pos" in observation:
            last_robot_joint7_rad = float(observation["joint_6.pos"])
        show_preview_if_available(observation)

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
    if args.joint7_min_deg >= args.joint7_max_deg:
        raise ValueError("--joint7-min-deg must be smaller than --joint7-max-deg")

    if not args.camera and not args.realsense_id:
        args.camera = [f"{name}={serial}" for name, serial in DEFAULT_CAMERAS.items()]
        logger.info("Using default RealSense cameras: %s", args.camera)

    cal_path = args.calibration_json
    logger.info("Setting up Franka RealSense wrist scan recording...")
    logger.info("Calibration: %s", cal_path.resolve())
    logger.info("Leader: degrees=%s (--leader-normalized if calibrated as [-100,100])", not args.leader_normalized)
    logger.info("Dataset: %s", args.dataset_name)
    logger.info("Task: %s", args.task)
    logger.info("Camera specs: %s", args.camera if args.camera else [f"{args.camera_name}={args.realsense_id}"])
    logger.info("Unified recording fps (control + camera + dataset): %s", args.fps)
    logger.info(
        "Joint 7 scan: angle=%.1f deg, speed=%.1f deg/s, limits=[%.1f, %.1f] deg",
        args.scan_angle_deg,
        abs(args.scan_speed_deg_s),
        args.joint7_min_deg,
        args.joint7_max_deg,
    )

    robot = FrankaFER(build_robot_config(args))
    teleop = build_subarm_teleop_from_calibration(
        cal_path,
        leader_port=args.leader_port,
        leader_normalized=args.leader_normalized,
    )
    scan_controller = Joint7ScanController(
        scan_angle_rad=math.radians(args.scan_angle_deg),
        scan_speed_rad_s=math.radians(abs(args.scan_speed_deg_s)),
        joint_min_rad=math.radians(args.joint7_min_deg),
        joint_max_rad=math.radians(args.joint7_max_deg),
    )

    logger.info("Setting up dataset...")
    dataset_features = build_recording_dataset_features(robot, use_videos=True)

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
        if dataset_path.exists() and args.resume and not args.overwrite:
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
            use_videos=True,
            image_writer_threads=4,
        )
    ensure_episode_buffer(dataset)

    listener = None
    try:
        logger.info("Connecting robot...")
        robot.connect(calibrate=False)
        _init_rerun(session_name="franka_fer_realsense_scan_record")
        listener, events = init_recording_keyboard_listener()

        if not robot.is_connected:
            raise ValueError("Robot is not connected!")

        logger.info("Connecting teleoperation...")
        teleop.connect(calibrate=args.calibrate_leader)
        time.sleep(1.0)

        logger.info("Starting recording session...")
        print("\nKeyboard controls:")
        print("  r: start/stop writing frames into the current buffer")
        print("  s: start/stop automatic joint 7 scan")
        print("  n: save current buffer and move to the next episode")
        print("  x: clear current buffer")
        print("  Esc: stop recording session")
        print("  Live preview: enabled (OpenCV window)")
        print(
            "  Joint 7 scan: "
            f"{args.scan_angle_deg:.1f} deg at {abs(args.scan_speed_deg_s):.1f} deg/s, "
            f"limits [{args.joint7_min_deg:.1f}, {args.joint7_max_deg:.1f}] deg"
        )
        log_say(f"Ready for episode {dataset.num_episodes + 1}. Press r to start recording, s to scan.")

        run_record_loop(
            robot=robot,
            teleop=teleop,
            dataset=dataset,
            dataset_features=dataset_features,
            events=events,
            fps=args.fps,
            task=args.task,
            dataset_path=dataset_path,
            scan_controller=scan_controller,
        )
    finally:
        logger.info("Cleaning up...")
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


if __name__ == "__main__":
    main()
