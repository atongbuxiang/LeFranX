#!/usr/bin/env python3

"""
Record a TUM RGB-D style scan dataset while teleoperating Franka with the SoFranka subarm.

Output layout:
  dataset/
    rgb.txt
    depth.txt
    groundtruth.txt
    camera_info.txt
    rgb/<timestamp>.png
    depth/<timestamp>.png

Keyboard:
  r: pause/resume writing frames
  Esc or q in the OpenCV window: stop
"""

import argparse
import copy
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from common import build_robot_config, ee_pose_flat_to_xyz_quat

from lerobot.robots.franka_fer import FrankaFER
from lerobot.teleoperators.franka_fer_subarm import (
    FrankaFERSubarmTeleoperator,
    FrankaFERSubarmTeleoperatorConfig,
)
from lerobot.utils.control_utils import is_headless
from lerobot.utils.utils import init_logging, log_say


DEFAULT_FPS = 10
DEFAULT_CAMERA_FPS = 30
DEFAULT_DATASET_PATH = Path("data/franka_fer_subarm_scan")
DEFAULT_ROBOT_IP = "172.16.0.1"
DEFAULT_ROBOT_PORT = 5000
DEFAULT_LEADER_PORT = "/dev/ttyACM0"
DEFAULT_CAMERA_NAME = "realsense_wrist"
DEFAULT_REALSENSE_ID = "241122305042"
DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480
DEFAULT_MAX_RELATIVE_TARGET_RAD = None
_DEFAULT_CAL = Path(__file__).resolve().parents[1] / "franka_gripper" / "subarm_cal.json"


init_logging()
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing TUM RGB-D dataset directory.",
    )
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument(
        "--camera-fps",
        type=int,
        default=DEFAULT_CAMERA_FPS,
        help="RealSense stream FPS. Dataset/control recording still uses --fps.",
    )
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--robot-port", type=int, default=DEFAULT_ROBOT_PORT)
    parser.add_argument("--leader-port", default=DEFAULT_LEADER_PORT)
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
    parser.add_argument("--camera-name", default=DEFAULT_CAMERA_NAME)
    parser.add_argument("--realsense-id", default=DEFAULT_REALSENSE_ID)
    parser.add_argument("--camera-width", type=int, default=DEFAULT_CAMERA_WIDTH)
    parser.add_argument("--camera-height", type=int, default=DEFAULT_CAMERA_HEIGHT)
    parser.add_argument(
        "--max-relative-target-rad",
        type=float,
        default=DEFAULT_MAX_RELATIVE_TARGET_RAD,
        help=(
            "Optional per-step joint target clamp in radians. Default is disabled so calibrated "
            "SoFranka absolute joint targets follow directly."
        ),
    )
    args = parser.parse_args()
    args.camera = [f"{args.camera_name}={args.realsense_id}"]
    args.use_depth = True
    return args


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
    arm_kwargs = {
        "port": leader_port,
        "use_degrees": leader_use_degrees,
    }
    if "joint_scale" in arm_cal:
        arm_kwargs["joint_scale"] = tuple(float(x) for x in arm_cal["joint_scale"])
    if "joint_offset_rad" in arm_cal:
        arm_kwargs["joint_offset_rad"] = tuple(float(x) for x in arm_cal["joint_offset_rad"])

    return FrankaFERSubarmTeleoperator(FrankaFERSubarmTeleoperatorConfig(**arm_kwargs))


def init_keyboard_listener() -> tuple[object | None, dict[str, bool]]:
    events = {
        "recording": True,
        "toggle_recording": False,
        "stop": False,
    }

    if is_headless():
        logger.warning("Headless environment detected. Use Ctrl-C to stop; keyboard hotkeys are unavailable.")
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
        if normalized == "r":
            events["toggle_recording"] = True
        elif key == keyboard.Key.esc:
            events["stop"] = True

    def on_release(key):
        pressed_keys.discard(normalize_key(key))

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    return listener, events


def prepare_dataset(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Dataset already exists: {path}. Use --overwrite to recreate it.")
        logger.warning("Removing existing dataset directory: %s", path)
        shutil.rmtree(path)

    (path / "rgb").mkdir(parents=True, exist_ok=True)
    (path / "depth").mkdir(parents=True, exist_ok=True)

    (path / "rgb.txt").write_text("# color images\n# timestamp filename\n", encoding="utf-8")
    (path / "depth.txt").write_text("# depth maps\n# timestamp filename\n", encoding="utf-8")
    (path / "groundtruth.txt").write_text(
        "# ground truth trajectory\n# timestamp tx ty tz qx qy qz qw\n",
        encoding="utf-8",
    )


def extract_rgbd(observation: dict, camera_name: str) -> tuple[np.ndarray, np.ndarray]:
    rgb = observation[camera_name]
    depth = observation[f"{camera_name}_depth"]

    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if depth.ndim == 3:
        depth = depth[..., 0]
    if depth.dtype != np.uint16:
        depth = np.clip(depth, 0, np.iinfo(np.uint16).max).astype(np.uint16)

    return rgb, depth


def extract_eepose(observation: dict) -> np.ndarray:
    flat = [observation[f"ee_pose.{i:02d}"] for i in range(16)]
    return ee_pose_flat_to_xyz_quat(flat)


def append_line(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(line)


def _format_intrinsics(prefix: str, intrinsics) -> list[str]:
    return [
        f"{prefix}.width {intrinsics.width}",
        f"{prefix}.height {intrinsics.height}",
        f"{prefix}.fx {intrinsics.fx:.10f}",
        f"{prefix}.fy {intrinsics.fy:.10f}",
        f"{prefix}.ppx {intrinsics.ppx:.10f}",
        f"{prefix}.ppy {intrinsics.ppy:.10f}",
        f"{prefix}.model {intrinsics.model}",
        f"{prefix}.coeffs {' '.join(f'{c:.10f}' for c in intrinsics.coeffs)}",
    ]


def write_camera_info(dataset_path: Path, robot: FrankaFER, args) -> None:
    """Write RealSense RGB/depth stream metadata and intrinsics in a simple txt file."""
    lines = [
        "# RealSense RGB-D camera parameters",
        f"created_unix_time {time.time():.6f}",
        f"camera.name {args.camera_name}",
        f"camera.serial {args.realsense_id}",
        f"record.fps {args.fps}",
        f"requested.camera_fps {args.camera_fps}",
        f"requested.width {args.camera_width}",
        f"requested.height {args.camera_height}",
        "rgb.file_format png",
        "rgb.dtype uint8",
        "rgb.color_order rgb",
        "depth.file_format png",
        "depth.dtype uint16",
        "depth.value_unit raw_realsense_z16",
    ]

    cam = robot.cameras.get(args.camera_name)
    if cam is None:
        lines.append(f"warning camera {args.camera_name!r} not found in robot.cameras")
    else:
        lines.extend(
            [
                f"actual.camera_fps {getattr(cam, 'fps', 'unknown')}",
                f"actual.width {getattr(cam, 'width', 'unknown')}",
                f"actual.height {getattr(cam, 'height', 'unknown')}",
                f"actual.capture_width {getattr(cam, 'capture_width', 'unknown')}",
                f"actual.capture_height {getattr(cam, 'capture_height', 'unknown')}",
                f"actual.color_mode {getattr(cam, 'color_mode', 'unknown')}",
                f"actual.use_depth {getattr(cam, 'use_depth', 'unknown')}",
            ]
        )

        try:
            import pyrealsense2 as rs

            profile = cam.rs_profile
            color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()

            lines.extend(
                [
                    f"rgb.stream_name {color_profile.stream_name()}",
                    f"rgb.format {color_profile.format().name}",
                    f"rgb.fps {color_profile.fps()}",
                    f"depth.stream_name {depth_profile.stream_name()}",
                    f"depth.format {depth_profile.format().name}",
                    f"depth.fps {depth_profile.fps()}",
                ]
            )
            lines.extend(_format_intrinsics("rgb.intrinsics", color_profile.get_intrinsics()))
            lines.extend(_format_intrinsics("depth.intrinsics", depth_profile.get_intrinsics()))

            device = profile.get_device()
            depth_sensor = device.first_depth_sensor()
            lines.append(f"depth.scale_m_per_unit {depth_sensor.get_depth_scale():.12f}")
        except Exception as exc:
            lines.append(f"warning failed_to_read_realsense_intrinsics {type(exc).__name__}: {exc}")

    (dataset_path / "camera_info.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Camera metadata written to: %s", (dataset_path / "camera_info.txt").resolve())


def save_frame(dataset_path: Path, timestamp: float, observation: dict, *, camera_name: str) -> None:
    import cv2

    rgb, depth = extract_rgbd(observation, camera_name)
    eepose = extract_eepose(observation)

    timestamp_text = f"{timestamp:.6f}"
    rgb_rel = f"rgb/{timestamp_text}.png"
    depth_rel = f"depth/{timestamp_text}.png"

    cv2.imwrite(str(dataset_path / rgb_rel), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(dataset_path / depth_rel), depth)

    append_line(dataset_path / "rgb.txt", f"{timestamp_text} {rgb_rel}\n")
    append_line(dataset_path / "depth.txt", f"{timestamp_text} {depth_rel}\n")
    append_line(
        dataset_path / "groundtruth.txt",
        (
            f"{timestamp_text} "
            f"{eepose[0]:.7f} {eepose[1]:.7f} {eepose[2]:.7f} "
            f"{eepose[3]:.7f} {eepose[4]:.7f} {eepose[5]:.7f} {eepose[6]:.7f}\n"
        ),
    )


def show_preview(rgb: np.ndarray, *, camera_name: str) -> bool:
    import cv2

    cv2.imshow(f"{camera_name} rgb", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    key = cv2.waitKey(1) & 0xFF
    return key not in (ord("q"), 27)


def run_loop(
    robot: FrankaFER,
    teleop: FrankaFERSubarmTeleoperator,
    dataset_path: Path,
    events: dict[str, bool],
    *,
    fps: int,
    camera_name: str,
) -> None:
    dt = 1.0 / fps
    frame_count = 0

    logger.info("Controls: r=pause/resume recording, Esc/q=stop")
    log_say("Recording TUM RGB-D frames. Press r to pause/resume.")

    while not events["stop"]:
        loop_start = time.perf_counter()

        if events["toggle_recording"]:
            events["toggle_recording"] = False
            events["recording"] = not events["recording"]
            state = "resumed" if events["recording"] else "paused"
            log_say(f"Recording {state}. Frames written: {frame_count}.")

        action = teleop.get_action()
        try:
            robot.send_action(action)
            observation = robot.get_observation()
        except (ConnectionError, RuntimeError) as exc:
            if not robot.is_connected:
                logger.error("Robot disconnected during recording loop: %s", exc)
                break
            raise

        rgb, _ = extract_rgbd(observation, camera_name)
        if not show_preview(rgb, camera_name=camera_name):
            break

        if events["recording"]:
            timestamp = time.time()
            save_frame(dataset_path, timestamp, observation, camera_name=camera_name)
            frame_count += 1

        elapsed = time.perf_counter() - loop_start
        if elapsed < dt:
            time.sleep(dt - elapsed)

    logger.info("Frames written: %s", frame_count)


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset_path
    cal_path = args.calibration_json

    logger.info("Dataset path: %s", dataset_path.resolve())
    logger.info("Robot: %s:%s", args.robot_ip, args.robot_port)
    logger.info("Leader: %s, calibration=%s", args.leader_port, cal_path)
    logger.info("Leader: degrees=%s (--leader-normalized if calibrated as [-100,100])", not args.leader_normalized)
    logger.info(
        "RealSense: %s=%s, %sx%s, RGB+D, camera=%s Hz, record=%s Hz",
        args.camera_name,
        args.realsense_id,
        args.camera_width,
        args.camera_height,
        args.camera_fps,
        args.fps,
    )

    prepare_dataset(dataset_path, overwrite=args.overwrite)
    robot_args = copy.copy(args)
    robot_args.fps = args.camera_fps
    robot = FrankaFER(build_robot_config(robot_args))
    teleop = build_subarm_teleop_from_calibration(
        cal_path,
        leader_port=args.leader_port,
        leader_normalized=args.leader_normalized,
    )
    listener = None

    try:
        robot.connect(calibrate=False)
        write_camera_info(dataset_path, robot, args)
        teleop.connect(calibrate=args.calibrate_leader)
        time.sleep(1.0)
        listener, events = init_keyboard_listener()
        run_loop(robot, teleop, dataset_path, events, fps=args.fps, camera_name=args.camera_name)
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    finally:
        try:
            import cv2

            cv2.destroyAllWindows()
        except Exception:
            pass
        if listener is not None:
            listener.stop()
        if teleop.is_connected:
            teleop.disconnect()
        try:
            robot.disconnect()
        except Exception as exc:
            logger.warning("Robot cleanup encountered an error: %s", exc)


if __name__ == "__main__":
    main()
