import logging
import time
from functools import cached_property
from typing import Any, Dict

from lerobot.cameras.realsense import RealSenseCamera
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.robots.franka_fer import FrankaFER
from lerobot.robots.gripper import Gripper
from lerobot.robots.robot import Robot

from .franka_fer_gripper_config import FrankaFERGripperConfig

logger = logging.getLogger(__name__)


class FrankaFERGripper(Robot):
    config_class = FrankaFERGripperConfig
    name = "franka_fer_gripper"

    def __init__(self, config: FrankaFERGripperConfig):
        super().__init__(config)
        self.config = config
        self.arm = FrankaFER(config.arm_config)
        self.gripper = Gripper(config.gripper_config)
        self.cameras = make_cameras_from_configs(config.cameras)
        self._is_connected = False

    @cached_property
    def observation_features(self) -> Dict[str, type | tuple]:
        features: Dict[str, type | tuple] = {}

        for key, value in self.arm.observation_features.items():
            if not key.startswith(("camera", "cam")):
                features[f"arm_{key}"] = value

        for key, value in self.gripper.observation_features.items():
            features[key] = value

        for cam_name, cam_config in self.config.cameras.items():
            features[cam_name] = (cam_config.height, cam_config.width, 3)
            if getattr(cam_config, "use_depth", False):
                features[f"{cam_name}_depth"] = (cam_config.height, cam_config.width, 1)

        return features

    @cached_property
    def action_features(self) -> Dict[str, type]:
        features = {}

        for key, value in self.arm.action_features.items():
            features[f"arm_{key}"] = value

        for key, value in self.gripper.action_features.items():
            features[key] = value

        return features

    @property
    def is_connected(self) -> bool:
        return self._is_connected and self.arm.is_connected and self.gripper.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        try:
            self.arm.connect(calibrate=calibrate)
            self.gripper.connect(calibrate=calibrate)
            for cam in self.cameras.values():
                cam.connect()
            self._is_connected = True
            logger.info("%s connected successfully", self)
        except Exception as exc:
            logger.error("Failed to connect %s: %s", self, exc)
            try:
                if self.arm.is_connected:
                    self.arm.disconnect()
                if self.gripper.is_connected:
                    self.gripper.disconnect()
                for cam in self.cameras.values():
                    if cam.is_connected:
                        cam.disconnect()
            except Exception:
                pass
            raise ConnectionError(f"Failed to connect {self.name}: {exc}") from exc

    @property
    def is_calibrated(self) -> bool:
        return self.arm.is_calibrated and self.gripper.is_calibrated

    def configure(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

    def calibrate(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        self.arm.calibrate()
        self.gripper.calibrate()

    def get_observation(self) -> Dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        obs_dict = {}

        start = time.perf_counter()
        arm_obs = self.arm.get_observation()
        arm_time = time.perf_counter() - start
        for key, value in arm_obs.items():
            if not key.startswith(("camera", "cam")):
                obs_dict[f"arm_{key}"] = value

        start = time.perf_counter()
        gripper_obs = self.gripper.get_observation()
        gripper_time = time.perf_counter() - start
        obs_dict.update(gripper_obs)

        start = time.perf_counter()
        for cam_name, cam in self.cameras.items():
            obs_dict[cam_name] = cam.read()
            if isinstance(cam, RealSenseCamera) and getattr(cam.config, "use_depth", False):
                depth = cam.read_depth()
                if depth.ndim == 2:
                    depth = depth[..., None]
                obs_dict[f"{cam_name}_depth"] = depth
        cam_time = time.perf_counter() - start

        logger.debug(
            "Franka arm obs: %.1fms, gripper obs: %.1fms, cameras: %.1fms",
            arm_time * 1000,
            gripper_time * 1000,
            cam_time * 1000,
        )
        return obs_dict

    def send_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        arm_action = {}
        gripper_action = {}

        for key, value in action.items():
            if key.startswith("arm_"):
                arm_action[key[4:]] = value
            elif key in self.gripper.action_features:
                gripper_action[key] = value
            else:
                logger.warning("Unknown action key: %s", key)

        performed_action = {}

        try:
            if arm_action:
                arm_result = self.arm.send_action(arm_action)
                for key, value in arm_result.items():
                    performed_action[f"arm_{key}"] = value

            if gripper_action:
                gripper_result = self.gripper.send_action(gripper_action)
                performed_action.update(gripper_result)
        except Exception as exc:
            logger.error("Failed to send composite action: %s", exc)
            if self.config.emergency_stop_both:
                self.stop()
            raise

        return performed_action

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        try:
            self.arm.disconnect()
        finally:
            try:
                self.gripper.disconnect()
            finally:
                for cam in self.cameras.values():
                    try:
                        cam.disconnect()
                    except Exception:
                        pass
                self._is_connected = False

    def reset_to_home(self) -> bool:
        if not self.is_connected:
            return False
        arm_success = self.arm.reset_to_home()
        gripper_success = self.gripper.reset_to_home()
        return arm_success and gripper_success

    def stop(self) -> bool:
        arm_success = self.arm.stop() if self.arm.is_connected else True
        gripper_success = self.gripper.stop() if self.gripper.is_connected else True
        return arm_success and gripper_success
