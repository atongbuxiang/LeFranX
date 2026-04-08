import logging
from typing import Any, Dict

import numpy as np

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.franka_fer_spacemouse.franka_fer_spacemouse_teleoperator import (
    FrankaFERSpaceMouseTeleoperator,
)
from lerobot.teleoperators.franka_fer_spacemouse.spacemouse_reader import SpaceMouseStateReader

from .config_franka_fer_gripper_spacemouse import FrankaFERGripperSpaceMouseTeleoperatorConfig

logger = logging.getLogger(__name__)


class FrankaFERGripperSpaceMouseTeleoperator(Teleoperator):
    config_class = FrankaFERGripperSpaceMouseTeleoperatorConfig
    name = "franka_fer_gripper_spacemouse"

    def __init__(self, config: FrankaFERGripperSpaceMouseTeleoperatorConfig):
        super().__init__(config)
        self.config = config
        self._reader = SpaceMouseStateReader()
        self.arm_teleop = FrankaFERSpaceMouseTeleoperator(config.arm_config, reader=self._reader)
        self._is_connected = False
        self._robot_reference = None
        self._gripper_pos = float(np.clip(config.initial_gripper_pos, 0.0, 1.0))
        self._last_sent_gripper_pos: float | None = None

    @property
    def action_features(self) -> dict[str, type]:
        features = {f"arm_joint_{i}.pos": float for i in range(7)}
        features["gripper.pos"] = float
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected and self.arm_teleop.is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        self._reader.start()
        self.arm_teleop.connect(calibrate=calibrate)
        self._is_connected = True

    def disconnect(self) -> None:
        self.arm_teleop.disconnect()
        self._reader.stop()
        self._robot_reference = None
        self._is_connected = False
        self._last_sent_gripper_pos = None

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    def send_feedback(self, feedback: Dict[str, Any]) -> None:
        del feedback
        return

    def set_robot(self, robot):
        self._robot_reference = robot
        if hasattr(robot, "arm"):
            self.arm_teleop.set_robot(robot.arm)
        else:
            self.arm_teleop.set_robot(robot)
        if hasattr(robot, "get_observation"):
            try:
                obs = robot.get_observation()
                if "gripper.pos" in obs:
                    self._gripper_pos = float(obs["gripper.pos"])
            except Exception:
                pass
        self._last_sent_gripper_pos = None

    def get_action(self) -> Dict[str, Any]:
        if not self.is_connected:
            raise RuntimeError("FrankaFERGripperSpaceMouseTeleoperator is not connected")

        arm_action = self.arm_teleop.get_action()
        action = {f"arm_{key}": value for key, value in arm_action.items()}

        state = self._reader.get_state()
        if state is not None:
            buttons = state.get("buttons", [])
            close_pressed = self._button_pressed(buttons, self.config.close_button_index)
            open_pressed = self._button_pressed(buttons, self.config.open_button_index)
            if close_pressed != open_pressed:
                self._gripper_pos = 0.0 if close_pressed else 1.0

        if self._gripper_pos != self._last_sent_gripper_pos:
            action["gripper.pos"] = self._gripper_pos
            self._last_sent_gripper_pos = self._gripper_pos
        return action

    def reset_initial_pose(self) -> bool:
        return self.arm_teleop.reset_initial_pose()

    @staticmethod
    def _button_pressed(buttons: list[Any], index: int) -> bool:
        return index < len(buttons) and bool(buttons[index])
