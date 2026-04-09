from typing import Any

import numpy as np

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.sofranka import SoFranka, SoFrankaConfig

from .config_franka_fer_gripper_subarm import FrankaFERGripperSubarmTeleoperatorConfig
from ..franka_fer_subarm.franka_fer_subarm_teleoperator import FrankaFERSubarmTeleoperator


class FrankaFERGripperSubarmTeleoperator(Teleoperator):
    config_class = FrankaFERGripperSubarmTeleoperatorConfig
    name = "franka_fer_gripper_subarm"

    def __init__(self, config: FrankaFERGripperSubarmTeleoperatorConfig):
        super().__init__(config)
        self.config = config
        self.leader = SoFranka(
            SoFrankaConfig(
                port=config.arm_config.port,
                use_degrees=config.arm_config.use_degrees,
            )
        )
        self.arm_teleop = FrankaFERSubarmTeleoperator(config.arm_config, leader=self.leader)

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
        return self.leader.is_connected

    @property
    def is_calibrated(self) -> bool:
        return self.leader.is_calibrated

    def connect(self, calibrate: bool = True) -> None:
        self.leader.connect(calibrate=calibrate)

    def disconnect(self) -> None:
        if self.leader.is_connected:
            self.leader.disconnect()

    def calibrate(self) -> None:
        self.leader.calibrate()

    def configure(self) -> None:
        self.leader.configure()

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        del feedback
        return

    def get_action(self) -> dict[str, float]:
        arm_action = self.arm_teleop.get_action()
        action = {f"arm_joint_{i}.pos": arm_action[f"joint_{i}.pos"] for i in range(7)}

        raw_action = self.leader.get_action()
        g_raw = float(raw_action["gripper.pos"])
        c = self.config.gripper_raw_at_closed
        o = self.config.gripper_raw_at_open
        if abs(o - c) < 1e-9:
            g01 = g_raw / 100.0
        elif c <= o:
            g01 = (g_raw - c) / (o - c)
        else:
            # closed raw > open raw: map closed→0, open→1 with reversed span
            g01 = (c - g_raw) / (c - o)
        action["gripper.pos"] = float(np.clip(g01, 0.0, 1.0))
        return action
