import math
from typing import Any

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.sofranka import SoFranka, SoFrankaConfig

from .config_franka_fer_subarm import FrankaFERSubarmTeleoperatorConfig


class FrankaFERSubarmTeleoperator(Teleoperator):
    config_class = FrankaFERSubarmTeleoperatorConfig
    name = "franka_fer_subarm"

    def __init__(self, config: FrankaFERSubarmTeleoperatorConfig, leader: SoFranka | None = None):
        super().__init__(config)
        self.config = config
        self.leader = leader or SoFranka(SoFrankaConfig(port=config.port, use_degrees=config.use_degrees))
        self._owns_leader = leader is None

    @property
    def action_features(self) -> dict[str, type]:
        return {f"joint_{i}.pos": float for i in range(7)}

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
        if self._owns_leader:
            self.leader.connect(calibrate=calibrate)

    def disconnect(self) -> None:
        if self._owns_leader and self.leader.is_connected:
            self.leader.disconnect()

    def calibrate(self) -> None:
        if self._owns_leader:
            self.leader.calibrate()

    def configure(self) -> None:
        if self._owns_leader:
            self.leader.configure()

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        del feedback
        return

    def get_action(self) -> dict[str, float]:
        raw_action = self.leader.get_action()
        scale = self.config.joint_scale
        offset = self.config.joint_offset_rad

        def leader_scalar_for_franka(raw: float) -> float:
            # With use_degrees, bus reports °; joint_offset_rad is paired with leader in rad.
            return math.radians(raw) if self.config.use_degrees else raw

        return {
            f"joint_{i}.pos": float(scale[i]) * leader_scalar_for_franka(float(raw_action[f"joint_{i + 1}.pos"]))
            + float(offset[i])
            for i in range(7)
        }
