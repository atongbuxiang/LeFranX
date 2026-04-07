from dataclasses import dataclass
from typing import List, Optional

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("franka_fer_spacemouse")
@dataclass
class FrankaFERSpaceMouseTeleoperatorConfig(TeleoperatorConfig):
    translation_scale: float = 0.003
    rotation_scale: float = 0.035
    deadzone: float = 0.05
    smoothing_factor: float = 0.2

    manipulability_weight: float = 1.0
    neutral_distance_weight: float = 2.0
    current_distance_weight: float = 2.0
    joint_weights: Optional[List[float]] = None

    q7_min: float = -2.89
    q7_max: float = 2.89
    verbose: bool = False

    def __post_init__(self):
        if self.joint_weights is None:
            self.joint_weights = [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0]
