from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.franka_fer_spacemouse.config_franka_fer_spacemouse import (
    FrankaFERSpaceMouseTeleoperatorConfig,
)


@TeleoperatorConfig.register_subclass("franka_fer_gripper_spacemouse")
@dataclass
class FrankaFERGripperSpaceMouseTeleoperatorConfig(TeleoperatorConfig):
    arm_config: FrankaFERSpaceMouseTeleoperatorConfig = field(
        default_factory=FrankaFERSpaceMouseTeleoperatorConfig
    )
    gripper_step: float = 0.03
    initial_gripper_pos: float = 1.0
    close_button_index: int = 0
    open_button_index: int = 1
