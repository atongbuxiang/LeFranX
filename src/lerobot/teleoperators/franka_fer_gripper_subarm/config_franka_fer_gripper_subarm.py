from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.franka_fer_subarm.config_franka_fer_subarm import (
    FrankaFERSubarmTeleoperatorConfig,
)


@TeleoperatorConfig.register_subclass("franka_fer_gripper_subarm")
@dataclass
class FrankaFERGripperSubarmTeleoperatorConfig(TeleoperatorConfig):
    arm_config: FrankaFERSubarmTeleoperatorConfig = field(
        default_factory=lambda: FrankaFERSubarmTeleoperatorConfig(port="/dev/ttyACM0")
    )
