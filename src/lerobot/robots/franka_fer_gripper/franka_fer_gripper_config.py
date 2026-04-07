from dataclasses import dataclass, field
from typing import Dict

from lerobot.cameras.utils import CameraConfig
from lerobot.robots.config import RobotConfig
from lerobot.robots.franka_fer.franka_fer_config import FrankaFERConfig
from lerobot.robots.gripper.config_gripper import GripperConfig


@RobotConfig.register_subclass("franka_fer_gripper")
@dataclass
class FrankaFERGripperConfig(RobotConfig):
    arm_config: FrankaFERConfig = field(default_factory=lambda: FrankaFERConfig(cameras={}))
    gripper_config: GripperConfig = field(default_factory=GripperConfig)

    cameras: Dict[str, CameraConfig] = field(default_factory=dict)

    synchronize_actions: bool = True
    action_timeout: float = 0.1
    emergency_stop_both: bool = True

    def __post_init__(self):
        super().__post_init__()

        if not isinstance(self.arm_config, FrankaFERConfig):
            raise TypeError("arm_config must be a FrankaFERConfig")
        if not isinstance(self.gripper_config, GripperConfig):
            raise TypeError("gripper_config must be a GripperConfig")
