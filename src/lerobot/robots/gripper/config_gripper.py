from dataclasses import dataclass

from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("gripper")
@dataclass
class GripperConfig(RobotConfig):
    serial_port: str = "/dev/ttyACM0"
    baud_rate: int = 115200
    slave_id: int = 1
    timeout_s: float = 1.0

    default_force: int = 30
    home_position: float = 1.0
