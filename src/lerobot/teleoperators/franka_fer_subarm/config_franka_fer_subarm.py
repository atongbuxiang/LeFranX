from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("franka_fer_subarm")
@dataclass
class FrankaFERSubarmTeleoperatorConfig(TeleoperatorConfig):
    port: str = "/dev/ttyACM0"
    use_degrees: bool = False
