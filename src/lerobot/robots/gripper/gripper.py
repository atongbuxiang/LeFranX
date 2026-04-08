import logging
from functools import cached_property
from typing import Any


from ..robot import Robot
from .config_gripper import GripperConfig
from .zhixing_driver import ZhixingDriver

logger = logging.getLogger(__name__)


class Gripper(Robot):
    config_class = GripperConfig
    name = "gripper"

    def __init__(self, config: GripperConfig):
        super().__init__(config)
        self.config = config
        self.driver: ZhixingDriver | None = None
        self._is_connected = False

    @cached_property
    def observation_features(self) -> dict[str, type]:
        return {"gripper.pos": float}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {"gripper.pos": float}

    @property
    def is_connected(self) -> bool:
        return self._is_connected and self.driver is not None

    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        if self.is_connected:
            raise RuntimeError(f"{self} already connected")

        self.driver = ZhixingDriver(
            serial_dev=self.config.serial_port,
            baud=self.config.baud_rate,
            slave_id=self.config.slave_id,
            timeout_s=self.config.timeout_s,
        )
        self.driver.start()
        self._is_connected = True
        self.configure()
        logger.info("%s connected", self)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected")
        if self.config.default_force is not None:
            self.driver.set_force(int(self.config.default_force))

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected")
        return {"gripper.pos": float(self.driver.read_pos())}

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected")
        if "gripper.pos" not in action:
            raise ValueError("Missing action key 'gripper.pos'")

        target = float(action["gripper.pos"])
        self.driver.move_to(target)
        return {"gripper.pos": target}

    def disconnect(self) -> None:
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected")
        self.driver.stop()
        self.driver = None
        self._is_connected = False
        logger.info("%s disconnected", self)

    def reset_to_home(self) -> bool:
        if not self.is_connected:
            return False
        self.driver.move_to(self.config.home_position)
        return True

    def stop(self) -> bool:
        return self.is_connected
