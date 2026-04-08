import abc
from pathlib import Path
from typing import Any

import draccus

from lerobot.constants import HF_LEROBOT_CALIBRATION
from lerobot.motors import MotorCalibration

from .config import TeleoperatorConfig

TELEOPERATORS = "teleoperators"


class Teleoperator(abc.ABC):
    config_class: type[TeleoperatorConfig]
    name: str

    def __init__(self, config: TeleoperatorConfig):
        self.teleoperator_type = self.name
        self.id = config.id
        self.calibration_dir = (
            config.calibration_dir
            if config.calibration_dir
            else HF_LEROBOT_CALIBRATION / TELEOPERATORS / self.name
        )
        self.calibration_dir.mkdir(parents=True, exist_ok=True)
        calibration_name = self.id if self.id else "default"
        self.calibration_fpath = self.calibration_dir / f"{calibration_name}.json"
        self.calibration: dict[str, MotorCalibration] = {}
        if self.calibration_fpath.is_file():
            self._load_calibration()

    def __str__(self) -> str:
        prefix = f"{self.id} " if self.id else ""
        return f"{prefix}{self.__class__.__name__}"

    @property
    @abc.abstractmethod
    def action_features(self) -> dict:
        pass

    @property
    @abc.abstractmethod
    def feedback_features(self) -> dict:
        pass

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        pass

    @abc.abstractmethod
    def connect(self, calibrate: bool = True) -> None:
        pass

    @property
    @abc.abstractmethod
    def is_calibrated(self) -> bool:
        pass

    @abc.abstractmethod
    def calibrate(self) -> None:
        pass

    def _load_calibration(self, fpath: Path | None = None) -> None:
        fpath = self.calibration_fpath if fpath is None else fpath
        with open(fpath) as f, draccus.config_type("json"):
            self.calibration = draccus.load(dict[str, MotorCalibration], f)

    def _save_calibration(self, fpath: Path | None = None) -> None:
        fpath = self.calibration_fpath if fpath is None else fpath
        with open(fpath, "w") as f, draccus.config_type("json"):
            draccus.dump(self.calibration, f, indent=4)

    @abc.abstractmethod
    def configure(self) -> None:
        pass

    @abc.abstractmethod
    def get_action(self) -> dict[str, Any]:
        pass

    @abc.abstractmethod
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        pass

    def reset_initial_pose(self) -> bool:
        return False
