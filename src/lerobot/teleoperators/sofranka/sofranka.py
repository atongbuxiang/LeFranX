#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.

import logging
import time

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode

from ..teleoperator import Teleoperator
from .config_sofranka import SoFrankaConfig

logger = logging.getLogger(__name__)

class SoFranka(Teleoperator):
    """
    SoFranka Leader Arm - 基于 Franka FR3 结构的魔改版本。
    包含 7 个关节电机和 1 个夹爪电机。
    """

    config_class = SoFrankaConfig
    name = "sofranka"

    def __init__(self, config: SoFrankaConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        
        # 对应 Franka FR3 的 7 轴结构 + 夹爪
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "joint_1": Motor(1, "sts3215", norm_mode_body),
                "joint_2": Motor(2, "sts3215", norm_mode_body),
                "joint_3": Motor(3, "sts3215", norm_mode_body),
                "joint_4": Motor(4, "sts3215", norm_mode_body),
                "joint_5": Motor(5, "sts3215", norm_mode_body),
                "joint_6": Motor(6, "sts3215", norm_mode_body),
                "joint_7": Motor(7, "sts3215", norm_mode_body),
                "gripper": Motor(8, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise RuntimeError(f"{self} already connected")

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info("未检测到有效校准文件，准备开始校准...")
            self.calibrate()

        self.configure()
        logger.info(f"{self} (Franka Style) connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        if self.calibration:
            user_input = input(f"按下回车使用现有校准文件 ({self.id})，或输入 'c' 重新校准: ")
            if user_input.strip().lower() != "c":
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\n正在校准 SoFranka: 请确保机械臂处于自由活动状态")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(f"请将机械臂移动至【零位/中间位置】(Franka Home Pose) 然后按回车...")
        homing_offsets = self.bus.set_half_turn_homings()

        print("依次移动所有关节至其极限范围以记录行程。完成后按回车...")
        range_mins, range_maxes = self.bus.record_ranges_of_motion()

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print(f"校准已保存至 {self.calibration_fpath}")

    def configure(self) -> None:
        self.bus.disable_torque()
        self.bus.configure_motors()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    def setup_motors(self) -> None:
        """用于初始化 ID (1-8)"""
        for motor in reversed(self.bus.motors):
            input(f"请【仅连接】准备设为 '{motor}' 的电机，然后按回车...")
            self.bus.setup_motor(motor)
            print(f"电机 '{motor}' ID 已设置为 {self.bus.motors[motor].id}")

    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        action = self.bus.sync_read("Present_Position")
        action = {f"{motor}.pos": val for motor, val in action.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} 读取耗时: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError("SoFranka 暂不支持力反馈")

    def disconnect(self) -> None:
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected.")
        self.bus.disconnect()
        logger.info(f"{self} disconnected.")