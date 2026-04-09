from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("franka_fer_subarm")
@dataclass
class FrankaFERSubarmTeleoperatorConfig(TeleoperatorConfig):
    port: str = "/dev/ttyACM0"
    # True: SoFranka joint reads are ° (MotorNormMode.DEGREES) → teleop converts to rad for Franka.
    # False: reads are [-100,100] normalized (RANGE_M100_100); offsets must be calibrated in that unit.
    use_degrees: bool = True

    # Map leader joint readings to Franka command radians (single °→rad if use_degrees):
    #   leader_q = radians(leader[joint_{i+1}]) if use_degrees else raw leader
    #   franka_joint_i = joint_scale[i] * leader_q + joint_offset_rad[i]
    # Calibrate: offset_rad[i] = franka_rad[i] - joint_scale[i]*leader_q (same pose).
    joint_scale: tuple[float, ...] = field(default_factory=lambda: (1.0,) * 7)
    joint_offset_rad: tuple[float, ...] = field(default_factory=lambda: (0.0,) * 7)
