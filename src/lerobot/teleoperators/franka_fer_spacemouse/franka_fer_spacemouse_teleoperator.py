import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from lerobot.teleoperators.teleoperator import Teleoperator

from .config_franka_fer_spacemouse import FrankaFERSpaceMouseTeleoperatorConfig
from .spacemouse_reader import SpaceMouseStateReader

logger = logging.getLogger(__name__)


def _normalize_quaternion_xyzw(quaternion: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quaternion)
    if norm <= 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return quaternion / norm


def _quaternion_multiply_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=float,
    )


@dataclass(slots=True)
class DeltaPoseCommand:
    translation: list[float]
    quaternion: list[float]


@dataclass(slots=True)
class WristPoseInput:
    position: list[float]
    quaternion: list[float]
    fist_state: str = "open"
    valid: bool = True


class DeltaPoseAdapter:
    def __init__(
        self,
        position_scale: float = 1.0,
        position_deadzone: float = 0.0,
        orientation_deadzone: float = 0.0,
    ) -> None:
        self.position_scale = float(position_scale)
        self.position_deadzone = float(position_deadzone)
        self.orientation_deadzone = float(orientation_deadzone)
        self.reset()

    def reset(self) -> None:
        self._cumulative_position = np.zeros(3, dtype=float)
        self._cumulative_quaternion = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        self._last_sequence: int | None = None
        self._cached_wrist_pose = WristPoseInput(
            position=self._cumulative_position.tolist(),
            quaternion=self._cumulative_quaternion.tolist(),
            valid=True,
        )

    def to_wrist_data(self, command: DeltaPoseCommand, sequence: int) -> WristPoseInput:
        if self._last_sequence != sequence:
            delta_translation = np.array(command.translation, dtype=float) * self.position_scale
            if np.linalg.norm(delta_translation) < self.position_deadzone:
                delta_translation = np.zeros(3, dtype=float)

            delta_quaternion = _normalize_quaternion_xyzw(np.array(command.quaternion, dtype=float))
            rotation_angle = 2.0 * math.acos(np.clip(delta_quaternion[3], -1.0, 1.0))
            if rotation_angle < self.orientation_deadzone:
                delta_quaternion = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

            self._cumulative_position = self._cumulative_position + delta_translation
            self._cumulative_quaternion = _normalize_quaternion_xyzw(
                _quaternion_multiply_xyzw(self._cumulative_quaternion, delta_quaternion)
            )
            self._cached_wrist_pose = WristPoseInput(
                position=self._cumulative_position.tolist(),
                quaternion=self._cumulative_quaternion.tolist(),
                valid=True,
            )
            self._last_sequence = sequence

        return self._cached_wrist_pose


class FrankaFERSpaceMouseTeleoperator(Teleoperator):
    config_class = FrankaFERSpaceMouseTeleoperatorConfig
    name = "franka_fer_spacemouse"

    def __init__(
        self,
        config: FrankaFERSpaceMouseTeleoperatorConfig,
        reader: SpaceMouseStateReader | None = None,
    ):
        super().__init__(config)
        self.config = config
        self._reader = reader or SpaceMouseStateReader()
        self._owns_reader = reader is None
        self._robot_reference = None
        self._initialized = False
        self._is_connected = False
        self._last_action = None
        self._delta_pose_adapter = DeltaPoseAdapter()

        from ..franka_fer_vr.arm_ik_processor import ArmIKProcessor

        self.arm_ik_processor = ArmIKProcessor(
            {
                "verbose": config.verbose,
                "smoothing_factor": config.smoothing_factor,
            }
        )

    @property
    def action_features(self) -> dict[str, type]:
        return {f"joint_{i}.pos": float for i in range(7)}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        if self._owns_reader:
            self._reader.start()
        self._is_connected = True

    def disconnect(self) -> None:
        if self._owns_reader:
            self._reader.stop()
        self._robot_reference = None
        self._initialized = False
        self._is_connected = False
        self._delta_pose_adapter.reset()
        self._reset_ik_state()

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    def send_feedback(self, feedback: Dict[str, Any]) -> None:
        del feedback
        return

    def set_robot(self, robot):
        self._robot_reference = robot

    def get_action(self) -> Tuple[Dict[str, Any], List[bool]]:
        if not self._is_connected:
            raise RuntimeError("FrankaFERSpaceMouseTeleoperator is not connected")
        if self._robot_reference is None:
            return {f"joint_{i}.pos": 0.0 for i in range(7)}

        current_obs = self._robot_reference.get_observation()
        current_joints = [current_obs[f"joint_{i}.pos"] for i in range(7)]

        if not self._initialized and not self._initialize_ik_solver(current_obs, current_joints):
            return {f"joint_{i}.pos": float(current_joints[i]) for i in range(7)}

        state = self._reader.get_state()
        if state is None:
            return {f"joint_{i}.pos": float(current_joints[i]) for i in range(7)}

        translation = self._apply_deadzone(
            np.array([state["x"], state["z"], state["y"]], dtype=float)
        ) * 0.2 * 0.02
        rotation = self._apply_deadzone(
            np.array([state["pitch"], state["yaw"], -state["roll"]], dtype=float)
        ) * 0.5 * 0.02

        try:
            command = DeltaPoseCommand(
                translation=translation.tolist(),
                quaternion=self._euler_xyz_to_quaternion_xyzw(rotation).tolist(),
            )
            wrist_data = self._delta_pose_adapter.to_wrist_data(command, int(state.get("sequence", 0)))
            arm_action = self.arm_ik_processor.process_wrist_data(wrist_data, current_joints)
            action = {f"joint_{i}.pos": float(arm_action[f"arm_joint_{i}.pos"]) for i in range(7)}
            self._last_action = action
            return action
        except Exception as exc:
            logger.error("SpaceMouse IK failed: %s", exc)

        return {f"joint_{i}.pos": float(current_joints[i]) for i in range(7)}

    def reset_initial_pose(self) -> bool:
        if not self._is_connected or self._robot_reference is None:
            return False
        current_obs = self._robot_reference.get_observation()
        current_joints = [current_obs[f"joint_{i}.pos"] for i in range(7)]
        return self._initialize_ik_solver(current_obs, current_joints)

    def _initialize_ik_solver(self, current_obs: dict[str, Any], current_joints: list[float]) -> bool:
        try:
            ee_pose = [current_obs[f"ee_pose.{i:02d}"] for i in range(16)]
            success = self.arm_ik_processor.setup(
                neutral_pose=current_joints,
                initial_robot_pose=ee_pose,
                manipulability_weight=self.config.manipulability_weight,
                neutral_distance_weight=self.config.neutral_distance_weight,
                current_distance_weight=self.config.current_distance_weight,
                joint_weights=self.config.joint_weights,
                q7_min=self.config.q7_min,
                q7_max=self.config.q7_max,
            )
            if not success:
                return False
            self._initialized = True
            return True
        except Exception as exc:
            logger.error("Failed to initialize SpaceMouse IK: %s", exc)
            return False

    def _apply_deadzone(self, values: np.ndarray) -> np.ndarray:
        masked = values.copy()
        masked[np.abs(masked) < self.config.deadzone] = 0.0
        return masked

    @staticmethod
    def _euler_xyz_to_quaternion_xyzw(euler_xyz: np.ndarray) -> np.ndarray:
        rx, ry, rz = euler_xyz
        hx, hy, hz = rx * 0.5, ry * 0.5, rz * 0.5
        cx, sx = np.cos(hx), np.sin(hx)
        cy, sy = np.cos(hy), np.sin(hy)
        cz, sz = np.cos(hz), np.sin(hz)
        return _normalize_quaternion_xyzw(
            np.array(
                [
                    sx * cy * cz - cx * sy * sz,
                    cx * sy * cz + sx * cy * sz,
                    cx * cy * sz - sx * sy * cz,
                    cx * cy * cz + sx * sy * sz,
                ],
                dtype=float,
            )
        )

    def _reset_ik_state(self) -> None:
        self.arm_ik_processor.initial_vr_pose = None
        self.arm_ik_processor.vr_initialized = False
        self.arm_ik_processor.last_target_joints = None
