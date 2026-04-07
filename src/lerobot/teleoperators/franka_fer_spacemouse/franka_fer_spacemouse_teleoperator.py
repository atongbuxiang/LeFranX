import logging
from typing import Any, Dict

import numpy as np

from lerobot.teleoperators.teleoperator import Teleoperator

from .config_franka_fer_spacemouse import FrankaFERSpaceMouseTeleoperatorConfig
from .spacemouse_reader import SpaceMouseStateReader

logger = logging.getLogger(__name__)


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
        self._target_position: np.ndarray | None = None
        self._target_rotation: np.ndarray | None = None
        self._last_action = None

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
        self._target_position = None
        self._target_rotation = None

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    def send_feedback(self, feedback: Dict[str, Any]) -> None:
        del feedback
        return

    def set_robot(self, robot):
        self._robot_reference = robot

    def get_action(self) -> Dict[str, Any]:
        if not self._is_connected:
            raise RuntimeError("FrankaFERSpaceMouseTeleoperator is not connected")
        if self._robot_reference is None:
            return self._last_action or {f"joint_{i}.pos": 0.0 for i in range(7)}

        current_obs = self._robot_reference.get_observation()
        current_joints = [current_obs[f"joint_{i}.pos"] for i in range(7)]

        if not self._initialized and not self._initialize_ik_solver(current_obs, current_joints):
            return self._last_action or {f"joint_{i}.pos": float(current_joints[i]) for i in range(7)}

        state = self._reader.get_state()
        if state is None:
            return {f"joint_{i}.pos": float(current_joints[i]) for i in range(7)}

        translation = self._apply_deadzone(
            np.array([state["y"], -state["x"], state["z"]], dtype=float)
        ) * self.config.translation_scale
        rotation = self._apply_deadzone(
            np.array([state["pitch"], -state["roll"], state["yaw"]], dtype=float)
        ) * self.config.rotation_scale

        self._target_position = self._target_position + translation
        self._target_rotation = self._euler_to_matrix(rotation) @ self._target_rotation

        try:
            ik_result = self.arm_ik_processor.ik_solver.solve_q7_optimized(
                target_position=self._target_position.tolist(),
                target_orientation=self._target_rotation.flatten().tolist(),
                current_pose=current_joints,
                q7_min=self.config.q7_min,
                q7_max=self.config.q7_max,
                tolerance=1e-6,
                max_iterations=100,
            )
            if ik_result.success:
                target_joints = list(ik_result.joint_angles)
                action = {f"joint_{i}.pos": float(target_joints[i]) for i in range(7)}
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

            ee_pose_matrix = np.array(ee_pose, dtype=float).reshape(4, 4)
            self._target_position = ee_pose_matrix[:3, 3].copy()
            self._target_rotation = ee_pose_matrix[:3, :3].copy()
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
    def _euler_to_matrix(euler_xyz: np.ndarray) -> np.ndarray:
        rx, ry, rz = euler_xyz
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)

        rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return rot_z @ rot_y @ rot_x
