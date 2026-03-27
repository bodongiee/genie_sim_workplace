import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R

# Ensure IK-SDK and export utils are importable
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_IK_SDK_DIR = os.path.normpath(os.path.join(_THIS_DIR, "../utils/IK-SDK"))
_EXPORT_DIR = os.path.normpath(os.path.join(_THIS_DIR, "../export"))

for _p in (_IK_SDK_DIR, _EXPORT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ik_solver
from ikfk_solver_export import capture_c_stdout


class DualArmIKSolver:
    # Dual-arm IK solver wrapper

    def __init__(
        self,
        urdf_path: str,
        config_path: str,
        left_q0: np.ndarray,
        right_q0: np.ndarray,
        alpha_per_joint: np.ndarray = None,
        shoulder_weight : float = 0.23,
        elbow_weight : float = 0.3,
        wrist_weight : float = 0.3,
    ):
        with capture_c_stdout():
            self._ik_left = ik_solver.Solver(part=ik_solver.RobotPart.LEFT_ARM, urdf_path=urdf_path, config_path=config_path,)
            self._ik_right = ik_solver.Solver(part=ik_solver.RobotPart.RIGHT_ARM, urdf_path=urdf_path, config_path=config_path,)

        self._ik_left.sync_target_with_joints(left_q0)
        self._ik_right.sync_target_with_joints(right_q0)

        self.target_q_left = left_q0.copy()
        self.target_q_right = right_q0.copy()
        self._smooth_q_left = left_q0.copy()
        self._smooth_q_right = right_q0.copy()
        self._initialized = True
        self.shoulder_weight = shoulder_weight
        self.elbow_weight = elbow_weight
        self.wrist_weight = wrist_weight
        self.DEFAULT_ALPHA = np.array([shoulder_weight, shoulder_weight, shoulder_weight, elbow_weight, wrist_weight, wrist_weight, wrist_weight], dtype=np.float32)
        self._alpha = alpha_per_joint if alpha_per_joint is not None else self.DEFAULT_ALPHA.copy()

    @classmethod
    def from_robot_cfg(cls, robot_cfg: str, left_q0: np.ndarray, right_q0: np.ndarray, **kwargs):
        # Create solver by robot config name
        cfg_map = {
            "ffw_sg2_follower": ("ffw_sg2_follower.urdf", "ffw_sg2_follower_solver.yaml"),
            "ffw_sh5_follower": ("ffw_sh5_follower.urdf", "ffw_sh5_follower_solver.yaml"),
            "G2": ("G2_NO_GRIPPER.urdf", "g2_solver.yaml"),
            "G1": ("G1_NO_GRIPPER.urdf", "g1_solver.yaml"),
        }
        for key, (urdf_name, config_name) in cfg_map.items():
            if key in robot_cfg:
                break
        else:
            urdf_name, config_name = "G1_NO_GRIPPER.urdf", "g1_solver.yaml"

        urdf_path = os.path.join(_IK_SDK_DIR, urdf_name)
        config_path = os.path.join(_IK_SDK_DIR, config_name)
        return cls(urdf_path, config_path, left_q0, right_q0, **kwargs)

    def _get_solver(self, side: str):
        return self._ik_left if "left" in side else self._ik_right

    def activate(self, side: str, cmd_q: np.ndarray):
        # Sync solver with commanded joint positions on grip activation.
        solver = self._get_solver(side)
        solver.sync_target_with_joints(cmd_q)
        if "left" in side:
            self.target_q_left = cmd_q.copy()
        else:
            self.target_q_right = cmd_q.copy()

    def update_target(self, side: str, pos_base: np.ndarray, quat_base: np.ndarray):
        # Set IK target in arm-base frame (position + quaternion xyzw)
        solver = self._get_solver(side)
        solver.update_target_quat(
            target_pos=pos_base.astype(np.float32),
            target_quat=quat_base.astype(np.float32),
        )

    def solve(self, left_active: bool, right_active: bool) -> tuple:
        # Solve IK for active arms, apply per-joint exponential smoothing.
        # Returns (smooth_q_left, smooth_q_right).
        
        if left_active:
            self.target_q_left = self._ik_left.solve()
        if right_active:
            self.target_q_right = self._ik_right.solve()

        self._smooth_q_left = (self._alpha * self.target_q_left + (1 - self._alpha) * self._smooth_q_left)
        self._smooth_q_right = (self._alpha * self.target_q_right + (1 - self._alpha) * self._smooth_q_right)

        return self._smooth_q_left, self._smooth_q_right

    def sync_from_current(self, left_q: np.ndarray, right_q: np.ndarray):
        # Re-sync solver and smoothing state (on first frame)
        self._ik_left.sync_target_with_joints(left_q)
        self._ik_right.sync_target_with_joints(right_q)
        self.target_q_left = left_q.copy()
        self.target_q_right = right_q.copy()
        self._smooth_q_left = left_q.copy()
        self._smooth_q_right = right_q.copy()

    @property
    def smooth_q_left(self):
        return self._smooth_q_left

    @property
    def smooth_q_right(self):
        return self._smooth_q_right

    def step(self, arm_inputs: dict) -> tuple:
        # arm_inputs
        # "left_hand" : {
        #   "active" : True
        #   "newly_activated" : True
        #   "cmd_q" -> smooth joint angle when activated
        #   "pos_base" -> target pose
        #   "quat_base" -> target orientation

        left_active = False
        right_active = False

        for side_name, inp in arm_inputs.items():
            is_left = "left" in side_name
            if is_left:
                left_active = inp["active"]
            else:
                right_active = inp["active"]

            if inp["active"]:
                if inp.get("newly_activated"):
                    self.activate(side_name, inp["cmd_q"])

                self.update_target(side_name, inp["pos_base"], inp["quat_base"])

        return self.solve(left_active, right_active)

    def compute_fk(self, side: str, joints: np.ndarray) -> np.ndarray:
        # Forward kinematics — returns 4x4 transform matrix.
        solver = self._get_solver(side)
        return np.asarray(solver.compute_fk(joints), dtype=np.float64).reshape(4, 4)
