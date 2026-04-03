# Author: Kwon
# License: Mozilla Public License Version 2.0

from typing import Callable, Dict, List, Optional

import numpy as np
import omni
import omni.kit.app

try:
    from geniesim.plugins.logger import Logger
    logger = Logger()
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.grippers.gripper import Gripper
from pxr import UsdPhysics


# 5 fingers x 4 joints = 20 joints per hand
# Finger layout: [splay, curl1, curl2, curl3] per finger
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]
JOINTS_PER_FINGER = 4
FINGER_JOINT_SLICES = {
    "thumb":  slice(0, 4),    # joint1-4
    "index":  slice(4, 8),    # joint5-8
    "middle": slice(8, 12),   # joint9-12
    "ring":   slice(12, 16),  # joint13-16
    "pinky":  slice(16, 20),  # joint17-20
}
NUM_HAND_JOINTS = 20

class HandGripper(Gripper):
    # NOTE : Controller for a 5-finger dexterous hand (e.g. HX5-D20).
    # - Each hand has 20 revolute joints: 5 fingers with 4 joints each
    # - (1 splay/abduction + 3 curl/flexion).
    # Supports both teleop (per-joint position control) and simple open/close.

    # Args:
    #    end_effector_prim_path: prim path of the hand base (e.g. hx5_r_base).
    #    joint_prim_names: list of 20 joint prim names in order.
    #    joint_opened_positions: 20-element array of joint positions when fully open.
    #    joint_closed_positions: 20-element array of joint positions when fully closed.
    #    joint_closed_velocities: 20-element array of joint velocities for force-close.
    #    gripper_type: drive type ("angular" or "linear").
    #    gripper_max_force: max force for  close action.

    def __init__(
        self,
        end_effector_prim_path: str,
        joint_prim_names: List[str],
        joint_opened_positions: np.ndarray = None,
        joint_closed_positions: np.ndarray = None,
        joint_closed_velocities: np.ndarray = None,
        gripper_type: str = "angular",
        gripper_max_force: float = 5.0,
    ) -> None:
        Gripper.__init__(self, end_effector_prim_path=end_effector_prim_path)

        assert len(joint_prim_names) == NUM_HAND_JOINTS, (
            f"Expected {NUM_HAND_JOINTS} joint names, got {len(joint_prim_names)}")

        self._joint_prim_names = joint_prim_names
        self._joint_dof_indices: List[Optional[int]] = [None] * NUM_HAND_JOINTS
        self._joint_opened_positions = (
            np.array(joint_opened_positions) if joint_opened_positions is not None
            else np.zeros(NUM_HAND_JOINTS))

        self._joint_closed_positions = (
            np.array(joint_closed_positions) if joint_closed_positions is not None
            else np.ones(NUM_HAND_JOINTS))

        self._joint_closed_velocities = (
            np.array(joint_closed_velocities) if joint_closed_velocities is not None
            else -np.ones(NUM_HAND_JOINTS))

        self._get_joint_positions_func = None
        self._set_joint_positions_func = None
        self._articulation_apply_action_func = None
        self._articulation_num_dofs = None

        self.is_reached = False
        self._frozen_positions = None
        self.gripper_type = gripper_type
        self.gripper_max_force = gripper_max_force

        self.physics_material = PhysicsMaterial(
            prim_path="/World/hand_gripper_physics",
            static_friction=2,
            dynamic_friction=2,
            restitution=0.1,
        )
        self.object_material = PhysicsMaterial(
            prim_path="/World/object_physics",
            static_friction=1,
            dynamic_friction=1,
            restitution=0.1,
        )
        self._modify_friction_mode("/World/hand_gripper_physics")
        self._modify_friction_mode("/World/object_physics")
        return

    # ==================================================================
    # Properties
    # ==================================================================

    @property
    def joint_opened_positions(self) -> np.ndarray:
        return self._joint_opened_positions

    @property
    def joint_closed_positions(self) -> np.ndarray:
        return self._joint_closed_positions

    @property
    def joint_dof_indices(self) -> List[Optional[int]]:
        return self._joint_dof_indices

    @property
    def joint_prim_names(self) -> List[str]:
        return self._joint_prim_names

        
    # ==================================================================
    # Initialization
    # ==================================================================

    def _modify_friction_mode(self, prim_path: str):
        from pxr import PhysxSchema

        stage = omni.usd.get_context().get_stage()
        obj_physics_prim = stage.GetPrimAtPath(prim_path)
        physx_material_api = PhysxSchema.PhysxMaterialAPI(obj_physics_prim)
        if physx_material_api is not None:
            fric_combine_mode = physx_material_api.GetFrictionCombineModeAttr().Get()
            if fric_combine_mode is None:
                physx_material_api.CreateFrictionCombineModeAttr().Set("max")
            elif fric_combine_mode != "max":
                physx_material_api.GetFrictionCombineModeAttr().Set("max")

    def initialize(
        self,
        articulation_apply_action_func: Callable,
        get_joint_positions_func: Callable,
        set_joint_positions_func: Callable,
        dof_names: List[str],
        physics_sim_view: omni.physics.tensors.SimulationView = None,
    ) -> None:

        Gripper.initialize(self)
        self._get_joint_positions_func = get_joint_positions_func
        self._set_joint_positions_func = set_joint_positions_func
        self._articulation_apply_action_func = articulation_apply_action_func
        self._articulation_num_dofs = len(dof_names)

        # Resolve joint names to DOF indices
        dof_name_to_idx = {name: i for i, name in enumerate(dof_names)}
        for j, name in enumerate(self._joint_prim_names):
            if name in dof_name_to_idx:
                self._joint_dof_indices[j] = dof_name_to_idx[name]

        unresolved = [n for n, idx in zip(self._joint_prim_names, self._joint_dof_indices) if idx is None]
        if unresolved:
            raise Exception(f"Could not resolve hand joint DOF indices for: {unresolved}")

        current_positions = get_joint_positions_func()
        if self._default_state is None:
            self._default_state = np.array([
                current_positions[idx] for idx in self._joint_dof_indices])
        return

    # ==================================================================
    # Teleop: per-joint position control
    # ==================================================================

    def freeze(self) -> None:
        """Capture current joint positions as the frozen target."""
        current = self._get_joint_positions_func()
        self._frozen_positions = np.array([current[idx] for idx in self._joint_dof_indices])

    def hold_current_positions(self) -> None:
        """Re-apply frozen positions as targets to prevent gradual sagging."""
        if self._frozen_positions is not None:
            self.set_target_positions(self._frozen_positions)
        else:
            self.freeze()
            self.set_target_positions(self._frozen_positions)

    def set_target_positions(self, positions: np.ndarray) -> None:
        # Set target positions for all 20 hand joints (used by teleop).
        # Args
        #  - positions: 20-element array of target joint positions.
        assert len(positions) == NUM_HAND_JOINTS
        target = [None] * self._articulation_num_dofs
        for j in range(NUM_HAND_JOINTS):
            target[self._joint_dof_indices[j]] = float(positions[j])
        self._articulation_apply_action_func(
            control_actions=ArticulationAction(joint_positions=target))
        return

    def set_finger_positions(self, finger_name: str, positions: np.ndarray) -> None:
        # Set target positions for a single finger (4 joints).

        # Args:
        #  - finger_name: one of "thumb", "index", "middle", "ring", "pinky".
        # positions: 4-element array of target joint positions.
        
        assert finger_name in FINGER_JOINT_SLICES, f"Unknown finger: {finger_name}"
        assert len(positions) == JOINTS_PER_FINGER
        s = FINGER_JOINT_SLICES[finger_name]
        indices = self._joint_dof_indices[s]

        target = [None] * self._articulation_num_dofs
        for idx, pos in zip(indices, positions):
            target[idx] = float(pos)
        self._articulation_apply_action_func(
            control_actions=ArticulationAction(joint_positions=target))
        return
    # ==================================================================
    # Simple open / close 
    # ==================================================================

    def open(self) -> None:
        # Open all fingers to their opened positions.
        self._articulation_apply_action_func(self.forward(action="open"))

    def close(self) -> None:
        # Close all fingers using velocity/force control.
        self._articulation_apply_action_func(self.forward(action="close"))

    def forward(self, action: str) -> ArticulationAction:
        # Compute ArticulationAction for open/close.
        #  - Args:
        #       action: "open" or "close".
        #  - Returns:
        #       ArticulationAction for the full articulation.
        target_action = None

        if action == "open":
            # Position mode
            self.is_reached = False
            target_joint_positions = [None] * self._articulation_num_dofs
            for j in range(NUM_HAND_JOINTS):
                target_joint_positions[self._joint_dof_indices[j]] = float(self._joint_opened_positions[j])
            target_action = ArticulationAction(joint_positions=target_joint_positions)
        elif action == "close":
            # Force mode
            self.is_reached = False
            target_vel = [None] * self._articulation_num_dofs
            for j in range(NUM_HAND_JOINTS):
                target_vel[self._joint_dof_indices[j]] = float(self._joint_closed_velocities[j])
            target_action = ArticulationAction(joint_velocities=target_vel)
        else:
            raise Exception(f"action '{action}' is not defined for HandGripper")
        return target_action

    def apply_default_action(self) -> None:
        # Apply opened positions as the default action.
        self.set_target_positions(self._joint_opened_positions)

    # ==================================================================
    # State management
    # ==================================================================

    def get_joint_positions(self) -> np.ndarray:
        # Get current positions of all 20 hand joints.
        return self._get_joint_positions_func(
            joint_indices=self._joint_dof_indices)

    def set_joint_positions(self, positions: np.ndarray) -> None:
        # Directly set joint positions (not as action, but as state).
        self._set_joint_positions_func(
            positions=positions,
            joint_indices=self._joint_dof_indices,)

    def get_finger_positions(self, finger_name: str) -> np.ndarray:
        # Get current positions for a single finger.
        s = FINGER_JOINT_SLICES[finger_name]
        all_pos = self.get_joint_positions()
        return all_pos[s]

    def set_default_state(self, joint_positions: np.ndarray) -> None:
        self._default_state = joint_positions
        return

    def get_default_state(self) -> np.ndarray:
        return self._default_state

    def post_reset(self) -> None:
        Gripper.post_reset(self)
        self.set_joint_positions(self._default_state)
        return
    # ==================================================================
    # Articulation action passthrough
    # ==================================================================

    def apply_action(self, control_actions: ArticulationAction) -> None:
        # Map a 20-joint hand action to the full articulation action.
        joint_actions = ArticulationAction()
        if control_actions.joint_positions is not None:
            joint_actions.joint_positions = [None] * self._articulation_num_dofs
            for j in range(NUM_HAND_JOINTS):
                joint_actions.joint_positions[self._joint_dof_indices[j]] = control_actions.joint_positions[j]
        if control_actions.joint_velocities is not None:
            joint_actions.joint_velocities = [None] * self._articulation_num_dofs
            for j in range(NUM_HAND_JOINTS):
                joint_actions.joint_velocities[self._joint_dof_indices[j]] = control_actions.joint_velocities[j]
        if control_actions.joint_efforts is not None:
            joint_actions.joint_efforts = [None] * self._articulation_num_dofs
            for j in range(NUM_HAND_JOINTS):
                joint_actions.joint_efforts[self._joint_dof_indices[j]] = control_actions.joint_efforts[j]
        self._articulation_apply_action_func(control_actions=joint_actions)

        return
