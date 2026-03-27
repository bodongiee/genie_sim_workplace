import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

# Isaac Sim imports (available after SimulationApp is created)
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.types import ArticulationAction
from meshcat import transformations as tf

# Sibling modules under custom_utils
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CUSTOM_UTILS_DIR = os.path.normpath(os.path.join(_THIS_DIR, ".."))
if _CUSTOM_UTILS_DIR not in sys.path:
    sys.path.insert(0, _CUSTOM_UTILS_DIR)

from ikfk_solver.ikfk_solver import DualArmIKSolver
from teleop.teleop_input import TeleopInput


class DualArmIsaacTeleopController:
    # Dual-arm XR teleop controller for Isaac Sim.
    # - Pipeline per frame:
    #    - 1. Teleoperation  -- read XR input, compute delta poses
    #    - 2. IK Solver      -- solve joint angles from target poses
    #    - 3. Action          -- apply joint positions to articulation

    def __init__(self):
        pass

    @classmethod
    def create(
        cls,
        world: World,
        articulation: SingleArticulation,
        robot_prim_path: str,
        manipulator_config: Dict[str, Dict[str, Any]],
        xr_client,
        left_arm_joints: List[str],
        right_arm_joints: List[str],
        T_link7_to_gripper_base: np.ndarray,
        ik_urdf_path: str,
        ik_config_path: str,
        simulation_app=None,
        xrt_module=None,
        R_headset_world: np.ndarray = None,
        scale_factor: float = 1.0,
        hold_joint_positions: Dict[str, float] = None,
        camera_config: Optional[Dict] = None,
        input_mode: str = "controller",
    ):
        self = cls()
        self.world = world
        self.articulation = articulation
        self.robot_prim_path = robot_prim_path
        self.manipulator_config = manipulator_config
        self._left_arm_joints = left_arm_joints
        self._right_arm_joints = right_arm_joints
        self._T_link7_to_gripper_base = T_link7_to_gripper_base
        self._simulation_app = simulation_app
        self._xrt = xrt_module
        self._hold_joint_positions = hold_joint_positions or {}
        self._stop_event = threading.Event()
        self._input_mode = input_mode
        self._ik_initialized = False

        R_hw = R_headset_world if R_headset_world is not None else np.eye(3)

        # ===== Build joint index maps =====
        self.dof_names = list(self.articulation.dof_names)
        self._left_joint_indices = [self.dof_names.index(j) for j in left_arm_joints]
        self._right_joint_indices = [self.dof_names.index(j) for j in right_arm_joints]

        # ===== Cache gripper joint limits =====
        self._gripper_lower = {}
        self._gripper_upper = {}
        lower = self.articulation.dof_properties["lower"]
        upper = self.articulation.dof_properties["upper"]
        for name, config in manipulator_config.items():
            if "gripper_config" not in config:
                continue
            for jname in config["gripper_config"]["joint_names"]:
                if jname in self.dof_names:
                    idx = self.dof_names.index(jname)
                    self._gripper_lower[idx] = float(lower[idx])
                    self._gripper_upper[idx] = float(upper[idx])

        # ==== 1. Teleoperation input module =====
        self.teleop_input = TeleopInput(
            manipulator_config=manipulator_config,
            xr_client=xr_client,
            R_headset_world=R_hw,
            scale_factor=scale_factor,
            input_mode=input_mode,
        )

        # ===== 2. IK Solver module =====
        left_q0, right_q0 = self._read_arm_joints()
        self.ik_solver = DualArmIKSolver(ik_urdf_path, ik_config_path, left_q0, right_q0)

        # ===== Camera setup =====
        if camera_config is not None:
            self._setup_cameras(camera_config)

        return self

    # =========================================================================
    # Pose helpers (Isaac Sim specific)
    # =========================================================================
    def _get_link_pose(self, body_name: str):
        from isaacsim.core.utils.xforms import get_world_pose
        prim_path = self._find_body_prim_path(body_name)
        position, orientation = get_world_pose(prim_path)
        return np.array(position), np.array(orientation)

    def _find_body_prim_path(self, body_name: str) -> str:
        from pxr import Usd
        stage = get_current_stage()
        robot_prim = stage.GetPrimAtPath(self.robot_prim_path)
        for prim in Usd.PrimRange(robot_prim):
            if prim.GetName() == body_name:
                return str(prim.GetPath())
        raise ValueError(f"Body '{body_name}' not found under {self.robot_prim_path}")

    def _get_gripper_base_pose(self, link_name: str) -> np.ndarray:
        xyz, quat_wxyz = self._get_link_pose(link_name)
        T_world_link = tf.quaternion_matrix(quat_wxyz)
        T_world_link[:3, 3] = xyz
        return T_world_link @ self._T_link7_to_gripper_base

    def _get_arm_base_transform(self) -> np.ndarray:
        try:
            xyz, quat_wxyz = self._get_link_pose("arm_base_link")
            T = tf.quaternion_matrix(quat_wxyz)
            T[:3, 3] = xyz
            return T
        except ValueError:
            return np.eye(4)

    # =========================================================================
    # Joint read / write  (Action layer)
    # =========================================================================
    def _read_arm_joints(self):
        all_positions = self.articulation.get_joint_positions()
        left_q = np.array([all_positions[i] for i in self._left_joint_indices], dtype=np.float32)
        right_q = np.array([all_positions[i] for i in self._right_joint_indices], dtype=np.float32)
        return left_q, right_q

    def _write_arm_joints(self, left_q, right_q):
        all_positions = self.articulation.get_joint_positions().copy()

        for idx, val in zip(self._left_joint_indices, left_q):
            all_positions[idx] = float(val)
        for idx, val in zip(self._right_joint_indices, right_q):
            all_positions[idx] = float(val)

        # Gripper targets (clamped to joint limits)
        for targets in self.teleop_input.gripper_pos_target.values():
            for joint_name, joint_pos in targets.items():
                if joint_name in self.dof_names:
                    idx = self.dof_names.index(joint_name)
                    all_positions[idx] = np.clip(float(joint_pos),
                                                 self._gripper_lower[idx],
                                                 self._gripper_upper[idx])

        # Hold waist/head joints at fixed positions
        for joint_name, joint_pos in self._hold_joint_positions.items():
            if joint_name in self.dof_names:
                all_positions[self.dof_names.index(joint_name)] = float(joint_pos)

        action = ArticulationAction(joint_positions=all_positions)
        self.articulation.apply_action(action)

    # =========================================================================
    # Camera setup
    # =========================================================================
    def _setup_cameras(self, camera_config):
        from omni.kit.viewport.utility import get_active_viewport_and_window
        try:
            from omni.kit.viewport.window import ViewportWindow
        except ImportError:
            from omni.kit.viewport.utility import create_viewport_window
            ViewportWindow = None

        if "head_camera" in camera_config:
            viewport, _ = get_active_viewport_and_window()
            head_cam = camera_config["head_camera"].format(robot=self.robot_prim_path)
            viewport.set_active_camera(head_cam)

        self._extra_viewports = []
        for title, cam_path_template in camera_config.get("extra_cameras", []):
            cam_path = cam_path_template.format(robot=self.robot_prim_path)
            try:
                if ViewportWindow is not None:
                    vp_win = ViewportWindow(title, width=320, height=240)
                else:
                    vp_win = create_viewport_window(title, width=320, height=240)
                vp_win.viewport_api.set_active_camera(cam_path)
                self._extra_viewports.append(vp_win)
                print(f"  Viewport window: {title} -> {cam_path}")
            except Exception as e:
                print(f"  WARNING: Could not create viewport for {title}: {e}")

    # =========================================================================
    # Per-frame update  — the core pipeline
    # =========================================================================
    def _update(self):
        # Single-frame update: teleoperation -> IK solver -> action.

        # First-frame IK sync
        if not self._ik_initialized:
            left_q, right_q = self._read_arm_joints()
            self.ik_solver.sync_from_current(left_q, right_q)
            self._ik_initialized = True

        T_base_world = np.linalg.inv(self._get_arm_base_transform())

        # ===== Step 1: Teleoperation — read XR input, compute delta poses =====
        ee_poses = {
            name: self._get_gripper_base_pose(cfg["link_name"])
            for name, cfg in self.manipulator_config.items()
        }
        smooth_qs = {
            name: (self.ik_solver.smooth_q_left.copy() if "left" in name
                   else self.ik_solver.smooth_q_right.copy())
            for name in self.manipulator_config
        }
        arm_inputs = self.teleop_input.get_arm_inputs(ee_poses, smooth_qs, T_base_world)

        # ===== Step 2: IK Solver — compute joint angles =====
        smooth_left, smooth_right = self.ik_solver.step(arm_inputs)

        # ===== Step 3: Action — apply joint positions to articulation =====
        if self._input_mode == "controller":
            self.teleop_input.update_gripper_targets()
        self._write_arm_joints(smooth_left, smooth_right)

    # =========================================================================
    # Debug logging
    # =========================================================================
    def _log_debug(self):
        ts = self._xrt.get_time_stamp_ns() if self._xrt else 0
        actual_l, actual_r = self._read_arm_joints()

        if self._input_mode == "controller":
            grips = {}
            for src_name, config in self.manipulator_config.items():
                grips[src_name] = self.teleop_input.xr_client.get_key_value_by_name(config["control_trigger"])
            print(f"[Debug] grip={grips}  ts={ts}")
        else:
            active_hands = {n: self.teleop_input.active[n] for n in self.manipulator_config}
            print(f"[Debug] hand_active={active_hands}  ts={ts}")

        print(f"  L shoulder IK_target={np.rad2deg(self.ik_solver.target_q_left[:3]).round(1)}"
              f"  cmd={np.rad2deg(self.ik_solver.smooth_q_left[:3]).round(1)}"
              f"  actual={np.rad2deg(actual_l[:3]).round(1)}"
              f"  err={np.rad2deg(self.ik_solver.smooth_q_left[:3] - actual_l[:3]).round(2)}")
        print(f"  R shoulder IK_target={np.rad2deg(self.ik_solver.target_q_right[:3]).round(1)}"
              f"  cmd={np.rad2deg(self.ik_solver.smooth_q_right[:3]).round(1)}"
              f"  actual={np.rad2deg(actual_r[:3]).round(1)}"
              f"  err={np.rad2deg(self.ik_solver.smooth_q_right[:3] - actual_r[:3]).round(2)}")

    # =========================================================================
    # Main loop
    # =========================================================================
    def run(self):
        _last_time = 0.0
        app = self._simulation_app
        run_check = app.is_running if app is not None else lambda: True

        while run_check() and not self._stop_event.is_set():
            try:
                self._update()
                self.world.step(render=True)

                now = time.time()
                if now - _last_time > 0.5:
                    _last_time = now
                    self._log_debug()

            except KeyboardInterrupt:
                print("\nTeleoperation stopped.")
                self._stop_event.set()

        if app is not None:
            app.close()

    def stop(self):
        self._stop_event.set()
