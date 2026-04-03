import os
import sys
import numpy as np
from typing import Any, Dict, Optional, Set, Tuple

from meshcat import transformations as tf

# Hand config (sibling package under custom_utils)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_HAND_CONFIG_DIR = os.path.normpath(os.path.join(_THIS_DIR, "../hand_config"))
if _HAND_CONFIG_DIR not in sys.path:
    sys.path.insert(0, _HAND_CONFIG_DIR)

from pico_vr_finger_chain import (
    PICO_FINGER_CHAINS,
    Joint_ANGLE_NAMES,
    CALIB_JOINT,
    angle_between_vecotrs,
    normalize,
)


class KeyboardMonitor:
    """Isaac Sim carb.input keyboard toggle monitor.

    Arrow keys  → toggle left hand active/inactive
    Numpad 0    → toggle right hand active/inactive
    """

    def __init__(self):
        self._state = {"left": False, "right": False}
        self._sub = None

    @property
    def left_active(self):
        return self._state["left"]

    @property
    def right_active(self):
        return self._state["right"]

    def start(self):
        import carb.input
        import omni.appwindow

        input_iface = carb.input.acquire_input_interface()
        app_window = omni.appwindow.get_default_app_window()
        kb = app_window.get_keyboard()

        state = self._state

        def _on_kb_event(event, *args, **kwargs):
            if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                if event.input == carb.input.KeyboardInput.A:
                    state["left"] = not state["left"]
                    print(f"[Keyboard] Left hand {'ON' if state['left'] else 'OFF'}")
                elif event.input == carb.input.KeyboardInput.S:
                    state["right"] = not state["right"]
                    print(f"[Keyboard] Right hand {'ON' if state['right'] else 'OFF'}")
            return True

        self._sub = input_iface.subscribe_to_keyboard_events(kb, _on_kb_event)

    def stop(self):
        if self._sub is not None:
            import carb.input
            input_iface = carb.input.acquire_input_interface()
            input_iface.unsubscribe_to_keyboard_events(self._sub)
            self._sub = None

    def poll(self):
        pass  # carb.input events are dispatched by Isaac Sim's event loop

    def is_left_control_key_pressed(self) -> bool:
        return self.left_active

    def is_right_control_key_pressed(self) -> bool:
        return self.right_active


def quat_diff_as_angle_axis(q_from, q_to):
    # Compute angle-axis difference between two quaternions (wxyz format)
    from xrobotoolkit_teleop.utils.geometry import quat_diff_as_angle_axis as _quat_diff
    return _quat_diff(q_from, q_to)


def apply_delta_pose(ref_xyz, ref_quat, delta_xyz, delta_rot):
    # Apply delta pose to reference pose
    from xrobotoolkit_teleop.utils.geometry import apply_delta_pose as _apply
    return _apply(ref_xyz, ref_quat, delta_xyz, delta_rot)


class TeleopInput:
    # Compute delta pose from tracking data
    # - "controller" : VR controller pose + gripper trigger
    # - "hand_tracking" : hand wrist pose
    # manipulator config
    # - "left_hand"
    #    - "link_name" : "target_lint"
    #    - "pose_source" : "left_controller"
    #    - "control_trigger" : "left_grip"
    #    - "hand_side" : "left"
    #    - "gripper_config"
    #        - "type" : "parallel"
    #        - "gripper_trigger" : "left_trigger"
    #        - "open_pos" : [0.1]
    #        - "close_pos" : [1.0]
    # - "right_hand"

    def __init__(
        self,
        manipulator_config: Dict[str, Dict[str, Any]],
        xr_client,
        R_headset_world: np.ndarray,
        scale_factor: float = 1.0,
        input_mode: str = "controller",
    ):
        self.manipulator_config = manipulator_config
        self.xr_client = xr_client
        self.R_headset_world = R_headset_world
        self.scale_factor = scale_factor
        self.input_mode = input_mode

        self.ref_ee_xyz: Dict[str, Optional[np.ndarray]] = {n: None for n in manipulator_config}
        self.ref_ee_quat: Dict[str, Optional[np.ndarray]] = {n: None for n in manipulator_config}
        self.ref_controller_xyz: Dict[str, Optional[np.ndarray]] = {n: None for n in manipulator_config}
        self.ref_controller_quat: Dict[str, Optional[np.ndarray]] = {n: None for n in manipulator_config}
        self.active: Dict[str, bool] = {n: False for n in manipulator_config}

        # Keyboard monitor (lazy-initialized when keyboard mode is first used)
        self._keyboard: Optional[KeyboardMonitor] = None
        if input_mode == "keyboard":
            self._keyboard = KeyboardMonitor()
            self._keyboard.start()

        # Gripper targets (controller mode)
        self.gripper_pos_target: Dict[str, Dict[str, float]] = {}
        for name, config in manipulator_config.items():
            if "gripper_config" in config:
                gc = config["gripper_config"]
                self.gripper_pos_target[name] = dict(zip(gc["joint_names"], gc["open_pos"]))

    # =========================================================================
    # Public API
    # =========================================================================
    def get_arm_inputs(self, ee_poses: Dict[str, np.ndarray], smooth_qs: Dict[str, np.ndarray], T_base_world: np.ndarray,) -> Dict[str, dict]:
        # Read XR input
        # Returns:
        #     - arm_inputs dict keyed by manipulator name, each containing:
        #     - active, newly_activated, (cmd_q, pos_base, quat_base if active).

        if self.input_mode == "hand_tracking":
            return self._arm_inputs_hand_tracking(ee_poses, smooth_qs, T_base_world)
        elif self.input_mode == "keyboard":
            return self._arm_inputs_keyboard(ee_poses, smooth_qs, T_base_world)
        else:
            return self._arm_inputs_controller(ee_poses, smooth_qs, T_base_world)

    def compute_delta_pose(self, current_xyz: np.ndarray, current_quat: np.ndarray, ref_xyz: np.ndarray, ref_quat: np.ndarray,) -> Tuple[np.ndarray, np.ndarray]:
        # Compute (delta_xyz, delta_rot) from reference to current pose.
        delta_xyz = (current_xyz - ref_xyz) * self.scale_factor
        delta_rot = quat_diff_as_angle_axis(ref_quat, current_quat)
        return delta_xyz, delta_rot

    def update_gripper_targets(self):
        # Update gripper joint targets from VR trigger values (controller mode).
        gripper_alpha = 0.2
        for gname, config in self.manipulator_config.items():
            if "gripper_config" not in config:
                continue
            gc = config["gripper_config"]
            trigger = self.xr_client.get_key_value_by_name(gc["gripper_trigger"])
            for jname, open_p, close_p in zip(gc["joint_names"], gc["open_pos"], gc["close_pos"]):
                desired = open_p + (close_p - open_p) * trigger
                current = self.gripper_pos_target[gname][jname]
                self.gripper_pos_target[gname][jname] = current + gripper_alpha * (desired - current)

    # =========================================================================
    # XR pose reading helpers
    # =========================================================================
    def _transform_to_world(self, xyz: np.ndarray, quat_xyzw: np.ndarray):
        #Transform pose from headset frame to world frame. Returns (xyz, quat_wxyz).
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        #xyz is referenced from headset
        xyz_world = self.R_headset_world @ xyz
        #xyzz is referenced from world
        R_transform = np.eye(4)
        R_transform[:3, :3] = self.R_headset_world
        R_quat = tf.quaternion_from_matrix(R_transform)
        quat_world = tf.quaternion_multiply(tf.quaternion_multiply(R_quat, quat_wxyz), tf.quaternion_conjugate(R_quat),)
        return xyz_world, quat_world

    def _get_controller_pose(self, pose_source: str):
        # Get VR controller pose in world frame. Returns (xyz, quat_wxyz).
        xr_pose = self.xr_client.get_pose_by_name(pose_source)
        xyz = np.array([xr_pose[0], xr_pose[1], xr_pose[2]])
        quat_xyzw = np.array([xr_pose[3], xr_pose[4], xr_pose[5], xr_pose[6]])
        return self._transform_to_world(xyz, quat_xyzw)

    def _detect_fist(self, hand_state: np.ndarray, side: str, threshold: float = 0.35) -> bool:
        """Detect fist gesture. Returns True if all finger chains' normalized values < threshold."""
        positions = hand_state[:, :3]  # (26, 3) xyz only

        for finger, chain in PICO_FINGER_CHAINS.items():
            for i, joint_name in enumerate(Joint_ANGLE_NAMES[finger]):
                a, b, c = positions[chain[i]], positions[chain[i + 1]], positions[chain[i + 2]]
                angle = angle_between_vecotrs(a - b, c - b)

                calib_key = f"{side}_{finger}_{joint_name}"
                if calib_key not in CALIB_JOINT:
                    continue
                open_val, close_val = CALIB_JOINT[calib_key]
                norm_val = normalize(angle, open_val, close_val)

                if norm_val >= threshold:
                    return False  # Any joint still open → not a fist

        return True

    def _get_hand_wrist_pose(self, side: str):
        # Get hand wrist pose from VR hand tracking.
        # Returns (xyz, quat_wxyz) in world frame, or (None, None) if inactive.

        hand_state = self.xr_client.get_hand_tracking_state(side)
        if hand_state is None:
            return None, None

        hand_state = np.array(hand_state)
        wrist = hand_state[0]  # index 0 = wrist joint: [x, y, z, qx, qy, qz, qw]
        xyz = np.array([wrist[0], wrist[1], wrist[2]])
        quat_xyzw = np.array([wrist[3], wrist[4], wrist[5], wrist[6]])
        return self._transform_to_world(xyz, quat_xyzw)

    # =========================================================================
    # Activation helpers
    # =========================================================================
    def _activate(self, name: str, ee_pose: np.ndarray, smooth_q: np.ndarray,
                  controller_xyz: np.ndarray, controller_quat: np.ndarray) -> dict:
        # Handle first-frame activation for a manipulator.
        self.ref_ee_xyz[name] = ee_pose[:3, 3].copy()
        self.ref_ee_quat[name] = tf.quaternion_from_matrix(ee_pose)
        self.ref_controller_xyz[name] = controller_xyz.copy()
        self.ref_controller_quat[name] = controller_quat.copy()
        print(f"{name} activated.")
        return {"newly_activated": True, "cmd_q": smooth_q.copy()}

    def _deactivate(self, name: str):
        # Reset reference state on deactivation.
        print(f"{name} deactivated.")
        self.ref_ee_xyz[name] = None
        self.ref_ee_quat[name] = None
        self.ref_controller_xyz[name] = None
        self.ref_controller_quat[name] = None

    def _compute_target_in_base(self, name, controller_xyz, controller_quat, T_base_world):
        # Compute IK target (pos, quat) in arm-base frame from current XR input.
        from scipy.spatial.transform import Rotation as Rot

        delta_xyz, delta_rot = self.compute_delta_pose(controller_xyz, controller_quat,  self.ref_controller_xyz[name], self.ref_controller_quat[name],)
        
        #target pose, orientation based on world frame
        target_xyz, target_quat = apply_delta_pose(self.ref_ee_xyz[name], self.ref_ee_quat[name], delta_xyz, delta_rot,)

        T_world_target = tf.quaternion_matrix(target_quat)
        T_world_target[:3, 3] = target_xyz

        #target pose, orientation based on robot
        T_base_target = T_base_world @ T_world_target

        pos_base = T_base_target[:3, 3].astype(np.float32)
        quat_base = Rot.from_matrix(T_base_target[:3, :3]).as_quat(scalar_first=False).astype(np.float32)
        return pos_base, quat_base

    # =========================================================================
    # Controller mode
    # =========================================================================
    def _arm_inputs_controller(self, ee_poses, smooth_qs, T_base_world):
        arm_inputs = {}
        for name, config in self.manipulator_config.items():
            grip_val = self.xr_client.get_key_value_by_name(config["control_trigger"])
            is_active = grip_val > 0.9
            was_active = self.active[name]
            self.active[name] = is_active

            inp = {"active": is_active, "newly_activated": False}

            if is_active:
                controller_xyz, controller_quat = self._get_controller_pose(config["pose_source"])

                if not was_active or self.ref_ee_xyz[name] is None:
                    extra = self._activate(name, ee_poses[name], smooth_qs[name], controller_xyz, controller_quat)
                    inp.update(extra)

                inp["pos_base"], inp["quat_base"] = self._compute_target_in_base(name, controller_xyz, controller_quat, T_base_world)
            else:
                if was_active:
                    self._deactivate(name)

            arm_inputs[name] = inp
        return arm_inputs

    # =========================================================================
    # Hand tracking mode
    # =========================================================================
    def _arm_inputs_hand_tracking(self, ee_poses, smooth_qs, T_base_world):
        arm_inputs = {}
        for name, config in self.manipulator_config.items():
            hand_side = config.get("hand_side", "left" if "left" in name else "right")

            # Read hand tracking state and check fist gesture
            hand_state = self.xr_client.get_hand_tracking_state(hand_side)
            if hand_state is not None:
                hand_state = np.array(hand_state)
                is_fist = self._detect_fist(hand_state, hand_side)
                wrist = hand_state[0]
                xyz = np.array([wrist[0], wrist[1], wrist[2]])
                quat_xyzw = np.array([wrist[3], wrist[4], wrist[5], wrist[6]])
                wrist_xyz, wrist_quat = self._transform_to_world(xyz, quat_xyzw)
                is_active = not is_fist
            else:
                wrist_xyz, wrist_quat = None, None
                is_active = False

            was_active = self.active[name]
            self.active[name] = is_active

            inp = {"active": is_active, "newly_activated": False}

            if is_active:
                if not was_active or self.ref_ee_xyz[name] is None:
                    extra = self._activate(name, ee_poses[name], smooth_qs[name], wrist_xyz, wrist_quat)
                    inp.update(extra)

                inp["pos_base"], inp["quat_base"] = self._compute_target_in_base(name, wrist_xyz, wrist_quat, T_base_world)
            else:
                if was_active:
                    self._deactivate(name)

            arm_inputs[name] = inp
        return arm_inputs

    # =========================================================================
    # Keyboard mode
    # =========================================================================
    def _arm_inputs_keyboard(self, ee_poses, smooth_qs, T_base_world):
        self._keyboard.poll()

        arm_inputs = {}
        for name, config in self.manipulator_config.items():
            hand_side = config.get("hand_side", "left" if "left" in name else "right")

            # Keyboard activation gate
            if "left" in name:
                key_active = self._keyboard.is_left_control_key_pressed()
            else:
                key_active = self._keyboard.is_right_control_key_pressed()

            # Read hand tracking for pose if available
            wrist_xyz, wrist_quat = None, None
            hand_state = self.xr_client.get_hand_tracking_state(hand_side)
            #if key_active and hand_state is None:
            #    print(f"[KB_DEBUG] {name}: hand_state is None (no tracking data)")
            if hand_state is not None:
                hand_state = np.array(hand_state)
                wrist = hand_state[0]
                xyz = np.array([wrist[0], wrist[1], wrist[2]])
                quat_xyzw = np.array([wrist[3], wrist[4], wrist[5], wrist[6]])
                wrist_xyz, wrist_quat = self._transform_to_world(xyz, quat_xyzw)

            is_active = key_active

            was_active = self.active[name]
            self.active[name] = is_active

            inp = {"active": is_active, "newly_activated": False}

            if is_active:
                # No hand tracking: use ref position so delta stays 0 (hold in place)
                if wrist_xyz is None:
                    if self.ref_controller_xyz[name] is not None:
                        wrist_xyz = self.ref_controller_xyz[name]
                        wrist_quat = self.ref_controller_quat[name]
                    else:
                        ee = ee_poses[name]
                        wrist_xyz = ee[:3, 3].copy()
                        wrist_quat = tf.quaternion_from_matrix(ee)

                if not was_active or self.ref_ee_xyz[name] is None:
                    extra = self._activate(name, ee_poses[name], smooth_qs[name], wrist_xyz, wrist_quat)
                    inp.update(extra)

                inp["pos_base"], inp["quat_base"] = self._compute_target_in_base(name, wrist_xyz, wrist_quat, T_base_world)
            else:
                if was_active:
                    self._deactivate(name)

            arm_inputs[name] = inp
        return arm_inputs

    def shutdown(self):
        """Clean up resources (restore terminal settings, etc.)."""
        if self._keyboard is not None:
            self._keyboard.stop()
