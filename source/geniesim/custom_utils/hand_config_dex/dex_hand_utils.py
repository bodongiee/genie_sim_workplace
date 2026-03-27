# =============================================================================
# Dex-retargeting hand control utilities for HX5-D20
# Ported from hand_control_ffw_sh5_v2.py (XRoboToolkit workspace)
# =============================================================================
import os
import time

import numpy as np
import pinocchio as pin
from dex_retargeting.constants import HandType, RetargetingType
from dex_retargeting.retargeting_config import RetargetingConfig
from pathlib import Path
from typing import Optional

# =============================================================================
# PICO ↔ MediaPipe keypoint mapping (from dex_hand_utils.py)
# =============================================================================
from dex_retargeting.constants import OPERATOR2MANO

pico_to_mediapipe = {
    1: 0,   # Wrist
    2: 1,   # Thumb_metacarpal -> THUMB_CMC
    3: 2,   # Thumb_proximal   -> THUMB_MCP
    4: 3,   # Thumb_distal     -> THUMB_IP
    5: 4,   # Thumb_tip        -> THUMB_TIP
    7: 5,   # Index_proximal   -> INDEX_FINGER_MCP
    8: 6,   # Index_intermediate -> INDEX_FINGER_PIP
    9: 7,   # Index_distal     -> INDEX_FINGER_DIP
    10: 8,  # Index_tip        -> INDEX_FINGER_TIP
    12: 9,  # Middle_proximal  -> MIDDLE_FINGER_MCP
    13: 10, # Middle_intermediate -> MIDDLE_FINGER_PIP
    14: 11, # Middle_distal    -> MIDDLE_FINGER_DIP
    15: 12, # Middle_tip       -> MIDDLE_FINGER_TIP
    17: 13, # Ring_proximal    -> RING_FINGER_MCP
    18: 14, # Ring_intermediate -> RING_FINGER_PIP
    19: 15, # Ring_distal      -> RING_FINGER_DIP
    20: 16, # Ring_tip         -> RING_FINGER_TIP
    22: 17, # Little_proximal  -> PINKY_MCP
    23: 18, # Little_intermediate -> PINKY_PIP
    24: 19, # Little_distal    -> PINKY_DIP
    25: 20, # Little_tip       -> PINKY_TIP
}


def pico_hand_state_to_mediapipe(hand_state: np.ndarray) -> np.ndarray:
    """Convert PICO VR 27-joint (N,7) hand state → MediaPipe 21-joint (21,3).
    Positions are centred at the wrist."""
    mediapipe_state = np.zeros((21, 3), dtype=float)
    for pico_idx, mediapipe_idx in pico_to_mediapipe.items():
        mediapipe_state[mediapipe_idx] = hand_state[pico_idx, :3]
    return mediapipe_state - mediapipe_state[0:1, :]


def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
    """Compute wrist coordinate frame (MANO convention) from 21 MediaPipe keypoints."""
    assert keypoint_3d_array.shape == (21, 3)
    points = keypoint_3d_array[[0, 5, 9], :]
    x_vector = points[0] - points[2]
    points_centered = points - np.mean(points, axis=0, keepdims=True)
    _, _, v = np.linalg.svd(points_centered)
    normal = v[2, :]
    x = x_vector - np.sum(x_vector * normal) * normal
    x = x / (np.linalg.norm(x) + 1e-6)
    z = np.cross(x, normal)
    if np.sum(z * (points[1] - points[2])) < 0:
        normal *= -1
        z *= -1
    return np.stack([x, normal, z], axis=1)


# =============================================================================
# DexHandTracker — wraps dex-retargeting for vector / position retargeting
# =============================================================================
class DexHandTracker:
    def __init__(
        self,
        urdf_path: str = "",
        hand_type: HandType = HandType.right,
        retargeting_type: RetargetingType = RetargetingType.vector,
        config_dict: dict = None,
    ):
        self.retargeting_type = retargeting_type
        self.hand_type = hand_type
        self.urdf_path = urdf_path
        self.OPERATOR2MANO = OPERATOR2MANO[hand_type]

        if config_dict is not None:
            RetargetingConfig.set_default_urdf_dir(str(Path(urdf_path).parent))
            self.retargeting = RetargetingConfig.from_dict(config_dict).build()
        else:
            raise ValueError("config_dict is required for HX5 DexHandTracker")

    def retarget(self, hand_pos: np.ndarray, wrist_rot: np.ndarray = None) -> Optional[np.ndarray]:
        """Retarget MediaPipe (21,3) keypoints → robot joint qpos (pin_q)."""
        if hand_pos is None or hand_pos.shape != (21, 3):
            return None
        if wrist_rot is None:
            wrist_rot = estimate_frame_from_hand_points(hand_pos)
        transformed_pos = hand_pos @ wrist_rot @ self.OPERATOR2MANO

        indices = self.retargeting.optimizer.target_link_human_indices
        if self.retargeting_type == RetargetingType.position:
            ref_value = transformed_pos[indices, :]
        elif self.retargeting_type == RetargetingType.vector:
            origin_indices = indices[0, :]
            task_indices = indices[1, :]
            ref_value = transformed_pos[task_indices, :] - transformed_pos[origin_indices, :]
        else:
            raise NotImplementedError(f"Unsupported retargeting type: {self.retargeting_type}")

        try:
            qpos = self.retargeting.retarget(ref_value)
            return qpos
        except (RuntimeWarning, RuntimeError) as e:
            print(f"Warning: Retargeting failed: {e}")
            return None


# =============================================================================
# HX5-D20 retargeting config (11 vectors with intermediate keypoints)
# =============================================================================
HX5_RIGHT_RETARGETING_CONFIG = {
    "type": "vector",
    "urdf_path": "hx5_d20_right.urdf",
    "target_origin_link_names": [
        "hx5_d20_right_base", "hx5_d20_right_base", "hx5_d20_right_base",
        "hx5_d20_right_base", "hx5_d20_right_base",
        "hx5_d20_right_base", "hx5_d20_right_base",
        "hx5_d20_right_base", "hx5_d20_right_base",
        "hx5_d20_right_base", "hx5_d20_right_base",
    ],
    "target_task_link_names": [
        # Thumb: MCP(link2), IP(link3), tip(end1)
        "finger_r_link2", "finger_r_link3", "finger_end_r_link1",
        # Index: PIP(link6), tip(end2)
        "finger_r_link6", "finger_end_r_link2",
        # Middle: PIP(link10), tip(end3)
        "finger_r_link10", "finger_end_r_link3",
        # Ring: PIP(link14), tip(end4)
        "finger_r_link14", "finger_end_r_link4",
        # Pinky: PIP(link18), tip(end5)
        "finger_r_link18", "finger_end_r_link5",
    ],
    "target_link_human_indices": [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    ],
    "scaling_factor": 1.4,
    "low_pass_alpha": 0.4,
}

HX5_LEFT_RETARGETING_CONFIG = {
    "type": "vector",
    "urdf_path": "hx5_d20_left.urdf",
    "target_origin_link_names": [
        "hx5_d20_left_base", "hx5_d20_left_base", "hx5_d20_left_base",
        "hx5_d20_left_base", "hx5_d20_left_base",
        "hx5_d20_left_base", "hx5_d20_left_base",
        "hx5_d20_left_base", "hx5_d20_left_base",
        "hx5_d20_left_base", "hx5_d20_left_base",
    ],
    "target_task_link_names": [
        # Thumb: MCP(link2), IP(link3), tip(end1)
        "finger_l_link2", "finger_l_link3", "finger_end_l_link1",
        # Index: PIP(link6), tip(end2)
        "finger_l_link6", "finger_end_l_link2",
        # Middle: PIP(link10), tip(end3)
        "finger_l_link10", "finger_end_l_link3",
        # Ring: PIP(link14), tip(end4)
        "finger_l_link14", "finger_end_l_link4",
        # Pinky: PIP(link18), tip(end5)
        "finger_l_link18", "finger_end_l_link5",
    ],
    "target_link_human_indices": [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    ],
    "scaling_factor": 1.4,
    "low_pass_alpha": 0.4,
}


# =============================================================================
# HX5 actuator range (tuned joint limits)
# =============================================================================
HX5_RIGHT_ACTUATOR_RANGE = {
    # Thumb
    "finger_r_joint1":  (-0.5, 0.5),
    "finger_r_joint2":  (-1.57, 0.0),
    "finger_r_joint3":  (0.0, 0.6),
    "finger_r_joint4":  (0.0, 1.5708),
    # Index
    "finger_r_joint5":  (-0.15, 0.15),
    "finger_r_joint6":  (0.0, 2.0071),
    "finger_r_joint7":  (0.0, 1.5708),
    "finger_r_joint8":  (0.0, 1.5708),
    # Middle
    "finger_r_joint9":  (-0.15, 0.15),
    "finger_r_joint10": (0.0, 2.0071),
    "finger_r_joint11": (0.0, 1.5708),
    "finger_r_joint12": (0.0, 1.5708),
    # Ring
    "finger_r_joint13": (-0.15, 0.15),
    "finger_r_joint14": (0.0, 2.0071),
    "finger_r_joint15": (0.0, 1.5708),
    "finger_r_joint16": (0.0, 1.5708),
    # Pinky
    "finger_r_joint17": (-0.15, 0.15),
    "finger_r_joint18": (0.0, 2.0071),
    "finger_r_joint19": (0.0, 1.5708),
    "finger_r_joint20": (0.0, 1.5708),
}

# Build left-hand actuator range from right-hand (mirror thumb joints 1-4)
HX5_LEFT_ACTUATOR_RANGE = {}
for _rname, (_lo, _hi) in HX5_RIGHT_ACTUATOR_RANGE.items():
    _lname = _rname.replace("finger_r_", "finger_l_")
    _idx = int(_rname.split("joint")[1])
    if _idx <= 4:  # thumb joints: mirror the range
        HX5_LEFT_ACTUATOR_RANGE[_lname] = (-_hi, -_lo)
    else:
        HX5_LEFT_ACTUATOR_RANGE[_lname] = (_lo, _hi)


# =============================================================================
# Adaptive keypoint scaling
# =============================================================================
WRIST = 0
MIDDLE_TIP = 12

def adaptive_scale_keypoints(kp: np.ndarray, scale_open: float, scale_close: float, open_length: float) -> np.ndarray:
    """Scale keypoints relative to wrist based on hand openness."""
    wrist = kp[WRIST]
    cur_length = np.linalg.norm(kp[MIDDLE_TIP] - wrist)
    openness = np.clip(cur_length / open_length, 0.0, 1.0)
    scale = scale_close + (scale_open - scale_close) * openness
    return wrist + (kp - wrist) * scale


# =============================================================================
# Ctrl Smoother (time-based rate limit + EMA)
# =============================================================================
class CtrlSmoother:
    def __init__(self, alpha=0.5, max_speed=2.0):
        self.alpha = alpha
        self.max_speed = max_speed
        self.prev_ctrl = None
        self.prev_time = None

    def apply(self, ctrl: np.ndarray) -> np.ndarray:
        now = time.time()
        if self.prev_ctrl is None:
            self.prev_ctrl = ctrl.copy()
            self.prev_time = now
            return ctrl
        dt = max(now - self.prev_time, 1e-6)
        self.prev_time = now
        smoothed = self.alpha * ctrl + (1.0 - self.alpha) * self.prev_ctrl
        max_delta = self.max_speed * dt
        delta = np.clip(smoothed - self.prev_ctrl, -max_delta, max_delta)
        result = self.prev_ctrl + delta
        self.prev_ctrl = result.copy()
        return result

    def reset(self):
        self.prev_ctrl = None
        self.prev_time = None


# =============================================================================
# Hand calibration YAML loader
# =============================================================================
def load_hand_sizes(yaml_path: str) -> tuple:
    """Load human_hand_size and robot_hand_size from calibration YAML."""
    import yaml
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    calib = data["calibration"]
    human_size = calib["human_hand"]["size"]
    robot_size = calib["robot_hand"]["shadow"]["size"]
    return human_size, robot_size


# =============================================================================
# pin_q → Isaac Sim joint target mapping
# =============================================================================
def pin_q_to_isaac_hand_targets(
    pin_model: pin.Model,
    pin_q: np.ndarray,
    hand_joint_names: list,
) -> np.ndarray:
    """Convert Pinocchio joint config (pin_q) to Isaac Sim hand joint targets.

    The pin_q from dex-retargeting follows the URDF joint order.
    We extract only the actuated joints (skip universe) and map them
    by name to the Isaac Sim hand joint order.

    Args:
        pin_model: Pinocchio model built from the hand URDF.
        pin_q: Joint configuration from DexHandTracker.retarget().
        hand_joint_names: List of Isaac Sim hand joint names in order.

    Returns:
        np.ndarray of joint positions matching hand_joint_names order.
    """
    # Build name → pin_q value mapping
    pin_joint_names = [n for n in pin_model.names if n not in ("universe", "root_joint")]
    pin_q_map = {}
    for i, name in enumerate(pin_joint_names):
        pin_q_map[name] = float(pin_q[i])

    # Map to Isaac Sim order
    targets = np.zeros(len(hand_joint_names))
    for i, jname in enumerate(hand_joint_names):
        if jname in pin_q_map:
            targets[i] = pin_q_map[jname]
    return targets


# =============================================================================
# Convenience: get URDF paths relative to this module
# =============================================================================
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_right_urdf_path() -> str:
    return os.path.join(_MODULE_DIR, "hx5_d20_right.urdf")

def get_left_urdf_path() -> str:
    return os.path.join(_MODULE_DIR, "hx5_d20_left.urdf")

def get_calibration_yaml_path() -> str:
    return os.path.join(_MODULE_DIR, "hand_calibration.yaml")
