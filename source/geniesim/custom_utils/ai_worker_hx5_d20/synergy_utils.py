# Synergy hand control utilities (no MuJoCo dependency).
# Extracted from synergy_control_hx5_d20.py for use in Isaac Sim teleop.

import os
import time

import numpy as np
import yaml


CALIBRATION_YAML_PATH = os.path.join(os.path.dirname(__file__), "synergy_calibration.yaml")

# PICO joint indices
PICO_THUMB_TIP = 5
PICO_INDEX_TIP = 10
PICO_INDEX_CHAIN = [6, 7, 8, 9, 10]
PICO_FINGER_CHAINS = {
    "middle": [11, 12, 13, 14, 15],
    "ring": [16, 17, 18, 19, 20],
    "pinky": [21, 22, 23, 24, 25],
}


# =============================================================================
# Calibration loader
# =============================================================================
def load_synergy_calibration(yaml_path: str, hand_type: str = "right") -> dict:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    calib = data["synergy_calibration"][hand_type]
    return {
        "pinch_open_dist": calib["pinch"]["open_dist"],
        "pinch_close_dist": calib["pinch"]["close_dist"],
        "grip_open_mcp": np.mean([calib["grip"][f]["open_mcp"] for f in ["middle", "ring", "pinky"]]),
        "grip_close_mcp": np.mean([calib["grip"][f]["close_mcp"] for f in ["middle", "ring", "pinky"]]),
    }


# =============================================================================
# Extract pinchVal & gripVal from VR hand state
# =============================================================================
def angle_between_vectors(v1, v2):
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def compute_index_pip_curl(pico_pos):
    a = pico_pos[PICO_INDEX_CHAIN[1]]  # proximal
    b = pico_pos[PICO_INDEX_CHAIN[2]]  # intermediate
    c = pico_pos[PICO_INDEX_CHAIN[3]]  # distal
    pip_angle = angle_between_vectors(a - b, c - b)
    curl = 1.0 - (pip_angle - 80.0) / (180.0 - 80.0)
    return float(np.clip(curl, 0.0, 1.0))


def extract_synergy_values(pico_pos, calib):
    pinch_dist = float(np.linalg.norm(pico_pos[PICO_THUMB_TIP] - pico_pos[PICO_INDEX_TIP]))
    pinch_range = calib["pinch_open_dist"] - calib["pinch_close_dist"]
    if pinch_range > 1e-6:
        pinch_val_raw = 1.0 - (pinch_dist - calib["pinch_close_dist"]) / pinch_range
    else:
        pinch_val_raw = 0.0
    pinch_val_raw = float(np.clip(pinch_val_raw, 0.0, 1.0))

    mcp_angles = []
    for finger, chain in PICO_FINGER_CHAINS.items():
        a = pico_pos[chain[0]]
        b = pico_pos[chain[1]]
        c = pico_pos[chain[2]]
        mcp_angles.append(angle_between_vectors(a - b, c - b))
    avg_mcp = np.mean(mcp_angles)

    grip_range = calib["grip_open_mcp"] - calib["grip_close_mcp"]
    if abs(grip_range) > 1e-6:
        grip_val = 1.0 - (avg_mcp - calib["grip_close_mcp"]) / grip_range
    else:
        grip_val = 0.0
    grip_val = float(np.clip(grip_val, 0.0, 1.0))

    index_curl = compute_index_pip_curl(pico_pos)
    pinch_gate = 1.0 - np.clip((index_curl - 0.5) / (0.8 - 0.5), 0.0, 1.0)
    pinch_val = float(np.clip(pinch_val_raw * pinch_gate, 0.0, 1.0))
    index_grip_blend = float(np.clip((index_curl - 0.5) / (0.8 - 0.5), 0.0, 1.0))

    return pinch_val, grip_val, index_grip_blend


# =============================================================================
# Pose blending
# =============================================================================
class GripStateTracker:
    def __init__(self, enter_thresh=0.8, exit_thresh=0.2):
        self.enter_thresh = enter_thresh
        self.exit_thresh = exit_thresh
        self.grip_established = False

    def update(self, grip_val):
        if grip_val > self.enter_thresh:
            self.grip_established = True
        elif grip_val < self.exit_thresh:
            self.grip_established = False
        return self.grip_established


THUMB_JOINTS = slice(0, 4)


def blend_synergy_pose(pinch_val, grip_val, index_grip_blend, grip_established, poses, thumb_pinch_lead=0.7):
    p_open = poses["open"]
    p_grip = poses["grip"]
    p_pinch_open = poses["pinch_other_open"]
    p_pinch_close = poses["pinch_other_close"]
    p_pinch_open_other_close = poses["pinch_open_other_close"]

    if grip_established:
        no_pinch_grip = p_pinch_open_other_close + (p_grip - p_pinch_open_other_close) * index_grip_blend
    else:
        no_pinch_grip = p_grip

    no_pinch = p_open + (no_pinch_grip - p_open) * grip_val
    pinch = p_pinch_open + (p_pinch_close - p_pinch_open) * grip_val

    thumb_pinch_val = float(np.clip(pinch_val ** (1.0 - thumb_pinch_lead), 0.0, 1.0)) if pinch_val > 0 else 0.0

    target = no_pinch + (pinch - no_pinch) * pinch_val
    target[THUMB_JOINTS] = no_pinch[THUMB_JOINTS] + (pinch[THUMB_JOINTS] - no_pinch[THUMB_JOINTS]) * thumb_pinch_val
    return target


# =============================================================================
# Ctrl Smoother
# =============================================================================
class CtrlSmoother:
    def __init__(self, alpha=0.5, max_speed=2.0):
        self.alpha = alpha
        self.max_speed = max_speed
        self.prev_ctrl = None
        self.prev_time = None

    def apply(self, ctrl):
        now = time.time()
        if self.prev_ctrl is None:
            self.prev_ctrl = ctrl.copy()
            self.prev_time = now
            return ctrl
        dt = now - self.prev_time
        if dt < 1e-6:
            dt = 1e-6
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
# HX5 actuator range
# =============================================================================
HX5_ACTUATOR_RANGE = {
    "finger_r_joint1": (-1.5708, 1.5708),
    "finger_r_joint2": (-3.1416, 0.0),
    "finger_r_joint3": (0.0, 1.5708),
    "finger_r_joint4": (0.0, 1.5708),
    "finger_r_joint5": (-0.6109, 0.6109),
    "finger_r_joint6": (0.0, 2.0071),
    "finger_r_joint7": (0.0, 1.5708),
    "finger_r_joint8": (0.0, 1.5708),
    "finger_r_joint9": (-0.6109, 0.6109),
    "finger_r_joint10": (0.0, 2.0071),
    "finger_r_joint11": (0.0, 1.5708),
    "finger_r_joint12": (0.0, 1.5708),
    "finger_r_joint13": (-0.6109, 0.6109),
    "finger_r_joint14": (0.0, 2.0071),
    "finger_r_joint15": (0.0, 1.5708),
    "finger_r_joint16": (0.0, 1.5708),
    "finger_r_joint17": (-0.6109, 0.6109),
    "finger_r_joint18": (0.0, 2.0071),
    "finger_r_joint19": (0.0, 1.5708),
    "finger_r_joint20": (0.0, 1.5708),
}
