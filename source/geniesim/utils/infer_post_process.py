# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from copy import deepcopy
from hmac import new
import numpy as np
import time


def relabel_gripper_state(obs, limit):
    state_dict = obs["states"]
    # The last 2 elements of the state vector are the gripper states (G1/G2: 14, 15 | Aloha: 12, 13)
    if "G2" in obs.get("robot_cfg", ""):
        # For G2 with waist, gripper might be earlier, but assume last two for now or offset from end
        pass # Actually states length is dynamic. We should use the last two elements before waist if present or just -1/-2.
    
    # Safely index the last two elements of the arm before waist if applicable. 
    # But usually gripper states are always added last, or right after arm.
    # In pi_env.py, G2 waist joints are appended *after* gripper joints.
    
    # A safer approach: just use negative indexing based on how many gripper states there are? 
    # Actually wait, in pi_env.py:
    # 1. G1/G2: Arm (14) + Gripper (2) + Waist (0 or 5)
    # 2. Aloha (OpenPI format): Left Arm (6) + Left Gripper (1) + Right Arm (6) + Right Gripper (1) -> Total 14

    if "aloha" in obs.get("robot_cfg", "aloha") or len(state_dict) == 14:
        # OpenPI 14-D format interleaves them
        g_idx1 = 6
        g_idx2 = 13
    else:
        # Default G1/G2 format where grippers are after arm 14
        g_idx1 = 14
        g_idx2 = 15
    
    state_dict[g_idx1] = min(max(1 - state_dict[g_idx1] / limit, 0), 1) * 100 + 20
    state_dict[g_idx2] = min(max(1 - state_dict[g_idx2] / limit, 0), 1) * 100 + 20


def relabel_gripper_action(action, limit):
    new_action = np.zeros(2)
    new_action[0] = (1 - action[0]) * limit
    new_action[1] = (1 - action[1]) * limit

    return new_action


def abs_ee_to_abs_joint(ikfk_solver, arm_joint_state, action: np.ndarray):
    abs_eef_action = [action]
    joint_actions = ikfk_solver.eef_actions_to_joint(abs_eef_action, arm_joint_state, [0, 0])
    return joint_actions[0]


def process_action(ikfk_solver, arm_joint_state, action: np.ndarray, type, smooth_alpha=1.0):
    if type == "delta_ee":
        raise ValueError("Delta EE to Abs Joint is not supported")
    elif type == "abs_ee":
        return abs_ee_to_abs_joint(ikfk_solver, arm_joint_state, action)
    elif type == "abs_joint":
        return filter_abs_joint(arm_joint_state, action, smooth_alpha)
    else:
        raise ValueError(f"Failed to process unknown action type: {type}")


def filter_abs_joint(arm_joint_state, action, alpha):
    """
    Smooths the absolute joint action using an EMA (Exponential Moving Average) filter.
    This prevents jerky movements by moving only a fraction `alpha` of the way
    from the current state towards the target action in each step.
    """
    state_len = len(arm_joint_state)
    action_np = np.asarray(action)
    smoothed_part = (1 - alpha) * np.asarray(arm_joint_state) + alpha * action_np[:state_len]
    return np.concatenate([smoothed_part, action_np[state_len:]]).tolist()


class MovingAVGFilter:
    def __init__(self, qpos, trajectory, alpha=0.03, repeat=1, freq=30, sleep=None):
        self.trajectory = np.array(trajectory)
        self.i = 0
        self.curr_traj = np.array(qpos)
        self.alpha = alpha
        self.repeat = repeat
        self.sleep = 1 / freq / repeat if sleep is None else sleep

    def move(self, fn):
        for r in range(self.repeat):
            self.curr_traj = (1 - self.alpha) * self.curr_traj + self.alpha * self.trajectory[self.i]
            fn(self.curr_traj.tolist())
            time.sleep(self.sleep)
        self.i += 1
