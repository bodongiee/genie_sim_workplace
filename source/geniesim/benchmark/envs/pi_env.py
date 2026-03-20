# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import json
import time
import glob
import pickle
import numpy as np
import os
from copy import deepcopy
from scipy.spatial.transform import Rotation

from .dummy_env import DummyEnv

from geniesim.plugins.logger import Logger

logger = Logger()  # Create singleton instance

from geniesim.benchmark.tasks.llm_task import LLMTask
from geniesim.utils.name_utils import *
from geniesim.utils.infer_pre_process import *
from geniesim.utils.infer_post_process import *

GRIPPER_CLOSE = 0.021
GRIPPER_OPEN = 0.057

# ffw_sg2 gripper: revolute joint range [0.0, 1.1] rad
FFW_SG2_GRIPPER_CLOSE = 0.0
FFW_SG2_GRIPPER_OPEN = 1.1

class PiEnv(DummyEnv):
    def __init__(
        self,
        api_core,
        task_file: str,
        init_task_config,
        need_setup=True,
    ):
        super().__init__(
            api_core,
            task_file,
            init_task_config,
            need_setup,
        )
        self.LIMIT_VAL = 1.0
        self.FFW_SG2_LIMIT_VAL = FFW_SG2_GRIPPER_OPEN  # 1.1 rad
        self.load_task_setup()

    def load_task_setup(self):
        self.task = LLMTask(self)

    def get_observation(self):
        for i in range(10):
            images = self.data_courier.get_observation_image()
            if images == {}:
                time.sleep(0.1)
            else:
                break
        full_joint_states = self.data_courier.get_joint_state_dict()
        states = []
        raw_states = []
        if "G1" in self.robot_cfg:
            for name in G1_DUAL_ARM_JOINT_NAMES:
                val = full_joint_states[name]
                states.append(val)
                raw_states.append(val)
        elif "G2" in self.robot_cfg:
            for name in G2_DUAL_ARM_JOINT_NAMES:
                val = full_joint_states[name]
                states.append(val)
                raw_states.append(val)
        elif "aloha" in self.robot_cfg:
            # OpenPI Format: [Left Arm (6), Left Gripper (1), Right Arm (6), Right Gripper (1)]
            raw_left_arm = []
            for i, name in enumerate(ALOHA_LEFT_ARM_JOINT_NAMES):
                val = full_joint_states[name]
                raw_left_arm.append(val)
                states.append(val)

            # Normalize gripper to [0,1] for OpenPI's _gripper_to_angular()
            # Isaac Sim slide joint range: 0.021 (closed) ~ 0.057 (open)
            # gym_aloha convention: 0=close, 1=open
            left_gripper_raw = full_joint_states[ALOHA_GRIPPER_NAMES[0]]  # left_left_finger (positive)
            states.append((left_gripper_raw - GRIPPER_CLOSE) / (GRIPPER_OPEN - GRIPPER_CLOSE))

            raw_right_arm = []
            for i, name in enumerate(ALOHA_RIGHT_ARM_JOINT_NAMES):
                val = full_joint_states[name]
                raw_right_arm.append(val)
                states.append(val)
            right_gripper_raw = full_joint_states[ALOHA_GRIPPER_NAMES[2]]  # right_left_finger (positive)
            states.append((right_gripper_raw - GRIPPER_CLOSE) / (GRIPPER_OPEN - GRIPPER_CLOSE))
            # Use interleaved format [L_arm, L_grip, R_arm, R_grip] to match policy standard
            raw_states = raw_left_arm + [left_gripper_raw] + raw_right_arm + [right_gripper_raw]

            pass
        elif "ffw_sg2_follower" in self.robot_cfg:
            for name in FFW_SG2_DUAL_ARM_JOINT_NAMES:
                val = full_joint_states[name]
                states.append(val)
                raw_states.append(val)
        else:
            raise ValueError(f"Invalid robot cfg: {self.robot_cfg}")

        self.cur_arm = deepcopy(raw_states)

        if "aloha" not in self.robot_cfg:
            if "ffw_sg2_follower" in self.robot_cfg:
                for name in FFW_SG2_GRIPPER_JOINTS_NAMES:
                    states.append(full_joint_states[name])
            else:
                for name in OMNIPICKER_AJ_NAMES:
                    states.append(full_joint_states[name])

        if "G2" in self.robot_cfg:
            for name in G2_WAIST_JOINT_NAMES[::-1]:
                states.append(full_joint_states[name])

        if "ffw_sg2_follower" in self.robot_cfg:
            # Use only first 5 waist joints (reversed) to match G2 state dimension (21)
            for name in FFW_SG2_WAIST_JOINT_NAMES[:5][::-1]:
                states.append(full_joint_states[name])

        obs = {"images": images, "states": states}

        # Left/right gripper center eef pose [x, y, z, qw, qx, qy, qz]
        # Compute EEF using raw sim states (matching URDF)
        if "aloha" in self.robot_cfg:
            obs["eef"] = {"left": [0.0] * 7, "right": [0.0] * 7}
        else:
            print("[DEBUG]Solving IK")
            obs["eef"] = self.ikfk_solver.compute_eef(self.cur_arm)
            print("[DEBUG]Donea solving IK")
        if "aloha" not in self.robot_cfg:
            relabel_gripper_state(obs, self.LIMIT_VAL)
        return obs

    def reset(self):
        self._followed_objects = set()  # Clear on new scene/episode (scene generalization)
        self.last_update_time = time.time()
        self.has_done = False
        self.task.reset(self)
        self.robot_joint_indices = self.api_core.get_robot_joint_indices()
        eps = 1e-2
        for i in range(10):
            print("Robot reset...")
            if self.robot_cfg == "aloha":
                init_gripper = self.init_gripper
            else:
                init_gripper = [1 - v for v in self.init_gripper]

            # fmt: off
            if self.robot_cfg == "G1_omnipicker":
                self.api_core.set_joint_positions(self.init_arm,joint_indices=[self.robot_joint_indices[v] for v in G1_DUAL_ARM_JOINT_NAMES],is_trajectory=False)
                self.api_core.set_joint_positions(self.init_waist,joint_indices=[self.robot_joint_indices[v] for v in G1_WAIST_JOINT_NAMES],is_trajectory=False)
                self.api_core.set_joint_positions(self.init_head,joint_indices=[self.robot_joint_indices[v] for v in G1_HEAD_JOINT_NAMES],is_trajectory=False)
                self.api_core.set_joint_positions(init_gripper, joint_indices=[self.robot_joint_indices[v] for v in OMNIPICKER_AJ_NAMES], is_trajectory=False)
            elif self.robot_cfg == "G2_omnipicker":
                self.api_core.set_joint_positions(self.init_arm,joint_indices=[self.robot_joint_indices[v] for v in G2_DUAL_ARM_JOINT_NAMES], is_trajectory=False)
                self.api_core.set_joint_positions(self.init_waist, joint_indices=[self.robot_joint_indices[v] for v in G2_WAIST_JOINT_NAMES], is_trajectory=False)
                self.api_core.set_joint_positions(self.init_head,joint_indices=[self.robot_joint_indices[v] for v in G2_HEAD_JOINT_NAMES], is_trajectory=False)
                self.api_core.set_joint_positions(init_gripper, joint_indices=[self.robot_joint_indices[v] for v in OMNIPICKER_AJ_NAMES], is_trajectory=False)
            elif self.robot_cfg == "aloha":
                self.api_core.set_joint_positions(self.init_arm, joint_indices=[self.robot_joint_indices[v] for v in ALOHA_DUAL_ARM_JOINT_NAMES], is_trajectory=False)
                self.api_core.set_joint_positions(init_gripper, joint_indices=[self.robot_joint_indices[v] for v in ALOHA_GRIPPER_NAMES], is_trajectory=False)
            elif self.robot_cfg == "ffw_sg2_follower":
                self.api_core.set_joint_positions(self.init_arm, joint_indices=[self.robot_joint_indices[v] for v in FFW_SG2_DUAL_ARM_JOINT_NAMES], is_trajectory=False)
                self.api_core.set_joint_positions(self.init_waist,joint_indices=[self.robot_joint_indices[v] for v in FFW_SG2_WAIST_JOINT_NAMES],is_trajectory=False)
                self.api_core.set_joint_positions(self.init_head,joint_indices=[self.robot_joint_indices[v] for v in FFW_SG2_HEAD_JOINT_NAMES],is_trajectory=False)
                self.api_core.set_joint_positions(init_gripper, joint_indices=[self.robot_joint_indices[v] for v in FFW_SG2_GRIPPER_JOINTS_NAMES], is_trajectory=False)

            # fmt: on

            time.sleep(0.1)

            for i in range(10):
                full_joint_states = self.data_courier.get_joint_state_dict()
                if full_joint_states != {}:
                    break
                time.sleep(0.5)

            arm_position, waist_position = [], []
            if self.robot_cfg == "G1_omnipicker":
                for name in G1_DUAL_ARM_JOINT_NAMES:
                    arm_position.append(full_joint_states[name])
                for name in G1_WAIST_JOINT_NAMES:
                    waist_position.append(full_joint_states[name])
            elif self.robot_cfg == "G2_omnipicker":
                for name in G2_DUAL_ARM_JOINT_NAMES:
                    arm_position.append(full_joint_states[name])
                for name in G2_WAIST_JOINT_NAMES:
                    waist_position.append(full_joint_states[name])
            elif self.robot_cfg == "aloha":
                for name in ALOHA_DUAL_ARM_JOINT_NAMES:
                    arm_position.append(full_joint_states[name])

            elif self.robot_cfg == "ffw_sg2_follower":
                for name in FFW_SG2_DUAL_ARM_JOINT_NAMES:
                    arm_position.append(full_joint_states[name])
                for name in FFW_SG2_WAIST_JOINT_NAMES:
                    waist_position.append(full_joint_states[name])
                

            c1 = np.max(np.abs(np.array(arm_position) - np.array(self.init_arm))) < eps
            c2 = len(waist_position) == 0 or np.max(np.abs(np.array(waist_position) - np.array(self.init_waist))) < eps

            if c1 and c2:
                break

        logger.info("Finish reset robot...")
        time.sleep(1)
        self.api_core.reset_env()
        obs = self.get_observation()
        logger.info("Finish reset env...")
        return obs

    def step(self, action):
        self.current_step += 1
        need_update = False
        if self.current_step != 1 and self.current_step % 30 == 0:
            self.task.step(self)
            self.action_update()
            need_update = True

        if self.robot_cfg == "aloha":
            # OpenPI interleaves dimensions: 0:6 (Left Arm), 6 (Left Gripper), 7:13 (Right Arm), 13 (Right Gripper)
            left_arm_sim = action[0:6]
            right_arm_sim = action[7:13]

            # AlohaOutputs._gripper_from_angular returns ≈[0,1] where 0≈close, 1≈open
            # Convert to Isaac Sim linear position: 0→GRIPPER_CLOSE(0.021), 1→GRIPPER_OPEN(0.057)
            left_gripper_sim = np.clip(action[6], 0, 1) * (GRIPPER_OPEN - GRIPPER_CLOSE) + GRIPPER_CLOSE
            right_gripper_sim = np.clip(action[13], 0, 1) * (GRIPPER_OPEN - GRIPPER_CLOSE) + GRIPPER_CLOSE

            # Smooth arm joints only (alpha=0.5), gripper uses higher alpha for responsive grasping
            arm_action = np.concatenate([left_arm_sim, [left_gripper_sim], right_arm_sim, [right_gripper_sim]])
            arm_action = process_action(None, self.cur_arm, arm_action, type="abs_joint", smooth_alpha=0.5)

            aloha_arm_action = np.concatenate([arm_action[0:6], arm_action[7:13]])
            # Gripper: use less smoothing (alpha=0.9) so it closes/opens quickly for grasping
            left_gripper = 0.1 * self.cur_arm[6] + 0.9 * left_gripper_sim
            right_gripper = 0.1 * self.cur_arm[13] + 0.9 * right_gripper_sim

            # Map to 4 Aloha finger joints (Isaac Sim convention)
            # [0],[2] are positive direction, [1],[3] are negative (mimic)
            aloha_gripper_commands = [
                left_gripper,  # [0] left_left_finger (positive)
                -left_gripper,   # [1] left_right_finger (negative mimic)
                right_gripper,   # [2] right_left_finger (positive)
                -right_gripper,  # [3] right_right_finger (negative mimic)
            ]
            self.api_core.set_joint_positions([float(v) for v in aloha_arm_action], joint_indices=[self.robot_joint_indices[v] for v in ALOHA_DUAL_ARM_JOINT_NAMES], is_trajectory=True)
            self.api_core.set_joint_positions([float(v) for v in aloha_gripper_commands], joint_indices=[self.robot_joint_indices[v] for v in ALOHA_GRIPPER_NAMES], is_trajectory=True)
        else:
            action = process_action(self.ikfk_solver, self.cur_arm, action, type="abs_ee", smooth_alpha=0.5)
            gripper_action = relabel_gripper_action(action[14:16], self.LIMIT_VAL)
            if self.robot_cfg == "G1_omnipicker":
                self.api_core.set_joint_positions([float(v) for v in action[:14]],joint_indices=[self.robot_joint_indices[v] for v in G1_DUAL_ARM_JOINT_NAMES],is_trajectory=True)
                self.api_core.set_joint_positions([float(v) for v in gripper_action], joint_indices=[self.robot_joint_indices[v] for v in OMNIPICKER_AJ_NAMES], is_trajectory=True)
            elif self.robot_cfg == "G2_omnipicker":
                self.api_core.set_joint_positions([float(v) for v in action[:14]],joint_indices=[self.robot_joint_indices[v] for v in G2_DUAL_ARM_JOINT_NAMES],is_trajectory=True)
                self.api_core.set_joint_positions([float(v) for v in gripper_action], joint_indices=[self.robot_joint_indices[v] for v in OMNIPICKER_AJ_NAMES], is_trajectory=True)
                if len(action) > 16: # Including waist control
                    self.api_core.set_joint_positions([float(v) for v in action[20:21]],joint_indices=[self.robot_joint_indices[v] for v in G2_WAIST_JOINT_NAMES[0:1]],is_trajectory=True)
            elif self.robot_cfg == "ffw_sg2_follower":
                self.api_core.set_joint_positions([float(v) for v in action[:14]],joint_indices=[self.robot_joint_indices[v] for v in FFW_SG2_DUAL_ARM_JOINT_NAMES],is_trajectory=True)
                self.api_core.set_joint_positions([float(v) for v in gripper_action], joint_indices=[self.robot_joint_indices[v] for v in FFW_SG2_GRIPPER_JOINTS_NAMES], is_trajectory=True)
                if len(action) > 16: # Including waist control
                    self.api_core.set_joint_positions([float(v) for v in action[20:21]],joint_indices=[self.robot_joint_indices[v] for v in FFW_SG2_WAIST_JOINT_NAMES[0:1]],is_trajectory=True)
        # fmt: on
        next_obs = self.get_observation()
        if self.data_courier.enable_ros:
            self.data_courier.sim_ros_node.publish_image()
        return next_obs, self.has_done, need_update, self.task.task_progress
