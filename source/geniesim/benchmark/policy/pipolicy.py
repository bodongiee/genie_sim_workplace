# -*- coding: utf-8 -*-
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os, time
import json
from .base import BasePolicy

from scipy.spatial.transform import Rotation as R


from geniesim.plugins.logger import Logger
import geniesim.utils.system_utils as system_utils
from geniesim.utils.comm.websocket_client import WebsocketClientPolicy
from collections import deque
import numpy as np
import geniesim.utils.system_utils as system_utils

logger = Logger()  # Create singleton instance

ROOT_DIR = os.environ.get("SIM_REPO_ROOT")

TIMEOUT_SEC = 30


class PiPolicy(BasePolicy):
    def __init__(self, task_name, host_ip, port, sub_task_name=""):
        super().__init__(task_name=task_name, sub_task_name=sub_task_name)
        self.load_robot_task_config()
        self.ts_str = time.strftime("%Y%m%d_%H%M", time.localtime(time.time()))
        self.initialized = False
        self.policy = WebsocketClientPolicy(host=host_ip, port=port)
        self.debug = True
        self.infer_cnt = 0

    def get_payload(self, obs, task_instruction):
        s = np.asarray(obs["states"]).flatten()
        if self.robot_name == "aloha":
            states = s[:14]  # ALOHA: OpenPI AlohaInputs expects 14-dim state
        else:
            states = np.concatenate([s, np.zeros(32)])[:32]

        eef = obs["eef"]
        left_eef = np.asarray(eef["left"], dtype=np.float64)
        right_eef = np.asarray(eef["right"], dtype=np.float64)
        payload = {
            "state": states,
            "eef": {
                "left": left_eef.tolist(),
                "right": right_eef.tolist(),
            },
            "images": self._build_images(obs),
            "prompt": task_instruction,
            "task_name": self.sub_task_name,
        }
        if self.debug:
            print("task_name", payload["task_name"])
            print("state", payload["state"])
            print("eef", payload["eef"])
            print("prompt", payload["prompt"])
            # Save debug images on first inference
            if self.infer_cnt == 0:
                import cv2
                debug_dir = os.path.join(ROOT_DIR, "aloha_debug") if ROOT_DIR else "/tmp/aloha_debug"
                os.makedirs(debug_dir, exist_ok=True)
                for k, img in payload["images"].items():
                    # img is (C, H, W), convert to (H, W, C) BGR for cv2
                    path = os.path.join(debug_dir, f"{k}.png")
                    cv2.imwrite(path, np.transpose(img, (1, 2, 0))[:, :, ::-1])
                    print(f"[DEBUG] Saved image: {path}")
        return payload

    def _build_images(self, obs):
        images = obs["images"]
        if self.robot_name == "aloha":
            # ALOHA: OpenPI expects cam_high, cam_left_wrist, cam_right_wrist
            return {
                "cam_high": np.transpose(images["head"], (2, 0, 1)),
                "cam_left_wrist": np.transpose(images["left_hand"], (2, 0, 1)),
                "cam_right_wrist": np.transpose(images["right_hand"], (2, 0, 1)),
            }
        else:
            return {
                "top_head": np.transpose(images["head"], (2, 0, 1)),
                "hand_left": np.transpose(images["left_hand"], (2, 0, 1)),
                "hand_right": np.transpose(images["right_hand"], (2, 0, 1)),
            }

    def reset(self):
        self.action_buffer.clear()

    def infer(self, payload):
        try:
            result = self.policy.infer(payload)
            actions = result["actions"]
            if self.debug:
                print(f"[DEBUG] actions shape: {np.array(actions).shape}")
                print(f"[DEBUG] actions[0] (first step): {actions[0]}")
                print(f"[DEBUG] actions min/max: {np.min(actions):.4f} / {np.max(actions):.4f}")
            n = max(len(actions), 1)
            self.action_buffer = deque(actions, maxlen=n)
            return True
        except Exception as e:
            logger.warning(f"Model inference failed: {str(e)}")
            time.sleep(1)
            return False

    def act(self, observation, **kwargs):
        if len(self.action_buffer) == 0:
            logger.info("policy.at call model infer")
            task_instruction = kwargs.get("task_instruction", "")
            logger.info(f"\nInstruction: {task_instruction}\n")
            payload = self.get_payload(observation, task_instruction)
            infer_success = self.infer(payload)
            infer_start = time.time()

            while not infer_success:
                infer_success = self.infer(payload)
                if time.time() - infer_start > TIMEOUT_SEC:
                    logger.error("Model inference timeout, please check if the policy server is running normally")
                    os.system("pkill -9 -f app.py")
            self.infer_cnt += 1

        return self.action_buffer.popleft()
