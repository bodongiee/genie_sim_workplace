
# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from pathlib import Path
from typing import Any, Dict, List, Union
import itertools


class DummyInfer:
    def __init__(
        self,
        model_name_or_path: Union[str, Path],
        data_statistic_path: Union[str, Path] = None,
        session_length: int = 4096,
        image_size: int = 448,
        down_sample_ratio: float = 0.5,
        patch_size: int = 14,
        ctrl_freq: int = 30,
        dynamic_image_size: bool = False,
        verbose: bool = False,
        end_effector: str = "g1_gripper",
        control_mode: str = "joint",
        use_state: bool = True,
        use_action_mask: bool = True,
        action_dim: int = 20,
    ) -> Path:
        """Load and init model."""
        pass

    def predict_action(self, payload: Dict[str, Any]) -> str:
        # 1. Parse payload components

        # 2. Generate input

        # 3. Input into model

        # 4. Return actions
        pass


def get_actions_lerobot(model, params, use_state=True, control_mode: str = "joint"):
    """
    input lerobot format, output g1 action
    """

    cam_head = params["observation.images.cam_top"]
    cam_right = params["observation.images.cam_right_wrist"]
    cam_left = params["observation.images.cam_left_wrist"]

    payload = {
        "cam_tensor_head_top": cam_head,
        "cam_tensor_wrist_right": cam_right,
        "cam_tensor_wrist_left": cam_left,
        "instruction": params["instruction"],
        "state": params["observation.state"],
    }

    actions = model.predict_action(payload)

    output_dict = {
        "state": params["observation.state"],
        "action_mask": model.action_mask,
        "pred": actions,
    }

    res = convert_g1_actions(model.end_effector, actions, control_mode), output_dict
    return res


def convert_g1_actions(end_effector, actions, control_mode: str):
    if actions.shape[-1] == 44:
        # action dim 44
        if control_mode == "joint":
            l_control_start = 2
            l_control_end = 9
            r_control_start = 22
            r_control_end = 29
        elif control_mode == "eef":
            l_control_start = 15
            l_control_end = 21
            r_control_start = 35
            r_control_end = 41
        if end_effector == "g1_gripper":
            l_effector_start = 21
            l_effector_end = 22
            r_effector_start = 41
            r_effector_end = 42
        elif end_effector == "g1_dex_hand":
            l_effector_start = 9
            l_effector_end = 15
            r_effector_start = 29
            r_effector_end = 35
    elif actions.shape[-1] == 20:
        # action dim 20
        if control_mode == "joint":
            l_control_start = 2
            l_control_end = 9
            r_control_start = 10
            r_control_end = 17
        if end_effector == "g1_gripper":
            l_effector_start = 9
            l_effector_end = 10
            r_effector_start = 17
            r_effector_end = 18
        # action dim 14
        if control_mode == "eef":
            l_control_start = 0
            l_control_end = 6
            r_control_start = 7
            r_control_end = 13

        if end_effector == "g1_gripper":
            l_effector_start = 6
            l_effector_end = 7
            r_effector_start = 13
            r_effector_end = 14

    indices = list(
        itertools.chain(
            range(l_control_start, l_control_end),
            range(r_control_start, r_control_end),
            range(l_effector_start, l_effector_end),
            range(r_effector_start, r_effector_end),
        )
    )
    results = actions[0][:, indices]

    return results.tolist()
