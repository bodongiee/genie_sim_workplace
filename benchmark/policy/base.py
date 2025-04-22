# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np


class BasePolicy:
    def __init__(self) -> None:
        pass

    def reset(self):
        """Called at the beginning of an episode."""
        pass

    def set_robot(self, robot, cam_dict):
        pass

    def act(self, observations, **kwargs) -> np.ndarray:
        """Act based on the observations."""
        pass


class RandomPolicy(BasePolicy):
    def __init__(self, action_space=1):
        self.action_space = action_space

    def act(self, observations, **kwargs):
        action = np.random.uniform(low=-1, high=1, size=(self.action_dim,))
        return action

    @classmethod
    def get_obs_mode(cls, env_id: str) -> str:
        return "rgbd"

    @classmethod
    def get_control_mode(cls, env_id: str) -> str:
        return None
