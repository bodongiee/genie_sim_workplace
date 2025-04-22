# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from abc import ABCMeta, abstractmethod


class BaseTask:
    """
    Base Task class.
    Task-specific reset_scene, reset_agent, get_task_obs, step methods are implemented in subclasses
    Subclasses are expected to populate self.reward_functions and self.termination_conditions
    """

    def __init__(self, env):
        self.config = env.task_info
        self.reward_functions = []
        self.termination_conditions = []

    @abstractmethod
    def reset_scene(self, env):
        """
        Task-specific scene reset

        :param env: environment instance
        """
        raise NotImplementedError()

    @abstractmethod
    def reset_agent(self, env):
        """
        Task-specific agent reset

        :param env: environment instance
        """
        raise NotImplementedError()

    def reset_variables(self, env):
        """
        Task-specific variable reset

        :param env: environment instance
        """
        return

    def reset(self, env):
        self.reset_variables(env)
        for termination_condition in self.termination_conditions:
            termination_condition.reset(self, env)

    def get_reward(self, env, info={}):
        """
        Aggreate reward functions

        :param env: environment instance
        :param collision_links: collision links after executing action
        :param action: the executed action
        :param info: additional info
        :return reward: total reward of the current timestep
        :return info: additional info
        """
        reward = 0.0
        for reward_function in self.reward_functions:
            reward += reward_function.get_reward(self, env)

        return reward, info

    def get_termination(self, env, info={}):
        """
        Aggreate termination conditions

        :param env: environment instance
        :param collision_links: collision links after executing action
        :param action: the executed action
        :param info: additional info
        :return done: whether the episode has terminated
        :return info: additional info
        """
        done = False
        success = False
        for condition in self.termination_conditions:
            d, s = condition.get_termination(self, env)
            done = done or d
            success = success or s
            if d:
                info["done_cond_name"] = type(condition).__name__
                break

        info["done"] = done
        info["success"] = success

        return done, info

    @abstractmethod
    def get_task_obs(self, env):
        """
        Get task-specific observation

        :param env: environment instance
        :return: task-specific observation (numpy array)
        """
        raise NotImplementedError()

    def step(self, env):
        """
        Perform task-specific step for every timestep

        :param env: environment instance
        """
        return
