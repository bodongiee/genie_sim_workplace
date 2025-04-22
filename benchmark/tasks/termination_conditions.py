
# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from abc import ABCMeta, abstractmethod


class BaseTerminationCondition:
    """
    Base TerminationCondition class
    Condition-specific get_termination method is implemented in subclasses
    """

    __metaclass__ = ABCMeta

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def get_termination(self, task, env):
        """
        Return whether the episode should terminate. Overwritten by subclasses.

        :param task: task instance
        :param env: environment instance
        :return: done, info
        """
        raise NotImplementedError()

    def reset(self, task, env):
        """
        Termination condition-specific reset

        :param task: task instance
        :param env: environment instance
        """
        return


class PredicateGoal(BaseTerminationCondition):
    """
    PredicateGoal used for BehaviorTask
    Episode terminates if all the predicates are satisfied
    """

    def __init__(self, config):
        super(PredicateGoal, self).__init__(config)

    def get_termination(self, task, env):
        """
        Return whether the episode should terminate.
        Terminate if point goal is reached (distance below threshold)

        :param task: task instance
        :param env: environment instance
        :return: done, info
        """
        done, _ = task.check_success()
        success = done
        return done, success


class Timeout(BaseTerminationCondition):
    """
    Timeout
    Episode terminates if max_step steps have passed
    """

    def __init__(self, config):
        super(Timeout, self).__init__(config)
        self.max_step = self.config.get("max_step", 6000)

    def get_termination(self, task, env):
        """
        Return whether the episode should terminate.
        Terminate if max_step steps have passed

        :param task: task instance
        :param env: environment instance
        :return: done, info
        """
        done = env.current_step >= self.max_step
        success = False
        return done, success
