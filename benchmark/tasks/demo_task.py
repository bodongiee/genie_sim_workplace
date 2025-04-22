
# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np
import copy


from .base_task import BaseTask
from .termination_conditions import Timeout, PredicateGoal
from .backend import IsaacSimBDDLBackend

from bddl.activity import (
    Conditions,
    evaluate_goal_conditions,
    get_goal_conditions,
    get_ground_goal_state_options,
    get_initial_conditions,
    get_natural_goal_conditions,
    get_object_scope,
)


class DemoTask(BaseTask):
    def __init__(self, env):
        super().__init__(env)
        self.termination_conditions = [
            Timeout(self.config),
            PredicateGoal(self.config),
        ]

        self.reward_functions = []

        self.backend = IsaacSimBDDLBackend()

        self.update_problem(
            env.specific_task_name,
            activity_definition=self.config.get("task_id", 0),
            predefined_problem=self.config.get("predefined_problem", None),
        )
        self.initialize(env)
        self.state_history = {}

    def update_problem(
        self, behavior_activity, activity_definition, predefined_problem
    ):
        self.behavior_activity = behavior_activity
        self.activity_definition = activity_definition
        self.conds = Conditions(
            behavior_activity,
            activity_definition,
            simulator_name="isaacsim",
            predefined_problem=predefined_problem,
        )
        self.object_scope = get_object_scope(self.conds)
        self.obj_inst_to_obj_cat = {
            obj_inst: obj_cat
            for obj_cat in self.conds.parsed_objects
            for obj_inst in self.conds.parsed_objects[obj_cat]
        }

        # Generate initial and goal conditions
        # self.initial_conditions = get_initial_conditions(self.conds, self.backend, self.object_scope)
        self.goal_conditions = get_goal_conditions(
            self.conds, self.backend, self.object_scope
        )  # goal condition, every head has multple condition
        self.ground_goal_state_options = get_ground_goal_state_options(
            self.conds, self.backend, self.object_scope, self.goal_conditions
        )  # goal condition -> initial condition / goal state

        self.instruction_order = np.arange(len(self.conds.parsed_goal_conditions))
        np.random.shuffle(self.instruction_order)
        self.currently_viewed_index = 0
        self.currently_viewed_instruction = self.instruction_order[
            self.currently_viewed_index
        ]
        self.current_success = False
        self.current_goal_status = {"satisfied": [], "unsatisfied": []}
        self.previous_goal_status = copy.deepcopy(self.current_goal_status)

    def initialize(self, env):
        self.assign_object_scope_with_cache(env)
        self.goal_conditions = get_goal_conditions(
            self.conds, self.backend, self.object_scope
        )
        self.ground_goal_state_options = get_ground_goal_state_options(
            self.conds, self.backend, self.object_scope, self.goal_conditions
        )

    def assign_object_scope_with_cache(self, env):
        for obj_inst in self.object_scope:
            matched_sim_obj = None
            for _, sim_obj in env.states_objects_by_name.items():
                if sim_obj.name.lower() == obj_inst:
                    matched_sim_obj = sim_obj
                    break
            self.object_scope[obj_inst] = matched_sim_obj

    def step(self, env):
        self.previous_goal_status = self.current_goal_status

    def check_success(self):
        self.current_success, self.current_goal_status = evaluate_goal_conditions(
            self.goal_conditions
        )
        return self.current_success, self.current_goal_status

    def get_termination(self, env, info={}):
        """
        Aggreate termination conditions and fill info
        """
        done, info = super(DemoTask, self).get_termination(env, info)
        info["goal_status"] = self.current_goal_status
        return done, info
