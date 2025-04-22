
# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np
import json

from object_states import get_default_states, get_states_for_ability


class BaseObject(object):

    def __init__(self, obj_info_file="", robot=None, obj_info=None):
        if obj_info is not None and obj_info["model_path"] is not None:
            self.name = obj_info["object_id"]
            self.size = np.array(obj_info["size"]) / 1000.0
            self.model_path = obj_info["model_path"]
            self.is_articulated = obj_info.get("is_articulated", False)
            if self.is_articulated:
                self.link_ids = obj_info["link_id"]
                self.part_size = obj_info["part_size"]
                for name, size in self.part_size.items():
                    self.part_size[name] = np.array(size) / 1000.0
        else:
            # Load object info from object parameter file
            obj_cfg = json.load(open(obj_info_file))
            self.name = obj_cfg["object_id"]
            self.size = np.array(obj_cfg["size"]) / 1000.0
            self.model_path = obj_cfg["model_path"]
            self.is_articulated = obj_cfg.get("is_articulated", False)
            if self.is_articulated:
                self.link_ids = obj_cfg["link_id"]
                self.part_size = obj_cfg["part_size"]
                for name, size in self.part_size.items():
                    self.part_size[name] = np.array(size) / 1000.0

        self.robot = robot

    def load(self):
        pass


class StatefulObject(BaseObject):
    def __init__(self, obj_info_file="", robot=None, abilities=None, obj_info=None):
        super().__init__(obj_info_file, robot, obj_info)
        self.states = {}
        self.prepare_object_states(abilities)

    def prepare_object_states(self, abilities):
        state_types_and_params = [(state, {}) for state in get_default_states()]
        if abilities is not None:
            for ability, params in abilities.items():
                state_types_and_params.extend(
                    (state_name, params)
                    for state_name in get_states_for_ability(ability)
                )

        for state_type, params in reversed(state_types_and_params):
            self.states[state_type] = state_type(self, self.robot)


class USDObject(StatefulObject):
    def __init__(self, obj_info_file="", robot=None, abilities=None, obj_info=None):
        super().__init__(obj_info_file, robot, abilities, obj_info)
        if self.is_articulated:
            self.add_state("openable")
            self.init_links_pose, _ = self.get_link_info()

    def add_state(self, ability):
        state_types_and_params = [
            (state, {}) for state in get_states_for_ability(ability)
        ]
        for state_type, params in reversed(state_types_and_params):
            self.states[state_type] = state_type(self, self.robot)

    def get_link_info(self):
        links_pose = {}
        links_aabb = {}

        for link_id in self.link_ids:
            pose = self.robot.get_prim_world_pose(
                "/World/Objects/%s/%s" % (self.name, link_id)
            )
            links_pose[link_id] = pose
            # obj bbox
            link_size = self.part_size[link_id]
            x_min, y_min, z_min = (
                -link_size[0] / 2,
                -link_size[1] / 2,
                -link_size[2] / 2,
            )
            x_max, y_max, z_max = link_size[0] / 2, link_size[1] / 2, link_size[2] / 2
            bbox_3d = np.array(
                [
                    [x_min, y_min, z_min],
                    [x_max, y_min, z_min],
                    [x_max, y_max, z_min],
                    [x_min, y_max, z_min],
                    [x_min, y_min, z_max],
                    [x_max, y_min, z_max],
                    [x_max, y_max, z_max],
                    [x_min, y_max, z_max],
                ]
            )
            # obj bbox at global cord
            p3d_world = np.dot(pose[:3, :3], bbox_3d.T) + pose[:3, 3:]

            x_max = np.max(p3d_world[0, :])
            x_min = np.min(p3d_world[0, :])
            y_max = np.max(p3d_world[1, :])
            y_min = np.min(p3d_world[1, :])
            z_max = np.max(p3d_world[2, :])
            z_min = np.min(p3d_world[2, :])

            room_aabb_low, room_aabb_hi = np.array([x_min, y_min, z_min]), np.array(
                [x_max, y_max, z_max]
            )
            links_aabb[link_id] = (room_aabb_low, room_aabb_hi)

        return links_pose, links_aabb

    def load(self):
        pass
