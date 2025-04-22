# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np
from .base_state import BaseObjState


class AABB(BaseObjState):
    """
    Note: Objects and hinged objects in scenes are not currently supported
    """

    def __init__(self, obj, robot):
        super().__init__(obj, robot)

    def get_value(self):
        # Position of objects in world coordinate system
        pose = self.robot.get_prim_world_pose("/World/Objects/%s" % self.obj_name)
        # Boundary box of objects
        x_min, y_min, z_min = -self.size[0] / 2, -self.size[1] / 2, -self.size[2] / 2
        x_max, y_max, z_max = self.size[0] / 2, self.size[1] / 2, self.size[2] / 2
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
        # The coordinates of the bounding box of an object under the world coordinate system
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
        return room_aabb_low, room_aabb_hi
