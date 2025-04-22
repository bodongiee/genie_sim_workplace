
# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np
import math
from . import Pose, BaseObjState


from base_utils.logger import Logger

logger = Logger()  # Create singleton instance

GREEN = "\033[92m"
RESET = "\033[0m"


class PourWater(BaseObjState):
    def __init__(self, obj, robot):
        super().__init__(obj, robot)
        self.robot = robot
        self.obj_name = obj.name
        self.kettle_lifted = False
        self.kettle_approach = False
        self.kettle_restore = False
        self.frame_lifted = 0
        self.frame_near_cup = 0
        self.frame_restore = 0

    def get_value(self, other):
        [kx, ky, kz] = self.robot.get_prim_world_pose("/World/Objects/%s" % self.obj_name)[0:3, 3]
        [cx, cy, cz] = self.robot.get_prim_world_pose("/World/Objects/%s" % other.name)[0:3, 3]
        [dx, dy, dz] = self.robot.get_prim_world_pose("/World/Objects/benchmark_coaster_015")[0:3, 3]

        kc_dist_horizon = np.linalg.norm(np.array([kx-cx, ky-cy]))
        kd_dist_horizon = np.linalg.norm(np.array([kx-dx, ky-dy]))

        if kz > 1:
            logger.info("kettle lifted")
            self.frame_lifted += 1
        if kc_dist_horizon < 0.15:
            logger.info("kettle near cup")
            self.frame_near_cup +=1

        if self.frame_lifted > 10:
            self.kettle_lifted = True

        if self.frame_near_cup > 10:
            self.kettle_approach = True

        if self.kettle_lifted and self.kettle_approach:
            if kd_dist_horizon < 0.1 and dz < 1:
                logger.info("kettle placed back")
                self.frame_restore += 1

        if self.frame_restore > 5:
            logger.info(
                "\n======================\n\n\nTask Success!!!\n\n\n======================"
            )
            return 1
        return 0

    def _set_value(self, new_value):
        raise NotImplementedError("Pour water state currently does not support setting.")
