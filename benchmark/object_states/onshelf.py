# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np
from . import Pose, BaseObjState


from base_utils.logger import Logger

logger = Logger()  # Create singleton instance

GREEN = "\033[92m"
RESET = "\033[0m"


class OnShelf(BaseObjState):
    def __init__(self, obj, robot):
        super().__init__(obj, robot)
        self.robot = robot
        self.obj_name = obj.name
        self.success_frame = 0
        self.success_time = 1

    def get_value(self, threshold=0.2):
        [x, y, z] = self.robot.get_prim_world_pose("/World/Objects/%s" % self.obj_name)[
            0:3, 3
        ]
        if "004" in self.obj_name:
            if abs(x + 3.75) < 0.05 and abs(y + 0.06) < 0.04 and abs(z - 1.11) < 0.03:
                self.success_frame += 1
        if self.success_frame >= self.success_time:
            self.success_frame = 0
            logger.info(
                "\n======================\n\n\nTask Success!!!\n\n\n======================"
            )
            return 1
        else:
            return 0

    def _set_value(self, new_value):
        raise NotImplementedError("On Shelf state currently does not support setting.")
