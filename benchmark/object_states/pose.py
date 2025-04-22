
# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np


class Pose:
    def __init__(self, obj, robot):
        self.robot = robot
        self.obj = obj
        self.obj_name = obj.name

    def get_value(self):
        pose = self.robot.get_prim_world_pose("/World/Objects/%s" % self.obj_name)
        return np.array(pose)

    def _set_value(self, new_value):
        raise NotImplementedError("Pose state currently does not support setting.")
