
# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np
from . import Pose, BaseObjState


class Pass2People(BaseObjState):
    def __init__(self, obj, robot):
        super().__init__(obj, robot)
        pose = self.robot.get_prim_world_pose("/World/people")
        self.people_pose = np.array(pose)

    @staticmethod
    def get_dependencies():
        return [Pose]

    def get_value(self, threshold=0.2):
        pose_obj_y = self.obj.states[Pose].get_value()[1, 3]
        pose_p_y = self.people_pose[1, 3]
        if abs(pose_obj_y - pose_p_y) < threshold:
            return 1
        return 0

    def _set_value(self, new_value):
        raise NotImplementedError(
            "Pass2People state currently does not support setting."
        )
