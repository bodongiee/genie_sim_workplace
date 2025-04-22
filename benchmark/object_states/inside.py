# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np
from .pose import Pose
from .aabb import AABB


def aabb_contains_point(point, container):
    lower, upper = container
    return np.less_equal(lower, point).all() and np.less_equal(point, upper).all()


class Inside:
    """
    better critic of aabb vs aabb, not simple intersection check
    """

    def __init__(self, obj, robot):
        self.obj = obj
        self.robot = robot

    @staticmethod
    def get_dependencies():
        return [AABB, Pose]

    def get_value(self, other):
        pose_A = self.obj.states[Pose].get_value()
        pos_A = pose_A[:3, 3].reshape(
            -1,
        )
        aa_B, bb_B = other.states[AABB].get_value()

        if other.is_articulated:
            _, links_aabb = other.get_link_info()
            for link_id, link_aabb in links_aabb.items():
                if aabb_contains_point(pos_A, link_aabb):
                    return True
            return False

        return aabb_contains_point(pos_A, (aa_B, bb_B))
