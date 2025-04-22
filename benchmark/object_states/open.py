# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np
from . import BaseObjState, AABB, Pose


class Open(BaseObjState):
    def __init__(self, obj, robot):
        super().__init__(obj, robot)

    @staticmethod
    def get_dependencies():
        return [AABB, Pose]

    def compute_diff(self, pose1, pose2):
        trans_dist = np.linalg.norm(pose1[:3, 3] - pose2[:3, 3])
        rotation_diff = np.dot(pose1[:3, :3], pose2[:3, :3].T)
        trace = np.trace(rotation_diff)
        trace = trace if trace <= 3 else 3
        angular_dist = np.rad2deg(np.arccos((trace - 1.0) / 2.0))
        return trans_dist, angular_dist

    def get_value(self, threshold=0.01):
        # assuming init state is closed, more info is needed, more joint info
        init_links_pose = self.obj.init_links_pose
        current_links_pose, _ = self.obj.get_link_info()

        for link_name, init_link_pose in init_links_pose.items():
            current_link_pose = current_links_pose[link_name]
            pos_diff, rot_diff = self.compute_diff(init_link_pose, current_link_pose)
            if pos_diff > threshold:
                return True

        return False
