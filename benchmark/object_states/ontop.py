
# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from . import AABB, Pose, BaseObjState


class OnTop(BaseObjState):
    def __init__(self, obj, robot):
        super().__init__(obj, robot)

    @staticmethod
    def get_dependencies():
        return [AABB, Pose]

    def get_value(self, other, threshold=0.5):
        pose_A_z = self.obj.states[Pose].get_value()[2, 3]
        pos_B_z = other.states[Pose].get_value()[2, 3]

        aa_A, bb_A = self.obj.states[AABB].get_value()
        aa_B, bb_B = other.states[AABB].get_value()

        z_diff_A = bb_A[2] - aa_A[2]
        z_diff_B = bb_B[2] - aa_B[2]

        if pose_A_z <= pos_B_z or (pose_A_z - pos_B_z) > z_diff_A + z_diff_B:
            return False

        inter_bbox = [
            max(aa_A[0], aa_B[0]),
            max(aa_A[1], aa_B[1]),
            min(bb_A[0], bb_B[0]),
            min(bb_A[1], bb_B[1]),
        ]
        inter_area = (inter_bbox[2] - inter_bbox[0]) * (inter_bbox[3] - inter_bbox[1])
        area_A = (bb_A[0] - aa_A[0]) * (bb_A[1] - aa_A[1])
        if inter_area / area_A > threshold:
            return True
        else:
            return False
