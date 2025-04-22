
# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np


class InSideStatus:
    def __init__(self, obj):
        self.obj = obj

    def evaluate(self, container, env, partial_check=False):
        fixtr_p0, fixtr_px, fixtr_py, fixtr_pz = container.get_int_sites(relative=False)
        u = fixtr_px - fixtr_p0
        v = fixtr_py - fixtr_p0
        w = fixtr_pz - fixtr_p0

        # get the position and quaternion of object
        obj_pos = np.array(env.sim.data.body_xpos[env.obj_body_id[obj.name]])
        obj_quat = T.convert_quat(
            env.sim.data.body_xquat[env.obj_body_id[obj.name]], to="xyzw"
        )

        if partial_check:
            obj_points_to_check = [obj_pos]
            th = 0.0
        else:
            # calculate 8 boundary points of object
            obj_points_to_check = obj.get_bbox_points(trans=obj_pos, rot=obj_quat)
            # threshold to mitigate false negatives: even if the bounding box point is out of bounds,
            th = 0.05

        inside_of = True
        for obj_p in obj_points_to_check:
            check1 = (
                np.dot(u, fixtr_p0) - th <= np.dot(u, obj_p) <= np.dot(u, fixtr_px) + th
            )
            check2 = (
                np.dot(v, fixtr_p0) - th <= np.dot(v, obj_p) <= np.dot(v, fixtr_py) + th
            )
            check3 = (
                np.dot(w, fixtr_p0) - th <= np.dot(w, obj_p) <= np.dot(w, fixtr_pz) + th
            )

            if not (check1 and check2 and check3):
                inside_of = False
                break

        return inside_of
