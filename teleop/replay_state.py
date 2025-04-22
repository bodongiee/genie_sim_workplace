# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np
import os
import json
import argparse
import time
import sys
sys.path.append(os.environ.get("SIM_REPO_ROOT"))
from robot.isaac_sim.client import Rpc_Client
from robot.utils import (
    get_quaternion_from_rotation_matrix,
    get_quaternion_from_euler,
    matrix_to_euler_angles,
)
import argparse

from base_utils.logger import Logger

logger = Logger()

class Recording:
    def __init__(
        self,
        client_host="localhost:50051",
        state_file="task/task.json",
        task_file="task/task.json",
        fps=60,
        use_recording=False,
    ):
        self.client = Rpc_Client(client_host)
        self.data_root = os.path.dirname(__file__) + "/assets"
        self.fps = 30
        self.use_recording = use_recording

        with open(state_file, "r") as f:
            self.state = json.load(f)

        with open(task_file, "r") as f:
            self.task = json.load(f)

        translation_matrix = np.array(self.state["frames"][0]["robot"]["pose"])
        scene_usd = self.state["scene"]["scene_usd"]
        target_rotation_matrix = translation_matrix[:3, :3]
        target_position = translation_matrix[:3, 3]
        target_rotation = get_quaternion_from_euler(
            matrix_to_euler_angles(target_rotation_matrix), order="ZYX"
        )
        self.init_translation_matrix = translation_matrix
        self.client.InitRobot(
            robot_cfg="G1_120s.json",
            robot_usd="",
            scene_usd=scene_usd,
            init_position=target_position,
            init_rotation=target_rotation,
        )
        self.robot_pose = self.state["frames"][0]["robot"]["pose"]
        self.object_info = self.state["frames"][0]["objects"]
        self.articulated_object_info = self.state["frames"][0]["articulated_object"]
        self.run()
        self.client.Exit()

    def fetch_object_info(self):
        self.task_objects = {}
        object_category = [self.task["objects"]["extra_objects"], self.task["objects"]["fix_objects"], self.task["objects"]["task_related_objects"], self.task["objects"].get("articulated_objects", {})]
        for object_set in object_category:
            for object in object_set:
                data_info_dir = object["data_info_dir"]
                if "metadata" in object:
                    model_path = object["metadata"]["info"]["model_path"]
                elif "model_path" in object:
                    model_path = object["model_path"]
                else:
                    continue
                self.task_objects[object["object_id"]] = {"data_info_dir": data_info_dir, "model_path": model_path}

    def add_task_objects(self):
        for key, value in self.object_info.items():
            if key in self.task_objects:
                usd_path = self.task_objects[key]["model_path"]
                target_matrix = self.init_translation_matrix @ (
                    np.linalg.inv(self.robot_pose) @ value["pose"]
                )
                target_rotation_matrix, target_position = (
                    target_matrix[:3, :3],
                    target_matrix[:3, 3],
                )
                target_rotation = get_quaternion_from_euler(
                    matrix_to_euler_angles(target_rotation_matrix), order="ZYX"
                )
                self.client.add_object(
                    usd_path=usd_path,
                    prim_path="/World/Objects/" + key,
                    label_name=key,
                    target_position=target_position,
                    target_quaternion=target_rotation,
                    target_scale=np.array([1, 1, 1]),
                    color=np.array([1, 1, 1]),
                    material="general",
                    mass=0.01,
                )
            else:
                logger.warning("Object {} not found in task_objects".format(key))

    def add_articulated_objects(self):
        for key, value in self.articulated_object_info.items():
            if key in self.task_objects:
                usd_path = self.task_objects[key]["model_path"]
                if "pose" in value:
                    object_pose = value["pose"]
                elif "selfModeling_table_001_Base" in self.object_info:
                    object_pose = self.object_info["selfModeling_table_001_Base"]["pose"]
                else:
                    object_pose = self.object_info["Base"]["pose"]
                target_matrix = self.init_translation_matrix @ (
                    np.linalg.inv(self.robot_pose) @ object_pose
                )
                target_rotation_matrix, target_position = (
                    target_matrix[:3, :3],
                    target_matrix[:3, 3],
                )
                target_rotation = get_quaternion_from_euler(
                    matrix_to_euler_angles(target_rotation_matrix), order="ZYX"
                )
                self.client.add_object(
                    usd_path=usd_path,
                    prim_path="/World/Objects/" + key,
                    label_name=key,
                    target_position=target_position,
                    target_quaternion=target_rotation,
                    target_scale=np.array([1] * 3),
                    color=np.array([1, 1, 1]),
                    material="general",
                    add_particle=False,
                    mass=0.01,
                )

    def run(self):
        state = self.state["frames"]
        self.camera_list = self.task["recording_setting"]["camera_list"]
        task_name = self.task["task"]
        target_joint = state[0]["robot"]["joints"]["joint_position"]
        joint_names = []
        joint_positions = self.client.get_joint_positions().states
        for key in joint_positions:
            joint_names.append(key.name)

        joint_num = len(joint_positions)
        joint_indices = []
        for key in target_joint:
            if key in joint_names:
                index = joint_names.index(key)
                joint_indices.append(index)

        self.fetch_object_info()
        self.add_task_objects()
        self.add_articulated_objects()

        target_joint_positions = [0] * joint_num
        joint_state = list(state[0]["robot"]["joints"]["joint_position"])
        if "Left_Left_RevoluteJoint" in state[0]["robot"]["joints"]["joint_name"]:
            joint_list = joint_state
        else:
            joint_list = []
            for joint in joint_state:
                joint_list.append(joint)
            joint_list[20] = joint_state[22]
            joint_list[21] = joint_state[23]
        self.client.set_joint_positions(joint_list, False)
        time.sleep(1)
        if self.use_recording:
            self.client.start_recording(
                task_name=task_name,
                fps=self.fps,
                data_keys={
                    "camera": {
                        "camera_prim_list": self.camera_list,
                        "render_depth": False,
                        "render_semantic": False,
                    },
                    "pose": ["/World/G1/gripper_center"],
                    "joint_position": False,
                    "gripper": False,
                },
            )
        robot_position_last, robot_rotation_last = None, None
        BASE_MOVE_THRESH = 1E-3
        for idx in range(len(state)):
            # set robot pose
            robot_pose_mat = np.array(state[idx]["robot"]["pose"])
            robot_position = robot_pose_mat[:3, 3]
            robot_rotation = get_quaternion_from_euler(
                matrix_to_euler_angles(robot_pose_mat[:3, :3]), order="ZYX"
            )
            if idx != 0:
                if np.linalg.norm(robot_position - robot_position_last) > BASE_MOVE_THRESH or np.linalg.norm(robot_rotation - robot_rotation_last) > BASE_MOVE_THRESH:
                    self.client.SetObjectPose([{
                        "prim_path": "robot",
                        "position": robot_position,
                        "rotation": robot_rotation,
                    }], [])
                    target_joint_positions = [0] * joint_num
                    joint_state= list()
                    # self.robot_pose = robot_pose_mat

            robot_position_last = robot_position
            robot_rotation_last = robot_rotation

            # set object pose
            object_info = state[idx]["objects"]
            object_poses = []
            for key, value in object_info.items():
                if key in self.task_objects:
                    object_pose = {}
                    target_matrix = self.init_translation_matrix @ (
                        np.linalg.inv(self.robot_pose) @ value["pose"]
                    )
                    target_rotation_matrix, target_position = (
                        target_matrix[:3, :3],
                        target_matrix[:3, 3],
                    )
                    target_rotation = get_quaternion_from_euler(
                        matrix_to_euler_angles(target_rotation_matrix), order="ZYX"
                    )
                    object_pose["prim_path"] = "/World/Objects/" + key
                    object_pose["position"] = target_position
                    object_pose["rotation"] = target_rotation
                    object_poses.append(object_pose)

            target_joint_positions = [0] * joint_num
            joint_state = list(state[idx]["robot"]["joints"]["joint_position"])
            if "Left_Left_RevoluteJoint" in state[0]["robot"]["joints"]["joint_name"]:
                joint_list = joint_state
            else:
                joint_list = []
                for joint in joint_state:
                    joint_list.append(joint)
                joint_list[20] = joint_state[22]
                joint_list[21] = joint_state[23]
            for _idx in range(len(joint_indices)):
                target_joint_positions[joint_indices[_idx]] = joint_list[_idx]
            object_joints = []
            for key, value in state[idx]["articulated_object"].items():
                object_joint = {}
                object_joint["prim_path"] = "/World/Objects/" + key
                object_joint["joint_cmd"] = value["joints"]["joint_position"]
                object_joints.append(object_joint)
            self.client.SetObjectPose(object_poses, joint_list, object_joints)
        self.client.stop_recording()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SimGraspingAgent Command Line Interface"
    )
    parser.add_argument(
        "--client_host",
        type=str,
        default="localhost:50051",
        help="The client host for SimGraspingAgent (default: localhost:50051)",
    )
    parser.add_argument(
        "--state_file",
        type=str,
        default="task/task.json",
        help="",
    )
    parser.add_argument("--task_file", type=str, default="", help="")
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="fps of the video",
    )
    parser.add_argument("--record", action="store_true")

    args = parser.parse_args()
    recording = Recording(
        args.client_host,
        state_file=args.state_file,
        task_file=args.task_file,
        fps=args.fps,
        use_recording=args.record,
    )
