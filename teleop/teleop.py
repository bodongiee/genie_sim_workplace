# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import argparse
import os, sys
import json
import numpy as np
import glob

root_dir = os.environ.get("SIM_REPO_ROOT")
work_dir = os.path.join(root_dir, "teleop")
sys.path.append(root_dir)
sys.path.append(work_dir)
sys.path.append(os.path.join(root_dir, "benchmark"))
from benchmark.envs.demo_env import DemoEnv
from robot.genie_robot import IsaacSimRpcRobot
from layout.task_generate import TaskGenerator
from vr_server import VRServer
from scipy.spatial.transform import Rotation as R
import math
from pynput import keyboard
import threading

from base_utils.logger import Logger

logger = Logger()


def scale_quat_to_euler(q, s):
    rotation = R.from_quat(q)
    rpy = rotation.as_euler("xzy", degrees=False)
    r, p, y = rpy
    if r >= 0:
        r_trans = math.pi - r
    else:
        r_trans = -math.pi - r
    p_trans = -p
    y_trans = y
    rpy_trans = np.array([p_trans, -r_trans, y_trans])
    rpy_scaled = rpy_trans * s

    return rpy_scaled


class TeleOp(object):
    def __init__(self, args):
        if args.task_name != "":
            self.task_name = args.task_name
        else:
            raise ValueError("Invalid task_name or task_id")
        self.args = args
        self.task_config = None
        self.mode = args.mode
        self.record = args.record
        self.fps = args.fps
        self.episodes_per_instance = 1
        self.host_ip = args.host_ip
        self.port = args.port
        if self.mode == "pico":
            self.init_pico_control()
        elif self.mode == "keyboard":
            self.init_keyboard_control()
        else:
            raise ValueError("Invalid mode")
        self.lock = threading.Lock()
        self.reset_flg = False
        self.switch_flg = False

    def init_pico_control(self):
        self.vr_server = VRServer(self.host_ip, self.port)

    def init_keyboard_control(self):
        self.reset_command()
        self.gripper_command = "open"
        self.current_gripper_state = "open"
        self.current_arm_type = "right"
        default_cmd = {
            "pos": np.array([0.0, 0.0, 0.0]),
            "rot": np.array([0.0, 0.0, 0.0]),
        }
        self.last_command = [default_cmd, default_cmd]

    def reset_command(self):
        self.command_ = np.array([0.0, 0.0, 0.0])
        self.rotation_command = np.array([0.0, 0.0, 0.0])
        self.robot_command = np.array([0.0, 0.0, 0.0])
        self.robot_rotation_command = np.array([0.0, 0.0, 0.0])
        self.waist_command = np.array([0.0, 0.0])
        self.head_command = np.array([0.0, 0.0])

    def sub_keyboard_event(self):
        self.pressed_keys = set()

        def on_press(key):
            try:
                if key.char == 'w':
                    self.command_ += np.array([0.01, 0.0, 0.0])
                elif key.char == 's':
                    self.command_ += np.array([-0.01, 0.0, 0.0])
                elif key.char == 'a':
                    self.command_ += np.array([0.0, 0.01, 0.0])
                elif key.char == 'd':
                    self.command_ += np.array([0.0, -0.01, 0.0])
                elif key.char == 'q':
                    self.command_ += np.array([0.0, 0.0, 0.01])
                elif key.char == 'e':
                    self.command_ += np.array([0.0, 0.0, -0.01])
                elif key.char == 'j':
                    self.rotation_command += np.array([-0.02, 0.0, 0.0])
                elif key.char == 'l':
                    self.rotation_command += np.array([0.02, 0.0, 0.0])
                elif key.char == 'i':
                    self.rotation_command += np.array([0.0, 0.02, 0.0])
                elif key.char == 'k':
                    self.rotation_command += np.array([0.0, -0.02, 0.0])
                elif key.char == 'u':
                    self.rotation_command += np.array([0.0, 0.0, 0.02])
                elif key.char == 'o':
                    self.rotation_command += np.array([0.0, 0.0, -0.02])
                elif key.char == 'c':
                    if keyboard.Key.ctrl in self.pressed_keys:
                        self.gripper_command = "open"
                    else:
                        self.gripper_command = "close"
                elif key.char == 'r':
                    self.reset_flg = True
            except AttributeError:
                self.pressed_keys.add(key)
                if key == keyboard.Key.up:
                    if keyboard.Key.shift in self.pressed_keys:
                        self.head_command += np.array([0.0, -0.01])
                    elif keyboard.Key.ctrl in self.pressed_keys:
                        self.waist_command += np.array([0.01, 0.0])
                    else:
                        self.robot_command += np.array([0.02, 0.0, 0.0])
                elif key == keyboard.Key.down:
                    if keyboard.Key.shift in self.pressed_keys:
                        self.head_command += np.array([0.0, 0.01])
                    elif keyboard.Key.ctrl in self.pressed_keys:
                        self.waist_command += np.array([-0.01, 0.0])
                    else:
                        self.robot_command += np.array([-0.02, 0.0, 0.0])
                elif key == keyboard.Key.left:
                    if keyboard.Key.shift in self.pressed_keys:
                        self.head_command += np.array([0.01, 0.0])
                    elif keyboard.Key.ctrl in self.pressed_keys:
                        self.waist_command += np.array([0.0, 0.01])
                    else:
                        self.robot_rotation_command += np.array([0.0, 0.0, 0.02])
                elif key == keyboard.Key.right:
                    if keyboard.Key.shift in self.pressed_keys:
                        self.head_command += np.array([-0.01, 0.0])
                    elif keyboard.Key.ctrl in self.pressed_keys:
                        self.waist_command += np.array([0.0, -0.01])
                    else:
                        self.robot_rotation_command += np.array([0.0, 0.0, -0.02])
                elif key == keyboard.Key.tab and keyboard.Key.ctrl in self.pressed_keys:
                    if self.current_arm_type == "right":
                        self.current_arm_type = "left"
                        self.last_command[1] = {
                            "pos": self.command_,
                            "rot": self.rotation_command,
                        }
                    else:
                        self.current_arm_type = "right"
                        self.last_command[0] = {
                            "pos": self.command_,
                            "rot": self.rotation_command,
                        }
                    self.switch_flg = True

        def on_release(key):
            self.pressed_keys.discard(key)
            if key == keyboard.Key.up or key == keyboard.Key.down:
                self.robot_command = np.array([0.0, 0.0, 0.0])
            elif key == keyboard.Key.left or key == keyboard.Key.right:
                self.robot_rotation_command = np.array([0.0, 0.0, 0.0])

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()

    def parse_joint_pose(self, precision=3):
        l_arm, l_arm_name, r_arm, r_arm_name, ori_pos = [], [], [], [], []
        joint_pos = self.env.robot.get_joint_pose()
        for jp in joint_pos.states:
            name = jp.name
            angle = jp.position
            if "Joint" in name:
                if "_l" in name:
                    l_arm.append(angle)
                    l_arm_name.append(name)
                elif "_r" in name:
                    r_arm.append(angle)
                    r_arm_name.append(name)
            ori_pos.append(angle)
        return {
            "l_arm": [round(v, precision) for v in l_arm],
            "r_arm": [round(v, precision) for v in r_arm],
            "l_arm_name": l_arm_name,
            "r_arm_name": r_arm_name,
            "ori_pos": [round(v, precision) for v in ori_pos],
        }

    def parse_pico_command(self, content):
        """
        Args:
            content (list(dict)): parse pico control command to local frame
        """
        [l_sig, r_sig] = content
        ret = {"l": {}, "r": {}}
        ret["l"]["position"] = np.array(
            [l_sig["position"]["z"], -l_sig["position"]["x"], l_sig["position"]["y"]]
        )
        ret["l"]["quaternion"] = np.array(
            [
                l_sig["rotation"]["w"],
                l_sig["rotation"]["x"],
                l_sig["rotation"]["y"],
                l_sig["rotation"]["z"],
            ]
        )
        ret["l"]["axisMode"] = "reset" if l_sig["axisClick"] == "true" else "move"
        ret["l"]["axisX"] = l_sig["axisX"]
        ret["l"]["axisY"] = l_sig["axisY"]
        ret["l"]["gripper"] = 1 - l_sig["indexTrig"]
        ret["l"]["On"] = l_sig["handTrig"]
        ret["l"]["reset"] = l_sig["keyOne"] == "true"
        ret["l"]["reserved"] = l_sig["keyTwo"] == "true"

        ret["r"]["position"] = np.array(
            [r_sig["position"]["z"], -r_sig["position"]["x"], r_sig["position"]["y"]]
        )
        ret["r"]["quaternion"] = np.array(
            [
                r_sig["rotation"]["w"],
                r_sig["rotation"]["x"],
                r_sig["rotation"]["y"],
                r_sig["rotation"]["z"],
            ]
        )
        ret["r"]["axisX"] = r_sig["axisX"]
        ret["r"]["axisY"] = r_sig["axisY"]
        ret["r"]["axisMode"] = "head" if r_sig["axisClick"] == "true" else "waist"
        ret["r"]["gripper"] = 1 - r_sig["indexTrig"]
        ret["r"]["On"] = r_sig["handTrig"]
        ret["r"]["reset"] = r_sig["keyOne"] == "true"
        ret["r"]["reserved"] = r_sig["keyTwo"] == "true"
        self.pico_command = ret

    def parse_head_control(self):
        if self.pico_command["r"]["axisMode"] == "head":
            delta_angle = 0.05
            thresh = 0.5
            head_yaw = self.pico_command["r"]["axisX"]
            head_pitch = self.pico_command["r"]["axisY"]
            if head_yaw > thresh:
                self.init_pos[2] -= delta_angle
            if head_yaw < -thresh:
                self.init_pos[2] += delta_angle
            if head_pitch > thresh:
                self.init_pos[3] -= delta_angle
            if head_pitch < -thresh:
                self.init_pos[3] += delta_angle

    def parse_waist_control(self):
        if self.pico_command["r"]["axisMode"] == "waist":
            delta_angle = 0.01
            thresh = 0.5
            waist_lift = self.pico_command["r"]["axisY"]
            waist_pitch = self.pico_command["r"]["axisX"]
            if waist_lift > thresh:
                self.init_pos[0] += delta_angle
            if waist_lift < -thresh:
                self.init_pos[0] -= delta_angle
            if waist_pitch > thresh:
                self.init_pos[1] += delta_angle * 5
            if waist_pitch < -thresh:
                self.init_pos[1] -= delta_angle * 5

    def parse_arm_control(self, coef_pos, coef_quat):
        on_l, on_r = self.pico_command["l"]["On"], self.pico_command["r"]["On"]
        rescaled_pos_l = coef_pos * self.pico_command["l"]["position"]
        rescaled_pos_r = coef_pos * self.pico_command["r"]["position"]
        rescaled_rpy_l = scale_quat_to_euler(
            self.pico_command["l"]["quaternion"], coef_quat
        )
        rescaled_rpy_r = scale_quat_to_euler(
            self.pico_command["r"]["quaternion"], coef_quat
        )
        reset_l = self.pico_command["l"]["reset"]
        reset_r = self.pico_command["r"]["reset"]

        jp_l = self.env.robot.get_joint_from_deltapos(
            xyz=rescaled_pos_l, rpy=rescaled_rpy_l, id="left", isOn=on_l
        )
        jp_r = self.env.robot.get_joint_from_deltapos(
            xyz=rescaled_pos_r, rpy=rescaled_rpy_r, id="right", isOn=on_r
        )

        if jp_l.any() != 0.0:
            self.init_pos[4:18:2] = jp_l
        if jp_r.any() != 0.0:
            self.init_pos[5:19:2] = jp_r

        if reset_l:
            logger.info("reset left arm...")
            self.init_pos[4:18:2] = self.init_l_arm
            joint_info = self.parse_joint_pose()
            joint_info["l_arm"] = self.init_l_arm
            self.env.robot.initialize_solver(joint_info)
        if reset_r:
            logger.info("reset right arm...")
            self.init_pos[5:19:2] = self.init_r_arm
            joint_info = self.parse_joint_pose()
            joint_info["r_arm"] = self.init_r_arm
            self.env.robot.initialize_solver(joint_info)

    def parse_gripper_control(self):
        self.init_pos[18] = self.pico_command["l"]["gripper"]
        self.init_pos[20] = self.pico_command["r"]["gripper"]

    def apply_base_control(self):
        x = self.pico_command["l"]["axisY"]
        y = -self.pico_command["l"]["axisX"]
        mode = self.pico_command["l"]["axisMode"]
        if mode == "move":
            target_x = 0.02 * x
            target_yaw = 0
            if y < -0.5:
                target_yaw = -0.05
            if y > 0.5:
                target_yaw = 0.05

            base_pos_local = np.array([target_x, 0, 0])
            base_rot_local = np.array([0, 0, target_yaw])
            self.env.robot.update_transform()
            target_base_pos, target_base_rot = (
                self.env.robot.transform_from_base_to_world(
                    base_pos_local, base_rot_local
                )
            )
            self.env.robot.set_base_pose(
                target_pos=target_base_pos, target_rot=target_base_rot
            )
        else:
            self.reset_robot_pose()

    def initialize(self):
        joint_info = self.parse_joint_pose()
        self.init_pos = joint_info["ori_pos"]
        self.init_l_arm = joint_info["l_arm"]
        self.init_r_arm = joint_info["r_arm"]
        self.env.robot.initialize_solver(joint_info)
        self.robot_init_pos, self.robot_init_rot = self.env.robot.get_init_pose()
        self.reset_command()

    def reset_robot_pose(self):
        self.env.robot.set_base_pose(
            target_pos=self.robot_init_pos, target_rot=self.robot_init_rot
        )

    def run_pico_control(self, with_physics=True):
        self.initialize()
        coef_pos = 0.8
        coef_quat = 0.8
        try:
            while True:
                pico_cmd = self.vr_server.on_update()
                if pico_cmd:
                    self.parse_pico_command(pico_cmd)
                    self.parse_arm_control(coef_pos, coef_quat)
                    self.parse_gripper_control()
                    self.parse_head_control()
                    self.parse_waist_control()
                    self.apply_base_control()
                    self.env.robot.set_joint_pose(
                        target_joint_position=self.init_pos, is_trajectory=with_physics
                    )

        except KeyboardInterrupt:
            logger.warning("keyboard interrupt")

    def run_keyboard_control(self, with_physics=True):
        self.initialize()
        self.sub_keyboard_event()

        try:
            while True:
                with self.lock:
                    if self.reset_flg:
                        self.reset_command()
                        self.reset_robot_pose()
                        self.reset_flg = False
                    if self.switch_flg:
                        idx = 0 if self.current_arm_type == "left" else 1
                        self.command_ = self.last_command[idx]["pos"]
                        self.rotation_command = self.last_command[idx]["rot"]
                        self.switch_flg = False
                    if self.current_arm_type == "right":
                        jp = self.env.robot.get_joint_from_deltapos(
                            xyz=self.command_,
                            rpy=self.rotation_command,
                            id="right",
                            isOn=True,
                        )
                        if jp.any() != 0.0:
                            self.init_pos[5:19:2] = jp
                        self.init_pos[20] = (
                            0.0 if self.gripper_command == "close" else 1.0
                        )
                    else:
                        jp = self.env.robot.get_joint_from_deltapos(
                            xyz=self.command_,
                            rpy=self.rotation_command,
                            id="left",
                            isOn=True,
                        )
                        if jp.any() != 0.0:
                            self.init_pos[4:18:2] = jp
                        self.init_pos[18] = (
                            0.0 if self.gripper_command == "close" else 1.0
                        )

                    base_pos_local = self.robot_command
                    base_rot_local = self.robot_rotation_command
                    logger.info(
                        f"Base moving speed: {base_pos_local[0]:.2f}, yaw rate: {base_rot_local[2]:.2f}"
                    )
                    self.env.robot.update_transform()
                    target_base_pos, target_base_rot = (
                        self.env.robot.transform_from_base_to_world(
                            base_pos_local, base_rot_local
                        )
                    )
                    self.init_pos[0:2] += self.waist_command
                    self.init_pos[2:4] += self.head_command
                    self.env.robot.set_base_pose(target_base_pos, target_base_rot)
                    self.env.robot.set_joint_pose(
                        target_joint_position=self.init_pos, is_trajectory=with_physics
                    )

        except KeyboardInterrupt:
            logger.warning("keyboard interrupt, exiting keyboard control")

    def load_task_config(self, task):
        task_config_file = os.path.join(work_dir, "tasks", task + ".json")
        logger.info(f"task config file {task_config_file}")
        if not os.path.exists(task_config_file):
            raise ValueError("Task config file not found: {}".format(task_config_file))
        with open(task_config_file) as f:
            self.task_config = json.load(f)

    def run(self):
        self.load_task_config(self.task_name)
        robot_cfg = "G1_120s_dual_high.json"

        # init robot and scene
        scene_info = self.task_config["scene"]
        workspace_id = scene_info["scene_id"].split("/")[-1]
        if workspace_id in scene_info["function_space_objects"]:
            robot_init_pose = self.task_config["robot"]["robot_init_pose"][workspace_id]
        else:
            robot_init_pose = self.task_config["robot"]["robot_init_pose"]
        self.task_config["specific_task_name"] = self.task_name

        logger.info(f"scene_usd {self.task_config['scene']['scene_usd']}")
        robot = IsaacSimRpcRobot(
            robot_cfg=robot_cfg,
            scene_usd=self.task_config["scene"]["scene_usd"],
            client_host=self.args.client_host,
            position=robot_init_pose["position"],
            rotation=robot_init_pose["quaternion"],
        )

        # init state
        task_generator = TaskGenerator(self.task_config)
        task_folder = "saved_task/%s" % (self.task_config["task"])
        task_generator.generate_tasks(
            save_path=task_folder,
            task_num=self.episodes_per_instance,
            task_name=self.task_config["task"],
        )
        robot_position = task_generator.robot_init_pose["position"]
        robot_rotation = task_generator.robot_init_pose["quaternion"]
        self.task_config["robot"]["robot_init_pose"]["position"] = robot_position
        self.task_config["robot"]["robot_init_pose"]["quaternion"] = robot_rotation
        specific_task_files = glob.glob(task_folder + "/*.json")
        for episode_id in range(self.episodes_per_instance):
            episode_file_path = specific_task_files[episode_id]
            env = DemoEnv(robot, episode_file_path, self.task_config)
            env.load(episode_file_path)
            self.env = env
            if self.record:
                self.env.start_recording(
                    task_name=self.task_name,
                    camera_prim_list=[],
                    fps=self.fps,
                )

            if self.mode == "pico":
                self.run_pico_control()
            elif self.mode == "keyboard":
                self.run_keyboard_control()
            else:
                raise ValueError("Invalid mode: {}".format(self.mode))

            if self.record:
                self.env.stop_recording(True)
        self.env.robot.client.Exit()


def main():
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument("--client_host", type=str, default="localhost:50051", help="The client")
    parser.add_argument("--fps", type=int, default=30, help="Set fps of the recording")
    parser.add_argument("--task_name", default="genie_task_supermarket", type=str, help="Selected task to run")
    parser.add_argument("--mode", type=str, default="pico", help="Choose teleop mode: pico or keyboard")
    parser.add_argument("--record", action="store_true", help="Enable data recording")
    parser.add_argument("--host_ip", type=str, default="192.168.111.177", help="Set vr host ip")
    parser.add_argument("--port", type=int, default=8080, help="Set vr port")
    # fmt: on
    args = parser.parse_args()
    task = TeleOp(args)
    task.run()


if __name__ == "__main__":
    main()
