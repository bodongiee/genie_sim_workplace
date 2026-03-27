# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os
import sys
import contextlib
import tempfile

import ik_solver
import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    print(f"DEBUG: ik_solver module loaded from: {ik_solver.__file__}")
except AttributeError:
    print("DEBUG: ik_solver module does not have a __file__ attribute (might be built-in).")

current_dir = os.path.dirname(os.path.abspath(__file__))

@contextlib.contextmanager
def capture_c_stdout():
    # Capture stdout from C++ libraries and print via Python cai1345!
    fd = sys.stdout.fileno() if hasattr(sys.stdout, 'fileno') else 1
    with tempfile.TemporaryFile(mode='w+b') as tfile:
        old_stdout = os.dup(fd)
        try:
            sys.stdout.flush()
            os.dup2(tfile.fileno(), fd)
            yield
        finally:
            os.dup2(old_stdout, fd)
            os.close(old_stdout)
            tfile.seek(0)
            output = tfile.read().decode('utf-8', errors='replace')
            if output.strip():
                translations = {
                    "========== 初始化 Solver ==========": "========== Initialize Solver ==========",
                    "机器人部件:": "Robot Part:",
                    "正在加载配置文件:": "Loading config file:",
                    "配置文件加载成功": "Config file loaded successfully",
                    "正在初始化RobotWrapper...": "Initializing RobotWrapper...",
                    "成功加载完整模型，路径为:": "Successfully loaded full model, path:",
                    "成功加载配置文件:": "Successfully loaded config file:",
                    "目标维度设置为:": "Target dimension set to:",
                    "已加载右臂关节配置": "Right arm joint config loaded",
                    "已加载左臂关节配置": "Left arm joint config loaded",
                    "========== 简化模型信息 ==========": "========== Simplified Model Info ==========",
                    "自由度:": "DOF:",
                    "----- 可动关节列表 -----": "----- Active Joint List -----",
                    "成功加载简化模型": "Successfully loaded simplified model",
                    "已设置关节限位：": "Joint limits set:",
                    "位置下限:": "Pos Lower Limit:",
                    "位置上限:": "Pos Upper Limit:",
                    "速度限制:": "Vel Limit:",
                    "力矩限制:": "Torque Limit:",
                    "RobotWrapper初始化完成": "RobotWrapper initialization complete",
                    "状态维度:": "State dim:",
                    "目标维度:": "Target dim:",
                    "Variables初始化完成": "Variables initialization complete",
                    "初始化完成": "Initialization complete",
                }
                for cn, en in translations.items():
                    output = output.replace(cn, en)
                print(f"[C++ STDOUT]:\n{output}")

def xyzquat_to_xyzrpy(xyzquat):
    xyz = xyzquat[:3]
    rpy = R.from_quat(xyzquat[3:], scalar_first=True).as_euler("xyz", degrees=False)
    xyzrpy = np.concatenate([xyz, rpy])
    return xyzrpy


def xyzrpy_to_xyzquat(xyzrpy):
    xyz = xyzrpy[:3]
    quat = R.from_euler("xyz", xyzrpy[3:]).as_quat(scalar_first=False)
    xyzquat = np.concatenate([xyz, quat])
    return xyzquat


def xyzrpy2mat(xyzrpy):
    rot = R.from_euler("xyz", xyzrpy[3:6]).as_matrix()
    mat = np.eye(4)
    mat[0:3, 0:3] = rot
    mat[0:3, 3] = xyzrpy[0:3]
    return mat


def mat2xyzrpy(mat):
    rpy = R.from_matrix(mat[0:3, 0:3]).as_euler("xyz", degrees=False)
    xyz = mat[0:3, 3]
    xyzrpy = np.concatenate([xyz, rpy])
    return xyzrpy


class IKFKSolver:
    def __init__(self, arm_init_joint_position, head_init_position, waist_init_position, robot_cfg="G1_omnipicker"):
        self.robot_cfg = robot_cfg

        if "aloha" in robot_cfg:
            # Aloha uses vx300s - no G1/G2 IK/FK solver needed
            self.left_solver = None
            self.right_solver = None
            self._arm_init = arm_init_joint_position
            return

        if "G2" in robot_cfg:
            urdf_name, config_name = "G2_NO_GRIPPER.urdf", "g2_solver.yaml"
        elif "ffw_sg2_follower" in robot_cfg:
            urdf_name, config_name = "ffw_sg2_follower.urdf", "ffw_sg2_follower_solver.yaml"
        elif "ffw_sh5_follower" in robot_cfg:
            urdf_name, config_name = "ffw_sh5_follower.urdf", "ffw_sh5_follower_solver.yaml"
        else:
            urdf_name, config_name = "G1_NO_GRIPPER.urdf", "g1_solver.yaml"

        sdk_dir = os.path.join(current_dir, "IK-SDK")
        urdf_path = os.path.join(sdk_dir, urdf_name)
        config_path = os.path.join(sdk_dir, config_name)

        with capture_c_stdout():
            self.left_solver = ik_solver.Solver(
                part=ik_solver.RobotPart.LEFT_ARM,
                urdf_path=urdf_path,
                config_path=config_path,
            )
            self.right_solver = ik_solver.Solver(
                part=ik_solver.RobotPart.RIGHT_ARM,
                urdf_path=urdf_path,
                config_path=config_path,
            )

        self.left_solver.set_debug_mode(False)
        self.right_solver.set_debug_mode(False)
        self.left_solver.sync_target_with_joints(arm_init_joint_position[:7])
        self.right_solver.sync_target_with_joints(arm_init_joint_position[7:14])

    def quaternion_to_euler(self, w, x, y, z):
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x**2 + y**2)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_Cosp = 1 - 2 * (y**2 + z**2)
        yaw = np.arctan2(siny_cosp, cosy_Cosp)

        return np.stack([roll, pitch, yaw], axis=-1)

    def compute_eef(self, arm_joint_states):
        """Compute left/right EEF poses via FK from current arm joint states.

        Returns dict {"left": [x,y,z,qw,qx,qy,qz], "right": [x,y,z,qw,qx,qy,qz]}
        in the arm_base_link frame.
        """
        if self.left_solver is None:
            return {"left": [0.0] * 7, "right": [0.0] * 7}
        left_joints = np.asarray(arm_joint_states[:7], dtype=np.float32)
        right_joints = np.asarray(arm_joint_states[7:14], dtype=np.float32)

        left_mat = np.asarray(self.left_solver.compute_fk(left_joints), dtype=np.float64).reshape(4, 4)
        right_mat = np.asarray(self.right_solver.compute_fk(right_joints), dtype=np.float64).reshape(4, 4)

        left_pos = left_mat[:3, 3]
        left_quat = R.from_matrix(left_mat[:3, :3]).as_quat(scalar_first=True)
        right_pos = right_mat[:3, 3]
        right_quat = R.from_matrix(right_mat[:3, :3]).as_quat(scalar_first=True)

        return {
            "left": np.concatenate([left_pos, left_quat]).tolist(),
            "right": np.concatenate([right_pos, right_quat]).tolist(),
        }

    def eef_actions_to_joint(self, eef_actions, arm_joint_states, head_init_position):
        if self.left_solver is None:
            return []
        joint_actions = []
        self.left_solver.sync_target_with_joints(arm_joint_states[:7])
        self.right_solver.sync_target_with_joints(arm_joint_states[7:14])

        for _, action in enumerate(eef_actions):
            eefrot_left_cur = np.array(action[:6], dtype=np.float32)
            eefrot_right_cur = np.array(action[6:12], dtype=np.float32)

            target_pos_left = xyzrpy_to_xyzquat(eefrot_left_cur)
            target_pos_right = xyzrpy_to_xyzquat(eefrot_right_cur)
            self.left_solver.update_target_quat(
                target_pos=target_pos_left[:3],
                target_quat=target_pos_left[3:],
            )
            self.right_solver.update_target_quat(
                target_pos=target_pos_right[:3],
                target_quat=target_pos_right[3:],
            )

            left_joints = self.left_solver.solve()
            right_joints = self.right_solver.solve()

            l_gripper = action[12:13] if type(action) == list else action[12:13].tolist()
            r_gripper = action[13:14] if type(action) == list else action[13:14].tolist()
            joint_actions.append(left_joints.tolist() + right_joints.tolist() + l_gripper + r_gripper)

        return joint_actions
