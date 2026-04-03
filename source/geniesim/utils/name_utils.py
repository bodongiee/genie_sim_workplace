# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

G1_JOINT_NAMES = [
    "idx21_arm_l_joint1",
    "idx22_arm_l_joint2",
    "idx23_arm_l_joint3",
    "idx24_arm_l_joint4",
    "idx25_arm_l_joint5",
    "idx26_arm_l_joint6",
    "idx27_arm_l_joint7",
    "idx61_arm_r_joint1",
    "idx62_arm_r_joint2",
    "idx63_arm_r_joint3",
    "idx64_arm_r_joint4",
    "idx65_arm_r_joint5",
    "idx66_arm_r_joint6",
    "idx67_arm_r_joint7",
    "idx11_head_joint1",
    "idx12_head_joint2",
    "idx02_body_joint2",
    "idx01_body_joint1",
]

G1_LEFT_ARM_JOINT_NAMES = [
    "idx21_arm_l_joint1",
    "idx22_arm_l_joint2",
    "idx23_arm_l_joint3",
    "idx24_arm_l_joint4",
    "idx25_arm_l_joint5",
    "idx26_arm_l_joint6",
    "idx27_arm_l_joint7",
]

G1_RIGHT_ARM_JOINT_NAMES = [
    "idx61_arm_r_joint1",
    "idx62_arm_r_joint2",
    "idx63_arm_r_joint3",
    "idx64_arm_r_joint4",
    "idx65_arm_r_joint5",
    "idx66_arm_r_joint6",
    "idx67_arm_r_joint7",
]
G1_DUAL_ARM_JOINT_NAMES = G1_LEFT_ARM_JOINT_NAMES + G1_RIGHT_ARM_JOINT_NAMES

G1_HEAD_JOINT_NAMES = [
    "idx11_head_joint1",
    "idx12_head_joint2",
]

G1_WAIST_JOINT_NAMES = [
    "idx02_body_joint2",
    "idx01_body_joint1",
]


G2_JOINT_NAMES = [
    "idx21_arm_l_joint1",
    "idx22_arm_l_joint2",
    "idx23_arm_l_joint3",
    "idx24_arm_l_joint4",
    "idx25_arm_l_joint5",
    "idx26_arm_l_joint6",
    "idx27_arm_l_joint7",
    "idx61_arm_r_joint1",
    "idx62_arm_r_joint2",
    "idx63_arm_r_joint3",
    "idx64_arm_r_joint4",
    "idx65_arm_r_joint5",
    "idx66_arm_r_joint6",
    "idx67_arm_r_joint7",
]

G2_LEFT_ARM_JOINT_NAMES = [
    "idx21_arm_l_joint1",
    "idx22_arm_l_joint2",
    "idx23_arm_l_joint3",
    "idx24_arm_l_joint4",
    "idx25_arm_l_joint5",
    "idx26_arm_l_joint6",
    "idx27_arm_l_joint7",
]
G2_RIGHT_ARM_JOINT_NAMES = [
    "idx61_arm_r_joint1",
    "idx62_arm_r_joint2",
    "idx63_arm_r_joint3",
    "idx64_arm_r_joint4",
    "idx65_arm_r_joint5",
    "idx66_arm_r_joint6",
    "idx67_arm_r_joint7",
]

G2_DUAL_ARM_JOINT_NAMES = G2_LEFT_ARM_JOINT_NAMES + G2_RIGHT_ARM_JOINT_NAMES

G2_HEAD_JOINT_NAMES = [
    "idx11_head_joint1",
    "idx12_head_joint2",
    "idx13_head_joint3",
]
G2_WAIST_JOINT_NAMES = [
    "idx05_body_joint5",
    "idx04_body_joint4",
    "idx03_body_joint3",
    "idx02_body_joint2",
    "idx01_body_joint1",
]

OMNIPICKER_AJ_NAMES = [
    "idx41_gripper_l_outer_joint1",
    "idx81_gripper_r_outer_joint1",
]

ALOHA_JOINT_NAMES = [
    "vx300s_left_waist",
    "vx300s_left_shoulder",
    "vx300s_left_elbow",
    "vx300s_left_forearm_roll",
    "vx300s_left_wrist_angle",
    "vx300s_left_wrist_rotate",
    "vx300s_right_waist",
    "vx300s_right_shoulder",
    "vx300s_right_elbow",
    "vx300s_right_forearm_roll",
    "vx300s_right_wrist_angle",
    "vx300s_right_wrist_rotate",
]

ALOHA_LEFT_ARM_JOINT_NAMES = [
    "vx300s_left_waist",
    "vx300s_left_shoulder",
    "vx300s_left_elbow",
    "vx300s_left_forearm_roll",
    "vx300s_left_wrist_angle",
    "vx300s_left_wrist_rotate",
]

ALOHA_RIGHT_ARM_JOINT_NAMES = [
    "vx300s_right_waist",
    "vx300s_right_shoulder",
    "vx300s_right_elbow",
    "vx300s_right_forearm_roll",
    "vx300s_right_wrist_angle",
    "vx300s_right_wrist_rotate",
]

ALOHA_DUAL_ARM_JOINT_NAMES = ALOHA_LEFT_ARM_JOINT_NAMES + ALOHA_RIGHT_ARM_JOINT_NAMES

ALOHA_GRIPPER_NAMES = [
    "vx300s_left_left_finger",
    "vx300s_left_right_finger",
    "vx300s_right_left_finger",
    "vx300s_right_right_finger",
]

G1_CHASSIS = [
    "base_linear_joint_x",
    "base_linear_joint_y",
    "base_angular_joint",
]


FFW_SG2_JOINT_NAMES = [
    "arm_l_joint1",
    "arm_l_joint2",
    "arm_l_joint3",
    "arm_l_joint4",
    "arm_l_joint5",
    "arm_l_joint6",
    "arm_l_joint7",
    "arm_r_joint1",
    "arm_r_joint2",
    "arm_r_joint3",
    "arm_r_joint4",
    "arm_r_joint5",
    "arm_r_joint6",
    "arm_r_joint7",
]

FFW_SG2_LEFT_ARM_JOINT_NAMES = [
    "arm_l_joint1",
    "arm_l_joint2",
    "arm_l_joint3",
    "arm_l_joint4",
    "arm_l_joint5",
    "arm_l_joint6",
    "arm_l_joint7",
]

FFW_SG2_RIGHT_ARM_JOINT_NAMES = [
    "arm_r_joint1",
    "arm_r_joint2",
    "arm_r_joint3",
    "arm_r_joint4",
    "arm_r_joint5",
    "arm_r_joint6",
    "arm_r_joint7",
]

FFW_SG2_DUAL_ARM_JOINT_NAMES = FFW_SG2_LEFT_ARM_JOINT_NAMES + FFW_SG2_RIGHT_ARM_JOINT_NAMES

FFW_SG2_HEAD_JOINT_NAMES = [
    "head_joint1",
    "head_joint2",
]

FFW_SG2_WAIST_JOINT_NAMES = [
    "lift_joint",
    "left_wheel_steer",
    "left_wheel_drive",
    "right_wheel_steer",
    "right_wheel_drive",
    "rear_wheel_steer",
    "rear_wheel_drive",
]


FFW_SG2_GRIPPER_JOINTS_NAMES = [
    "gripper_l_joint1",
    "gripper_r_joint1"
]

FFW_SG2_JOINT_NAMES = FFW_SG2_DUAL_ARM_JOINT_NAMES + FFW_SG2_HEAD_JOINT_NAMES



FFW_SH5_JOINT_NAMES = [
    "arm_l_joint1",
    "arm_l_joint2",
    "arm_l_joint3",
    "arm_l_joint4",
    "arm_l_joint5",
    "arm_l_joint6",
    "arm_l_joint7",
    "arm_r_joint1",
    "arm_r_joint2",
    "arm_r_joint3",
    "arm_r_joint4",
    "arm_r_joint5",
    "arm_r_joint6",
    "arm_r_joint7",
]

FFW_SH5_LEFT_ARM_JOINT_NAMES = [
    "arm_l_joint1",
    "arm_l_joint2",
    "arm_l_joint3",
    "arm_l_joint4",
    "arm_l_joint5",
    "arm_l_joint6",
    "arm_l_joint7",
]

FFW_SH5_RIGHT_ARM_JOINT_NAMES = [
    "arm_r_joint1",
    "arm_r_joint2",
    "arm_r_joint3",
    "arm_r_joint4",
    "arm_r_joint5",
    "arm_r_joint6",
    "arm_r_joint7",
]

FFW_SH5_DUAL_ARM_JOINT_NAMES = FFW_SH5_LEFT_ARM_JOINT_NAMES + FFW_SH5_RIGHT_ARM_JOINT_NAMES

FFW_SH5_HEAD_JOINT_NAMES = [
    "head_joint1",
    "head_joint2",
]

FFW_SH5_WAIST_JOINT_NAMES = [
    "lift_joint",
    "left_wheel_steer_joint",
    "left_wheel_drive_joint",
    "right_wheel_steer_joint",
    "right_wheel_drive_joint",
    "rear_wheel_steer_joint",
    "rear_wheel_drive_joint",
]

FFW_SH5_LEFT_HAND_JOINT_NAMES = [
    "finger_l_joint1",
    "finger_l_joint2",
    "finger_l_joint3",
    "finger_l_joint4",
    "finger_l_joint5",
    "finger_l_joint6",
    "finger_l_joint7",
    "finger_l_joint8",
    "finger_l_joint9",
    "finger_l_joint10",
    "finger_l_joint11",
    "finger_l_joint12",
    "finger_l_joint13",
    "finger_l_joint14",
    "finger_l_joint15",
    "finger_l_joint16",
    "finger_l_joint17",
    "finger_l_joint18",
    "finger_l_joint19",
    "finger_l_joint20",
]

FFW_SH5_RIGHT_HAND_JOINT_NAMES = [
    "finger_r_joint1",
    "finger_r_joint2",
    "finger_r_joint3",
    "finger_r_joint4",
    "finger_r_joint5",
    "finger_r_joint6",
    "finger_r_joint7",
    "finger_r_joint8",
    "finger_r_joint9",
    "finger_r_joint10",
    "finger_r_joint11",
    "finger_r_joint12",
    "finger_r_joint13",
    "finger_r_joint14",
    "finger_r_joint15",
    "finger_r_joint16",
    "finger_r_joint17",
    "finger_r_joint18",
    "finger_r_joint19",
    "finger_r_joint20",
]

FFW_SH5_JOINT_NAMES = FFW_SH5_DUAL_ARM_JOINT_NAMES + FFW_SH5_HEAD_JOINT_NAMES
FFW_SH5_HAND_JOINT_NAMES = FFW_SH5_LEFT_HAND_JOINT_NAMES + FFW_SH5_RIGHT_HAND_JOINT_NAMES

AMBER_BIMANUAL_LEFT_ARM_JOINT_NAMES = [
    "left_Rev1",
    "left_Rev2",
    "left_Rev3",
    "left_Rev4",
    "left_Rev5",
    "left_Rev6",
    "left_Rev7",
]

AMBER_BIMANUAL_RIGHT_ARM_JOINT_NAMES = [
    "right_Rev1",
    "right_Rev2",
    "right_Rev3",
    "right_Rev4",
    "right_Rev5",
    "right_Rev6",
    "right_Rev7",
]

AMBER_BIMANUAL_ARM_JOINT_NAMES = AMBER_BIMANUAL_LEFT_ARM_JOINT_NAMES + AMBER_BIMANUAL_RIGHT_ARM_JOINT_NAMES

def robot_type_mapping(robot_type):
    if "G1_omnipicker" in robot_type:
        return "G1_omnipicker"
    elif "G2_omnipicker" in robot_type:
        return "G2_omnipicker"
    elif "aloha" in robot_type:
        return "aloha"
    elif "ffw_sg2_follower" in robot_type:
        return "ffw_sg2_follower"
    elif "ffw_sh5_follower" in robot_type:
        return "ffw_sh5_follower"
    else:
        raise ValueError(f"Invalid robot type: {robot_type}")
