import os
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import placo
import tyro
from meshcat import transformations as tf

from xrobotoolkit_teleop.simulation.mujoco_teleop_controller import (MujocoTeleopController,)
from xrobotoolkit_teleop.utils.geometry import apply_delta_pose

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def main(
    xml_path: str = os.path.join(SCRIPT_DIR, "ai_worker/ffw_sg2.xml"),
    robot_urdf_path: str = os.path.join(SCRIPT_DIR, "ai_worker/ffw_description/urdf/ffw_sg2_rev1_follower/ffw_sg2_follower.urdf"),
    scale_factor: float = 1.5,
    visualize_placo: bool = True,
):
    """
    Main function to run the ai_worker ffw_sg2 follower teleoperation in MuJoCo.
    """
    config = {
        "right_hand": {
            "link_name": "arm_r_link7",
            "pose_source": "right_controller",
            "control_trigger": "right_grip",
            "vis_target": "right_target",
        },
        "left_hand": {
            "link_name": "arm_l_link7",
            "pose_source": "left_controller",
            "control_trigger": "left_grip",
            "vis_target": "left_target",
        },
    }

    controller = MujocoTeleopController(
        xml_path=xml_path,
        robot_urdf_path=robot_urdf_path,
        manipulator_config=config,
        scale_factor=scale_factor,
        visualize_placo=visualize_placo,
    )
    joints_task = controller.solver.add_joints_task()
    joints_task.set_joints({joint: 0.0 for joint in controller.placo_robot.joint_names()})
    joints_task.configure("joints_regularization", "soft", 1e-4)

    controller.run()


if __name__ == "__main__":
    tyro.cli(main)
