# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os, sys
from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image
import omni
from genie.sim.lab.controllers.parallel_gripper import ParallelGripper
from genie.sim.lab.utils import RobotCfg


from base_utils.logger import Logger

logger = Logger()  # Create singleton instance


if os.getenv("ISAACSIM_VERSION") == "v45":
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.api.materials import PhysicsMaterial, OmniPBR, OmniGlass
    from isaacsim.core.api.objects import cuboid, cylinder
    from isaacsim.core.prims import SingleXFormPrim, SingleGeometryPrim, SingleRigidPrim
    from isaacsim.core.utils.prims import get_prim_at_path, get_prim_object_type
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.robot.manipulators.grippers.surface_gripper import SurfaceGripper
    from isaacsim.core.utils.stage import get_current_stage
    from isaacsim.sensors.camera import Camera
    from isaacsim.sensors.physics import ContactSensor
else:
    from omni.isaac.manipulators.grippers.surface_gripper import SurfaceGripper
    from omni.isaac.core.utils.prims import get_prim_at_path, get_prim_object_type
    from omni.isaac.core.prims import (
        XFormPrim as SingleXFormPrim,
        GeometryPrim as SingleGeometryPrim,
        RigidPrim as SingleRigidPrim,
    )
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.articulations import Articulation as SingleArticulation
    from omni.isaac.core.materials import PhysicsMaterial, OmniPBR, OmniGlass
    from omni.isaac.core.objects import cuboid, cylinder
    from omni.isaac.core.utils.stage import get_current_stage
    from omni.isaac.sensor import Camera
    from omni.isaac.sensor import ContactSensor

import omni.replicator.core as rep
import omni.timeline
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from omni.physx.scripts import utils, physicsUtils, particleUtils
import omni.usd

import threading
import queue
import json
from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf, UsdPhysics, PhysxSchema
from genie.sim.lab.utils import material_changer, Light
import asyncio
from opentelemetry import trace
from genie.sim.lab.utils.utils import (
    get_rotation_matrix_from_quaternion,
    get_quaternion_from_euler,
    matrix_to_euler_angles,
    rotation_matrix_to_quaternion,
)

import subprocess
import signal

from genie.sim.lab.ros_publisher.base import USDBase

tracer = trace.get_tracer(__name__)


class CommandController:
    def __init__(
        self,
        ui_builder,
        enable_physics=False,
        enable_curobo=False,
        rendering_step=60,
        publish_ros=True,
        record_images=False,
    ):
        self.ui_builder = ui_builder
        self.data = None
        self.Command = 0
        self.data_to_send = None
        self.gripper_L = None
        self.gripper_R = None
        self.gripper_state_L = ""
        self.gripper_state_R = ""
        self.gripper_state = ""
        self.gripper_initialized = False
        self.condition = threading.Condition()
        self.result_queue = queue.Queue()
        self.target_position = np.array([0, 0, 0])
        self.target_rotation = np.array([0, 0, 0])
        self.target_joints_pose = None
        self.Start_Recording = False
        self.task_name = None
        self.cameras = {}
        self.step = 0
        self.step_server = 0
        self.path_to_save = None
        self.exit = False
        self.object_prims = []
        self.usd_objects = {}
        self.articulat_objects = {}
        self.enable_physics = enable_physics
        self.enable_curobo = enable_curobo
        self.trajectory_list = None
        self.trajectory_index = 0
        self.trajectory_reached = False
        self.target_joints_pose = []
        self.graph_path = []
        self.camera_graph_path = []
        self.loop_count = 0
        self.rendering_step = rendering_step
        self.process_pid = []
        self.target_point = None
        self.debug_view = {}
        self.timeline = omni.timeline.get_timeline_interface()
        self.publish_ros = publish_ros
        self.record_images = record_images

    def _init_robot_cfg(
        self,
        robot_cfg,
        scene_usd,
        is_mocap=False,
        batch_num=0,
        init_position=[0, 0, 0],
        init_rotation=[1, 0, 0, 0],
        stand_type="cylinder",
        size_x=0.1,
        size_y=0.1,
    ):
        main_path = Path(sys.modules["__main__"].__file__).resolve()
        main_dir = str(main_path.parent)
        robot = RobotCfg(main_dir + "/robot_cfg/" + robot_cfg)
        self.robot_cfg = robot
        robot_usd_path = main_dir + "/data/" + robot.robot_usd
        scene_usd_path = main_dir + "/data/" + scene_usd
        prim = get_prim_at_path("/World")
        self.batch_num = batch_num
        add_reference_to_stage(robot_usd_path, robot.robot_prim_path)
        add_reference_to_stage(scene_usd_path, "/World")
        self.usd_objects["robot"] = SingleXFormPrim(
            prim_path=robot.robot_prim_path,
            position=init_position,
            orientation=init_rotation,
        )
        self.robot_init_position = init_position
        self.robot_init_rotation = init_rotation
        self.scene_usd = scene_usd
        if "multispace" in scene_usd:
            self.scene_name = scene_usd.split("/")[-3] + "/" + scene_usd.split("/")[-2]
        else:
            self.scene_name = scene_usd.split("/")[-2]
        self.robot_name = scene_usd.split(".")[-2]
        self.material_changer = material_changer()
        camera_state = ViewportCameraState("/OmniverseKit_Persp")
        camera_state.set_position_world(
            Gf.Vec3d(init_position[0]+0.3, init_position[1]+0.5, init_position[2]+2.3), True
        )
        camera_state.set_target_world(
            Gf.Vec3d(init_position[0], init_position[1], init_position[2]), True
        )
        stage = omni.usd.get_context().get_stage()
        self.scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
        self.scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        self.scene.CreateGravityMagnitudeAttr().Set(9.81)
        robot_rep = rep.get.prims(
            path_pattern=robot.robot_prim_path, prim_types=["Xform"]
        )
        with robot_rep:
            rep.modify.semantics([("class", "robot")])

        stage = omni.usd.get_context().get_stage()

        def enable_collsion_api(prim_str):
            collision_prim = stage.GetPrimAtPath(prim_str)
            UsdPhysics.CollisionAPI.Apply(collision_prim)

        for p in [
            "/G1/right_Right_Pad_Link/collisions",
            "/G1/right_Left_Pad_Link/collisions",
        ]:
            enable_collsion_api(p)

        self._play()

    def _play(self):
        self.ui_builder.my_world.play()
        self._init_robot(self.robot_cfg, self.enable_curobo)

        self.frame_status = []

    def _init_robot(self, robot: RobotCfg, enable_curobo):
        self.robot_name = robot.robot_name
        self.robot_prim_path = robot.robot_prim_path
        self.end_effector_prim_path = robot.end_effector_prim_path
        self.end_effector_name = robot.end_effector_name

        self.finger_names = robot.finger_names
        self.gripper_controll_joint = robot.gripper_controll_joint
        self.opened_positions = robot.opened_positions
        self.closed_velocities = robot.closed_velocities
        self.cameras = robot.cameras
        self.is_single_gripper = robot.is_single
        self.gripper_type = robot.gripper_type
        self.gripper_max_force = robot.gripper_max_force
        self.init_joint_position = robot.init_joint_position
        self.ui_builder._init_solver(robot, enable_curobo, self.batch_num)
        self.goal_position, self.goal_rotation = self._get_ee_pose(True)
        self.past_position = [0, 0, 0]
        self.past_rotation = [1, 0, 0, 0]
        if self.publish_ros:
            self.sensor_base = USDBase()

    def _set_trajectory_list(self):
        if self.trajectory_list:
            if self.trajectory_index < len(self.trajectory_list):
                position, rotation = self.trajectory_list[self.trajectory_index]
                self.trajectory_index += 1
                self.ui_builder._on_time_trajectory(position, rotation)
            else:
                if self.trajectory_blocking:
                    self.trajectory_reached = True
                self.trajectory_list = None

    def on_physics_step(self):
        self.ui_builder._on_every_frame_trajectory_list()
        # curobo step
        if self.ui_builder.curoboMotion:
            self.ui_builder.curoboMotion.on_physics_step()

        self.on_command_step()
        # trajectory step
        self._set_trajectory_list()
        if self.Start_Recording:
            self._on_recording_step()

    def command_thread(self):
        step = 0
        while True:
            self.on_command_step()

    # update
    def on_command_step(self):
        if not self.data or not self.Command:
            return
        else:
            with tracer.start_as_current_span(
                f"rpc_server.step_command_{self.Command}"
            ) as span:
                if self.Command == 1:
                    prim_path = self.data["Cam_prim_path"]
                    isRGB = self.data["isRGB"]
                    isDepth = self.data["isDepth"]
                    isSemantic = self.data["isSemantic"]
                    isGN = self.data["isGN"]
                    self.data_to_send = self._capture_camera(
                        prim_path=prim_path,
                        isRGB=isRGB,
                        isDepth=isDepth,
                        isSemantic=isSemantic,
                        isGN=isGN,
                    )
                if self.Command == 2:
                    for value in self.articulat_objects.values():
                        value.initialize()
                    isSingle = False
                    self.data_to_send = None
                    target_position = self.data["target_position"]
                    target_rotation = self.data["target_rotation"]
                    is_backend = self.data["is_backend"]
                    is_Right = False
                    if self.data["isArmRight"]:
                        is_Right = True
                    if not is_backend:
                        self.ui_builder.rmp_flow = False
                        if (
                            np.linalg.norm(self.target_position - target_position)
                            != 0.0
                            or np.linalg.norm(self.target_rotation - target_rotation)
                            != 0.0
                            or self.ui_builder.curoboMotion.success is False
                        ):
                            self.target_position = target_position
                            self.target_rotation = target_rotation
                            self._hand_moveto(
                                position=target_position,
                                rotation=target_rotation,
                                isRight=is_Right,
                            )
                        if self.ui_builder.curoboMotion.reached:
                            self.data_to_send = self.ui_builder.curoboMotion.success
                    else:
                        if (
                            np.linalg.norm(self.target_position - target_position)
                            != 0.0
                            or np.linalg.norm(self.target_rotation - target_rotation)
                            != 0.0
                        ):
                            self.target_position = target_position
                            self.target_rotation = target_rotation
                            self.arm_move_rmp(
                                position=target_position,
                                rotation=target_rotation,
                                ee_interpolation=self.data["ee_interpolation"],
                                distance_frame=self.data["distance_frame"],
                                is_right=is_Right,
                            )
                        if self.ui_builder.reached:
                            self.data_to_send = True
                elif self.Command == 3:
                    target_joints_pose = self.data["target_joints_position"]
                    is_trajectory = self.data["is_trajectory"]
                    target_joint_indices = self.data["target_joints_indices"]
                    if not len(self.target_joints_pose):
                        for idx, value in enumerate(
                            list(self._get_joint_positions().values())
                        ):
                            if idx in target_joint_indices:
                                self.target_joints_pose.append(value)
                    if (
                        np.linalg.norm(self.target_joints_pose - target_joints_pose)
                        != 0
                    ):
                        self.target_joints_pose = target_joints_pose
                        self._joint_moveto(
                            target_joints_pose,
                            target_joint_indices=target_joint_indices,
                            is_trajectory=is_trajectory,
                        )
                    if not is_trajectory:
                        self.data_to_send = "move joints"
                        self.target_joints_pose = []
                    else:
                        if self.ui_builder.reached:
                            self.data_to_send = "move_joints"
                            self.target_joints_pose = []
                # GetObjectPose Get the position of an object
                elif self.Command == 5:
                    prim_path = self.data["object_prim_path"]
                    self.data_to_send = self._get_object_pose(prim_path)
                # AddUsdObject Add usd object
                elif self.Command == 6:
                    usd_path = self.data["usd_object_path"]
                    prim_path = self.data["usd_object_prim_path"]
                    label_name = self.data["usd_label_name"]
                    position = self.data["usd_object_position"]
                    rotation = self.data["usd_object_rotation"]
                    scale = self.data["usd_object_scale"]
                    object_color = self.data["object_color"]
                    object_material = self.data["object_material"]
                    object_mass = self.data["object_mass"]
                    add_particle = self.data["add_particle"]
                    particle_position = self.data["particle_position"]
                    particle_scale = self.data["particle_scale"]
                    particle_color = self.data["particle_color"]
                    object_com = self.data["object_com"]
                    model_type = self.data["model_type"]
                    static_friction = self.data["static_friction"]
                    dynamic_friction = self.data["dynamic_friction"]
                    self._add_usd_object(
                        usd_path=usd_path,
                        prim_path=prim_path,
                        label_name=label_name,
                        position=position,
                        rotation=rotation,
                        scale=scale,
                        object_color=object_color,
                        object_material=object_material,
                        object_mass=object_mass,
                        add_particle=add_particle,
                        particle_position=particle_position,
                        particle_scale=particle_scale,
                        object_com=object_com,
                        model_type=model_type,
                        static_friction=static_friction,
                        dynamic_friction=dynamic_friction,
                    )
                    self.data_to_send = "object added"
                elif self.Command == 8:
                    self.data_to_send = self._get_joint_positions()
                elif self.Command == 9:
                    state = self.data["gripper_state"]
                    isRight = self.data["is_gripper_right"]
                    width = self.data["opened_width"]
                    if self.gripper_state != state:
                        self._set_gripper_state(
                            state=state, isRight=isRight, width=width
                        )
                        self.gripper_state = state
                    if self.gripper_type == "surface":
                        is_reached = True
                    else:
                        if isRight:
                            is_reached = self.gripper_R.is_reached
                        else:
                            is_reached = self.gripper_L.is_reached
                    if is_reached:
                        self.gripper_state = ""
                        self.data_to_send = "gripper moving"

                elif self.Command == 11:
                    if self.data["startRecording"]:
                        self.task_name = self.data["task_name"]
                        self.fps = self.data["fps"]
                        main_dir = os.getenv("SIM_REPO_ROOT")
                        root_path = os.path.join(main_dir, "output/recording_data/")
                        recording_path = os.path.join(root_path, self.task_name)
                        if os.path.isdir(recording_path):
                            folder_index = 1
                            while os.path.isdir(
                                os.path.join(recording_path, str(folder_index))
                            ):
                                folder_index += 1
                            recording_path = os.path.join(
                                recording_path, str(folder_index)
                            )
                        self.path_to_save = recording_path
                        self.camera_info_list = {}
                        tf_target = []
                        for prim in self.data["camera_prim_list"]:
                            image = self._capture_camera(
                                prim_path=prim,
                                isRGB=False,
                                isDepth=False,
                                isSemantic=False,
                                isGN=False,
                            )
                            prim_name = prim.split("/")[-1]
                            self.camera_info_list[prim_name] = {
                                "intrinsic": image["camera_info"],
                                "output": {
                                    "rgb": "camera/"
                                    + "{frame_num}/"
                                    + f"{prim_name}.jpg",
                                    "video": f"{prim_name}.mp4",
                                },
                            }
                            if "fisheye" not in prim_name:
                                if "G1" in self.robot_name:
                                    self.camera_info_list[prim_name]["output"][
                                        "depth"
                                    ] = (
                                        "camera/" + "{frame_num}/" + f"{prim_name}.png"
                                    )
                                else:
                                    self.camera_info_list[prim_name]["output"][
                                        "depth"
                                    ] = (
                                        "camera/"
                                        + "{frame_num}/"
                                        + f"{prim_name}_depth.png"
                                    )

                            if self.data["render_semantic"]:
                                self.camera_info_list[prim_name]["output"][
                                    "semantic"
                                ] = (
                                    "camera/"
                                    + "{frame_num}/"
                                    + f"{prim_name}_semantic.png"
                                )
                            tf_target.append(prim)
                        if self.publish_ros:
                            command = [
                                "ros2",
                                "bag",
                                "record",
                                "-o",
                                recording_path,
                                "--exclude",
                                ".*rgb",
                                "-a",
                            ]
                            process = subprocess.Popen(command)
                            self.process_pid.append(process.pid)
                            frequency = (int)(60 / self.data["fps"])
                            self.sensor_base._init_sensor(self.loop_count)
                            for prim in self.object_prims:
                                if prim not in tf_target:
                                    tf_target.append(prim)
                            for prim in self.end_effector_prim_path.values():
                                if prim not in tf_target:
                                    tf_target.append(prim)
                            delta_time = 1 / (2 * self.rendering_step)
                            logger.info(self.end_effector_prim_path)
                            tf_target.append(f"{self.robot_prim_path}/base_link")
                            self.sensor_base.publish_tf(
                                robot_prim=self.robot_prim_path,
                                targets=tf_target,
                                approx_freq=frequency,
                                delta_time=delta_time,
                            )
                            for prim in self.articulat_objects.keys():
                                self.sensor_base.publish_joint(
                                    robot_prim=prim,
                                    approx_freq=frequency,
                                    delta_time=delta_time,
                                    topic_name=prim,
                                )
                            self.fps = self.data["fps"]
                            self.graph_path = [
                                "/World/RobotTFActionGraph",
                                "/World/RobotJointActionGraph",
                                "/ClockActionGraph",
                            ]

                            if not self.camera_graph_path:
                                for camera in self.data["camera_prim_list"]:

                                    camera_param = {
                                        "path": camera,
                                        "frequency": frequency,
                                        "resolution": {
                                            "width": self.cameras[camera][0],
                                            "height": self.cameras[camera][1],
                                        },
                                        "publish": [
                                            "rgb:/" + camera.split("/")[-1] + "_rgb",
                                            "depth:/" + camera.split("/")[-1],
                                        ],
                                    }

                                    if "Fisheye" in camera or "Top" in camera:
                                        camera_param["publish"] = [
                                            "rgb:/" + camera.split("/")[-1] + "_rgb"
                                        ]
                                    else:
                                        camera_param["publish"] = [
                                            "rgb:/" + camera.split("/")[-1] + "_rgb",
                                            "depth:/" + camera.split("/")[-1],
                                        ]
                                    if self.data["render_semantic"]:
                                        camera_param["publish"].append(
                                            "semantic:/"
                                            + camera.split("/")[-1]
                                            + "_semantic"
                                        )
                                    camera_graph = self.sensor_base._init_camera(
                                        camera_param
                                    )
                                    topic_name = "/" + camera.split("/")[-1] + "_rgb"
                                    compressed_name = topic_name + "_compressed"
                                    command = [
                                        "ros2",
                                        "run",
                                        "image_transport",
                                        "republish",
                                        "raw",
                                        "compressed",
                                        "--ros-args",
                                        "--remap",
                                        f"/in:={topic_name}",
                                        "--remap",
                                        f"/out:={compressed_name}",
                                    ]
                                    subpro = subprocess.Popen(command)
                                    self.process_pid.append(subpro.pid)
                                self.camera_graph_path.append(
                                    self.data["camera_prim_list"]
                                )
                            self.data_to_send = "Start"
                        else:
                            os.makedirs(recording_path, exist_ok=True)
                            self.recorder_req = {
                                "fps": self.data["fps"],
                                "isCam": self.data["isCam"],
                                "isJoint": self.data["isJoint"],
                                "isPose": self.data["isPose"],
                                "isGripper": self.data["isGripper"],
                                "camera_prim_list": self.data["camera_prim_list"],
                                "render_depth": self.data["render_depth"],
                                "render_semantic": self.data["render_semantic"],
                                "object_prim": self.data["object_prim"],
                            }
                            self.step = 0
                            self.state_info = {
                                "robot_base_pose": [],
                                "camera_pose": [],
                                "joint_position": [],
                                "ee_pose": [],
                                "object_pose": [],
                                "action": [],
                                "gripper_state": [],
                                "camera_image": [],
                                "record_duration": "",
                                "origin_steps_num": "",
                                "synced_steps_num": "",
                            }
                            self.Start_Recording = True
                            self.data_to_send = "Start"
                    elif self.data["stopRecording"]:
                        self.Start_Recording = False
                        if self.publish_ros:
                            for process_pid in self.process_pid:
                                os.kill(process_pid, signal.SIGTERM)

                            async def store_info():
                                if self.record_images:
                                    from genie.sim.lab.controllers.extract_ros_bag import (
                                        Ros_Extrater,
                                    )
                                else:
                                    from genie.sim.lab.controllers.extract_state_json import (
                                        Ros_Extrater,
                                    )
                                extract_ros = Ros_Extrater(
                                    bag_file=self.path_to_save,
                                    output_dir=self.path_to_save,
                                    robot_init_position=self.robot_init_position,
                                    robot_init_rotation=self.robot_init_rotation,
                                    camera_info=self.camera_info_list,
                                    scene_name=self.scene_name,
                                    scene_usd=self.scene_usd,
                                    object_names=self.object_prims,
                                    fps=self.fps,
                                    robot_name=self.robot_name,
                                    frame_status=self.frame_status,
                                )
                                while True:
                                    await asyncio.sleep(1)
                                    if os.path.isfile(
                                        self.path_to_save + "/metadata.yaml"
                                    ):
                                        extract_ros.extract()
                                        break

                            asyncio.run(store_info())
                            self.process_pid = []
                        self.data_to_send = "Stopped"
                    else:
                        isCam = self.data["isCam"]
                        isJoint = self.data["isJoint"]
                        isPose = self.data["isPose"]
                        isGripper = self.data["isGripper"]
                        data_to_send = {}

                        cam_datas = []
                        joint_datas = None
                        object_datas = []
                        gripper_datas = {}
                        if isCam:
                            prim_list = self.data["camera_prim_list"]
                            isDepth = self.data["render_depth"]
                            isSemantic = self.data["render_semantic"]
                            for prim in prim_list:
                                cam_datas.append(
                                    self._capture_camera(
                                        prim_path=prim,
                                        isRGB=True,
                                        isDepth=isDepth,
                                        isSemantic=isSemantic,
                                        isGN=False,
                                    )
                                )
                        if isJoint:
                            joint_datas = self._get_joint_positions()
                        if isPose:
                            obj_prim_list = self.data["object_prim"]
                            for prim in obj_prim_list:
                                object_datas.append(self._get_object_pose(prim))
                        if isGripper:
                            gripper_datas = {
                                "left": self._get_ee_pose(is_right=False),
                                "right": self._get_ee_pose(is_right=True),
                            }
                        self.data_to_send = {
                            "camera": cam_datas,
                            "joint": joint_datas,
                            "object": object_datas,
                            "gripper": gripper_datas,
                        }
                elif self.Command == 12:
                    self._on_reset()
                    self.loop_count += 1
                    self.data_to_send = "reset"
                elif self.Command == 13:
                    obj_prims = self.data["obj_prim_paths"]
                    is_right = self.data["is_right"]
                    items = []
                    stage = omni.usd.get_context().get_stage()
                    if stage:
                        for prim_path in obj_prims:
                            for prim in Usd.PrimRange(stage.GetPrimAtPath(prim_path)):
                                path = str(prim.GetPath())
                                prim = get_prim_at_path(path)
                                if prim.IsA(UsdGeom.Mesh):
                                    items.append(path)
                    result = self.ui_builder.attach_objs(items, is_right)
                    self.data_to_send = "attaching"
                elif self.Command == 14:
                    self.ui_builder.detach_objs()
                    self.data_to_send = "detaching"
                elif self.Command == 15:
                    is_plan = self.data["isPlan"]
                    poses = self.data["poses"]
                    index = self.data["plan_index"]
                    if is_plan:
                        result = self.ui_builder._caculate_multi_ik(poses)
                        self.data_to_send = {
                            "cmd_plans": result,
                            "msg": self.ui_builder.curoboMotion.success,
                        }
                    else:
                        self.ui_builder._multi_move(index)
                        if self.ui_builder.curoboMotion.reached:
                            self.data_to_send = self.ui_builder.curoboMotion.success
                elif self.Command == 16:
                    isSuccess = self.data["isSuccess"]
                    self.object_prims = []
                    if self.task_name is not None:
                        result = {
                            "task_name": self.task_name,
                            "fail_stage_step": self.data["failStep"],
                            "fps": self.fps,
                            "task_status": isSuccess,
                            "camera_info": self.camera_info_list,
                        }
                        with open(
                            self.path_to_save + "/task_result.json",
                            "w",
                            encoding="utf-8",
                        ) as f:
                            json.dump(result, f, ensure_ascii=False)
                        with open(
                            self.path_to_save + "/frame_state.json",
                            "w",
                            encoding="utf-8",
                        ) as f:
                            json.dump(self.frame_status, f, indent=4)
                    self.data_to_send = str(isSuccess)
                elif self.Command == 17:
                    self.exit = self.data["exit"]
                    self.data_to_send = "exit"
                elif self.Command == 18:
                    cmd = "get_ee_pose"
                    is_right = self.data["isRight"]
                    self.data_to_send = self._get_ee_pose(is_right)
                elif self.Command == 19:
                    target_poses = self.data["target_poses"]
                    ik_result = []
                    is_Right = self.data["isRight"]
                    ObsAvoid = self.data["ObsAvoid"]
                    n = 0
                    for pose in target_poses:
                        ik_result.append(
                            self._get_ik_status(
                                np.array(pose["position"]),
                                np.array(pose["rotation"]),
                                is_Right,
                                ObsAvoid,
                            )
                        )
                    self.data_to_send = ik_result
                elif self.Command == 20:
                    obj_type = self.data["obj_type"]
                    self.data_to_send = self._find_all_objects_of_type(obj_type)
                elif self.Command == 21:
                    robot_cfg_file = self.data["robot_cfg_file"]
                    scene_usd_path = self.data["scene_usd_path"]
                    self._init_robot_cfg(
                        robot_cfg=robot_cfg_file,
                        scene_usd=scene_usd_path,
                        init_position=self.data["robot_position"],
                        init_rotation=self.data["robot_rotation"],
                        stand_type=self.data["stand_type"],
                        size_x=self.data["stand_size_x"],
                        size_y=self.data["stand_size_y"],
                    )
                    self.data_to_send = "success"
                elif self.Command == 22:
                    camera_prim = self.data["camera_prim"]
                    camera_position = self.data["camera_position"]
                    camera_rotation = self.data["camera_rotation"]
                    self._add_camera(
                        camera_prim=camera_prim,
                        camera_position=camera_position,
                        camera_rotation=camera_rotation,
                        width=self.data["width"],
                        height=self.data["height"],
                        focal_length=self.data["focus_length"],
                        horizontal_aperture=self.data["horizontal_aperture"],
                        vertical_aperture=self.data["vertical_aperture"],
                        is_local=self.data["is_local"],
                    )
                    self.data_to_send = "success"
                elif self.Command == 23:
                    self.draw_lines(
                        self.data["point_list_1"],
                        self.data["point_list_2"],
                        self.data["colors"],
                        self.data["sizes"],
                        None,
                    )
                    self.data_to_send = "success"
                elif self.Command == 24:
                    self._set_object_pose(
                        self.data["object_poses"],
                        self.data["joint_position"],
                        self.data["object_joints"],
                    )
                    self.data_to_send = "success"
                elif self.Command == 25:
                    self.trajectory_blocking = self.data["is_block"]
                    if not self.trajectory_list:
                        self.trajectory_list = self.data["trajectory_list"]
                        self.trajectory_index = 0
                        self.trajectory_reached = False
                        if not self.data["is_block"]:
                            self.data_to_send = "success"
                    if self.trajectory_index >= len(self.trajectory_list):
                        self.data_to_send = "success"
                elif self.Command == 26:
                    self.data_to_send = self._get_object_joint(
                        self.data["object_prim_path"]
                    )
                elif self.Command == 27:
                    self.target_point = self.data["target_position"]
                    self.data_to_send = "success"
                elif self.Command == 28:
                    time_stamp = self.timeline.get_current_time()
                    self.frame_status.append(
                        {
                            "time_stamp": time_stamp,
                            "frame_state": json.loads(self.data["frame_state"]),
                        }
                    )
                    self.data_to_send = "success"
                elif self.Command == 29:
                    for material_info in self.data:
                        self._set_object_material(
                            material_info["object_prim"],
                            material_info["material_name"],
                            material_info["material_path"],
                            material_info["label_name"],
                        )
                    self.data_to_send = "success"
                elif self.Command == 30:
                    for light in self.data:
                        self._set_light(
                            light_type=light["light_type"],
                            light_prim=light["light_prim"],
                            light_temperature=light["light_temperature"],
                            light_intensity=light["light_intensity"],
                            light_rotation=light["light_rotation"],
                            light_texture=light["light_texture"],
                        )
                    self.data_to_send = "success"
                elif self.Command == 31:
                    self.clear_lines(self.data["name"])
                    self.data_to_send = "success"
                elif self.Command == 32:
                    omni.kit.commands.execute(
                        "ChangeProperty",
                        prop_path=Sdf.Path(self.data["prop_path"]),
                        value=self.data["value"],
                        prev=None,
                    )
                    self.data_to_send = "success"
        if self.Command:
            with self.condition:
                self.condition.notify_all()

    def _on_recording_step(self):
        def get_pose(xyz: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
            pose = np.eye(4)
            pose[:3, :3] = get_rotation_matrix_from_quaternion(quat_wxyz)
            pose[:3, 3] = xyz
            return pose

        def write_rgb_image(data, path, box):
            rgb_image = Image.fromarray(data, mode="RGBA")
            region = rgb_image.crop(box)
            region.save(path + ".png")

        self.step += 1
        fps = self.recorder_req["fps"]
        physics_dt = 30
        if self.enable_physics:
            physics_dt = 120
        num_step = max(1, int(physics_dt / fps))
        self.image_path = {}
        if self.step % num_step == 0:
            isCam = self.recorder_req["isCam"]
            idx = (int)(self.step / num_step) - 1
            step_folder = os.path.join(self.path_to_save, str(idx))
            os.makedirs(step_folder, exist_ok=True)
            if isCam:
                prim_list = self.recorder_req["camera_prim_list"]
                prim_index = 0
                for prim in prim_list:
                    prim_index += 1
                    image = self._capture_camera(
                        prim_path=prim,
                        isRGB=True,
                        isDepth=True,
                        isSemantic=False,
                        isGN=False,
                    )
                    file_name = prim.split("/")[-1]
                    box = (0, 0, self.cameras[prim][0], self.cameras[prim][1])
                    if box in self.cameras:
                        box = (0, 0, self.cameras[prim][0], self.cameras[prim][1])
                    write_rgb_image(
                        data=image["rgb"],
                        path=f"{step_folder}/{file_name}_color",
                        box=box,
                    )
                    np.savez_compressed(
                        step_folder + "/" + file_name + "_depth.npz", image["depth"]
                    )
                    if file_name not in self.image_path:
                        self.image_path[file_name] = {
                            "rgb": f"{file_name}_color.png",
                            "depth": f"{file_name}_depth.npz",
                        }
            robot_base_pose = get_pose(
                *self._get_object_pose("/World/G1/base_link")
            ).tolist()
            joint_position = self._get_joint_positions()
            camera_pose = {}
            rotation_x_180 = np.array(
                [
                    [1.0, 0.0, 0.0, 0],
                    [0.0, -1.0, 0.0, 0],
                    [0.0, 0.0, -1.0, 0],
                    [0, 0, 0, 1],
                ]
            )
            for cam_prim in self.recorder_req["camera_prim_list"]:
                camera_pose[cam_prim] = (
                    get_pose(*self._get_object_pose(cam_prim)) @ rotation_x_180
                ).tolist()
            object_pose = {}
            for prim in self.object_prims:
                object_pose[prim] = (get_pose(*self._get_object_pose(prim))).tolist()
            ee_pose = (get_pose(*self._get_ee_pose(True))).tolist()
            state_info = {
                "time_stamp": self.step / 30,
                "robot_base_pose": robot_base_pose,
                "joint_position": joint_position,
                "ee_pose": ee_pose,
                "camera_pose": camera_pose,
                "object_pose": object_pose,
            }
            self.state_info["robot_base_pose"].append(robot_base_pose)
            self.state_info["camera_pose"].append(camera_pose)
            self.state_info["joint_position"].append(joint_position)
            self.state_info["ee_pose"].append(ee_pose)
            self.state_info["object_pose"].append(object_pose)
            self.state_info["camera_image"].append(self.image_path)
            with open(step_folder + "/state.json", "w", encoding="utf-8") as f:
                json.dump(state_info, f, indent=4)

    def _generate_materials(self):
        self.materials = {}
        material_infos = {}
        path = os.path.dirname(__file__) + "/material_infos.json"
        with open(path, "r") as f:
            material_infos = json.load(f)

        for mat in material_infos:
            material = self.material_changer.assign_material(
                material_infos[mat]["material_path"], mat
            )
            self.materials[mat] = material

    def _get_observation(self):
        for camera in self.cameras:
            self._capture_camera(
                prim_path=camera,
                isRGB=True,
                isDepth=False,
                isSemantic=False,
                isGN=False,
            )

    def _on_reset(self):
        self._reset_stiffness()
        self.ui_builder._on_reset()
        self.usd_objects = {}
        self.target_position = [0, 0, 0]
        self._get_observation()
        self.articulat_objects = {}
        self.frame_status = []

    def _on_blocking_thread(self, data, Command):
        self.data = data
        self.Command = Command
        with self.condition:
            while self.data_to_send is None:
                self.condition.wait()
            result = self.data_to_send
            self.data_to_send = None
            self.Command = 0
            self.result_queue.put(result)

    def blocking_start_server(self, data, Command):
        self._on_blocking_thread(data, Command)
        if not self.result_queue.empty():
            result = self.result_queue.get()
            return result

    # debug_draw_line
    def draw_lines(self, point_list_1, point_list_2, colors, sizes, name):
        # draw = _debug_draw.acquire_debug_draw_interface()
        # draw.draw_lines(point_list_1, point_list_2, colors, sizes)
        # self.debug_view = draw
        pass

    def clear_lines(self, name):
        self.debug_view.clear_lines()

    # 1. Photo capturing function, prim path of Input camera in isaac side scene and whether to use Gaussian Noise, return
    def _capture_camera(self, prim_path: str, isRGB, isDepth, isSemantic, isGN: bool):
        isDepth = False
        isSemantic = False
        self.ui_builder._currentCamera = prim_path
        self.ui_builder._on_capture_cam(isRGB, isDepth, isSemantic)
        currentImage = self.ui_builder.currentImg
        return currentImage

    # 2. Move the left and right hands to the specified position, position(x,y,z), rotation(x,y,z) angle
    def _hand_moveto(self, position, rotation, isRight=True):
        self.ui_builder._followingPos = position
        self.ui_builder._followingOrientation = rotation
        self._initialize_articulation()
        self.ui_builder._follow_target(isRight)

    def _reset_stiffness(self):
        if not self.gripper_initialized:
            self._init_grippers()
        self.gripper_L.reset_stiffness()
        self.gripper_R.reset_stiffness()

    def arm_move_rmp(
        self, position, rotation, ee_interpolation, distance_frame, is_right=True
    ):
        self.ui_builder._followingPos = position
        self.ui_builder._followingOrientation = rotation
        self.ui_builder._trajectory_list_follow_target(
            position, rotation, is_right, ee_interpolation, distance_frame
        )

    def _initialize_articulation(self):
        return self.ui_builder.articulation

    # 3. The whole body joints move to the specified angle, Input:np.array([None])*28
    def _joint_moveto(self, joint_position, is_trajectory, target_joint_indices):
        self._initialize_articulation()
        self.ui_builder._move_to(joint_position, target_joint_indices, is_trajectory)

    def _add_camera(
        self,
        camera_prim,
        camera_position,
        camera_rotation,
        width=640,
        height=480,
        focal_length=18.14756,
        horizontal_aperture=20.955,
        vertical_aperture=15.2908,
        is_local=False,
    ):
        camera = Camera(prim_path=camera_prim, resolution=[width, height])
        camera.initialize()
        self._get_observation()
        self._capture_camera(
            prim_path=camera_prim,
            isRGB=True,
            isDepth=False,
            isSemantic=False,
            isGN=False,
        )
        if is_local:
            camera.set_local_pose(
                translation=camera_position,
                orientation=camera_rotation,
                camera_axes="usd",
            )
        else:
            camera.set_world_pose(
                position=camera_position, orientation=camera_rotation, camera_axes="usd"
            )
        self.cameras[camera_prim] = [width, height]
        self.ui_builder.cameras[camera_prim] = [width, height]
        _prim = get_prim_at_path(camera_prim)
        _prim.GetAttribute("focalLength").Set(focal_length)
        _prim.GetAttribute("horizontalAperture").Set(horizontal_aperture)
        _prim.GetAttribute("verticalAperture").Set(vertical_aperture)
        _prim.GetAttribute("clippingRange").Set((0.01, 100000))

    def _get_object_joint(self, prim_path):
        self.articulat_objects[prim_path].initialize()
        dof_names = self.articulat_objects[prim_path].dof_names
        positions = self.articulat_objects[prim_path].get_joint_positions()
        velocities = self.articulat_objects[prim_path].get_joint_velocities()
        return {
            "joint_names": dof_names,
            "joint_positions": positions,
            "joint_velocities": velocities,
        }

    def _set_object_joint(self, prim_path, target_positions):
        self.articulat_objects[prim_path].initialize()
        self.articulat_objects[prim_path].set_joint_positions(target_positions)

    def _add_particle(self, position, size):
        stage = get_current_stage()
        particle_system_path = Sdf.Path("/World/Objects/part/particleSystem")
        if stage.GetPrimAtPath(particle_system_path) != None:
            omni.kit.commands.execute(
                "DeletePrims", paths=[particle_system_path], destructive=False
            )
        # create a scene with gravity and up axis:
        scene = self.scene
        Particle_Contact_Offset = 0.0045
        Sample_Volume = 1
        particle_system = particleUtils.add_physx_particle_system(
            stage,
            particle_system_path,
            particle_system_enabled=True,
            simulation_owner=scene.GetPath(),
            particle_contact_offset=Particle_Contact_Offset,
            max_velocity=0.5,
        )
        # create particle material and assign it to the system:
        particle_material_path = Sdf.Path("/World/Objects/part/particleMaterial")
        particleUtils.add_pbd_particle_material(
            stage,
            particle_material_path,
            friction=0,
            density=1000.0,
            viscosity=0,
            cohesion=0.0,
            surface_tension=0.0,
            drag=0.0,
            lift=0.0,
        )  # Set the viscosity.

        physicsUtils.add_physics_material_to_prim(
            stage, stage.GetPrimAtPath(particle_system_path), particle_material_path
        )
        cube_mesh_path = Sdf.Path("/World/Objects/part/Cube")
        cube_resolution = 20  # resolution can be low because we'll sample the surface / volume only irrespective of the vertex count
        omni.kit.commands.execute(
            "CreateMeshPrimWithDefaultXform",
            prim_type="Cylinder",
            u_patches=cube_resolution,
            v_patches=cube_resolution,
            prim_path=cube_mesh_path,
        )
        cube_mesh = UsdGeom.Mesh.Get(stage, cube_mesh_path)
        physicsUtils.set_or_add_translate_op(
            cube_mesh, Gf.Vec3f(position[0], position[1], position[2])
        )
        physicsUtils.set_or_add_scale_op(cube_mesh, Gf.Vec3f(size[0], size[1], size[2]))
        particle_points_path = Sdf.Path("/World/Objects/part/sampledParticles")
        points = UsdGeom.Points.Define(stage, particle_points_path)
        point_prim = stage.GetPrimAtPath(particle_points_path)
        visibility_attribute = point_prim.GetAttribute("visibility")
        if visibility_attribute is not None:
            visibility_attribute.Set("invisible")
        geometry_prim = SingleGeometryPrim(
            prim_path="/World/Objects/part/particleSystem"
        )
        material_prim = "/World/Looks_01/OmniGlass"
        material = OmniGlass(
            prim_path=material_prim, color=np.array([0.645, 0.271, 0.075])
        )
        geometry_prim.apply_visual_material(material)
        particle_set_api = PhysxSchema.PhysxParticleSetAPI.Apply(points.GetPrim())
        PhysxSchema.PhysxParticleAPI(
            particle_set_api
        ).CreateParticleSystemRel().SetTargets([particle_system_path])
        fluid_rest_offset = 0.99 * 0.6 * Particle_Contact_Offset
        particle_sampler_distance = 2.0 * fluid_rest_offset
        sampling_api = PhysxSchema.PhysxParticleSamplingAPI.Apply(cube_mesh.GetPrim())
        sampling_api.CreateParticlesRel().AddTarget(particle_points_path)
        sampling_api.CreateSamplingDistanceAttr().Set(particle_sampler_distance)
        sampling_api.CreateMaxSamplesAttr().Set(5e5)
        sampling_api.CreateVolumeAttr().Set(Sample_Volume)
        particleUtils.add_physx_particle_isosurface(
            stage, particle_system_path, enabled=True
        )
        self.ui_builder.my_world.stop()
        self._play()

    # Add objects
    def _add_usd_object(
        self,
        usd_path: str,
        prim_path: str,
        label_name: str,
        position,
        rotation,
        scale,
        object_color,
        object_material,
        object_mass,
        add_particle=False,
        particle_position=[0, 0, 0],
        particle_scale=[0.1, 0.1, 0.1],
        particle_color=[1, 1, 1],
        object_com=[0, 0, 0],
        model_type="convexDecomposition",
        static_friction=0.5,
        dynamic_friction=0.5,
    ):
        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        if add_particle and self.enable_physics:
            particle_pos = [
                position[0] + particle_position[0],
                position[1] + particle_position[1],
                position[2] + particle_position[2],
            ]
            self._add_particle(particle_pos, particle_scale)
        usd_object = SingleXFormPrim(
            prim_path=prim_path, position=position, orientation=rotation, scale=scale
        )
        stage = omni.usd.get_context().get_stage()
        type = get_prim_object_type(prim_path)
        items = []
        if stage:
            for prim in Usd.PrimRange(stage.GetPrimAtPath(prim_path)):
                path = str(prim.GetPath())
                prim = get_prim_at_path(path)
                if prim.IsA(UsdGeom.Mesh):
                    items.append(path)
        self.object_prims.append(prim_path)
        object_rep = rep.get.prims(path_pattern=prim_path, prim_types=["Xform"])

        with object_rep:
            rep.modify.semantics([("class", label_name)])
        if type == "articulation":
            self.ui_builder.my_world.play()
            for path in items:
                if not self.enable_physics:
                    collisionAPI = UsdPhysics.CollisionAPI.Get(stage, path)
                    if collisionAPI:
                        collisionAPI.GetCollisionEnabledAttr().Set(False)
            articulation = SingleArticulation(prim_path)
            articulation.initialize()
            self.articulat_objects[prim_path] = articulation
            self.usd_objects[prim_path] = usd_object
        else:
            self.usd_objects[prim_path] = usd_object
            for _prim in items:
                geometry_prim = SingleGeometryPrim(prim_path=_prim)
                obj_physics_prim_path = f"{_prim}/object_physics"
                geometry_prim.apply_physics_material(
                    PhysicsMaterial(
                        prim_path=obj_physics_prim_path,
                        static_friction=static_friction,
                        dynamic_friction=dynamic_friction,
                        restitution=None,
                    )
                )
                # set friction combine mode to max to enable stable grasp
                obj_physics_prim = stage.GetPrimAtPath(obj_physics_prim_path)
                physx_material_api = PhysxSchema.PhysxMaterialAPI(obj_physics_prim)
                if physx_material_api is not None:
                    fric_combine_mode = (
                        physx_material_api.GetFrictionCombineModeAttr().Get()
                    )
                    if fric_combine_mode == None:
                        physx_material_api.CreateFrictionCombineModeAttr().Set("max")
                    elif fric_combine_mode != "max":
                        physx_material_api.GetFrictionCombineModeAttr().Set("max")

                if object_material != "general":
                    if object_material == "Glass":
                        material_prim = "/World/Materials/OmniGlass"
                        material = OmniGlass(prim_path=material_prim)
                        geometry_prim.apply_visual_material(material)
                    elif object_material not in self.materials:
                        material_prim = prim_path + "/Looks/DefaultMaterial"
                        material = OmniPBR(
                            prim_path=material_prim,
                            color=object_color,
                        )
                        material.set_metallic_constant(1)
                        material.set_reflection_roughness(0.4)
                        geometry_prim.apply_visual_material(material)
                    else:
                        Material = self.materials[object_material]
                        prim = stage.GetPrimAtPath(_prim)
                        UsdShade.MaterialBindingAPI(prim).Bind(Material)
            if self.enable_physics:
                prim = stage.GetPrimAtPath(prim_path)
                utils.setRigidBody(prim, model_type, False)
                rigid_prim = SingleRigidPrim(prim_path=prim_path)
                # Get Physics API
                physics_api = UsdPhysics.MassAPI.Apply(rigid_prim.prim)
                physics_api.CreateMassAttr().Set(object_mass)

            for _prim in items:
                # disable kettle lid collision
                if "part_02" in _prim:
                    collisionAPI = UsdPhysics.CollisionAPI.Get(stage, _prim)
                    if collisionAPI:
                        collisionAPI.GetCollisionEnabledAttr().Set(False)

    def _set_object_pose(
        self, object_poses, joint_position, object_joints=None, action=False
    ):
        for pose in object_poses:
            object = self.usd_objects[pose["prim_path"]]
            object.set_world_pose(pose["position"], pose["rotation"])
        if len(joint_position):
            joint_indices = []
            for i in range(len(joint_position)):
                joint_indices.append(i)
            self.ui_builder._move_to(joint_position, joint_indices, action)
        if object_joints is not None:
            for joint in object_joints:
                self._set_object_joint(
                    prim_path=joint["prim_path"], target_positions=joint["object_joint"]
                )

    def _set_object_material(
        self, prim_path, material_name, material_path, label_name=None
    ):
        items = []
        stage = omni.usd.get_context().get_stage()
        logger.info(label_name)
        if label_name:
            object_rep = rep.get.prims(path_pattern=prim_path, prim_types=["Xform"])
            with object_rep:
                rep.modify.semantics([("class", label_name)])
        if not stage:
            return
        if "Glass" in material_name or "glass" in material_name:
            material_prim = "/World/Materials/OmniGlass"
            material = OmniGlass(prim_path=material_prim)
            for prim in Usd.PrimRange(stage.GetPrimAtPath(prim_path)):
                path = str(prim.GetPath())
                prim = get_prim_at_path(path)
                if prim.IsA(UsdGeom.Mesh) or prim.GetTypeName() in "GeomSubset":
                    geometry_prim = SingleGeometryPrim(prim_path=path)
                    geometry_prim.apply_visual_material(material)

        else:
            material = self.material_changer.assign_material(
                material_path, material_name
            )
            for prim in Usd.PrimRange(stage.GetPrimAtPath(prim_path)):
                path = str(prim.GetPath())
                prim = get_prim_at_path(path)
                if prim.IsA(UsdGeom.Mesh) or prim.GetTypeName() in "GeomSubset":
                    UsdShade.MaterialBindingAPI(prim).Bind(material)

    def _set_light(
        self,
        light_type,
        light_prim,
        light_temperature,
        light_intensity,
        light_rotation,
        light_texture,
    ):
        stage = omni.usd.get_context().get_stage()
        light = Light(
            light_type=light_type,
            prim_path=light_prim,
            stage=stage,
            intensity=light_intensity,
            color=light_temperature,
            orientation=light_rotation,
            texture_file=light_texture,
        )
        light.initialize()

    def _get_joint_positions(self):
        self._initialize_articulation()
        articulation = self.ui_builder.articulation
        joint_positions = articulation.get_joint_positions()
        ids = {}
        for idx in range(len(articulation.dof_names)):
            name = articulation.dof_names[idx]
            ids[name] = float(joint_positions[idx])
        return ids

    def _find_all_objects_of_type(self, obj_type):
        items = []
        stage = omni.usd.get_context().get_stage()
        if stage:
            for prim in Usd.PrimRange(stage.GetPrimAtPath("/")):
                path = str(prim.GetPath())
                type = get_prim_object_type(path)
                if type == obj_type:
                    items.append(path)
        return items

    def _init_grippers(self):
        robot = self._initialize_articulation()
        num_dof = self.ui_builder.articulation.num_dof
        if self.gripper_type == "surface":
            self.gripper_L = SurfaceGripper(
                end_effector_prim_path=self.end_effector_prim_path["left"],
                translate=0.1,
                grip_threshold=0.02,
                direction="z",
                disable_gravity=False,
            )
            self.gripper_L.initialize(articulation_num_dofs=num_dof)
            self.gripper_R = SurfaceGripper(
                end_effector_prim_path=self.end_effector_prim_path["right"],
                translate=0.1,
                grip_threshold=0.02,
                direction="z",
                disable_gravity=False,
            )
            self.gripper_R.initialize(articulation_num_dofs=num_dof)
            return robot

        end_effector_prim_path = self.end_effector_prim_path["left"]
        right_end_effector_prim_path = self.end_effector_prim_path["right"]
        self.gripper_L = ParallelGripper(
            end_effector_prim_path=end_effector_prim_path,
            joint_prim_names=self.finger_names["left"],
            joint_closed_velocities=self.closed_velocities["left"],
            joint_opened_positions=self.opened_positions["left"],
            joint_controll_prim=self.gripper_controll_joint["left"],
            gripper_type=self.gripper_type,
            gripper_max_force=self.gripper_max_force,
        )
        self.gripper_L.initialize(
            articulation_apply_action_func=robot.apply_action,
            get_joint_positions_func=robot.get_joint_positions,
            set_joint_positions_func=robot.set_joint_positions,
            dof_names=robot.dof_names,
        )
        self.gripper_R = ParallelGripper(
            end_effector_prim_path=right_end_effector_prim_path,
            joint_prim_names=self.finger_names["right"],
            joint_closed_velocities=self.closed_velocities["right"],
            joint_opened_positions=self.opened_positions["right"],
            joint_controll_prim=self.gripper_controll_joint["right"],
            gripper_type=self.gripper_type,
            gripper_max_force=self.gripper_max_force,
        )
        self.gripper_R.initialize(
            articulation_apply_action_func=robot.apply_action,
            get_joint_positions_func=robot.get_joint_positions,
            set_joint_positions_func=robot.set_joint_positions,
            dof_names=robot.dof_names,
        )

        self.gripper_initialized = True

        return robot

    def _set_gripper_state(self, state: str, isRight: bool, width):
        if not self.gripper_initialized:
            self.robot = self._init_grippers()
            for value in self.articulat_objects.values():
                value.initialize()
        if isRight:
            if self.gripper_type == "surface":
                action = self.gripper_R.forward(action=state)
                self.robot.apply_action(action)
                return True
            self.gripper_state_R = state
            opened_position_R = self.gripper_R._joint_opened_positions[0]
            self.gripper_R._joint_opened_positions = [
                opened_position_R,
                opened_position_R,
            ]
            action = self.gripper_R.forward(action=self.gripper_state_R)
            self.robot.apply_action(action)
        else:
            if self.gripper_type == "surface":
                action = self.gripper_L.forward(action=state)
                self.robot.apply_action(action)
                return True
            self.gripper_state_L = state
            opened_position_L = self.gripper_L._joint_opened_positions[0]
            self.gripper_L._joint_opened_positions = [
                opened_position_L,
                opened_position_L,
            ]
            action = self.gripper_L.forward(action=self.gripper_state_L)
            self.robot.apply_action(action)

    # Get the position of any object, Input: prim_path
    def _get_object_pose(self, object_prim_path: str) -> Tuple[np.ndarray, np.ndarray]:
        for value in self.articulat_objects.values():
            value.initialize()
        if object_prim_path == "robot":
            position, rotation = self.usd_objects["robot"].get_world_pose()
        else:
            target_object = SingleXFormPrim(prim_path=object_prim_path)
            position, rotation = target_object.get_world_pose()
        for value in self.articulat_objects.values():
            value.initialize()
        return position, rotation

    def _get_ee_pose(self, is_right: bool) -> Tuple[np.ndarray, np.ndarray]:
        position, rotation_matrix = self.ui_builder._get_ee_pose(is_right)
        rotation = rotation_matrix_to_quaternion(rotation_matrix)
        return position, rotation

    def _get_ik_status(self, target_position, target_rotation, isRight, ObsAvoid=False):
        joint_positions = {}
        if self.ui_builder.curoboMotion == None:
            return False, joint_positions

        SingleXFormPrim("/ik_pos", position=target_position)
        if not ObsAvoid:
            is_success, joint_state = self.ui_builder._get_ik_status(
                target_position, target_rotation, isRight
            )
            joint_names = []
            all_names = self.ui_builder.articulation.dof_names
            for i, idx in enumerate(joint_state.joint_indices):
                joint_positions[all_names[idx]] = joint_state.joint_positions[i]
        else:
            init_rotation_matrix = get_rotation_matrix_from_quaternion(
                self.robot_init_rotation
            )
            translation_matrix = np.zeros((4, 4))
            translation_matrix[:3, :3] = init_rotation_matrix
            translation_matrix[:3, 3] = self.robot_init_position
            translation_matrix[3, 3] = 1
            target_rotation_world = get_rotation_matrix_from_quaternion(target_rotation)
            target_matrix_world = np.zeros((4, 4))
            target_matrix_world[:3, :3] = target_rotation_world
            target_matrix_world[:3, 3] = target_position
            target_matrix_world[3, 3] = 1
            target_matrix = np.linalg.inv(translation_matrix) @ target_matrix_world
            target_rotation_matrix, target_position_local = (
                target_matrix[:3, :3],
                target_matrix[:3, 3],
            )
            target_rotation_local = get_quaternion_from_euler(
                matrix_to_euler_angles(target_rotation_matrix), order="ZYX"
            )
            if isinstance(self.end_effector_name, dict):
                end_effector_name = self.end_effector_name["left"]
                if isRight:
                    end_effector_name = self.end_effector_name["right"]
            else:
                end_effector_name = self.end_effector_name
            is_success, joint_state = self.ui_builder.curoboMotion.solve_batch_ik(
                target_position_local, target_rotation_local, end_effector_name
            )
            for i, name in enumerate(joint_state.joint_names):
                joint_positions[name] = joint_state.position[0][0].cpu().tolist()[i]

        return is_success, joint_positions
