# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os
from omni.isaac.core.utils import stage
from omni.isaac.core.prims import XFormPrim
import omni.graph.core as og

from base_utils.logger import Logger

logger = Logger()  # Create singleton instance

if os.getenv("ISAACSIM_VERSION") == "v45":
    from isaacsim.core.nodes.scripts.utils import set_target_prims
else:
    from omni.isaac.core_nodes.scripts.utils import set_target_prims


class USDBase:
    def __init__(self):
        pass

    def _init_sensor(self, ros_domain_id):
        self.ros_domain_id = (int)(ros_domain_id)
        logger.info(self.ros_domain_id)
        self.publish_clock()

    def step(self):
        pass

    def init_all_sensors(self):
        if "sensors" in self.config:
            for sensor in self.config["sensors"]:
                if sensor["type"] == "Camera":
                    self._init_camera(sensor)
                if sensor["type"] == "Lidar":
                    self._init_lidar(sensor)
                if sensor["type"] == "IMU":
                    self._init_imu(sensor)

    def _init_lidar(self, param):
        from omni.isaac.sensor import LidarRtx
        from .lidar import publish_lidar_pointcloud, publish_lidar_scan

        lidar = LidarRtx(prim_path=param["path"])
        lidar.initialize()
        approx_freq = param["frequency"]
        for publish in param["publish"]:
            if publish is None:
                continue
            split = publish.split(":", 2)
            topic = ""
            if len(split) > 1:
                publish = split[0]
                topic = split[1]
            if publish == "pointcloud":
                publish_lidar_pointcloud(lidar, approx_freq, topic)
            elif publish == "scan":
                publish_lidar_scan(lidar, approx_freq, topic)

    def _init_camera(self, param):
        import omni
        from omni.isaac.sensor import Camera
        from .camera import (
            publish_camera_info,
            publish_rgb,
            publish_pointcloud_from_depth,
            publish_depth,
            publish_boundingbox2d_loose,
            publish_boundingbox2d_tight,
            publish_boundingbox3d,
            publish_semantic_segmant,
        )

        camera = Camera(
            prim_path=param["path"],
            frequency=param["frequency"],
            resolution=(param["resolution"]["width"], param["resolution"]["height"]),
        )
        camera.initialize()

        approx_freq = param["frequency"]
        camera_graph = []
        for publish in param["publish"]:
            if publish is None:
                continue
            split = publish.split(":", 2)
            topic = ""
            if len(split) > 1:
                publish = split[0]
                topic = split[1]
            if publish == "rgb":
                camera_graph.append(publish_rgb(camera, approx_freq, ""))
            elif publish == "info":
                camera_graph.append(publish_camera_info(camera, approx_freq, topic))
            elif publish == "pointcloud":
                publish_pointcloud_from_depth(camera, approx_freq, topic)
            elif publish == "depth":
                camera_graph.append(publish_depth(camera, approx_freq, ""))
            elif publish == "bbox2_loose":
                publish_boundingbox2d_loose(camera, approx_freq, topic)
            elif publish == "bbox2_tight":
                publish_boundingbox2d_tight(camera, approx_freq, topic)
            elif publish == "bbox3":
                publish_boundingbox3d(camera, approx_freq, topic)
            elif publish == "semantic":
                publish_semantic_segmant(camera, approx_freq, topic)
        camera.initialize()
        return camera_graph

    def _init_imu(self, param):
        from omni.isaac.sensor import IMUSensor
        from .imu import publish_imu

        imu = IMUSensor(prim_path=param["path"])
        # imu.initialize()
        approx_freq = param["frequency"]
        for publish in param["publish"]:
            if publish is None:
                continue
            split = publish.split(":", 2)
            topic = ""
            if len(split) > 1:
                publish = split[0]
                topic = split[1]
            else:
                topic = param["path"] + "_imu"
            if publish == "imu":
                publish_imu(imu, approx_freq, topic)

    def reset_graph(self):
        ros_tf_graph_path = "/World/RobotTFActionGraph"

        set_target_prims(
            primPath=ros_tf_graph_path + "/RosPublishTransformTree",
            inputName="inputs:targetPrims",
            targetPrimPaths=[],
        )

    def publish_tf(
        self, robot_prim, targets, approx_freq, delta_time, topic_name="/obj_tf"
    ):
        version_45 = os.getenv("ISAACSIM_VERSION") == "v45"
        ros_tf_graph_path = "/World/RobotTFActionGraph"
        step = (int)(approx_freq)
        og.Controller.edit(
            {
                "graph_path": ros_tf_graph_path,
                "evaluator_name": "execution",
                "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
            },
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    (
                        "IsaacSimulationGate",
                        (
                            "isaacsim.core.nodes.IsaacSimulationGate"
                            if version_45
                            else "omni.isaac.core_nodes.IsaacSimulationGate"
                        ),
                    ),
                    (
                        "RosPublishTransformTree",
                        (
                            "isaacsim.ros2.bridge.ROS2PublishTransformTree"
                            if version_45
                            else "omni.isaac.ros2_bridge.ROS2PublishTransformTree"
                        ),
                    ),
                    (
                        "ReadSimTime",
                        (
                            "isaacsim.core.nodes.IsaacReadSimulationTime"
                            if version_45
                            else "omni.isaac.core_nodes.IsaacReadSimulationTime"
                        ),
                    ),
                    (
                        "PublishJointState",
                        (
                            "isaacsim.ros2.bridge.ROS2PublishJointState"
                            if version_45
                            else "omni.isaac.ros2_bridge.ROS2PublishJointState"
                        ),
                    ),
                    (
                        "RosContext",
                        (
                            "isaacsim.ros2.bridge.ROS2Context"
                            if version_45
                            else "omni.isaac.ros2_bridge.ROS2Context"
                        ),
                    ),
                ],
                og.Controller.Keys.CONNECT: [
                    (
                        "OnPlaybackTick.outputs:tick",
                        "IsaacSimulationGate.inputs:execIn",
                    ),
                    (
                        "IsaacSimulationGate.outputs:execOut",
                        "RosPublishTransformTree.inputs:execIn",
                    ),
                    (
                        "RosContext.outputs:context",
                        "RosPublishTransformTree.inputs:context",
                    ),
                    (
                        "ReadSimTime.outputs:simulationTime",
                        "RosPublishTransformTree.inputs:timeStamp",
                    ),
                    (
                        "IsaacSimulationGate.outputs:execOut",
                        "PublishJointState.inputs:execIn",
                    ),
                    (
                        "ReadSimTime.outputs:simulationTime",
                        "PublishJointState.inputs:timeStamp",
                    ),
                    ("RosContext.outputs:context", "PublishJointState.inputs:context"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("OnPlaybackTick.outputs:deltaSeconds", delta_time),
                    ("PublishJointState.inputs:targetPrim", robot_prim),
                    ("IsaacSimulationGate.inputs:step", step),
                    ("RosPublishTransformTree.inputs:topicName", topic_name),
                ],
            },
        )
        set_target_prims(
            primPath=ros_tf_graph_path + "/RosPublishTransformTree",
            inputName="inputs:targetPrims",
            targetPrimPaths=targets,
        )

    def publish_joint(
        self, robot_prim, approx_freq, delta_time, topic_name="/joint_state"
    ):
        version_45 = os.getenv("ISAACSIM_VERSION") == "v45"
        step = (int)(approx_freq)
        og.Controller.edit(
            {
                "graph_path": "/World/RobotJointActionGraph",
                "evaluator_name": "execution",
            },
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    (
                        "IsaacSimulationGate",
                        (
                            "isaacsim.core.nodes.IsaacSimulationGate"
                            if version_45
                            else "omni.isaac.core_nodes.IsaacSimulationGate"
                        ),
                    ),
                    (
                        "PublishJointState",
                        (
                            "isaacsim.ros2.bridge.ROS2PublishJointState"
                            if version_45
                            else "omni.isaac.ros2_bridge.ROS2PublishJointState"
                        ),
                    ),
                    (
                        "ReadSimTime",
                        (
                            "isaacsim.core.nodes.IsaacReadSimulationTime"
                            if version_45
                            else "omni.isaac.core_nodes.IsaacReadSimulationTime"
                        ),
                    ),
                    (
                        "RosContext",
                        (
                            "isaacsim.ros2.bridge.ROS2Context"
                            if version_45
                            else "omni.isaac.ros2_bridge.ROS2Context"
                        ),
                    ),
                ],
                og.Controller.Keys.CONNECT: [
                    (
                        "OnPlaybackTick.outputs:tick",
                        "IsaacSimulationGate.inputs:execIn",
                    ),
                    (
                        "IsaacSimulationGate.outputs:execOut",
                        "PublishJointState.inputs:execIn",
                    ),
                    (
                        "ReadSimTime.outputs:simulationTime",
                        "PublishJointState.inputs:timeStamp",
                    ),
                    ("RosContext.outputs:context", "PublishJointState.inputs:context"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("OnPlaybackTick.outputs:deltaSeconds", delta_time),
                    ("PublishJointState.inputs:targetPrim", robot_prim),
                    ("IsaacSimulationGate.inputs:step", step),
                    ("PublishJointState.inputs:topicName", topic_name),
                ],
            },
        )

    def publish_clock(self, clock_graph_path="/ClockActionGraph"):
        version_45 = os.getenv("ISAACSIM_VERSION") == "v45"
        ros_clock_graph_path = clock_graph_path
        og.Controller.edit(
            {
                "graph_path": ros_clock_graph_path,
                "evaluator_name": "execution",
                "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
            },
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    (
                        "RosContext",
                        (
                            "isaacsim.ros2.bridge.ROS2Context"
                            if version_45
                            else "omni.isaac.ros2_bridge.ROS2Context"
                        ),
                    ),
                    (
                        "RosPublisher",
                        (
                            "isaacsim.ros2.bridge.ROS2PublishClock"
                            if version_45
                            else "omni.isaac.ros2_bridge.ROS2PublishClock"
                        ),
                    ),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "RosPublisher.inputs:execIn"),
                    ("OnPlaybackTick.outputs:time", "RosPublisher.inputs:timeStamp"),
                    ("RosContext.outputs:context", "RosPublisher.inputs:context"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("RosContext.inputs:domain_id", self.ros_domain_id),
                ],
            },
        )
