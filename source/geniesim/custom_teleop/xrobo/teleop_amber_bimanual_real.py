# =============================================================================
# Amber Bimanual Teleop in Isaac Sim with Genie Sim IK Solver.
# Isaac Sim's SimulationApp / World / SingleArticulation APIs.
# =============================================================================
import os
import sys
import time
import numpy as np
import argparse

# =============================================================================
# Args Helper
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--robot-usd", type=str, default=None)
parser.add_argument("--scale-factor", type=float, default=1)
parser.add_argument("--physics-dt", type=float, default=1.0 / 120.0)
parser.add_argument("--rendering-dt", type=float, default=1.0 / 60.0)
parser.add_argument("--input-mode", type=str, default="hand_tracking", choices=["controller", "hand_tracking", "keyboard"], help="Input mode: 'controller' for VR controller, 'hand_tracking' for VR hand wrist, 'keyboard' for keyboard-toggled hand tracking")
args, _unknown = parser.parse_known_args()

# =============================================================================
# XRoboToolkit SDK — init BEFORE SimulationApp so the SDK's background
# network thread is established before Isaac Sim takes over signal/network.
# =============================================================================

import xrobotoolkit_sdk as xrt

xrt.init()
_ts = xrt.get_time_stamp_ns()
_lg = xrt.get_left_grip()
_rg = xrt.get_right_grip()

# =============================================================================
# ===== Isaac Sim app — must be created before any omni imports ======
# =============================================================================
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless, "renderer": "RaytracedLighting",})

from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.utils.types import ArticulationAction
from pxr import UsdPhysics, PhysxSchema, Sdf, Gf

# =============================================================================
# ===== Paths & imports =====
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IK_SDK_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../../utils/IK-SDK"))
UTILS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../../utils"))
CUSTOM_UTILS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../../custom_utils"))
GENIESIM_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../.."))  # source/geniesim/
SOURCE_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../../../.."))  # source/

sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, UTILS_DIR)
sys.path.insert(0, CUSTOM_UTILS_DIR)
sys.path.insert(0, GENIESIM_DIR)
sys.path.insert(0, SOURCE_DIR)

from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.utils.geometry import R_HEADSET_TO_WORLD
from teleop.teleop_controller import DualArmIsaacTeleopController

from name_utils import (
    AMBER_BIMANUAL_LEFT_ARM_JOINT_NAMES, AMBER_BIMANUAL_RIGHT_ARM_JOINT_NAMES, AMBER_BIMANUAL_ARM_JOINT_NAMES)


BENCHMARK_CONFIG_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../../benchmark/config"))
sys.path.insert(0, BENCHMARK_CONFIG_DIR)
from robot_init_states import AMBER_BIMANUAL_DEFAULT_STATES

SOCKET_SENDER_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../../custom_utils/socket_sender"))
sys.path.insert(0, SOCKET_SENDER_DIR)
from socket_sender import UdpSocketSender

_T_LINK7_TO_GRIPPER = np.eye(4)


# =============================================================================
# ===== Entry point =====
# =============================================================================
def main():
    robot_usd = args.robot_usd
    if robot_usd is None:
        robot_usd = os.path.join(SCRIPT_DIR, "../../assets/robot/amber/amber_bimanual.usda")

    robot_prim_path = "/amber_bimanual"

    # ===== Phase 1: Create world =====
    world = World(stage_units_in_meters=1.0, physics_dt=args.physics_dt, rendering_dt=args.rendering_dt,)

    for _ in range(10):
        world.step(render=True)

    # Physics scene (same as api_core._init_robot_cfg lines 716-725)
    stage = get_current_stage()
    scene_prim = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
    scene_prim.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene_prim.CreateGravityMagnitudeAttr().Set(9.81)
    physics_scene = PhysxSchema.PhysxSceneAPI.Get(stage, "/physicsScene")
    physics_scene.CreateGpuMaxRigidContactCountAttr(8388608 * 2)
    physics_scene.CreateGpuMaxRigidPatchCountAttr(163840 * 2)
    physics_scene.CreateGpuFoundLostPairsCapacityAttr(2097152 * 4)
    physics_scene.CreateGpuFoundLostAggregatePairsCapacityAttr(33554432 * 2)
    physics_scene.CreateGpuTotalAggregatePairsCapacityAttr(2097152 * 4)
    physics_scene.CreateGpuCollisionStackSizeAttr(67108864 * 2)
    physics_scene.CreateEnableCCDAttr(True)

    #light
    from pxr import UsdLux, UsdGeom
    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    dome.CreateIntensityAttr(1000)
    dome.CreateColorTemperatureAttr(6500)
    dome.CreateEnableColorTemperatureAttr().Set(True)
    distant = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/DistantLight"))
    distant.CreateIntensityAttr(3000)
    distant.CreateColorTemperatureAttr(6500)
    distant.CreateEnableColorTemperatureAttr().Set(True)

    distant_xform = UsdGeom.Xformable(distant.GetPrim())
    distant_xform.AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))

    world.scene.add_default_ground_plane()

    # ===== Phase 2: Load robot =====
    add_reference_to_stage(usd_path=robot_usd, prim_path=robot_prim_path)

    from isaacsim.core.prims import SingleXFormPrim
    robot_xform = SingleXFormPrim(
        prim_path=robot_prim_path,
        position=np.array([0.0, 0.0, 0.0]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
    )

    # ===== Configure joint drive stiffness/damping ======
    from pxr import Usd
    robot_prim = stage.GetPrimAtPath(robot_prim_path)

    art_root_path = f"{robot_prim_path}/full_robot/base_link"
    art_root_prim = stage.GetPrimAtPath(art_root_path)
    physx_art_api = PhysxSchema.PhysxArticulationAPI.Apply(art_root_prim)
    physx_art_api.CreateSolverPositionIterationCountAttr(16)
    physx_art_api.CreateSolverVelocityIterationCountAttr(8)

    # Fix base to world at the robot's spawn position
    robot_pos = robot_xform.get_world_pose()[0]
    fixed_joint = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(f"{robot_prim_path}/FixedJoint"))
    fixed_joint.CreateBody1Rel().SetTargets([Sdf.Path(f"{robot_prim_path}/full_robot/base_link")])
    fixed_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(float(robot_pos[0]), float(robot_pos[1]), float(robot_pos[2])))
    fixed_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
    fixed_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
    fixed_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))

    for prim in Usd.PrimRange(robot_prim):

        for drive_type in ["angular", "linear"]:
            drive_api = UsdPhysics.DriveAPI.Get(prim, drive_type)
            if not drive_api:
                continue
            joint_name = prim.GetName()


            if any(k in joint_name for k in ["Rev1", "Rev2", "Rev3"]):
                drive_api.GetStiffnessAttr().Set(5e5)
                drive_api.GetDampingAttr().Set(1e3)
                drive_api.GetMaxForceAttr().Set(1e10)
                drive_api.GetTypeAttr().Set("acceleration")

            elif any(k in joint_name for k in ["Rev4"]):
                drive_api.GetStiffnessAttr().Set(5e5)
                drive_api.GetDampingAttr().Set(1e3)
                drive_api.GetMaxForceAttr().Set(1e10)
                drive_api.GetTypeAttr().Set("acceleration")

            elif any(k in joint_name for k in ["Rev5", "Rev6", "Rev7"]):
                drive_api.GetStiffnessAttr().Set(5e5)
                drive_api.GetDampingAttr().Set(1e3)
                drive_api.GetMaxForceAttr().Set(1e10)
                drive_api.GetTypeAttr().Set("acceleration")
            else:
                drive_api.GetStiffnessAttr().Set(5e5)
                drive_api.GetDampingAttr().Set(1e3)
            print(f"  Drive configured: {joint_name} ({drive_type})"
                  f" stiffness={drive_api.GetStiffnessAttr().Get()}"
                  f" damping={drive_api.GetDampingAttr().Get()}")

    # ===== Phase 3: Play + init articulation =====
    time.sleep(1)
    world.play()
    time.sleep(1)

    articulation = SingleArticulation(prim_path=art_root_path, name="amber_bimanual",)
    world.scene.add(articulation)
    articulation.initialize()

    dof_names = list(articulation.dof_names)
    INIT_ARM = np.deg2rad(AMBER_BIMANUAL_DEFAULT_STATES["init_arm"])

    ARM_NAMES = AMBER_BIMANUAL_ARM_JOINT_NAMES

    def set_joints(names, values):
        indices = []
        positions = []
        for name, val in zip(names, values):
            if name in dof_names:
                indices.append(dof_names.index(name))
                positions.append(float(val))
        if indices:
            action = ArticulationAction(joint_positions=np.array(positions),joint_indices=indices,)
            articulation.apply_action(action)

    set_joints(ARM_NAMES, INIT_ARM)

    for _ in range(120):
        world.step(render=True)
    print("Initial pose set.")

    # XR client (reuse already-initialized SDK)
    xr_client = XrClient.__new__(XrClient)
    print("XrClient attached (SDK already initialized).")

    camera_config = {
        "head_camera": "{robot}/camera_link/head_camera",
    }

    # ===== Phase 4: Create teleop controller and run =====
    manipulator_config = {
        "left_hand": {
            "link_name": "left_Hand_Dummy_Link",
            "hand_side": "left",
            "pose_source": "left_controller",
            "control_trigger": "left_grip",
        },
        "right_hand": {
            "link_name": "right_Hand_Dummy_Link",
            "hand_side": "right",
            "pose_source": "right_controller",
            "control_trigger": "right_grip",
        },
    }

    ik_urdf_path = os.path.join(IK_SDK_DIR, "amber_bimanual.urdf")
    ik_config_path = os.path.join(IK_SDK_DIR, "amber_bimanual.yaml")

    controller = DualArmIsaacTeleopController.create(
        world=world,
        articulation=articulation,
        robot_prim_path=robot_prim_path,
        manipulator_config=manipulator_config,
        xr_client=xr_client,
        left_arm_joints=AMBER_BIMANUAL_LEFT_ARM_JOINT_NAMES,
        right_arm_joints=AMBER_BIMANUAL_RIGHT_ARM_JOINT_NAMES,
        T_link7_to_gripper_base=_T_LINK7_TO_GRIPPER,
        ik_urdf_path=ik_urdf_path,
        ik_config_path=ik_config_path,
        simulation_app=simulation_app,
        xrt_module=xrt,
        R_headset_world=R_HEADSET_TO_WORLD,
        scale_factor=args.scale_factor,
        camera_config=camera_config,
        input_mode=args.input_mode,
        ik_solver_kwargs={
            "shoulder_weight" : 0.2,
            "elbow_weight" : 0.2,
            "wrist_weight" : 0.2

        }
            
    )

    # =================================================================
    # Run loop
    # =================================================================
    print(f"Teleop started (input_mode={args.input_mode}). Ctrl+C to exit...")

    _last_log_time = 0.0
    app = simulation_app
    _zero_gripper = [0.0, 0.0, 0.0]

    udp_sender = UdpSocketSender(host="127.0.0.1", port=5005)

    dof_names = list(articulation.dof_names)
    _left_indices  = [dof_names.index(n) for n in AMBER_BIMANUAL_LEFT_ARM_JOINT_NAMES]
    _right_indices = [dof_names.index(n) for n in AMBER_BIMANUAL_RIGHT_ARM_JOINT_NAMES]

    while app.is_running():
        try:
            controller._update()
            world.step(render=True)

            joint_positions = articulation.get_joint_positions()
            udp_sender.send(
                left_arm=joint_positions[_left_indices].tolist(),
                left_gripper=_zero_gripper,
                right_arm=joint_positions[_right_indices].tolist(),
                right_gripper=_zero_gripper,
            )

            now = time.time()
            if now - _last_log_time > 0.5:
                _last_log_time = now
                controller._log_debug()

        except KeyboardInterrupt:
            print("\nTeleoperation stopped.")
            break

    udp_sender.close()
    controller.teleop_input.shutdown()
    app.close()
if __name__ == "__main__":
    main()
