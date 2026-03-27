# =============================================================================
# FFW SG2 Follower Teleop in Isaac Sim with Genie Sim IK Solver.
# Isaac Sim's SimulationApp / World / SingleArticulation APIs.
# =============================================================================

import os
import sys
import time

import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--robot-usd", type=str, default=None)
parser.add_argument("--scale-factor", type=float, default=1)
parser.add_argument("--physics-dt", type=float, default=1.0 / 120.0)
parser.add_argument("--rendering-dt", type=float, default=1.0 / 60.0)
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
from pxr import UsdPhysics, PhysxSchema, Sdf, Gf, UsdShade
import omni.kit.commands
from scipy.spatial.transform import Rotation as R

# =============================================================================
# ===== Paths & imports =====
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IK_SDK_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../../utils/IK-SDK"))
UTILS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../../utils"))
CUSTOM_UTILS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../../custom_utils"))

sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, UTILS_DIR)
sys.path.insert(0, CUSTOM_UTILS_DIR)

from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.utils.geometry import R_HEADSET_TO_WORLD
from teleop.teleop_controller import DualArmIsaacTeleopController

from name_utils import (
    FFW_SG2_LEFT_ARM_JOINT_NAMES,
    FFW_SG2_RIGHT_ARM_JOINT_NAMES,
    FFW_SG2_DUAL_ARM_JOINT_NAMES,
    FFW_SG2_HEAD_JOINT_NAMES,
    FFW_SG2_WAIST_JOINT_NAMES,
    FFW_SG2_GRIPPER_JOINTS_NAMES,
)

# Fixed transform from arm_*_link7 -> gripper_*_rh_p12_rn_base (from URDF)
_T_LINK7_TO_GRIPPER_BASE = np.eye(4)
_T_LINK7_TO_GRIPPER_BASE[:3, :3] = R.from_euler("xyz", [0, np.pi, np.pi]).as_matrix()
_T_LINK7_TO_GRIPPER_BASE[:3, 3] = [0, 0, -0.078]


# =============================================================================
# ===== Entry point =====
# =============================================================================
def main():
    robot_usd = args.robot_usd
    if robot_usd is None:
        robot_usd = os.path.join(SCRIPT_DIR, "ai_worker/ffw_sg2.usd")

    robot_prim_path = "/ffw_sg2_follower"

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

    # Background scene from table_task_ffw_sg2.json
    ASSETS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../../assets"))
    scene_usd_path = os.path.join(ASSETS_DIR, "background/room/room_1/background.usda")
    add_reference_to_stage(usd_path=scene_usd_path, prim_path="/World")

    # Load sub_task scene (table + colored cubes)
    BENCHMARK_CFG_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../../benchmark/config"))
    sub_task_usd = os.path.join(BENCHMARK_CFG_DIR, "llm_task", "pick_block_color", "0", "scene.usda")
    add_reference_to_stage(usd_path=sub_task_usd, prim_path="/Workspace")
    print(f"Loaded sub_task scene: {sub_task_usd}")

    # ===== Phase 2: Load robot =====
    add_reference_to_stage(usd_path=robot_usd, prim_path=robot_prim_path)

    from isaacsim.core.prims import SingleXFormPrim
    robot_xform = SingleXFormPrim(
        prim_path=robot_prim_path,
        position=np.array([-0.48, -0.02646532841026783, -0.009999999776482582]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
    )

    # ===== Configure joint drive stiffness/damping ======
    from pxr import Usd
    robot_prim = stage.GetPrimAtPath(robot_prim_path)

    physx_art_api = PhysxSchema.PhysxArticulationAPI.Apply(robot_prim)
    physx_art_api.CreateSolverPositionIterationCountAttr(32)
    physx_art_api.CreateSolverVelocityIterationCountAttr(8)
    for prim in Usd.PrimRange(robot_prim):

        for drive_type in ["angular", "linear"]:
            drive_api = UsdPhysics.DriveAPI.Get(prim, drive_type)
            if not drive_api:
                continue
            joint_name = prim.GetName()
            if "gripper" in joint_name:
                drive_api.GetStiffnessAttr().Set(1e3)
                drive_api.GetDampingAttr().Set(5e2)

            elif any(k in joint_name for k in ["lift", "wheel", "head_joint"]):
                drive_api.GetStiffnessAttr().Set(1e6)
                drive_api.GetDampingAttr().Set(1e5)

            elif any(k in joint_name for k in ["joint1", "joint2", "joint3"]):
                drive_api.GetStiffnessAttr().Set(5e5)
                drive_api.GetDampingAttr().Set(1e3)
                drive_api.GetMaxForceAttr().Set(1e10)
                drive_api.GetTypeAttr().Set("acceleration")

            elif any(k in joint_name for k in ["joint4"]):
                drive_api.GetStiffnessAttr().Set(5e5)
                drive_api.GetDampingAttr().Set(1e3)
                drive_api.GetMaxForceAttr().Set(1e10)
                drive_api.GetTypeAttr().Set("acceleration")

            elif any(k in joint_name for k in ["joint5", "joint6", "joint7"]):
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

    # ===== Apply high-friction physics material to gripper collision meshes =====
    from pxr import Usd, UsdGeom

    from pxr import UsdPhysics as UsdPhysicsAPI
    grip_mat_path = Sdf.Path("/World/gripper_physics_material")
    UsdShade.Material.Define(stage, grip_mat_path)
    grip_mat_prim = stage.GetPrimAtPath(grip_mat_path)
    phys_mat_api = UsdPhysicsAPI.MaterialAPI.Apply(grip_mat_prim)
    phys_mat_api.CreateStaticFrictionAttr(1.0)
    phys_mat_api.CreateDynamicFrictionAttr(1.0)
    phys_mat_api.CreateRestitutionAttr(0.1)
    physx_mat_api = PhysxSchema.PhysxMaterialAPI.Apply(grip_mat_prim)
    physx_mat_api.CreateFrictionCombineModeAttr("max")
    grip_material = UsdShade.Material(grip_mat_prim)

    gripper_link_keywords = ["gripper_l_", "gripper_r_"]
    for prim in Usd.PrimRange(robot_prim):
        prim_path_str = str(prim.GetPath())
        if any(kw in prim_path_str for kw in gripper_link_keywords):
            if prim.HasAPI(UsdPhysicsAPI.CollisionAPI) or prim.IsA(UsdGeom.Mesh):
                binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
                binding_api.Bind(grip_material, UsdShade.Tokens.weakerThanDescendants, "physics")
                print(f"  Friction material bound to: {prim_path_str}")

    # ===== Phase 3: Play + init articulation =====
    time.sleep(1)
    world.play()
    time.sleep(1)

    articulation = SingleArticulation(prim_path=robot_prim_path, name="ffw_sg2_follower",)
    world.scene.add(articulation)
    articulation.initialize()

    dof_names = list(articulation.dof_names)
    # ===== FFW_SG2_DEFAULT_STATES from robot_init_states.py =====
    INIT_ARM = [
        # left arm (arm_l_joint1..7)
        -1.16, 1.27, -1.38, -2.39, 1.32, 0.612, 0,
        # right arm (arm_r_joint1..7)
        -1.16, -1.27, 1.38, -2.39, -1.32, 0.612, 0,
    ]
    INIT_HEAD = [0.7, 0.0]        # head_joint1, head_joint2
    INIT_WAIST = [               # lift_joint, wheel joints...
        -0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]
    INIT_GRIPPER = [0.0, 0.0]    # gripper_l_joint1, gripper_r_joint1

    ARM_NAMES = FFW_SG2_DUAL_ARM_JOINT_NAMES
    HEAD_NAMES = FFW_SG2_HEAD_JOINT_NAMES
    WAIST_NAMES = FFW_SG2_WAIST_JOINT_NAMES
    GRIPPER_NAMES = FFW_SG2_GRIPPER_JOINTS_NAMES

    def set_joints(names, values):
        indices = []
        positions = []
        for name, val in zip(names, values):
            if name in dof_names:
                indices.append(dof_names.index(name))
                positions.append(float(val))
        if indices:
            action = ArticulationAction(
                joint_positions=np.array(positions),
                joint_indices=indices,
            )
            articulation.apply_action(action)

    set_joints(ARM_NAMES, INIT_ARM)
    set_joints(HEAD_NAMES, INIT_HEAD)
    set_joints(WAIST_NAMES, INIT_WAIST)
    set_joints(GRIPPER_NAMES, INIT_GRIPPER)

    for _ in range(120):
        world.step(render=True)
    print("Initial pose set.")

    # ===== Phase 4: Create teleop controller and run =====
    manipulator_config = {
        "right_hand": {
            "link_name": "arm_r_link7",
            "pose_source": "right_controller",
            "control_trigger": "right_grip",
            "gripper_config": {
                "type": "parallel",
                "gripper_trigger": "right_trigger",
                "joint_names": ["gripper_r_joint1"],
                "open_pos": [0.1],
                "close_pos": [1.0],
            },
        },
        "left_hand": {
            "link_name": "arm_l_link7",
            "pose_source": "left_controller",
            "control_trigger": "left_grip",
            "gripper_config": {
                "type": "parallel",
                "gripper_trigger": "left_trigger",
                "joint_names": ["gripper_l_joint1"],
                "open_pos": [0.1],
                "close_pos": [1.0],
            },
        },
    }

    # Hold waist + head joints at init positions every frame to prevent vibration
    hold_positions = {}
    for name, val in zip(WAIST_NAMES, INIT_WAIST):
        hold_positions[name] = val
    for name, val in zip(HEAD_NAMES, INIT_HEAD):
        hold_positions[name] = val

    # XR client (reuse already-initialized SDK)
    xr_client = XrClient.__new__(XrClient)
    print("XrClient attached (SDK already initialized).")

    camera_config = {
        "head_camera": "{robot}/head_link2/zed/Head_Camera",
        "extra_cameras": [
            ("Left Camera",  "{robot}/arm_l_link7/camera_l_bottom_screw_frame/camera_l_link/Left_Camera"),
            ("Right Camera", "{robot}/arm_r_link7/camera_r_bottom_screw_frame/camera_r_link/Right_Camera"),
        ],
    }

    controller = DualArmIsaacTeleopController.create(
        world=world,
        articulation=articulation,
        robot_prim_path=robot_prim_path,
        manipulator_config=manipulator_config,
        xr_client=xr_client,
        left_arm_joints=FFW_SG2_LEFT_ARM_JOINT_NAMES,
        right_arm_joints=FFW_SG2_RIGHT_ARM_JOINT_NAMES,
        T_link7_to_gripper_base=_T_LINK7_TO_GRIPPER_BASE,
        ik_urdf_path=os.path.join(IK_SDK_DIR, "ffw_sg2_follower.urdf"),
        ik_config_path=os.path.join(IK_SDK_DIR, "ffw_sg2_follower_solver.yaml"),
        simulation_app=simulation_app,
        xrt_module=xrt,
        R_headset_world=R_HEADSET_TO_WORLD,
        scale_factor=args.scale_factor,
        hold_joint_positions=hold_positions,
        camera_config=camera_config,
    )

    controller.run()


if __name__ == "__main__":
    main()
