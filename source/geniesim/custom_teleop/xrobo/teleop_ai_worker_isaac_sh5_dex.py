# =============================================================================
# FFW SH5 Follower Teleop in Isaac Sim — Dex-Retargeting Hand Control
# Based on teleop_ai_worker_isaac_sh5.py with hand_control_ffw_sh5_v2.py method.
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
parser.add_argument("--rendering-dt", type=float, default=1.0 / 90.0)
parser.add_argument("--input-mode", type=str, default="hand_tracking", choices=["controller", "hand_tracking", "keyboard"], help="Input mode: 'controller' for VR controller, 'hand_tracking' for VR hand wrist, 'keyboard' for keyboard-toggled hand tracking")
parser.add_argument("--pico-ip", type=str, default="192.168.50.217", help="PICO VR IP address for video streaming")
parser.add_argument("--pico-stream-port", type=int, default=12345, help="PICO VR streaming port (0=disable)")
parser.add_argument("--stream-eye-res", type=str, default="1280x720", help="Per-eye resolution WxH (SBS output will be 2W x H)")
parser.add_argument("--stream-fps", type=int, default=90, help="Video stream FPS")
parser.add_argument("--stream-bitrate", type=int, default=8000000, help="Video stream bitrate (bps)")
parser.add_argument("--stream-skip", type=int, default=1, help="Encode every Nth render frame (1=every frame)")
parser.add_argument("--ipd", type=float, default=0.063, help="Inter-pupillary distance in meters (ZED Mini=0.063)")
parser.add_argument("--drama-ip", type=str, default="127.0.0.1", help="DRAMA server IP (None=disable socket sender)")
parser.add_argument("--drama-port", type=int, default=1111, help="DRAMA server WebSocket port")
# Dex-retargeting hand control parameters
parser.add_argument("--smooth-alpha", type=float, default=0.5, help="Ctrl smoother EMA alpha")
parser.add_argument("--smooth-max-speed", type=float, default=3.0, help="Ctrl smoother max joint speed (rad/s)")
parser.add_argument("--scale-open", type=float, default=1.0, help="Adaptive scaling factor when hand is open")
parser.add_argument("--scale-close", type=float, default=0.7, help="Adaptive scaling factor when hand is closed")
parser.add_argument("--open-length", type=float, default=0.20, help="Reference wrist-to-middle-tip length for open hand (m)")
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
GENIESIM_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../.."))  # source/geniesim/
SOURCE_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../../../.."))  # source/

sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, UTILS_DIR)
sys.path.insert(0, CUSTOM_UTILS_DIR)
sys.path.insert(0, GENIESIM_DIR)  # for custom_utils.video_sender etc.
sys.path.insert(0, SOURCE_DIR)  # for geniesim.plugins.logger etc.

from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.utils.geometry import R_HEADSET_TO_WORLD
from teleop.teleop_controller import DualArmIsaacTeleopController

from name_utils import (
    FFW_SH5_LEFT_ARM_JOINT_NAMES,
    FFW_SH5_RIGHT_ARM_JOINT_NAMES,
    FFW_SH5_DUAL_ARM_JOINT_NAMES,
    FFW_SH5_HEAD_JOINT_NAMES,
    FFW_SH5_WAIST_JOINT_NAMES,
    FFW_SH5_HAND_JOINT_NAMES,
    FFW_SH5_LEFT_HAND_JOINT_NAMES,
    FFW_SH5_RIGHT_HAND_JOINT_NAMES,)

# Hand gripper physics controller 
import importlib.util
_hg_spec = importlib.util.spec_from_file_location(
    "hand_gripper_hx5_d20",
    os.path.join(SCRIPT_DIR, "../../app/controllers/hand_gripper_hx5_d20.py"),)
_hg_mod = importlib.util.module_from_spec(_hg_spec)
_hg_spec.loader.exec_module(_hg_mod)
HandGripper = _hg_mod.HandGripper

# Dex-retargeting hand control imports are deferred to main()
# to avoid top-level ModuleNotFoundError in Isaac Sim Docker environment.

# Predefined open/grip poses (still used for HandGripper physics init)
from ai_worker_hx5_d20.synergy_preDef_pose import (
    RIGHT_HAND_OPEN, RIGHT_HAND_GRIP,
    LEFT_HAND_OPEN, LEFT_HAND_GRIP,)

BENCHMARK_CONFIG_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../../benchmark/config"))
sys.path.insert(0, BENCHMARK_CONFIG_DIR)
from robot_init_states import FFW_SH5_DEFAULT_STATES

# Fixed transform from hx5_base -> Control Offset
_T_HX5_BASE_TO_CONTROL_OFFSET = np.eye(4)
_T_HX5_BASE_TO_CONTROL_OFFSET[:3, 3] = [0, 0, -0.02]

from custom_utils.video_sender import VRStreamer
from custom_utils.video_sender.isaac_stereo_capture import IsaacStereoCapture
from custom_utils.socket_sender.socket_sender import DramaSocketSender

from meshcat import transformations as tf


# =============================================================================
# ===== Entry point =====
# =============================================================================
def main():
    # ===== Deferred imports: pinocchio / dex-retargeting =====
    import pinocchio as pin
    from dex_retargeting.constants import HandType, RetargetingType

    from custom_utils.hand_config_dex.dex_hand_utils import (
        DexHandTracker,
        HX5_RIGHT_RETARGETING_CONFIG,
        HX5_LEFT_RETARGETING_CONFIG,
        HX5_RIGHT_DEXPILOT_CONFIG,
        HX5_LEFT_DEXPILOT_CONFIG,
        HX5_RIGHT_ACTUATOR_RANGE,
        HX5_LEFT_ACTUATOR_RANGE,
        CtrlSmoother,
        adaptive_scale_keypoints,
        pico_hand_state_to_mediapipe,
        pin_q_to_isaac_hand_targets,
        load_hand_sizes,
        get_right_urdf_path,
        get_left_urdf_path,
        get_calibration_yaml_path,
    )
    from custom_utils.hand_config_dex.hand_tracking_filter import HandTrackingFilter

    robot_usd = args.robot_usd
    if robot_usd is None:
        robot_usd = os.path.join(SCRIPT_DIR, "../../assets/robot/ai_worker_sh5_geniesim/ffw_sh5.usda")

    robot_prim_path = "/ffw_sh5_follower"

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

    # Background scene from table_task_ffw_sg2.json
    ASSETS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../../assets"))
    scene_usd_path = os.path.join(ASSETS_DIR, "background/room/room_1/background.usda")
    add_reference_to_stage(usd_path=scene_usd_path, prim_path="/World")

    # Load sub_task scene (table + colored cubes)
    BENCHMARK_CFG_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../../benchmark/config"))
    sub_task_usd = os.path.join(BENCHMARK_CFG_DIR, "llm_task", "pick_specific_object", "0", "scene.usda")
    add_reference_to_stage(usd_path=sub_task_usd, prim_path="/Workspace")
    print(f"Loaded sub_task scene: {sub_task_usd}")

    # ===== Phase 2: Load robot =====
    add_reference_to_stage(usd_path=robot_usd, prim_path=robot_prim_path)

    from isaacsim.core.prims import SingleXFormPrim
    robot_xform = SingleXFormPrim(
        prim_path=robot_prim_path,
        position=np.array([-0.5, -0.02646532841026783, -0.009999999776482582]), 
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
    )

    # ===== Configure joint drive stiffness/damping ======
    from pxr import Usd
    robot_prim = stage.GetPrimAtPath(robot_prim_path)

    art_root_path = f"{robot_prim_path}/base_link"
    art_root_prim = stage.GetPrimAtPath(art_root_path)
    physx_art_api = PhysxSchema.PhysxArticulationAPI.Apply(art_root_prim)
    physx_art_api.CreateSolverPositionIterationCountAttr(32)
    physx_art_api.CreateSolverVelocityIterationCountAttr(8)

    # Fix base to world at the robot's spawn position
    robot_pos = robot_xform.get_world_pose()[0]
    fixed_joint = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(f"{robot_prim_path}/FixedJoint"))
    fixed_joint.CreateBody1Rel().SetTargets([Sdf.Path(f"{robot_prim_path}/base_link/base_link")])
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
            
            if "gripper" in joint_name or "finger" in joint_name:
                drive_api.GetStiffnessAttr().Set(1e3)
                drive_api.GetDampingAttr().Set(1e2)
                drive_api.GetMaxForceAttr().Set(60.0)

            elif any(k in joint_name for k in ["lift", "head_joint"]):
                drive_api.GetStiffnessAttr().Set(1e6)
                drive_api.GetDampingAttr().Set(1e5)
                drive_api.GetMaxForceAttr().Set(1e10)

            elif "wheel" in joint_name:
                drive_api.GetStiffnessAttr().Set(1e8)
                drive_api.GetDampingAttr().Set(1e6)
                drive_api.GetMaxForceAttr().Set(0)

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
            
    from pxr import Usd, UsdGeom

    # ===== Contact/Rest offset + CCD for hand collision meshes =====
    for prim in Usd.PrimRange(robot_prim):
        prim_name = prim.GetName()
        if ("hx5" in prim_name or "finger" in prim_name):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                physx_collision.CreateContactOffsetAttr(0.002)
                physx_collision.CreateRestOffsetAttr(0.001)

    # ===== Contact/Rest offset + CCD for workspace objects (ball etc.) =====
    workspace_prim = stage.GetPrimAtPath("/Workspace")
    if workspace_prim.IsValid():
        for prim in Usd.PrimRange(workspace_prim):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                physx_collision.CreateContactOffsetAttr(0.002)
                physx_collision.CreateRestOffsetAttr(0.001)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateEnableCCDAttr(True)

    # ===== Phase 3: Play + init articulation =====
    time.sleep(1)
    world.play()
    time.sleep(1)

    articulation = SingleArticulation(prim_path=art_root_path, name="ffw_sh5_follower",)
    world.scene.add(articulation)
    articulation.initialize()

    dof_names = list(articulation.dof_names)
    # ===== FFW_SH5_DEFAULT_STATES from robot_init_states.py =====
    INIT_ARM = FFW_SH5_DEFAULT_STATES["init_arm"]
    INIT_HAND = FFW_SH5_DEFAULT_STATES["init_hand"]
    INIT_HEAD = FFW_SH5_DEFAULT_STATES["head_state"]
    INIT_WAIST = FFW_SH5_DEFAULT_STATES["body_state"]

    ARM_NAMES = FFW_SH5_DUAL_ARM_JOINT_NAMES
    HAND_NAMES = FFW_SH5_HAND_JOINT_NAMES
    HEAD_NAMES = FFW_SH5_HEAD_JOINT_NAMES
    WAIST_NAMES = FFW_SH5_WAIST_JOINT_NAMES

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
    set_joints(HAND_NAMES, INIT_HAND)
    set_joints(HEAD_NAMES, INIT_HEAD)
    set_joints(WAIST_NAMES, INIT_WAIST)

    for _ in range(120):
        world.step(render=True)
    print("Initial pose set.")


    # Hold waist + head joints at init positions every frame to prevent vibration
    hold_positions = {}
    for name, val in zip(WAIST_NAMES, INIT_WAIST):
        hold_positions[name] = val
    for name, val in zip(HEAD_NAMES, INIT_HEAD):
        hold_positions[name] = val

    # ===== Hand gripper physics setup via HandGripper =====
    left_hand_gripper = HandGripper(
        end_effector_prim_path=f"{robot_prim_path}/base_link/hx5_l_base",
        joint_prim_names=FFW_SH5_LEFT_HAND_JOINT_NAMES,
        joint_opened_positions=np.array(LEFT_HAND_OPEN),
        joint_closed_positions=np.array(LEFT_HAND_GRIP),
    )
    right_hand_gripper = HandGripper(
        end_effector_prim_path=f"{robot_prim_path}/base_link/hx5_r_base",
        joint_prim_names=FFW_SH5_RIGHT_HAND_JOINT_NAMES,
        joint_opened_positions=np.array(RIGHT_HAND_OPEN),
        joint_closed_positions=np.array(RIGHT_HAND_GRIP),
    )
    left_hand_gripper.initialize(
        articulation_apply_action_func=articulation.apply_action,
        get_joint_positions_func=articulation.get_joint_positions,
        set_joint_positions_func=articulation.set_joint_positions,
        dof_names=dof_names,
    )
    right_hand_gripper.initialize(
        articulation_apply_action_func=articulation.apply_action,
        get_joint_positions_func=articulation.get_joint_positions,
        set_joint_positions_func=articulation.set_joint_positions,
        dof_names=dof_names,
    )

    # Bind high-friction physics material to hand collision meshes
    hand_material_path = left_hand_gripper.physics_material.prim_path
    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_prim_path)):
        prim_name = prim.GetName()
        if ("hx5" in prim_name or "finger" in prim_name) and prim.IsA(UsdGeom.Mesh):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
            binding_api.Bind(
                UsdShade.Material(stage.GetPrimAtPath(hand_material_path)),
                UsdShade.Tokens.weakerThanDescendants,
                "physics",
            )
        prim_path_str = str(prim.GetPath())
        if ("hx5" in prim_path_str or "finger" in prim_path_str):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
                binding_api.Bind(
                    UsdShade.Material(stage.GetPrimAtPath(hand_material_path)),
                    UsdShade.Tokens.weakerThanDescendants,
                    "physics",
                )
    print("Hand gripper physics materials applied.")

    # XR client (reuse already-initialized SDK)
    xr_client = XrClient.__new__(XrClient)
    print("XrClient attached (SDK already initialized).")

    camera_config = {
        "head_camera": "{robot}/base_link/head_link2/zed/Head_Camera",
        "extra_cameras": [
            ("Left Camera",  "{robot}/base_link/arm_l_link7/camera_l_bottom_screw_frame/camera_l_link/Left_Camera"),
            ("Right Camera", "{robot}/base_link/arm_r_link7/camera_r_bottom_screw_frame/camera_r_link/Right_Camera"),
        ],
    }

    # ===== Phase 4: Create teleop controller and run =====
    if args.input_mode == "keyboard":
        manipulator_config = {
            "left_hand": {
                "link_name": "hx5_l_base",
                "hand_side": "left",
            },
            "right_hand": {
                "link_name": "hx5_r_base",
                "hand_side": "right",
            },
        }
    else:
        manipulator_config = {
            "left_hand": {
                "link_name": "hx5_l_base",
                "hand_side": "left",
            },
            "right_hand": {
                "link_name": "hx5_r_base",
                "hand_side": "right",
            },
        }

    IK_SDK_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../../utils/IK-SDK"))
    ik_urdf_path = os.path.join(IK_SDK_DIR, "ffw_sh5_follower.urdf")
    ik_config_path = os.path.join(IK_SDK_DIR, "ffw_sh5_follower_solver.yaml")

    controller = DualArmIsaacTeleopController.create(
        world=world,
        articulation=articulation,
        robot_prim_path=robot_prim_path,
        manipulator_config=manipulator_config,
        xr_client=xr_client,
        left_arm_joints=FFW_SH5_LEFT_ARM_JOINT_NAMES,
        right_arm_joints=FFW_SH5_RIGHT_ARM_JOINT_NAMES,
        T_link7_to_gripper_base=_T_HX5_BASE_TO_CONTROL_OFFSET,
        ik_urdf_path=ik_urdf_path,
        ik_config_path=ik_config_path,
        simulation_app=simulation_app,
        xrt_module=xrt,
        R_headset_world=R_HEADSET_TO_WORLD,
        scale_factor=args.scale_factor,
        hold_joint_positions=hold_positions,
        camera_config=camera_config,
        input_mode=args.input_mode,
    )

    # =================================================================
    # Dex-retargeting hand control setup
    # =================================================================
    right_urdf_path = get_right_urdf_path()
    left_urdf_path = get_left_urdf_path()

    # Build Pinocchio models for pin_q → Isaac Sim joint mapping
    right_pin_model = pin.buildModelFromUrdf(right_urdf_path)
    left_pin_model = pin.buildModelFromUrdf(left_urdf_path)

    # DexHandTrackers
    right_dex_tracker = DexHandTracker(
        urdf_path=right_urdf_path,
        hand_type=HandType.right,
        retargeting_type=RetargetingType.dexpilot,
        config_dict=HX5_RIGHT_DEXPILOT_CONFIG,
    )
    left_dex_tracker = DexHandTracker(
        urdf_path=left_urdf_path,
        hand_type=HandType.left,
        retargeting_type=RetargetingType.dexpilot,
        config_dict=HX5_LEFT_DEXPILOT_CONFIG,
    )
    print("DexHandTrackers initialized (L & R).")

    # Hand tracking filters (One Euro + velocity clamp + hand size normalization)
    calib_yaml = get_calibration_yaml_path()
    human_hand_size, robot_hand_size = load_hand_sizes(calib_yaml)
    right_hand_filter = HandTrackingFilter(
        max_velocity=0.1, user_hand_size=human_hand_size, ref_hand_size=robot_hand_size)
    left_hand_filter = HandTrackingFilter(
        max_velocity=0.1, user_hand_size=human_hand_size, ref_hand_size=robot_hand_size)
    print(f"Hand calibration loaded: human={human_hand_size}, robot={robot_hand_size}")

    # Ctrl smoothers
    left_smoother = CtrlSmoother(alpha=args.smooth_alpha, max_speed=args.smooth_max_speed)
    right_smoother = CtrlSmoother(alpha=args.smooth_alpha, max_speed=args.smooth_max_speed)

    # Pinch joint2 override: rate-limited drive of thumb CMC joint during pinch
    _PINCH_J2_NAME  = {"right": "finger_r_joint2", "left": "finger_l_joint2"}
    _PINCH_J2_VAL   = {"right": -1.55, "left": 1.55}  # target angle (rad); left mirrors right
    _PINCH_THRESH   = 0.09   # metres — same as project_dist in yml
    _J2_SPEED       = 4.0   # rad/s — max transition speed
    _j2_idx         = {}
    _j2_commanded   = {}
    _j2_prev_time   = {}
    for _side, _tracker in [("right", right_dex_tracker), ("left", left_dex_tracker)]:
        _jname = _PINCH_J2_NAME[_side]
        _jnames = list(_tracker.retargeting.joint_names)
        _j2_idx[_side] = _jnames.index(_jname) if _jname in _jnames else None
        _j2_commanded[_side] = 0.0
        _j2_prev_time[_side] = time.time()

    # =================================================================
    # Stereo video streamer setup (Isaac Sim stereo camera -> PICO VR)
    # =================================================================
    streamer = None
    stereo_capture = None
    if args.pico_stream_port > 0:
        head_cam_path = camera_config["head_camera"].replace("{robot}", robot_prim_path)
        cam_prim = stage.GetPrimAtPath(head_cam_path)
        if cam_prim and cam_prim.IsValid():
            eye_w, eye_h = (int(x) for x in args.stream_eye_res.split("x"))
            # Create streamer (TCP connection) FIRST — only create render products if connected
            streamer = VRStreamer(
                target_ip=args.pico_ip,
                target_port=args.pico_stream_port,
                width=eye_w * 2, height=eye_h,
                fps=args.stream_fps,
                bitrate=args.stream_bitrate,
                send_every_n=args.stream_skip,
            )
            if streamer.active:
                stereo_capture = IsaacStereoCapture( stage=stage, parent_camera_path=head_cam_path, eye_resolution=(eye_w, eye_h), ipd=args.ipd,)
            else:
                print("[VideoStreamer] WARNING: TCP connection failed, stereo capture not created.")
                streamer = None
        else:
            print(f"[VideoStreamer] WARNING: Camera not found at {head_cam_path}, streaming disabled.")

    # =================================================================
    # DRAMA WebSocket sender setup
    # =================================================================
    drama_sender = None
    if args.drama_ip:
        drama_sender = DramaSocketSender(host=args.drama_ip, port=args.drama_port)
        try:
            drama_sender.connect()
        except Exception as e:
            print(f"[DRAMA] Connection failed: {e}")
            drama_sender = None

    def _build_drama_dict():
        """Build DRAMA protocol dict from current robot state."""
        # Arm joints (7 each)
        left_arm_q, right_arm_q = controller._read_arm_joints()

        # Hand joints (20 each) from articulation
        all_pos = articulation.get_joint_positions()
        left_hand_vals = [float(all_pos[dof_names.index(n)]) for n in FFW_SH5_LEFT_HAND_JOINT_NAMES]
        right_hand_vals = [float(all_pos[dof_names.index(n)]) for n in FFW_SH5_RIGHT_HAND_JOINT_NAMES]

        # Controller poses from XR input
        left_ctrl = [0.0] * 7
        right_ctrl = [0.0] * 7
        try:
            lxyz, lq = controller.teleop_input._get_controller_pose(
                controller.manipulator_config["left_hand"].get("pose_source", "left_controller"))
            left_ctrl = lxyz.tolist() + lq.tolist()  # [x,y,z,qw,qx,qy,qz]
        except Exception:
            pass
        try:
            rxyz, rq = controller.teleop_input._get_controller_pose(
                controller.manipulator_config["right_hand"].get("pose_source", "right_controller"))
            right_ctrl = rxyz.tolist() + rq.tolist()
        except Exception:
            pass

        # EEF poses from FK
        left_eef = [0.0] * 7
        right_eef = [0.0] * 7
        try:
            left_T = controller._get_gripper_base_pose(
                controller.manipulator_config["left_hand"]["link_name"])
            left_eef = left_T[:3, 3].tolist() + tf.quaternion_from_matrix(left_T).tolist()
        except Exception:
            pass
        try:
            right_T = controller._get_gripper_base_pose(
                controller.manipulator_config["right_hand"]["link_name"])
            right_eef = right_T[:3, 3].tolist() + tf.quaternion_from_matrix(right_T).tolist()
        except Exception:
            pass

        # Activation state
        left_active = controller.teleop_input.active.get("left_hand", False)
        right_active = controller.teleop_input.active.get("right_hand", False)

        return {
            "left": {
                "arm": left_arm_q.tolist(),
                "hand": left_hand_vals,
                "controller": left_ctrl,
                "eef": left_eef,
            },
            "right": {
                "arm": right_arm_q.tolist(),
                "hand": right_hand_vals,
                "controller": right_ctrl,
                "eef": right_eef,
            },
            "activated": {
                "left": left_active,
                "right": right_active,
            },
        }

    # =================================================================
    # Custom run loop with dex-retargeting hand control
    # =================================================================
    print(f"Teleop started (input_mode={args.input_mode}) with dex-retargeting hand control. Ctrl+C to exit...")

    _last_log_time = 0.0
    _hand_was_active = {"left": True, "right": True}  # start as True so first deactivate triggers freeze
    app = simulation_app

    while app.is_running():
        try:
            # --- Arm teleop update (IK pipeline) ---
            controller._update()

            # --- Dex-retargeting hand control ---
            # In keyboard mode, gate hand control by keyboard activation
            kb = controller.teleop_input._keyboard  # None when not keyboard mode

            for hand_side, dex_tracker, pin_model, hand_filter, smoother, act_range, hand_gripper, hand_joint_names in [
                ("left", left_dex_tracker, left_pin_model, left_hand_filter, left_smoother,
                 HX5_LEFT_ACTUATOR_RANGE, left_hand_gripper, FFW_SH5_LEFT_HAND_JOINT_NAMES),
                ("right", right_dex_tracker, right_pin_model, right_hand_filter, right_smoother,
                 HX5_RIGHT_ACTUATOR_RANGE, right_hand_gripper, FFW_SH5_RIGHT_HAND_JOINT_NAMES),
            ]:
                # Keyboard activation gate: freeze on deactivate, hold frozen positions
                if kb is not None:
                    if hand_side == "left":
                        is_hand_active = kb.is_left_control_key_pressed()
                    else:
                        is_hand_active = kb.is_right_control_key_pressed()

                    if not is_hand_active:
                        if _hand_was_active[hand_side]:
                            hand_gripper.freeze()
                            _hand_was_active[hand_side] = False
                        hand_gripper.hold_current_positions()
                        continue
                    else:
                        _hand_was_active[hand_side] = True

                hand_state = xr_client.get_hand_tracking_state(hand_side)
                if hand_state is None:
                    continue

                hand_state = np.array(hand_state)

                # ===== Step 1: VR → MediaPipe keypoints (SAPIEN-equivalent: no filter, no adaptive scale) =====
                kp = pico_hand_state_to_mediapipe(hand_state)

                # ===== Step 2: DexRetargeting IK → pin_q =====
                pin_q = dex_tracker.retarget(kp)
                if pin_q is None:
                    continue

                # ===== Step 2b: Rate-limited joint2 pinch override =====
                if _j2_idx[hand_side] is not None:
                    _now = time.time()
                    _dt = _now - _j2_prev_time[hand_side]
                    _j2_prev_time[hand_side] = _now
                    _thumb_tip = kp[4]
                    _pinching = any(
                        np.linalg.norm(_thumb_tip - kp[tip_idx]) < _PINCH_THRESH
                        for tip_idx in [8, 12, 16, 20]
                    )
                    _j2_target = _PINCH_J2_VAL[hand_side] if _pinching else pin_q[_j2_idx[hand_side]]
                    _max_delta = _J2_SPEED * _dt
                    _j2_commanded[hand_side] += np.clip(
                        _j2_target - _j2_commanded[hand_side], -_max_delta, _max_delta
                    )
                    pin_q[_j2_idx[hand_side]] = _j2_commanded[hand_side]
                    dex_tracker.retargeting.set_qpos(pin_q)

                # ===== Step 3: pin_q → Isaac Sim joint targets =====
                target = pin_q_to_isaac_hand_targets(pin_model, pin_q, hand_joint_names)

                # ===== Step 4: Clamp to actuator range =====
                act_names = list(act_range.keys())
                for i in range(len(target)):
                    if i < len(act_names):
                        lo, hi = act_range[act_names[i]]
                        target[i] = np.clip(target[i], lo, hi)

                hand_gripper.set_target_positions(target)

            world.step(render=True)

            # --- Stereo Video Stream: capture SBS frame and send ---
            if streamer and stereo_capture:
                try:
                    sbs_frame = stereo_capture.get_sbs_frame()
                    if sbs_frame is not None:
                        streamer.send_frame(sbs_frame)
                except Exception as e:
                    print(f"[VideoStreamer] Capture error: {e}")

            # --- DRAMA WebSocket: send robot state ---
            if drama_sender:
                if not drama_sender.is_connected:
                    try:
                        print("[DRAMA] Connection lost, reconnecting...")
                        drama_sender.reconnect()
                    except Exception as e:
                        print(f"[DRAMA] Reconnect failed: {e}")
                else:
                    try:
                        drama_sender.send_dict(_build_drama_dict())
                    except Exception as e:
                        print(f"[DRAMA] Send error: {e}")

            now = time.time()
            if now - _last_log_time > 0.5:
                _last_log_time = now
                controller._log_debug()

        except KeyboardInterrupt:
            print("\nTeleoperation stopped.")
            break

    controller.teleop_input.shutdown()
    if streamer:
        streamer.shutdown()
    if drama_sender:
        drama_sender.disconnect()
    app.close()
if __name__ == "__main__":
    main()
