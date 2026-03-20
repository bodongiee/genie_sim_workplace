"""
FFW SG2 Follower Teleop in Isaac Sim with Genie Sim IK Solver.

Isaac Sim's SimulationApp / World / SingleArticulation APIs.
"""

import contextlib
import os
import sys
import tempfile
import threading
import time
from typing import Any, Dict

import numpy as np
from scipy.spatial.transform import Rotation as R

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--robot-usd", type=str, default=None)
parser.add_argument("--scale-factor", type=float, default=1.5)
parser.add_argument("--debug-vr", action="store_true", default=False)
parser.add_argument("--physics-dt", type=float, default=1.0 / 120.0)
parser.add_argument("--rendering-dt", type=float, default=1.0 / 60.0)
parser.add_argument("--direct-pos", action="store_true", default=True,
                    help="True: direct position set (no gravity sag), False: PD control")
parser.add_argument("--no-direct-pos", dest="direct_pos", action="store_false",
                    help="Use PD control instead of direct position set")
parser.add_argument("--pico-ip", type=str, default="192.168.50.217",
                    help="PICO VR IP address for video streaming")
parser.add_argument("--pico-stream-port", type=int, default=12345,
                    help="PICO VR streaming port (0=disable)")
parser.add_argument("--stream-res", type=str, default="2560x2040",
                    help="Video stream resolution WxH")
parser.add_argument("--stream-fps", type=int, default=30,
                    help="Video stream FPS")
parser.add_argument("--stream-bitrate", type=int, default=4000000,
                    help="Video stream bitrate (bps)")
parser.add_argument("--stream-skip", type=int, default=2,
                    help="Encode every Nth render frame (higher=faster teleop, lower fps stream)")
args, _unknown = parser.parse_known_args()

# =============================================================================
# XRoboToolkit SDK — init BEFORE SimulationApp so the SDK's background
# network thread is established before Isaac Sim takes over signal/network.
# =============================================================================

import xrobotoolkit_sdk as xrt

print("Initializing XRoboToolkit SDK before SimulationApp...")
xrt.init()
print("XRoboToolkit SDK initialized. Waiting for device connection...")
# Give the SDK time to discover and connect to devices
time.sleep(2)
_ts = xrt.get_time_stamp_ns()
_lg = xrt.get_left_grip()
_rg = xrt.get_right_grip()
print(f"  SDK check: timestamp={_ts}, left_grip={_lg}, right_grip={_rg}")
if _ts == 0:
    print("  WARNING: No data received from XR device yet. Check device connection.")

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

import ik_solver
from meshcat import transformations as tf

# =============================================================================
# ===== XR client & geometry utils =====
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.utils.geometry import (
    R_HEADSET_TO_WORLD,
    apply_delta_pose,
    quat_diff_as_angle_axis,
)

IK_SDK_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../utils/IK-SDK"))

LEFT_ARM_JOINTS = [f"arm_l_joint{i}" for i in range(1, 8)]
RIGHT_ARM_JOINTS = [f"arm_r_joint{i}" for i in range(1, 8)]

# Fixed transform from arm_*_link7 -> gripper_*_rh_p12_rn_base (from URDF)
_T_LINK7_TO_GRIPPER_BASE = np.eye(4)
_T_LINK7_TO_GRIPPER_BASE[:3, :3] = R.from_euler("xyz", [0, np.pi, np.pi]).as_matrix()
_T_LINK7_TO_GRIPPER_BASE[:3, 3] = [0, 0, -0.078]

# =============================================================================
# ===== C++ stdout capture (reused from MuJoCo version) =====
# =============================================================================
@contextlib.contextmanager
def capture_c_stdout():
    fd = sys.stdout.fileno() if hasattr(sys.stdout, "fileno") else 1
    with tempfile.TemporaryFile(mode="w+b") as tfile:
        old_stdout = os.dup(fd)
        try:
            sys.stdout.flush()
            os.dup2(tfile.fileno(), fd)
            yield
        finally:
            os.dup2(old_stdout, fd)
            os.close(old_stdout)
            tfile.seek(0)
            output = tfile.read().decode("utf-8", errors="replace")
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
                print(f"[C++ IK Solver]:\n{output}")
# =============================================================================
# ===== Teleop Controller =====
# =============================================================================
class DualArmIsaacTeleopController:

    def __init__(self):
        pass

    @classmethod
    def create(
        cls,
        world: World,
        articulation: SingleArticulation,
        robot_prim_path: str,
        manipulator_config: Dict[str, Dict[str, Any]],
        R_headset_world: np.ndarray = R_HEADSET_TO_WORLD,
        scale_factor: float = 1.0,
        hold_joint_positions: Dict[str, float] = None,
        direct_pos: bool = True,
    ):
        self = cls()
        self.world = world
        self.articulation = articulation
        self.direct_pos = direct_pos
        self.robot_prim_path = robot_prim_path
        self.manipulator_config = manipulator_config
        self.R_headset_world = R_headset_world
        self.scale_factor = scale_factor
        self._hold_joint_positions = hold_joint_positions or {}
        self._stop_event = threading.Event()
        # Reuse already-initialized SDK (init was called before SimulationApp)
        self.xr_client = XrClient.__new__(XrClient)
        print("XrClient attached (SDK already initialized).")

        # Per-manipulator state
        self.ref_ee_xyz = {n: None for n in manipulator_config}
        self.ref_ee_quat = {n: None for n in manipulator_config}
        self.ref_controller_xyz = {n: None for n in manipulator_config}
        self.ref_controller_quat = {n: None for n in manipulator_config}
        self.active = {n: False for n in manipulator_config}
        self.gripper_pos_target: Dict[str, Dict[str, float]] = {}
        self._target_poses: Dict[str, np.ndarray] = {}
        self._debug_last_print = 0.0

        for name, config in manipulator_config.items():
            if "gripper_config" in config:
                gc = config["gripper_config"]
                self.gripper_pos_target[name] = dict(zip(gc["joint_names"], gc["open_pos"]))

        # Cache DOF info
        self.dof_names = list(self.articulation.dof_names)

        self._build_joint_map()
        self._home_positions = self.articulation.get_joint_positions()
        self._ik_setup()

        for name, config in manipulator_config.items():
            T = self._get_gripper_base_pose(config["link_name"])
            self._target_poses[name] = T

        print(f"Isaac Sim articulation DOFs ({len(self.dof_names)}):")
        for i, name in enumerate(self.dof_names):
            print(f"  [{i}] {name}")
        print(f"Home positions: {self._home_positions}")

        return self

    # =========================================================================
    # ===== Joint map =====
    # =========================================================================
    def _build_joint_map(self):
        self._left_joint_indices = []
        self._right_joint_indices = []

        for jname in LEFT_ARM_JOINTS:
            if jname not in self.dof_names:
                raise ValueError(f"Left arm joint '{jname}' not found.")
            self._left_joint_indices.append(self.dof_names.index(jname))

        for jname in RIGHT_ARM_JOINTS:
            if jname not in self.dof_names:
                raise ValueError(f"Right arm joint '{jname}' not found.")
            self._right_joint_indices.append(self.dof_names.index(jname))

    # =========================================================================
    # ===== IK solver =====
    # =========================================================================
    def _ik_setup(self):
        urdf_path = os.path.join(IK_SDK_DIR, "ffw_sg2_follower.urdf")
        config_path = os.path.join(IK_SDK_DIR, "ffw_sg2_follower_solver.yaml")

        with capture_c_stdout():
            self._ik_left = ik_solver.Solver(part=ik_solver.RobotPart.LEFT_ARM, urdf_path=urdf_path, config_path=config_path,)
            self._ik_right = ik_solver.Solver(part=ik_solver.RobotPart.RIGHT_ARM, urdf_path=urdf_path, config_path=config_path,)

        left_q0, right_q0 = self._read_arm_joints()
        self._ik_left.sync_target_with_joints(left_q0)
        self._ik_right.sync_target_with_joints(right_q0)
        print("Genie Sim IK solver initialised (left 7-DOF, right 7-DOF)")

    # =========================================================================
    # ===== Joint read / write =====
    # =========================================================================
    def _read_arm_joints(self):
        all_positions = self.articulation.get_joint_positions()
        left_q = np.array([all_positions[i] for i in self._left_joint_indices], dtype=np.float32)
        right_q = np.array([all_positions[i] for i in self._right_joint_indices], dtype=np.float32)
        return left_q, right_q

    def _write_arm_joints(self, left_q, right_q):
        # Build full joint position array — set_joint_positions requires all DOFs
        all_positions = self.articulation.get_joint_positions().copy()

        for idx, val in zip(self._left_joint_indices, left_q):
            all_positions[idx] = float(val)
        for idx, val in zip(self._right_joint_indices, right_q):
            all_positions[idx] = float(val)
        for gripper_name, targets in self.gripper_pos_target.items():
            for joint_name, joint_pos in targets.items():
                if joint_name not in self.dof_names:
                    raise ValueError(f"Gripper joint '{joint_name}' not found.")
                all_positions[self.dof_names.index(joint_name)] = float(joint_pos)

        # Hold waist/head joints at fixed positions
        for joint_name, joint_pos in self._hold_joint_positions.items():
            if joint_name in self.dof_names:
                all_positions[self.dof_names.index(joint_name)] = float(joint_pos)

        if self.direct_pos:
            # Direct position set (like MuJoCo's data.qpos) — no PD control lag/sag
            self.articulation.set_joint_positions(all_positions)
        else:
            # PD control — physics-based tracking with stiffness/damping
            action = ArticulationAction(joint_positions=all_positions,)
            self.articulation.apply_action(action)

    # =========================================================================
    # ===== Pose helpers =====
    # =========================================================================
    def _get_link_pose(self, body_name: str):
        from isaacsim.core.utils.xforms import get_world_pose
        prim_path = self._find_body_prim_path(body_name)
        position, orientation = get_world_pose(prim_path)
        return np.array(position), np.array(orientation)

    def _find_body_prim_path(self, body_name: str) -> str:
        from pxr import Usd
        stage = get_current_stage()
        robot_prim = stage.GetPrimAtPath(self.robot_prim_path)
        for prim in Usd.PrimRange(robot_prim):
            if prim.GetName() == body_name:
                return str(prim.GetPath())
        raise ValueError(f"Body '{body_name}' not found under {self.robot_prim_path}")

    def _get_gripper_base_pose(self, link7_name: str) -> np.ndarray:
        xyz, quat_wxyz = self._get_link_pose(link7_name)
        T_world_link7 = tf.quaternion_matrix(quat_wxyz)
        T_world_link7[:3, 3] = xyz
        return T_world_link7 @ _T_LINK7_TO_GRIPPER_BASE

    def _get_arm_base_transform(self) -> np.ndarray:
        try:
            xyz, quat_wxyz = self._get_link_pose("arm_base_link")
            T = tf.quaternion_matrix(quat_wxyz)
            T[:3, 3] = xyz
            return T
        except ValueError:
            return np.eye(4)

    # =========================================================================
    # ===== XR input processing =====
    # =========================================================================
    def _process_xr_pose(self, xr_pose, src_name):
        controller_xyz = np.array([xr_pose[0], xr_pose[1], xr_pose[2]])
        controller_quat = [xr_pose[6], xr_pose[3], xr_pose[4], xr_pose[5]]

        controller_xyz = self.R_headset_world @ controller_xyz

        R_transform = np.eye(4)
        R_transform[:3, :3] = self.R_headset_world
        R_quat = tf.quaternion_from_matrix(R_transform)
        controller_quat = tf.quaternion_multiply(tf.quaternion_multiply(R_quat, controller_quat), tf.quaternion_conjugate(R_quat),)

        if self.ref_controller_xyz[src_name] is None:
            self.ref_controller_xyz[src_name] = controller_xyz
            self.ref_controller_quat[src_name] = controller_quat
            return np.zeros(3), np.zeros(3)

        delta_xyz = (controller_xyz - self.ref_controller_xyz[src_name]) * self.scale_factor
        delta_rot = quat_diff_as_angle_axis(self.ref_controller_quat[src_name], controller_quat)
        return delta_xyz, delta_rot

    def _update_gripper_target(self):
        for gname, config in self.manipulator_config.items():
            if "gripper_config" not in config:
                continue
            gc = config["gripper_config"]
            if gc["type"] != "parallel":
                raise ValueError(f"Unsupported gripper type: {gc['type']}")
            trigger = self.xr_client.get_key_value_by_name(gc["gripper_trigger"])
            for jname, open_p, close_p in zip(gc["joint_names"], gc["open_pos"], gc["close_pos"]):
                self.gripper_pos_target[gname][jname] = open_p + (close_p - open_p) * trigger

    # =========================================================================
    # ===== IK update =====
    # =========================================================================
    def _update_ik(self):
        left_q, right_q = self._read_arm_joints()

        if not hasattr(self, "target_q_left"):
            self.target_q_left = left_q
            self.target_q_right = right_q
            self._smooth_q_left = left_q.copy()
            self._smooth_q_right = right_q.copy()

        T_world_base = self._get_arm_base_transform()
        T_base_world = np.linalg.inv(T_world_base)

        left_active = False
        right_active = False

        for src_name, config in self.manipulator_config.items():
            grip_val = self.xr_client.get_key_value_by_name(config["control_trigger"])
            self.active[src_name] = grip_val > 0.9

            if "left" in src_name:
                left_active = self.active[src_name]
            else:
                right_active = self.active[src_name]

            if self.active[src_name]:
                if self.ref_ee_xyz[src_name] is None:
                    print(f"{src_name} activated.")
                    # Use last commanded (smoothed) positions instead of actual
                    # physics positions to avoid gravity-sag jump at activation
                    if "left" in src_name:
                        cmd_q = self._smooth_q_left.copy()
                    else:
                        cmd_q = self._smooth_q_right.copy()
                    solver = self._ik_left if "left" in src_name else self._ik_right
                    solver.sync_target_with_joints(cmd_q)
                    if "left" in src_name:
                        self.target_q_left = cmd_q.copy()
                    else:
                        self.target_q_right = cmd_q.copy()
                    T_gripper = self._get_gripper_base_pose(config["link_name"])
                    self.ref_ee_xyz[src_name] = T_gripper[:3, 3]
                    self.ref_ee_quat[src_name] = tf.quaternion_from_matrix(T_gripper)

                xr_pose = self.xr_client.get_pose_by_name(config["pose_source"])
                delta_xyz, delta_rot = self._process_xr_pose(xr_pose, src_name)
                target_xyz, target_quat = apply_delta_pose(self.ref_ee_xyz[src_name], self.ref_ee_quat[src_name], delta_xyz, delta_rot,)
                T_world_target = tf.quaternion_matrix(target_quat)
                T_world_target[:3, 3] = target_xyz
                self._target_poses[src_name] = T_world_target

                T_base_target = T_base_world @ T_world_target
                pos_base = T_base_target[:3, 3].astype(np.float32)
                quat_base = (R.from_matrix(T_base_target[:3, :3]) .as_quat(scalar_first=False) .astype(np.float32))
                solver = self._ik_left if "left" in src_name else self._ik_right
                solver.update_target_quat(target_pos=pos_base, target_quat=quat_base)
            else:
                if self.ref_ee_xyz[src_name] is not None:
                    print(f"{src_name} deactivated.")
                    self.ref_ee_xyz[src_name] = None
                    self.ref_controller_xyz[src_name] = None

        if left_active:
            try:
                self.target_q_left = self._ik_left.solve()
            except RuntimeError as e:
                print(f"Left IK failed: {e}")

        if right_active:
            try:
                self.target_q_right = self._ik_right.solve()
            except RuntimeError as e:
                print(f"Right IK failed: {e}")

        # Per-joint exponential smoothing: shoulder joints (0-2) get stronger
        # smoothing to suppress jerk from heavy proximal links
        # alpha closer to 0 → smoother but more lag; alpha=1.0 → no smoothing
        alpha_per_joint = np.array([
            0.12, 0.12, 0.15,   # joints 1-3 (shoulder) — strong smoothing
            0.25,                # joint 4 (elbow) — moderate
            0.4, 0.4, 0.4,      # joints 5-7 (wrist) — lighter smoothing
        ], dtype=np.float32)
        self._smooth_q_left = alpha_per_joint * self.target_q_left + (1 - alpha_per_joint) * self._smooth_q_left
        self._smooth_q_right = alpha_per_joint * self.target_q_right + (1 - alpha_per_joint) * self._smooth_q_right

        self._write_arm_joints(self._smooth_q_left, self._smooth_q_right)

    # =========================================================================
    # ===== Main loop =====
    # =========================================================================
    def run(self):
        print("Starting Isaac Sim teleop loop. Press Ctrl+C to stop.")
        _last_heartbeat = 0.0
        while simulation_app.is_running() and not self._stop_event.is_set():
            try:
                self._update_ik()
                self._update_gripper_target()
                self.world.step(render=True)

                # Stream camera frame to PICO VR (after render)
                if hasattr(self, "_video_streamer") and self._video_streamer:
                    self._video_streamer.capture_and_send()

                # Periodic heartbeat: show grip/timestamp to diagnose XR connection
                now = time.time()
                if now - _last_heartbeat > 2.0:
                    _last_heartbeat = now
                    grips = {}
                    for src_name, config in self.manipulator_config.items():
                        grips[src_name] = self.xr_client.get_key_value_by_name(config["control_trigger"])
                    ts = xrt.get_time_stamp_ns()
                    print(f"[heartbeat] grip={grips}  ts={ts}")
            except KeyboardInterrupt:
                print("\nTeleoperation stopped.")
                self._stop_event.set()
        if hasattr(self, "_video_streamer") and self._video_streamer:
            self._video_streamer.shutdown()
        simulation_app.close()


# =============================================================================
# ===== Entry point =====
# =============================================================================
def main():
    robot_usd = args.robot_usd
    if robot_usd is None:
        robot_usd = os.path.join(SCRIPT_DIR, "ai_worker/ffw_sg2.usd")

    robot_prim_path = "/ffw_sg2_follower"

    # ===== Phase 1: Create world =====
    # The main thread runs world.step(render=True)
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
    # Ground plane
    from isaacsim.core.prims import SingleXFormPrim
    world.scene.add_default_ground_plane()

    # ===== Phase 2: Load robot =====
    # In production this runs on the render loop (between world.step calls)
    add_reference_to_stage(usd_path=robot_usd,prim_path=robot_prim_path,)

    # ===== Configure joint drive stiffness/damping ======
    # URDF only has damping=0.1, no stiffness — causes severe vibration.
    from pxr import Usd
    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    for prim in Usd.PrimRange(robot_prim):
        # Check for revolute or prismatic joint drives
        for drive_type in ["angular", "linear"]:
            drive_api = UsdPhysics.DriveAPI.Get(prim, drive_type)
            if not drive_api:
                continue
            joint_name = prim.GetName()
            # Waist/head joints: very stiff to prevent vibration
            if any(k in joint_name for k in ["lift", "wheel", "head_joint"]):
                drive_api.GetStiffnessAttr().Set(1e6)
                drive_api.GetDampingAttr().Set(1e5)
            elif any(k in joint_name for k in ["joint1", "joint2", "joint3"]):
                # Shoulder joints (1-3): high gains — heavy proximal links
                drive_api.GetStiffnessAttr().Set(5e5)
                drive_api.GetDampingAttr().Set(5e4)
            elif any(k in joint_name for k in ["joint4"]):
                # Elbow joint: medium-high
                drive_api.GetStiffnessAttr().Set(2e5)
                drive_api.GetDampingAttr().Set(2e4)
            elif any(k in joint_name for k in ["joint5", "joint6", "joint7"]):
                # Wrist joints (5-7): need sufficient stiffness to resist gravity sag
                drive_api.GetStiffnessAttr().Set(3e5)
                drive_api.GetDampingAttr().Set(3e4)
            else:
                # Other joints (gripper, etc.)
                drive_api.GetStiffnessAttr().Set(1e5)
                drive_api.GetDampingAttr().Set(1e4)
            print(f"  Drive configured: {joint_name} ({drive_type})"
                  f" stiffness={drive_api.GetStiffnessAttr().Get()}"
                  f" damping={drive_api.GetDampingAttr().Get()}")

    # ===== Phase 3: Play + init articulation (same as api_core._play) =====
    time.sleep(1)
    world.play()
    time.sleep(1)

    # ===== Same as UIBuilder.initialize_articulation() =====
    articulation = SingleArticulation(prim_path=robot_prim_path, name="ffw_sg2_follower",)
    world.scene.add(articulation)
    articulation.initialize()
    print("Robot loaded and articulation initialised!")

    # ===== Set initial pose (same as PiEnv.reset + FFW_SG2_DEFAULT_STATES) =====
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

    ARM_NAMES = [f"arm_l_joint{i}" for i in range(1, 8)] + [f"arm_r_joint{i}" for i in range(1, 8)]
    HEAD_NAMES = ["head_joint1", "head_joint2"]
    WAIST_NAMES = ["lift_joint", "left_wheel_steer", "left_wheel_drive",
                   "right_wheel_steer", "right_wheel_drive",
                   "rear_wheel_steer", "rear_wheel_drive"]
    GRIPPER_NAMES = ["gripper_l_joint1", "gripper_r_joint1"]

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

    # ===== Phase 3: Create teleop controller and run =====
    config = {
        "right_hand": {
            "link_name": "arm_r_link7",
            "pose_source": "right_controller",
            "control_trigger": "right_grip",
        },
        "left_hand": {
            "link_name": "arm_l_link7",
            "pose_source": "left_controller",
            "control_trigger": "left_grip",
        },
    }

    # ===== Hold waist + head joints at init positions every frame to prevent vibration =====
    hold_positions = {}
    for name, val in zip(WAIST_NAMES, INIT_WAIST):
        hold_positions[name] = val
    for name, val in zip(HEAD_NAMES, INIT_HEAD):
        hold_positions[name] = val

    controller = DualArmIsaacTeleopController.create(
        world=world,
        articulation=articulation,
        robot_prim_path=robot_prim_path,
        manipulator_config=config,
        scale_factor=args.scale_factor,
        hold_joint_positions=hold_positions,
        direct_pos=args.direct_pos,
    )

    # ===== Phase 4: Video streamer (Isaac Sim camera → PICO VR) =====
    video_streamer = None
    if args.pico_stream_port > 0:
        from isaac_video_streamer import IsaacVideoStreamer
        res_w, res_h = (int(x) for x in args.stream_res.split("x"))
        video_streamer = IsaacVideoStreamer(
            stage=stage,
            robot_prim_path=robot_prim_path,
            pico_ip=args.pico_ip,
            pico_stream_port=args.pico_stream_port,
            resolution=(res_w, res_h),
            fps=args.stream_fps,
            bitrate=args.stream_bitrate,
            send_every_n=args.stream_skip,
        )
        controller._video_streamer = video_streamer

    controller.run()


if __name__ == "__main__":
    main()
