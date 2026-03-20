# =============================================================================
# FFW SG2 Follower Teleop in Isaac Sim with Genie Sim IK Solver.
# Isaac Sim's SimulationApp / World / SingleArticulation APIs.
# =============================================================================

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
parser.add_argument("--scale-factor", type=float, default=1)
parser.add_argument("--physics-dt", type=float, default=1.0 / 120.0)
parser.add_argument("--rendering-dt", type=float, default=1.0 / 60.0)
args, _unknown = parser.parse_known_args()

# =============================================================================
# XRoboToolkit SDK — init BEFORE SimulationApp so the SDK's background
# network thread is established before Isaac Sim takes over signal/network.
# =============================================================================

import xrobotoolkit_sdk as xrt

print("Initializing XRoboToolkit SDK before SimulationApp...")
xrt.init()
print("XRoboToolkit SDK initialized. Waiting for device connection...")
_ts = xrt.get_time_stamp_ns()
_lg = xrt.get_left_grip()
_rg = xrt.get_right_grip()
print(f"  SDK check: timestamp={_ts}, left_grip={_lg}, right_grip={_rg}")
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
import ik_solver
from meshcat import transformations as tf

# =============================================================================
# ===== XR client & geometry utils =====
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.utils.geometry import (R_HEADSET_TO_WORLD, apply_delta_pose,quat_diff_as_angle_axis,)

IK_SDK_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../utils/IK-SDK"))
UTILS_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../utils"))
sys.path.insert(0, UTILS_DIR)

from name_utils import (
    FFW_SG2_LEFT_ARM_JOINT_NAMES,
    FFW_SG2_RIGHT_ARM_JOINT_NAMES,
    FFW_SG2_DUAL_ARM_JOINT_NAMES,
    FFW_SG2_HEAD_JOINT_NAMES,
    FFW_SG2_WAIST_JOINT_NAMES,
    FFW_SG2_GRIPPER_JOINTS_NAMES,
)

LEFT_ARM_JOINTS = FFW_SG2_LEFT_ARM_JOINT_NAMES
RIGHT_ARM_JOINTS = FFW_SG2_RIGHT_ARM_JOINT_NAMES

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
    def create(cls, world: World,articulation: SingleArticulation, robot_prim_path: str, manipulator_config: Dict[str, Dict[str, Any]], R_headset_world: np.ndarray = R_HEADSET_TO_WORLD, scale_factor: float = 1.0, hold_joint_positions: Dict[str, float] = None,):
        self = cls()
        self.world = world
        self.articulation = articulation
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

        # Initialize gripper with open position
        for name, config in manipulator_config.items():
            if "gripper_config" in config:
                gc = config["gripper_config"]
                self.gripper_pos_target[name] = dict(zip(gc["joint_names"], gc["open_pos"]))

        self.dof_names = list(self.articulation.dof_names)
        self._build_joint_map()
        self._home_positions = self.articulation.get_joint_positions()
        self._cache_gripper_limits()
        self._ik_setup()
        # !&%p
        # Custom Camera
        # !&%P
        self._setup_cameras()

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
            self._left_joint_indices.append(self.dof_names.index(jname))

        for jname in RIGHT_ARM_JOINTS:
            self._right_joint_indices.append(self.dof_names.index(jname))

    # =========================================================================
    # ===== Gripper joint limits =====
    # =========================================================================
    def _cache_gripper_limits(self):
        lower = self.articulation.dof_properties["lower"]
        upper = self.articulation.dof_properties["upper"]
        self._gripper_lower = {}
        self._gripper_upper = {}
        for name, config in self.manipulator_config.items():
            if "gripper_config" not in config:
                continue
            for jname in config["gripper_config"]["joint_names"]:
                if jname in self.dof_names:
                    idx = self.dof_names.index(jname)
                    self._gripper_lower[idx] = float(lower[idx])
                    self._gripper_upper[idx] = float(upper[idx])
                    print(f"  Gripper limit: {jname} [{self._gripper_lower[idx]:.3f}, {self._gripper_upper[idx]:.3f}]")

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
    # ===== Camera setup =====
    # =========================================================================
    def _setup_cameras(self):
        from omni.kit.viewport.utility import get_active_viewport_and_window
        try:
            from omni.kit.viewport.window import ViewportWindow
        except ImportError:
            from omni.kit.viewport.utility import create_viewport_window
            ViewportWindow = None

        # Main viewport -> Head Camera
        viewport, _ = get_active_viewport_and_window()
        head_cam = f"{self.robot_prim_path}/head_link2/zed/Head_Camera"
        viewport.set_active_camera(head_cam)

        # Additional viewport windows for Left and Right cameras
        cam_windows = [
            ("Left Camera",  f"{self.robot_prim_path}/arm_l_link7/camera_l_bottom_screw_frame/camera_l_link/Left_Camera"),
            ("Right Camera", f"{self.robot_prim_path}/arm_r_link7/camera_r_bottom_screw_frame/camera_r_link/Right_Camera"),
        ]

        self._extra_viewports = []
        for title, cam_path in cam_windows:
            try:
                if ViewportWindow is not None:
                    vp_win = ViewportWindow(title, width=320, height=240)
                else:
                    vp_win = create_viewport_window(title, width=320, height=240)
                vp_win.viewport_api.set_active_camera(cam_path)
                self._extra_viewports.append(vp_win)
                print(f"  Viewport window: {title} -> {cam_path}")
            except Exception as e:
                print(f"  WARNING: Could not create viewport for {title}: {e}")

    # =========================================================================
    # ===== Joint read / write =====
    # =========================================================================
    def _read_arm_joints(self):
        all_positions = self.articulation.get_joint_positions()
        left_q = np.array([all_positions[i] for i in self._left_joint_indices], dtype=np.float32)
        right_q = np.array([all_positions[i] for i in self._right_joint_indices], dtype=np.float32)
        return left_q, right_q

    def _write_arm_joints(self, left_q, right_q):
        all_positions = self.articulation.get_joint_positions().copy()

        for idx, val in zip(self._left_joint_indices, left_q):
            all_positions[idx] = float(val)
        for idx, val in zip(self._right_joint_indices, right_q):
            all_positions[idx] = float(val)

        # Gripper targets (clamped to joint limits)
        for gripper_name, targets in self.gripper_pos_target.items():
            for joint_name, joint_pos in targets.items():
                if joint_name in self.dof_names:
                    idx = self.dof_names.index(joint_name)
                    all_positions[idx] = np.clip(float(joint_pos),  self._gripper_lower[idx],  self._gripper_upper[idx])

        # Hold waist/head joints at fixed positions
        for joint_name, joint_pos in self._hold_joint_positions.items():
            if joint_name in self.dof_names:
                all_positions[self.dof_names.index(joint_name)] = float(joint_pos)


        action = ArticulationAction(joint_positions=all_positions) # Create action which contains all joint's target angle
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
        gripper_alpha = 0.2
        for gname, config in self.manipulator_config.items():
            if "gripper_config" not in config:
                continue
            gc = config["gripper_config"]
            if gc["type"] != "parallel":
                raise ValueError(f"Unsupported gripper type: {gc['type']}")
            trigger = self.xr_client.get_key_value_by_name(gc["gripper_trigger"])
            for jname, open_p, close_p in zip(gc["joint_names"], gc["open_pos"], gc["close_pos"]):
                desired = open_p + (close_p - open_p) * trigger
                current = self.gripper_pos_target[gname][jname]
                self.gripper_pos_target[gname][jname] = current + gripper_alpha * (desired - current)

    # =========================================================================
    # ===== IK update =====
    # =========================================================================
    def _update_ik(self):
        left_q, right_q = self._read_arm_joints()

        if not hasattr(self, "target_q_left"): # Initialize only in first frame
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
            self.target_q_left = self._ik_left.solve()

        if right_active:
            self.target_q_right = self._ik_right.solve()


        # Per-joint exponential smoothing: shoulder joints (0-2) get stronger
        # smoothing to suppress jerk from heavy proximal links
        # alpha closer to 0 → smoother but more lag; alpha=1.0 → no smoothing
        alpha_per_joint = np.array([
            0.23, 0.23, 0.23,      # joints 1-3 (shoulder)
            0.3,                # joint 4 (elbow)
            0.3, 0.3, 0.3,      # joints 5-7 (wrist)
        ], dtype=np.float32)
        self._smooth_q_left = alpha_per_joint * self.target_q_left + (1 - alpha_per_joint) * self._smooth_q_left
        self._smooth_q_right = alpha_per_joint * self.target_q_right + (1 - alpha_per_joint) * self._smooth_q_right

        self._write_arm_joints(self._smooth_q_left, self._smooth_q_right)

    # =========================================================================
    # ===== Main loop =====
    # =========================================================================
    def run(self):
        _last_time = 0.0
        while simulation_app.is_running() and not self._stop_event.is_set():
            try:
                self._update_ik()
                self._update_gripper_target()
                self.world.step(render=True)
                # Periodic heartbeat: show grip/timestamp to diagnose XR connection
                now = time.time()
                if now - _last_time > 0.5:
                    _last_time = now
                    grips = {}
                    for src_name, config in self.manipulator_config.items():
                        grips[src_name] = self.xr_client.get_key_value_by_name(config["control_trigger"])
                    ts = xrt.get_time_stamp_ns()
                    # Shoulder joint debug: IK target vs smoothed cmd vs actual physics
                    actual_l, actual_r = self._read_arm_joints()
                    print(f"[Debug] grip={grips}  ts={ts}")
                    print(f"  L shoulder IK_target={np.rad2deg(self.target_q_left[:3]).round(1)}"
                          f"  cmd={np.rad2deg(self._smooth_q_left[:3]).round(1)}"
                          f"  actual={np.rad2deg(actual_l[:3]).round(1)}"
                          f"  err={np.rad2deg(self._smooth_q_left[:3] - actual_l[:3]).round(2)}")
                    print(f"  R shoulder IK_target={np.rad2deg(self.target_q_right[:3]).round(1)}"
                          f"  cmd={np.rad2deg(self._smooth_q_right[:3]).round(1)}"
                          f"  actual={np.rad2deg(actual_r[:3]).round(1)}"
                          f"  err={np.rad2deg(self._smooth_q_right[:3] - actual_r[:3]).round(2)}")

            except KeyboardInterrupt:
                print("\nTeleoperation stopped.")
                self._stop_event.set()
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
    #from isaacsim.core.prims import SingleXFormPrim
    #world.scene.add_default_ground_plane()

    # Background scene from table_task_ffw_sg2.json
    ASSETS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../../assets"))
    scene_usd_path = os.path.join(ASSETS_DIR, "background/room/room_1/background.usda")
    add_reference_to_stage(usd_path=scene_usd_path, prim_path="/World")

    # Activate the table (disabled by default in background.usda)
    #table_prim = stage.GetPrimAtPath("/World/background/benchmark_table_019")
    #if table_prim.IsValid():
    #    table_prim.SetActive(True)
    #    print("Activated benchmark_table_019")

    # Load sub_task scene (table + colored cubes) — same as pick_block_color in select_color.yaml
    BENCHMARK_CFG_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../../benchmark/config"))
    sub_task_usd = os.path.join(BENCHMARK_CFG_DIR, "llm_task", "pick_block_color", "0", "scene.usda")
    add_reference_to_stage(usd_path=sub_task_usd, prim_path="/Workspace")
    print(f"Loaded sub_task scene: {sub_task_usd}")

    # ===== Phase 2: Load robot =====
    # In production this runs on the render loop (between world.step calls)
    add_reference_to_stage(usd_path=robot_usd, prim_path=robot_prim_path)

    # Set robot position from table_task_ffw_sg2.json robot_init_pose/workspace_00
    from isaacsim.core.prims import SingleXFormPrim
    robot_xform = SingleXFormPrim(
        prim_path=robot_prim_path,
        position=np.array([-0.48, -0.02646532841026783, -0.009999999776482582]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
    )

    # ===== Configure joint drive stiffness/damping ======
    # URDF only has damping=0.1
    from pxr import Usd
    robot_prim = stage.GetPrimAtPath(robot_prim_path)

    # Increase PhysX solver iterations for tighter joint tracking
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
                # acceleration mode: critical damping ≈ 2*sqrt(stiffness)
                # stiffness=1e5 → critical_damping ≈ 632, use ~1e3 for slight overdamp
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

    # Create physics material via USD API directly
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

    # Bind material to all gripper collision meshes
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

    # ===== Phase 3: Create teleop controller and run =====
    config = {
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
    )

    controller.run()


if __name__ == "__main__":
    main()


