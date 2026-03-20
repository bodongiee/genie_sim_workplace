import contextlib
import os
import sys
import tempfile
import threading
import time
from typing import Any, Dict

import ik_solver
import mujoco
import numpy as np
import tyro
from mujoco import viewer as mj_viewer
from scipy.spatial.transform import Rotation as R

from meshcat import transformations as tf

from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.utils.geometry import (R_HEADSET_TO_WORLD,apply_delta_pose,quat_diff_as_angle_axis,)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IK_SDK_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../utils/IK-SDK"))

LEFT_ARM_JOINTS = [f"arm_l_joint{i}" for i in range(1, 8)]
RIGHT_ARM_JOINTS = [f"arm_r_joint{i}" for i in range(1, 8)]

# Fixed transform from arm_*_link7 → gripper_*_rh_p12_rn_base (from URDF)
# <joint type="fixed"> rpy="0 π π" xyz="0 0 -0.078"
_T_LINK7_TO_GRIPPER_BASE = np.eye(4)
_T_LINK7_TO_GRIPPER_BASE[:3, :3] = R.from_euler("xyz", [0, np.pi, np.pi]).as_matrix() #3-dim rotation
_T_LINK7_TO_GRIPPER_BASE[:3, 3] = [0, 0, -0.078]

# 1  0  0  0  
# 0 -1  0  0  
# 0  0 -1 -0.078
# 0  0  0  1


# =================================================================================
# ===== C++ stdout capture =====
# =================================================================================
@contextlib.contextmanager
def capture_c_stdout():
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
                print(f"[C++ IK Solver]:\n{output}")

# =================================================================================
# ===== MuJoCo helpers =====
# =================================================================================
def _mj_joint_qpos_addr(mj_model, joint_name: str) -> int:
    """Return qpos address for a MuJoCo joint"""
    jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name) # jid is Model Joint Index
    return mj_model.jnt_qposadr[jid] # returns joint's lookup table start index

def _mj_ctrl_from_qpos(mj_model, qpos):
    """Convert qpos to ctrl assuming 1-DOF position actuators targeting joints."""
    ctrl = np.zeros(mj_model.nu)
    for i in range(mj_model.nu):
        target_id = mj_model.actuator_trnid[i, 0]
        ctrl[i] = qpos[mj_model.jnt_qposadr[target_id]]
    return ctrl


# =================================================================================
# ===== Controller : Teleop Controller on Mujoco using GenieSim IK Solver =====
# =================================================================================
class DualArmMujocoTeleopController:
    # =============================================================================
    # ===== Initialize =====
    # =============================================================================
    def __init__( self, xml_path: str, robot_urdf_path: str, manipulator_config: Dict[str, Dict[str, Any]], R_headset_world: np.ndarray = R_HEADSET_TO_WORLD, scale_factor: float = 1.0, dt: float = 0.01,debug_vr: bool = False,):
        #R_HEADSET_TO_WORLD : VR cordinates to Robot world cordinates
        self.xml_path = xml_path
        self.robot_urdf_path = robot_urdf_path
        self.manipulator_config = manipulator_config
        self.R_headset_world = R_headset_world
        self.scale_factor = scale_factor
        self.dt = dt
        self.debug_vr = debug_vr
        self._stop_event = threading.Event()

        # XR client
        self.xr_client = XrClient()

        # Initial Joint map
        self.ref_ee_xyz = {n: None for n in manipulator_config}
        self.ref_ee_quat = {n: None for n in manipulator_config}
        self.ref_controller_xyz = {n: None for n in manipulator_config}
        self.ref_controller_quat = {n: None for n in manipulator_config}
        self.active = {n: False for n in manipulator_config}
        self.gripper_pos_target: Dict[str, Dict[str, float]] = {}
        self._target_poses: Dict[str, np.ndarray] = {}  # 4×4 world-frame
        for name, config in manipulator_config.items():
            if "gripper_config" in config :
                gc = config["gripper_config"]
                self.gripper_pos_target[name] = dict(zip(gc["joint_names"], gc["open_pos"]))

        # Setup 
        self._mujoco_setup_model()
        self._build_joint_map()
        self._mujoco_setup_state()
        self._ik_setup()

        # Initialise target poses from current EEF
        for name, config in manipulator_config.items():
            T = self._get_gripper_base_pose(config["link_name"]) #arm_*_link7 -> gripper
            self._target_poses[name] = T

        # Debug timing
        self._debug_last_print = 0.0
        
    # =============================================================================
    # ===== Mujoco Configure =====
    # =============================================================================
    def _mujoco_setup_model(self):
        self.mj_model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        print("Mujoco joints:")
        for i in range(self.mj_model.njnt):
            name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            print(f"  {name}")

        self.mj_model.vis.headlight.ambient = [0.4, 0.4, 0.4]
        self.mj_model.vis.headlight.diffuse = [0.8, 0.8, 0.8]
        self.mj_model.vis.headlight.specular = [0.6, 0.6, 0.6]

    def _mujoco_setup_state(self):
        # Load initial pose from XML Keyframe "home"
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, self.mj_model.key("home").id)
        self.mj_data.ctrl[:] = _mj_ctrl_from_qpos(self.mj_model, self.mj_data.qpos)

        # Forward Kinematics -> reak EEF
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # [DEBUG] Mocap targets for visualisation
        self.target_mocap_idx: Dict[str, int] = {}
        for name, config in self.manipulator_config.items():
            vis = config.get("vis_target")
            if vis is None:
                self.target_mocap_idx[name] = -1
                continue
            body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, vis)
            mid = self.mj_model.body_mocapid[body_id]
            self.target_mocap_idx[name] = mid
            print(f"Mocap '{vis}' → idx {mid}")


    def _build_joint_map(self):
        self._left_qpos_addrs = []
        self._right_qpos_addrs = []

        for jname in LEFT_ARM_JOINTS:
            addr = _mj_joint_qpos_addr(self.mj_model, jname)
            if addr == -1:
                raise ValueError(f"Left arm joint '{jname}' not found in MuJoCo model.")
            self._left_qpos_addrs.append(addr)

        for jname in RIGHT_ARM_JOINTS:
            addr = _mj_joint_qpos_addr(self.mj_model, jname)
            if addr == -1:
                raise ValueError(f"Right arm joint '{jname}' not found in MuJoCo model.")
            self._right_qpos_addrs.append(addr)
    
    # =============================================================================
    # ===== IK solver =====
    # Use same URDF and XML with Genie Sim IK in Isaac Sim
    # =============================================================================

    def _ik_setup(self):
        urdf_path = os.path.join(IK_SDK_DIR, "ffw_sg2_follower.urdf")
        config_path = os.path.join(IK_SDK_DIR, "ffw_sg2_follower_solver.yaml")

        with capture_c_stdout():
            # Solve Left Arm IK
            self._ik_left = ik_solver.Solver(part=ik_solver.RobotPart.LEFT_ARM,urdf_path=urdf_path,config_path=config_path,)
            # Solve Right Arm IK
            self._ik_right = ik_solver.Solver(part=ik_solver.RobotPart.RIGHT_ARM, urdf_path=urdf_path, config_path=config_path,)
            
        self._ik_left.set_debug_mode(False)
        self._ik_right.set_debug_mode(False)

        left_q0, right_q0 = self._read_arm_joints()
        self._ik_left.sync_target_with_joints(left_q0)
        self._ik_right.sync_target_with_joints(right_q0)
        print("Genie Sim IK solver initialised (left 7-DOF, right 7-DOF)")

    # =============================================================================
    # ===== Joint read / write =====
    # =============================================================================

    def _read_arm_joints(self):
        left_q = np.array([self.mj_data.qpos[a] for a in self._left_qpos_addrs], dtype=np.float32)
        right_q = np.array([self.mj_data.qpos[a] for a in self._right_qpos_addrs], dtype=np.float32)
        return left_q, right_q

    def _write_arm_joints(self, left_q, right_q):
        qpos = self.mj_data.qpos.copy()
        for addr, val in zip(self._left_qpos_addrs, left_q):
            qpos[addr] = val
        for addr, val in zip(self._right_qpos_addrs, right_q):
            qpos[addr] = val

        # Gripper targets
        for gripper_name, targets in self.gripper_pos_target.items():
            for joint_name, joint_pos in targets.items():
                addr = _mj_joint_qpos_addr(self.mj_model, joint_name)
                if addr == -1:
                    raise ValueError(f"Gripper joint '{joint_name}' not found.")
                qpos[addr] = joint_pos
        self.mj_data.ctrl[:] = _mj_ctrl_from_qpos(self.mj_model, qpos)

    # =============================================================================
    # ===== Pose helpers =====
    # =============================================================================
    def _get_link_pose(self, body_name) -> [np.ndarray, np.ndarray]: # Get some links position
        bid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self.mj_data.xpos[bid].copy(), self.mj_data.xquat[bid].copy()

    def _get_gripper_base_pose(self, link7_name):
        xyz, quat = self._get_link_pose(link7_name)
        T_world_link7 = tf.quaternion_matrix(quat)
        T_world_link7[:3, 3] = xyz
        return T_world_link7 @ _T_LINK7_TO_GRIPPER_BASE

    def _get_arm_base_transform(self): # Get robot base position in World Frame
        bid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "arm_base_link")
        if bid == -1:
            return np.eye(4)
        T = tf.quaternion_matrix(self.mj_data.xquat[bid])
        T[:3, 3] = self.mj_data.xpos[bid]
        return T
    # =============================================================================
    # ===== XR input processing =====
    # =============================================================================
    def _process_xr_pose(self, xr_pose, src_name):
        # raw XR Controller Pose -> (delta_xyz, delta_rot)
        controller_xyz = np.array([xr_pose[0], xr_pose[1], xr_pose[2]])    # x, y, z
        controller_quat = [xr_pose[6], xr_pose[3], xr_pose[4], xr_pose[5]] # q, x, y, z

        controller_xyz = self.R_headset_world @ controller_xyz #Controller's position is calculated based on robot world 

        #XR rotation to world rotation
        R_transform = np.eye(4)
        R_transform[:3, :3] = self.R_headset_world
        R_quat = tf.quaternion_from_matrix(R_transform)
        controller_quat = tf.quaternion_multiply(tf.quaternion_multiply(R_quat, controller_quat),tf.quaternion_conjugate(R_quat),)

        if self.ref_controller_xyz[src_name] is None:
            self.ref_controller_xyz[src_name] = controller_xyz
            self.ref_controller_quat[src_name] = controller_quat
            return np.zeros(3), np.zeros(3)

        delta_xyz = (controller_xyz - self.ref_controller_xyz[src_name]) * self.scale_factor
        delta_rot = quat_diff_as_angle_axis(self.ref_controller_quat[src_name], controller_quat)
        return delta_xyz, delta_rot
    

    # ===== Gripper =====
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

    # =============================================================================
    # ===== Mocap visualisation =====
    # =============================================================================
    def _update_mocap_target(self):
        for name in self.manipulator_config:
            T = self._target_poses.get(name)
            mid = self.target_mocap_idx.get(name, -1)
            if mid != -1:
                self.mj_data.mocap_pos[mid] = T[:3, 3]
                self.mj_data.mocap_quat[mid] = tf.quaternion_from_matrix(T)

    # =============================================================================
    # ==== Target / EEF axes visualisation =====
    # =============================================================================
    @staticmethod
    def _add_axes_geom(scn, pos, rot_3x3, length=0.08, radius=0.003, alpha=0.9):
        # X : Red
        # Y : Green
        # Z : Blue  
        axes_cfg = [ (rot_3x3[:, 0], [1.0, 0.0, 0.0, alpha]), (rot_3x3[:, 1], [0.0, 1.0, 0.0, alpha]), (rot_3x3[:, 2], [0.0, 0.0, 1.0, alpha]),]

        for axis_dir, rgba in axes_cfg:
            if scn.ngeom >= scn.maxgeom:
                return
            g = scn.geoms[scn.ngeom]

            center = pos + axis_dir * (length / 2.0)
            z = axis_dir / (np.linalg.norm(axis_dir) + 1e-12)
            up = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
            x = np.cross(up, z)
            x /= np.linalg.norm(x) + 1e-12
            y = np.cross(z, x)
            mat3 = np.column_stack([x, y, z])  # 3×3 column-major for MuJoCo

            mujoco.mjv_initGeom(
                g,
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[radius, length / 2.0, 0],
                pos=center.astype(np.float64),
                mat=mat3.flatten().astype(np.float64),
                rgba=np.array(rgba, dtype=np.float32),
            )
            scn.ngeom += 1

    @staticmethod
    def _add_sphere_geom(scn, pos, radius=0.006, rgba=(1.0, 1.0, 1.0, 0.8)):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[radius, 0, 0],
            pos=np.asarray(pos, dtype=np.float64),
            mat=np.eye(3).flatten().astype(np.float64),
            rgba=np.array(rgba, dtype=np.float32),
        )
        scn.ngeom += 1

    def _update_target_vis(self, viewer):
        scn = viewer.user_scn
        scn.ngeom = 0  # reset every frame

        for name, config in self.manipulator_config.items():
            # ===== Target EEF axes (bright, full alpha) =====
            T_target = self._target_poses.get(name)
            if T_target is not None:
                self._add_sphere_geom(scn, T_target[:3, 3], radius=0.008,
                                      rgba=(1.0, 1.0, 0.0, 0.9))
                self._add_axes_geom(scn, T_target[:3, 3], T_target[:3, :3],
                                    length=0.10, radius=0.004, alpha=0.9)

            # ===== Current EEF axes at gripper_base frame (dim, lower alpha) =====
            T_cur = self._get_gripper_base_pose(config["link_name"])
            cur_pos = T_cur[:3, 3]
            self._add_sphere_geom(scn, cur_pos, radius=0.006,
                                  rgba=(0.8, 0.8, 0.8, 0.5))
            self._add_axes_geom(scn, cur_pos, T_cur[:3, :3],
                                length=0.07, radius=0.002, alpha=0.4)

    # =============================================================================
    # ==== VR debug print =====
    # =============================================================================

    def _print_vr_debug(self):
        # ===== Print VR input state at ~2 Hz =====
        now = time.time()
        if now - self._debug_last_print < 0.5:
            return
        self._debug_last_print = now

        lines = ["\n── VR Debug ──────────────────────────────"]
        for src_name, config in self.manipulator_config.items():
            grip_val = self.xr_client.get_key_value_by_name(config["control_trigger"])
            raw_pose = self.xr_client.get_pose_by_name(config["pose_source"])

            lines.append(f"[{src_name}]")
            lines.append(f"  active    : {self.active[src_name]}  (grip={grip_val:.3f})")
            lines.append(f"  raw XR    : pos={np.array(raw_pose[:3]).round(4)}"
                         f"  quat(xyzw)={np.array(raw_pose[3:7]).round(4)}")

            if self.ref_ee_xyz[src_name] is not None:
                lines.append(f"  ref_ee    : {self.ref_ee_xyz[src_name].round(4)}")

            T = self._target_poses.get(src_name)
            if T is not None:
                pos = T[:3, 3]
                quat = tf.quaternion_from_matrix(T)  # w,x,y,z
                lines.append(f"  target    : pos={pos.round(4)}  quat(wxyz)={np.array(quat).round(4)}")

            # Gripper
            if src_name in self.gripper_pos_target:
                for jn, jv in self.gripper_pos_target[src_name].items():
                    lines.append(f"  gripper   : {jn}={jv:.4f}")

        left_q, right_q = self._read_arm_joints()
        lines.append(f"  left_q    : {np.rad2deg(left_q).round(1)}")
        lines.append(f"  right_q   : {np.rad2deg(right_q).round(1)}")
        lines.append("───────────────────────────────────────────")
        print("\n".join(lines))
    # =============================================================================
    # ===== IK update =====
    # =============================================================================
    def _update_ik(self):
        left_q, right_q = self._read_arm_joints() #Read all joints : left q -> ndarray, right q -> ndarray

        # Fix current position if remote is deactivated
        if not hasattr(self, 'target_q_left'):
            self.target_q_left = left_q
            self.target_q_right = right_q

        T_world_base = self._get_arm_base_transform() #Get robot base at world frame
        T_base_world = np.linalg.inv(T_world_base)

        left_active = False
        right_active = False

        for src_name, config in self.manipulator_config.items():
            grip_val = self.xr_client.get_key_value_by_name(config["control_trigger"])
            # Check activated or not
            self.active[src_name] = grip_val > 0.9

            if "left" in src_name:
                left_active = self.active[src_name]
            else:
                right_active = self.active[src_name]

            if self.active[src_name]:
                if self.ref_ee_xyz[src_name] is None:
                    print(f"{src_name} activated.")

                    cur_q = left_q if "left" in src_name else right_q
                    solver = self._ik_left if "left" in src_name else self._ik_right
                    solver.sync_target_with_joints(cur_q)
                    
                    if "left" in src_name:
                        self.target_q_left = cur_q
                    else:
                        self.target_q_right = cur_q

                    # Reference pose = gripper_base frame (IK solver target)
                    # When remote is activated, save current position for read delta angle
                    T_gripper = self._get_gripper_base_pose(config["link_name"])
                    self.ref_ee_xyz[src_name] = T_gripper[:3, 3]
                    self.ref_ee_quat[src_name] = tf.quaternion_from_matrix(T_gripper)
                
                # Calculate delta
                xr_pose = self.xr_client.get_pose_by_name(config["pose_source"])
                delta_xyz, delta_rot = self._process_xr_pose(xr_pose, src_name)

                target_xyz, target_quat = apply_delta_pose(self.ref_ee_xyz[src_name], self.ref_ee_quat[src_name], delta_xyz, delta_rot, )


                # World-frame target at gripper_base (for vis + IK)
                T_world_target = tf.quaternion_matrix(target_quat)
                T_world_target[:3, 3] = target_xyz
                self._target_poses[src_name] = T_world_target

                # Transform to arm_base_link frame for IK solver
                T_base_target = T_base_world @ T_world_target
                pos_base = T_base_target[:3, 3].astype(np.float32)
                quat_base = R.from_matrix(T_base_target[:3, :3] ).as_quat(scalar_first=False).astype(np.float32)

                solver = self._ik_left if "left" in src_name else self._ik_right
                solver.update_target_quat(target_pos=pos_base, target_quat=quat_base)

            else:
                if self.ref_ee_xyz[src_name] is not None:
                    print(f"{src_name} deactivated.")
                    self.ref_ee_xyz[src_name] = None
                    self.ref_controller_xyz[src_name] = None

        # Solve IK if activated
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

        self._write_arm_joints(self.target_q_left, self.target_q_right)

    # =============================================================================
    # ===== Main loop =====
    # =============================================================================

    def run(self):
        import cv2
        renderer = mujoco.Renderer(self.mj_model, height=360, width=480)

        with mj_viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
            viewer.cam.azimuth = 0
            viewer.cam.elevation = -50
            viewer.cam.distance = 2.0
            viewer.cam.lookat[:] = [0.2, 0, 0]

            while not self._stop_event.is_set():
                try:
                    self._update_ik()
                    self._update_gripper_target()
                    self._update_mocap_target()
                    self._update_target_vis(viewer)

                    mujoco.mj_step(self.mj_model, self.mj_data)
                    viewer.sync()

                    if self.mj_model.ncam > 0:
                        renderer.update_scene(self.mj_data, camera=0)
                        pixels_sub = renderer.render()
                        cv2.namedWindow("Sub View (Camera 0)", cv2.WINDOW_NORMAL)
                        cv2.imshow("Sub View (Camera 0)", cv2.cvtColor(pixels_sub, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1) 

                    if self.debug_vr:
                        self._print_vr_debug()
                except KeyboardInterrupt:
                    print("\nTeleoperation stopped.")
                    self._stop_event.set()
        
        cv2.destroyAllWindows()


# =============================================================================
# ===== Entry point =====
# =============================================================================
def main(
    xml_path: str = os.path.join(SCRIPT_DIR, "ai_worker/ffw_sg2.xml"),
    robot_urdf_path: str = os.path.join(
        SCRIPT_DIR,
        "ai_worker/ffw_description/urdf/ffw_sg2_rev1_follower/ffw_sg2_follower.urdf",
    ),
    scale_factor: float = 1.5,
    debug_vr: bool = False,
):
    """FFW SG2 follower teleop in MuJoCo with Genie Sim IK solver."""
    # link_name = MuJoCo body (arm_*_link7).
    # IK solver targets gripper_*_rh_p12_rn_base, which is a fixed offset from link7.
    # The offset is applied internally via _get_gripper_base_pose().
    # 그리퍼 아직 없음
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

    controller = DualArmMujocoTeleopController(
        xml_path=xml_path,
        robot_urdf_path=robot_urdf_path,
        manipulator_config=config,
        scale_factor=scale_factor,
        debug_vr=debug_vr,
    )

    controller.run()


if __name__ == "__main__":
    tyro.cli(main)
