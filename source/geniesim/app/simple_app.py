# =============================================================================
# minimal_infer_app.py
# Genie Sim 3.0 — Minimal Inference Runner
# 
# 원본 app.py에서 benchmark, evaluation, recording, material generalization 등
# 부가 기능을 모두 제거하고, sim 환경에 로봇 + task(scene/objects)만 로드한 뒤
# 외부 inference 모델 서버(WebSocket)와 통신하는 최소 루프.
#
# Usage (Docker/Isaac Sim 환경 내에서):
#   omni_python minimal_infer_app.py \
#       --benchmark.task_name gm_task_pickplace \
#       --benchmark.infer_host localhost:8999 \
#       --benchmark.model_arc pi
#
# 필요 시 CLI로 config override 가능 (원본과 동일한 ParameterServer 활용)
# =============================================================================

import os, sys, time, json, threading, queue
from pathlib import Path

# ---------- 1. Path & Config Setup (원본 app.py 동일) ----------
project_root = Path(__file__).resolve().parent
# geniesim이 설치된 환경이면 아래 줄 불필요, standalone이면 필요
# sys.path.append(str(project_root / "source"))

import geniesim.utils.system_utils as system_utils
from geniesim.config.params import (
    ParameterServer, Config, AppConfig, BenchmarkConfig, LayoutConfig,
    fields, load_dataclass,
)

system_utils.check_and_fix_env()

ps = ParameterServer()
for f in fields(Config):
    ps.declare_parameter(f.name, None)
ps.set_parameters_from_yaml(system_utils.config_path() + "/config.yaml")
ps.override_from_cli()
cfg = load_dataclass(Config, ps)


# ---------- 2. Isaac Sim Launch (원본 app.py 동일) ----------
from geniesim.app.workflow import AppLauncher

app_launcher = AppLauncher(cfg.app)
simulation_app = app_launcher.app

import carb
import omni

from isaacsim.core.utils import extensions
extensions.enable_extension("isaacsim.ros2.bridge")


def wait_rclpy(timeout=10, tick=0.1):
    start = time.time()
    while True:
        try:
            import rclpy
            return rclpy
        except ModuleNotFoundError:
            if time.time() - start > timeout:
                raise RuntimeError("rclpy still not available")
            time.sleep(tick)


rclpy = wait_rclpy()
rclpy.init()

from isaacsim.core.api import World
from geniesim.app.controllers import APICore
from geniesim.app.workflow.ui_builder import UIBuilder


# ---------- 3. Minimal Inference Loop (benchmark 제거) ----------
def main():
    # ===== 3-1. World 생성 =====
    world = World(
        stage_units_in_meters=1,
        physics_dt=1.0 / cfg.app.physics_step,
        rendering_dt=1.0 / cfg.app.rendering_step,
    )
    if cfg.app.enable_gpu_dynamics:
        physx_interface = omni.physx.get_physx_interface()
        physx_interface.overwrite_gpu_setting(1)
        world._physics_context.enable_gpu_dynamics(flag=True)
        world._physics_context.enable_ccd(flag=True)

    # ===== 3-2. UI Builder + APICore (로봇/씬 관리의 핵심) =====
    ui_builder = UIBuilder(world=world)
    api_core = APICore(ui_builder=ui_builder, config=cfg)

    # ===== 3-3. Task config 로드 =====
    task_name = cfg.benchmark.task_name
    sub_task_name = cfg.benchmark.sub_task_name
    task_config_file = os.path.join(
        system_utils.benchmark_conf_path(), "eval_tasks", task_name + ".json"
    )
    task_config = system_utils.load_json(task_config_file)
    task_config["specific_task_name"] = task_name
    task_config["sub_task_name"] = sub_task_name

    # ===== 3-4. Task generation (scene layout) =====
    from geniesim.plugins.tgs import TaskGenerator
    from geniesim.utils.name_utils import robot_type_mapping

    task_generator = TaskGenerator(task_config)
    task_folder = os.path.join(
        system_utils.benchmark_root_path(),
        "saved_task/%s" % task_config["task"],
    )
    gen_config = task_config.get("generalization", {})
    task_generator.generate_tasks(
        save_path=task_folder,
        task_name=task_config["task"],
        gen_config=gen_config,
    )

    # Robot config
    robot_cfg_name = task_config.get("robot", {}).get("robot_cfg", "G1_120s.json")
    robot_position = task_generator.robot_init_pose["position"]
    robot_rotation = task_generator.robot_init_pose["quaternion"]
    task_config["robot"]["robot_init_pose"]["position"] = robot_position
    task_config["robot"]["robot_init_pose"]["quaternion"] = robot_rotation
    task_config["robot_cfg"] = robot_type_mapping(robot_cfg_name.split(".")[0])

    import glob
    episode_files = sorted(glob.glob(task_folder + "/*.json"))
    if not episode_files:
        raise FileNotFoundError(f"No episode files found in {task_folder}")
    episode_file = episode_files[0]

    # sub_task scene USD (LLM-generated scene이 있을 경우)
    sub_usd_path = ""
    if sub_task_name:
        sub_usd_path = os.path.join(
            system_utils.benchmark_conf_path(),
            "llm_task", sub_task_name, "0", "scene.usda",
        )

    # ===== 3-5. 로봇 + 씬 로드 =====
    api_core.init_robot_cfg(
        robot_cfg_name,
        task_config["scene"]["scene_usd"],
        robot_position,
        robot_rotation,
        sub_usd_path,
    )
    time.sleep(0.5)
    api_core.collect_init_physics()

    # ===== 3-6. DataCourier (observation/joint state 브릿지) =====
    from geniesim.utils.data_courier import DataCourier

    data_courier = DataCourier(
        api_core, cfg.benchmark.enable_ros, cfg.benchmark.model_arc
    )
    data_courier.set_robot_cfg(task_config["robot_cfg"])

    # ===== 3-7. Environment (PiEnv — observation/action 인터페이스) =====
    from geniesim.benchmark.envs.pi_env import PiEnv

    env = PiEnv(api_core, episode_file, task_config)
    env.set_data_courier(data_courier)
    env.set_scene_info(None)

    # Episode content에서 robot base pose & light 적용
    episode_content = system_utils.load_json(episode_file)
    api_core.update_robot_base(
        episode_content["generalization_config"]["robot_init_pose"]["position"],
        episode_content["generalization_config"]["robot_init_pose"]["quaternion"],
    )
    api_core.apply_light_config(
        episode_content["generalization_config"]["light_config"]
    )

    # Task instruction 설정
    env.set_current_task(0)
    instruction = task_config.get("instruction", "")
    if instruction:
        env.task.set_instruction(instruction)
    task_instruction = env.task.get_instruction()[0]
    print(f"[Inference] Task instruction: {task_instruction}")

    # ===== 3-8. Policy (WebSocket → Inference Server) =====
    from geniesim.benchmark.policy.pipolicy import PiPolicy

    host, port = _parse_infer_host(cfg.benchmark.infer_host)
    print(f"[Inference] Connecting to policy server at {host}:{port}")

    policy = PiPolicy(
        task_name=task_name,
        host_ip=host,
        port=port,
        sub_task_name=sub_task_name,
    )
    policy.set_data_courier(data_courier)

    # ===== 3-9. Physics callback (원본 동일 — 매 physics step마다 호출) =====
    _frame_count = 0
    _last_time = time.time()

    def callback_physics(step_size):
        nonlocal _frame_count, _last_time
        _frame_count += 1
        now = time.time()
        elapsed = now - _last_time
        if elapsed >= 1.0:
            hz = _frame_count / elapsed
            print(f"[Physics] {hz:.2f} Hz")
            _frame_count = 0
            _last_time = now

        api_core.physics_step()
        api_core.on_ros_tick(step_size)

    ui_builder.my_world.add_physics_callback("on_physics", callback_fn=callback_physics)

    # ===== 3-10. Inference Worker Thread =====
    #
    # 원본에서 TaskManager.worker → benchmark_main → evaluate_episode 로
    # 이어지던 흐름을 단순화.
    # 핵심: obs = env.reset() → policy.act(obs) → env.step(action) 루프만 유지.
    #
    inference_done = threading.Event()

    def inference_worker():
        try:
            print("[Inference] Resetting environment...")
            obs = env.reset()
            print("[Inference] Environment reset complete. Starting inference loop.")

            max_steps = 3000  # 안전 상한 (필요 시 조정)
            while data_courier.loop_ok() and not inference_done.is_set():
                # Policy inference (WebSocket으로 observation 전송 → action 수신)
                action = policy.act(
                    obs,
                    step_num=env.current_step,
                    task_instruction=task_instruction,
                )

                # Environment step (action → sim에 적용 → 다음 observation)
                obs, done, need_update, task_progress = env.step(action)
                print(f"[Inference] Step {env.current_step}  done={done}")

                if done or env.current_step >= max_steps:
                    print(f"[Inference] Episode finished. done={done}, steps={env.current_step}")
                    break

                data_courier.sleep()

        except KeyboardInterrupt:
            print("[Inference] Interrupted by user.")
        except Exception as e:
            print(f"[Inference] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            inference_done.set()

    worker = threading.Thread(target=inference_worker, daemon=True)
    worker.start()

    # ===== 3-11. Main Sim Loop (원본과 동일 구조) =====
    step = 0
    while simulation_app.is_running():
        ui_builder.my_world.step(render=True)
        api_core.render_step()

        if api_core.exit or inference_done.is_set():
            api_core.post_process()
            break

        if not ui_builder.my_world.is_playing():
            if step % 100 == 0:
                print("**** simulation paused ****")
            step += 1
            continue

    simulation_app.close()
    print("[Inference] Simulation closed.")


def _parse_infer_host(infer_host: str):
    """Parse 'host:port' → (host, port). Default port 8999."""
    if ":" in infer_host:
        host, port_str = infer_host.rsplit(":", 1)
        return host.strip(), int(port_str.strip())
    return infer_host.strip(), 8999


if __name__ == "__main__":
    main()
