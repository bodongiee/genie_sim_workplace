# =============================================================================
# Load background and assets only (no robot)
# =============================================================================
import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--physics-dt", type=float, default=1.0 / 120.0)
parser.add_argument("--rendering-dt", type=float, default=1.0 / 60.0)
args, _unknown = parser.parse_known_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless, "renderer": "RaytracedLighting"})

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from pxr import UsdPhysics, PhysxSchema, Sdf, Gf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    world = World(stage_units_in_meters=1.0, physics_dt=args.physics_dt, rendering_dt=args.rendering_dt)

    for _ in range(10):
        world.step(render=True)

    # Physics scene
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

    # Light
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
    #NOTE : White room with white desk -0.5
    scene_usd_path = os.path.join(ASSETS_DIR, "background/room/room_1/background.usda")
    #NOTE : Basic room with desk and shelf
    #scene_usd_path = os.path.join(ASSETS_DIR, "background/study_room/study_4/background.usda")
    #NOTE : KITCHEN 
    #scene_usd_path = os.path.join(ASSETS_DIR, "background/kitchen/kitchen_1/background.usda")
    
    add_reference_to_stage(usd_path=scene_usd_path, prim_path="/World")

    # Load sub_task scene (table + colored cubes)
    BENCHMARK_CFG_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../../benchmark/config"))
    TEST_ENV_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../../test_environment"))
    #sub_task_usd = os.path.join(BENCHMARK_CFG_DIR, "llm_task", "clean_the_desktop", "0", "scene.usda")
    #sub_task_usd = os.path.join(BENCHMARK_CFG_DIR, "llm_task", "bimanual_hold_ball", "0", "scene.usda")
    sub_task_usd = os.path.join(BENCHMARK_CFG_DIR, "llm_task", "straighten_object", "0", "scene.usda")
    #sub_task_usd = os.path.join(TEST_ENV_DIR, "kitchen", "scene.usda")

    add_reference_to_stage(usd_path=sub_task_usd, prim_path="/Workspace")
    print(f"Loaded sub_task scene: {sub_task_usd}")

    # KAPEX robot (Z축 90도 회전: qw=cos(45°), qz=sin(45°))
    kapex_usd = os.path.join(SCRIPT_DIR, "KAPEX.usdc")
    add_reference_to_stage(usd_path=kapex_usd, prim_path="/KAPEX")
    SingleXFormPrim(
        prim_path="/KAPEX",
        orientation=np.array([np.cos(np.pi / 4), 0.0, 0.0, -np.sin(np.pi / 4)]),  # [qw, qx, qy, qz]
    )
    print(f"Loaded KAPEX: {kapex_usd}")

    world.play()

    while simulation_app.is_running():
        world.step(render=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
