"""
Isaac Sim → VR video streamer.

Thin wrapper around custom_utils.video_sender that adds Isaac Sim
camera creation and annotator-based frame capture.

Usage:
    from isaac_video_streamer import IsaacVideoStreamer

    streamer = IsaacVideoStreamer(
        stage=stage,
        robot_prim_path="/my_robot",
        pico_ip="192.168.50.217",
        pico_stream_port=12345,
        resolution=(1280, 720),
    )
    # In your render loop (after world.step(render=True)):
    streamer.capture_and_send()
    # On exit:
    streamer.shutdown()
"""

import sys
import os
import numpy as np

# Ensure custom_utils is importable
_GENIESIM_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../.."))
if _GENIESIM_DIR not in sys.path:
    sys.path.insert(0, _GENIESIM_DIR)

from custom_utils.video_sender import VRStreamer


# ---------------------------------------------------------------------------
# Isaac Sim camera helper
# ---------------------------------------------------------------------------
def create_head_camera(stage, robot_prim_path: str,
                       camera_prim_path: str = None,
                       resolution: tuple = (1280, 720)):
    """Create or find a head-mounted camera and return an (annotator, render_product) pair."""
    from pxr import UsdGeom, Sdf, Gf, Usd

    if camera_prim_path is None:
        camera_prim_path = f"{robot_prim_path}/head_link2/zed/Head_Camera"

    cam_prim = stage.GetPrimAtPath(camera_prim_path)
    if cam_prim and cam_prim.IsValid():
        print(f"[IsaacVideoStreamer] Using existing camera: {camera_prim_path}")
    else:
        print(f"[IsaacVideoStreamer] Camera not found at {camera_prim_path}, creating...")
        robot_prim = stage.GetPrimAtPath(robot_prim_path)
        parent_path = None
        for prim in Usd.PrimRange(robot_prim):
            if prim.GetName() == "head_link2":
                parent_path = str(prim.GetPath())
                break
        if parent_path is None:
            raise ValueError(f"head_link2 not found under {robot_prim_path}")

        camera_prim_path = f"{parent_path}/vr_camera"
        camera = UsdGeom.Camera.Define(stage, Sdf.Path(camera_prim_path))
        camera.CreateFocalLengthAttr(8.484)
        camera.CreateHorizontalApertureAttr(20.955)
        camera.CreateVerticalApertureAttr(9.214)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.1, 9.0))
        camera.CreateProjectionAttr("perspective")

    import omni.replicator.core as rep
    rp = rep.create.render_product(camera_prim_path, resolution=resolution)
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach([rp])

    print(f"[IsaacVideoStreamer] Camera ready: {camera_prim_path}  resolution={resolution}")
    return annotator, rp


# ---------------------------------------------------------------------------
# Isaac Sim video streamer
# ---------------------------------------------------------------------------
class IsaacVideoStreamer:
    """
    Captures frames from an Isaac Sim camera annotator and streams
    them to a VR headset via the generic VRStreamer.
    """

    def __init__(self, stage, robot_prim_path: str,
                 pico_ip: str = "192.168.50.217",
                 pico_stream_port: int = 12345,
                 camera_prim_path: str = None,
                 resolution: tuple = (1280, 720),
                 fps: int = 30,
                 bitrate: int = 4_000_000,
                 send_every_n: int = 2):
        self._resolution = resolution

        # 1. Isaac Sim camera
        print(f"[IsaacVideoStreamer] Setting up camera...")
        self._annotator, self._rp = create_head_camera(
            stage, robot_prim_path,
            camera_prim_path=camera_prim_path,
            resolution=resolution)

        # 2. Generic VR streamer (handles encoding + TCP)
        self._streamer = VRStreamer(
            target_ip=pico_ip,
            target_port=pico_stream_port,
            width=resolution[0],
            height=resolution[1],
            fps=fps,
            bitrate=bitrate,
            send_every_n=send_every_n,
        )

    def capture_and_send(self):
        """Call every render frame. Grabs the annotator buffer and forwards it."""
        if not self._streamer.active:
            return

        try:
            data = self._annotator.get_data()
            if data is None:
                return
            if isinstance(data, dict):
                data = data.get("data", data)

            arr = np.asarray(data)
            if arr.size == 0:
                return

            w, h = self._resolution
            if arr.ndim == 1:
                expected = w * h * 4
                if arr.size == expected:
                    arr = arr.reshape((h, w, 4))
                else:
                    return

            self._streamer.send_frame(arr)
        except Exception as e:
            print(f"[IsaacVideoStreamer] Capture error: {e}")

    def shutdown(self):
        self._streamer.shutdown()
