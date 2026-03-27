"""
video_sender — Reusable VR video streaming module.

Encodes RGBA frames to H.264 and streams them over TCP.
Independent of any specific simulator (Isaac Sim, MuJoCo, etc.).

Quick start:
    from custom_utils.video_sender import VRStreamer

    streamer = VRStreamer(
        target_ip="192.168.50.217",
        target_port=12345,
        width=1280, height=720,
    )
    # render loop
    streamer.send_frame(rgba_array)
    # cleanup
    streamer.shutdown()
"""

from .encoder import H264Encoder
from .sender import FrameSender
from .stream_worker import StreamWorker
from .vr_streamer import VRStreamer

# Isaac Sim specific (optional — only importable when Isaac Sim is available)
try:
    from .isaac_stereo_capture import IsaacStereoCapture
except ImportError:
    pass

__all__ = ["H264Encoder", "FrameSender", "StreamWorker", "VRStreamer", "IsaacStereoCapture"]
