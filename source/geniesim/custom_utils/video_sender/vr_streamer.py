# =============================================================================
# VR video streamer.
#
# Happen in a background thread so the caller is never blocked.
#
#Usage:
#    streamer = VRStreamer(
#        target_ip="192.168.50.217",
#        target_port=12345,
#        width=1280, height=720,
#    )
#    # In your render loop:
#    streamer.send_frame(rgba_numpy_array)
#    # On exit:
#    streamer.shutdown()
# =============================================================================

from typing import Optional

import numpy as np

from .encoder import H264Encoder
from .sender import FrameSender
from .stream_worker import StreamWorker


class VRStreamer:


    def __init__( self, target_ip: str, target_port: int, width: int, height: int, fps: int = 30, bitrate: int = 4_000_000, gop_size: int = 15,send_every_n: int = 1,):
        # FPS : Used by the encoder to arrange frames based on time
        # BITRATE : Total amount of datas when use in per sec / encoding

        self._width = width
        self._height = height
        self._send_every_n = max(1, send_every_n)
        self._call_count = 0
        self._worker: Optional[StreamWorker] = None

        # 1. TCP connection
        print('\033[42m\033[37m'+f"[VRStreamer] Connecting to {target_ip}:{target_port}..."+'\033[0m')
        sender = FrameSender(target_ip, target_port)
        if not sender.connect():
            print("[VRStreamer] WARNING: Could not connect. Streaming disabled.")
            return

        # 2. Encoder
        try:
            encoder = H264Encoder(width, height, fps, bitrate, gop_size)
        except Exception as e:
            print(f"[VRStreamer] Encoder init failed: {e}")
            sender.disconnect()
            return

        # 3. Start background worker
        self._worker = StreamWorker(encoder, sender, target_ip, target_port)
        self._worker.start()
        print('\033[42m\033[37m'+f"[VRStreamer] Streaming ACTIVE: {width}x{height} @ {fps}fps "
              f"-> {target_ip}:{target_port} (encode every {self._send_every_n} frames)"+'\033[0m')

    @property
    def active(self) -> bool:
        return self._worker is not None

    @property
    def frame_count(self) -> int:
        return self._worker.frame_count if self._worker else 0

    def send_frame(self, rgba: np.ndarray):

        if not self._worker:
            return

        self._call_count += 1
        if self._call_count % self._send_every_n != 0:
            return

        frame = np.ascontiguousarray(rgba[:, :, :4], dtype=np.uint8)
        self._worker.submit(frame)

    def shutdown(self):
        if self._worker:
            self._worker.stop()
            self._worker.join(timeout=3)
            if self._worker.sender:
                self._worker.sender.disconnect()
            self._worker.encoder.close()
            count = self._worker.frame_count
            self._worker = None
            print(f"[VRStreamer] Shutdown ({count} frames sent)")
