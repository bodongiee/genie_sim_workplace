"""
Isaac Sim → PICO VR video streamer.

Captures frames from an Isaac Sim camera, encodes H.264 via PyAV
(NVENC hardware encoder), and sends the bitstream over TCP to the
PICO VR headset using the XRoboToolkit wire protocol.

Encoding and sending run in a background thread so the teleop loop
is not blocked.

Wire protocol (per encoded frame):
    [4-byte big-endian length][H.264 NAL payload]
"""

import queue
import socket
import struct
import threading
import time
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# PyAV H.264 encoder  (NVENC → libx264 fallback)
# ---------------------------------------------------------------------------
class _H264Encoder:
    def __init__(self, width: int, height: int, fps: int, bitrate: int):
        import av
        from fractions import Fraction

        for codec_name in ("h264_nvenc", "libx264"):
            try:
                self._codec = av.codec.Codec(codec_name, "w")
                print(f"[VideoStreamer] Using encoder: {codec_name}")
                break
            except Exception:
                continue
        else:
            raise RuntimeError("No H.264 encoder available")

        self._ctx = av.codec.CodecContext.create(self._codec, "w")
        self._ctx.width = width
        self._ctx.height = height
        self._ctx.time_base = Fraction(1, fps)
        self._ctx.framerate = Fraction(fps, 1)
        self._ctx.pix_fmt = "yuv420p"
        self._ctx.bit_rate = bitrate
        self._ctx.gop_size = 15

        if "nvenc" in self._codec.name:
            self._ctx.options = {
                "preset": "p3",
                "tune": "ull",
                "zerolatency": "1",
                "rc": "cbr",
                "repeat_headers": "1",
            }
        else:
            self._ctx.options = {
                "preset": "ultrafast",
                "tune": "zerolatency",
            }
            self._ctx.flags &= ~av.codec.context.Flags.GLOBAL_HEADER

        self._ctx.open()
        self._frame_id = 0
        self._fps = fps
        self._av = av
        print(f"[VideoStreamer] Encoder ready: {width}x{height} @ {fps}fps, "
              f"{bitrate/1e6:.1f} Mbps, codec={self._codec.name}")

    def encode(self, rgba: np.ndarray) -> list:
        """Encode RGBA frame, return list of bytes packets."""
        h, w = rgba.shape[:2]
        frame = self._av.VideoFrame(w, h, "rgba")
        frame.planes[0].update(rgba.tobytes())
        frame.pts = self._frame_id
        self._frame_id += 1

        yuv_frame = frame.reformat(format="yuv420p")
        return [bytes(pkt) for pkt in self._ctx.encode(yuv_frame) if bytes(pkt)]

    def flush(self):
        return [bytes(pkt) for pkt in self._ctx.encode(None) if bytes(pkt)]

    def stop(self):
        try:
            self.flush()
        except Exception:
            pass
        self._ctx.close()


# ---------------------------------------------------------------------------
# TCP frame sender
# ---------------------------------------------------------------------------
class _FrameSender:
    def __init__(self, ip: str, port: int):
        self.ip = ip
        self.port = port
        self._sock: Optional[socket.socket] = None

    def connect(self, retries: int = 10, retry_interval: float = 2.0):
        for i in range(retries):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)
                s.settimeout(5.0)
                s.connect((self.ip, self.port))
                s.settimeout(None)
                self._sock = s
                print(f"[VideoStreamer] Connected to PICO {self.ip}:{self.port}")
                return True
            except OSError as e:
                print(f"[VideoStreamer] Connect attempt {i+1}/{retries} → {e}")
                time.sleep(retry_interval)
        return False

    def send_frame(self, h264_data: bytes):
        if not self._sock:
            return False
        try:
            self._sock.sendall(struct.pack(">I", len(h264_data)) + h264_data)
            return True
        except OSError:
            self.disconnect()
            return False

    def disconnect(self):
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    @property
    def connected(self):
        return self._sock is not None


# ---------------------------------------------------------------------------
# Background encode + send thread
# ---------------------------------------------------------------------------
class _StreamWorker(threading.Thread):
    """Picks up RGBA frames from a queue, encodes, and sends over TCP."""

    def __init__(self, encoder: _H264Encoder, sender: _FrameSender,
                 pico_ip: str, pico_port: int):
        super().__init__(daemon=True)
        self.encoder = encoder
        self.sender = sender
        self.pico_ip = pico_ip
        self.pico_port = pico_port
        self.q: queue.Queue = queue.Queue(maxsize=2)  # drop-if-full
        self._stop_event = threading.Event()
        self.frame_count = 0
        self._last_log = 0.0

    def run(self):
        while not self._stop_event.is_set():
            try:
                frame = self.q.get(timeout=0.5)
            except queue.Empty:
                continue

            # Reconnect if needed
            if not self.sender.connected:
                print("[VideoStreamer] Reconnecting...")
                self.sender = _FrameSender(self.pico_ip, self.pico_port)
                if not self.sender.connect(retries=3, retry_interval=1.0):
                    continue
                self.encoder._frame_id = 0

            # Encode and send
            try:
                packets = self.encoder.encode(frame)
                for pkt_data in packets:
                    if not self.sender.send_frame(pkt_data):
                        break
                self.frame_count += 1

                now = time.time()
                if now - self._last_log > 5.0:
                    self._last_log = now
                    print(f"[VideoStreamer] Streamed {self.frame_count} frames "
                          f"(queue={self.q.qsize()})")
            except Exception as e:
                print(f"[VideoStreamer] Encode/send error: {e}")

    def submit(self, rgba: np.ndarray):
        """Submit frame. Drops oldest if queue full (keeps latency low)."""
        try:
            self.q.put_nowait(rgba)
        except queue.Full:
            # Drop oldest, push newest
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.q.put_nowait(rgba)
            except queue.Full:
                pass

    def stop(self):
        self._stop_event.set()


# ---------------------------------------------------------------------------
# Isaac Sim camera helper
# ---------------------------------------------------------------------------
def create_head_camera(stage, robot_prim_path: str,
                       camera_prim_path: str = None,
                       resolution: tuple = (1280, 720)):
    from pxr import UsdGeom, Sdf, Gf, Usd

    if camera_prim_path is None:
        camera_prim_path = f"{robot_prim_path}/head_link2/zed/Head_Camera"

    cam_prim = stage.GetPrimAtPath(camera_prim_path)
    if cam_prim and cam_prim.IsValid():
        print(f"[VideoStreamer] Using existing camera: {camera_prim_path}")
    else:
        print(f"[VideoStreamer] Camera not found at {camera_prim_path}, creating...")
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

    print(f"[VideoStreamer] Camera ready: {camera_prim_path}  resolution={resolution}")
    return annotator, rp


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
class IsaacVideoStreamer:
    """
    Non-blocking video streamer. capture_and_send() is cheap — it just
    grabs the annotator buffer and hands it to a background thread that
    does the heavy lifting (color convert + encode + TCP send).
    """

    def __init__(self, stage, robot_prim_path: str,
                 pico_ip: str = "192.168.50.217",
                 pico_stream_port: int = 12345,
                 camera_prim_path: str = None,
                 resolution: tuple = (1280, 720),
                 fps: int = 30,
                 bitrate: int = 4_000_000,
                 send_every_n: int = 2):
        """
        Args:
            send_every_n: Only encode every Nth render frame.
                          2 = encode at half the render rate (e.g. 30fps encode
                          when rendering at 60fps). Keeps teleop loop fast.
        """
        self._resolution = resolution
        self._send_every_n = max(1, send_every_n)
        self._render_count = 0
        self._worker: Optional[_StreamWorker] = None

        # 1. Camera
        print(f"[VideoStreamer] Setting up camera...")
        self._annotator, self._rp = create_head_camera(
            stage, robot_prim_path,
            camera_prim_path=camera_prim_path,
            resolution=resolution)

        # 2. TCP connection
        print(f"[VideoStreamer] Connecting to PICO {pico_ip}:{pico_stream_port}...")
        sender = _FrameSender(pico_ip, pico_stream_port)
        if not sender.connect():
            print("[VideoStreamer] WARNING: Could not connect. Streaming disabled.")
            return

        # 3. Encoder
        w, h = resolution
        try:
            encoder = _H264Encoder(w, h, fps, bitrate)
        except Exception as e:
            print(f"[VideoStreamer] Encoder init failed: {e}")
            sender.disconnect()
            return

        # 4. Start background worker
        self._worker = _StreamWorker(encoder, sender, pico_ip, pico_stream_port)
        self._worker.start()
        print(f"[VideoStreamer] *** Streaming ACTIVE ***  "
              f"{w}x{h} @ {fps}fps → {pico_ip}:{pico_stream_port}  "
              f"(encode every {self._send_every_n} frames)")

    def capture_and_send(self):
        """Call every render frame. Cheap — just copies the buffer."""
        if not self._worker:
            return

        self._render_count += 1
        if self._render_count % self._send_every_n != 0:
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

            frame = np.ascontiguousarray(arr[:, :, :4], dtype=np.uint8)
            self._worker.submit(frame)
        except Exception as e:
            print(f"[VideoStreamer] Capture error: {e}")

    def shutdown(self):
        if self._worker:
            self._worker.stop()
            self._worker.join(timeout=3)
            if self._worker.sender:
                self._worker.sender.disconnect()
            self._worker.encoder.stop()
            count = self._worker.frame_count
            self._worker = None
            print(f"[VideoStreamer] Shutdown ({count} frames sent)")
