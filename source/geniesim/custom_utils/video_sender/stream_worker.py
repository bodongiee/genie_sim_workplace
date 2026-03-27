# =============================================================================
#Background thread that picks up RGBA frames from a queue,
# =============================================================================
import queue
import threading
import time

import numpy as np

from .encoder import H264Encoder
from .sender import FrameSender


class StreamWorker(threading.Thread):

    def __init__(self, encoder: H264Encoder, sender: FrameSender, target_ip: str, target_port: int, max_queue: int = 2):
        super().__init__(daemon=True)
        self.encoder = encoder
        self.sender = sender
        self.target_ip = target_ip
        self.target_port = target_port
        self.q: queue.Queue = queue.Queue(maxsize=max_queue)
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
                print("[StreamWorker] Reconnecting...")
                self.sender = FrameSender(self.target_ip, self.target_port)
                if not self.sender.connect(retries=3, retry_interval=1.0):
                    continue
                self.encoder.frame_id = 0

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
                    print('\033[42m\033[37m' + f"[StreamWorker] Streamed {self.frame_count} frames " f"(queue={self.q.qsize()})" + '\033[0m')
            except Exception as e:
                print(f"[StreamWorker] Encode/send error: {e}")

    def submit(self, rgba: np.ndarray):
        try:
            self.q.put_nowait(rgba)
        except queue.Full:
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
