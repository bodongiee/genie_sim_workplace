# =============================================================================
# Send H.264 packet to PICO VR by TCP protocol
# =============================================================================
import socket
import struct
import time
from typing import Optional

class FrameSender:
    def __init__(self, ip: str, port: int):
        self.ip = ip
        self.port = port
        self._sock: Optional[socket.socket] = None

    def connect(self, retries: int = 10, retry_interval: float = 2.0) -> bool:
        for i in range(retries):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)
                s.settimeout(5.0)
                s.connect((self.ip, self.port))
                s.settimeout(None)
                self._sock = s
                print('\033[42m\033[37m' + f"[FrameSender] Connected to {self.ip}:{self.port}" + '\033[0m]')
                return True
            except OSError as e:
                print(f"[FrameSender] Connect attempt {i + 1}/{retries} -> {e}")
                time.sleep(retry_interval)
        return False

    def send_frame(self, data: bytes) -> bool:
        # Send a length-prefixed frame. Returns False on failure.
        if not self._sock:
            return False
        try:
            self._sock.sendall(struct.pack(">I", len(data)) + data)
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
    def connected(self) -> bool:
        return self._sock is not None
