import json
import time
import threading
import socket
import struct
import base64
import os


class DramaSocketSender:
    def __init__(self, host: str = "127.0.0.1", port: int = 1111, path: str = "/ws"):
        self.host = host
        self.port = port
        self.path = path
        self.url = f"ws://{host}:{port}{path}"
        self._sock = None
        self._lock = threading.Lock()
        self._connected = False
        self._recv_thread = None
        self._stop_event = threading.Event()

    def connect(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        self._sock.settimeout(5.0)
        self._sock.connect((self.host, self.port))

        # WebSocket handshake
        key = base64.b64encode(os.urandom(16)).decode()
        handshake = (
            f"GET {self.path} HTTP/1.1\r\n"
            f"Host: {self.host}:{self.port}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n"
            f"\r\n"
        )
        self._sock.sendall(handshake.encode())

        response = b""
        while b"\r\n\r\n" not in response:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise ConnectionError("WebSocket handshake failed: connection closed")
            response += chunk

        if b"101" not in response.split(b"\r\n")[0]:
            raise ConnectionError(f"WebSocket handshake failed: {response.decode()}")

        self._sock.settimeout(None)  # back to blocking for send
        self._connected = True
        self._stop_event.clear()

        # Background thread to handle server ping frames and detect disconnects
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()

        print(f"[DramaSocketSender] Connected to {self.url}")

    def _recv_loop(self):
        """Read incoming WebSocket frames (ping/close) and respond."""
        sock = self._sock
        try:
            sock.settimeout(1.0)
            while not self._stop_event.is_set():
                try:
                    head = sock.recv(2)
                    if not head:
                        break
                    opcode = head[0] & 0x0F
                    masked = (head[1] & 0x80) != 0
                    length = head[1] & 0x7F

                    if length == 126:
                        length = struct.unpack(">H", sock.recv(2))[0]
                    elif length == 127:
                        length = struct.unpack(">Q", sock.recv(8))[0]

                    if masked:
                        mask_key = sock.recv(4)
                        payload = bytearray(sock.recv(length))
                        for i in range(len(payload)):
                            payload[i] ^= mask_key[i % 4]
                    else:
                        payload = sock.recv(length) if length else b""

                    if opcode == 0x9:  # Ping
                        with self._lock:
                            self._send_ws_frame(bytes(payload), opcode=0xA)  # Pong
                    elif opcode == 0x8:  # Close
                        self._connected = False
                        break
                except socket.timeout:
                    continue
                except Exception:
                    break
        except Exception:
            pass
        self._connected = False

    def disconnect(self):
        self._stop_event.set()
        if self._sock:
            try:
                self._send_ws_frame(b"", opcode=0x8)
            except Exception:
                pass
            self._sock.close()
            self._sock = None
            self._connected = False
            print("[DramaSocketSender] Disconnected")

    def reconnect(self):
        """Disconnect and reconnect."""
        self.disconnect()
        time.sleep(0.5)
        self.connect()

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _send_ws_frame(self, data: bytes, opcode: int = 0x1):
        """Send a WebSocket frame (masked, as required by client)."""
        header = bytearray()
        header.append(0x80 | opcode)  # FIN + opcode

        length = len(data)
        if length < 126:
            header.append(0x80 | length)  # MASK bit set
        elif length < 65536:
            header.append(0x80 | 126)
            header.extend(struct.pack(">H", length))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack(">Q", length))

        mask = os.urandom(4)
        header.extend(mask)

        masked = bytearray(data)
        for i in range(len(masked)):
            masked[i] ^= mask[i % 4]

        self._sock.sendall(bytes(header) + bytes(masked))

    def send(
        self,
        left_arm=None,
        left_hand=None,
        left_controller=None,
        left_eef=None,
        right_arm=None,
        right_hand=None,
        right_controller=None,
        right_eef=None,
        left_activated=False,
        right_activated=False,
    ):
        msg = {
            "left": {
                "arm": left_arm or [0.0] * 7,
                "hand": left_hand or [0.0] * 20,
                "controller": left_controller or [0.0] * 7,
                "eef": left_eef or [0.0] * 7,
            },
            "right": {
                "arm": right_arm or [0.0] * 7,
                "hand": right_hand or [0.0] * 20,
                "controller": right_controller or [0.0] * 7,
                "eef": right_eef or [0.0] * 7,
            },
            "activated": {
                "left": left_activated,
                "right": right_activated,
            },
        }

        with self._lock:
            self._send_ws_frame(json.dumps(msg).encode("utf-8"))

    def send_dict(self, data: dict):
        with self._lock:
            self._send_ws_frame(json.dumps(data).encode("utf-8"))

    def send_loop(
        self,
        get_state_fn,
        hz: float = 10.0,
        stop_event: threading.Event = None,
    ):
        interval = 1.0 / hz
        if stop_event is None:
            stop_event = threading.Event()

        while not stop_event.is_set():
            t0 = time.time()
            state = get_state_fn()
            self.send(**state)
            elapsed = time.time() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False


class UdpSocketSender:
    """UDP sender for robot joint states.

    Sends a 20-float array as raw binary (little-endian):
        [left_arm x7] + [left_gripper x3] + [right_arm x7] + [right_gripper x3]
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5005):
        self.host = host
        self.port = port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._addr = (host, port)

    def send(
        self,
        left_arm,
        left_gripper,
        right_arm,
        right_gripper,
    ):

        data = list(left_arm) + list(left_gripper) + list(right_arm) + list(right_gripper)
        packet = struct.pack("<20f", *data)
        self._sock.sendto(packet, self._addr)

    def close(self):
        self._sock.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
