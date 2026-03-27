import numpy as np
from typing import Optional, Union
import time

# MediaPipe 21-joint hand landmark indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20


class OneEuroFilter:
    # ===== One Euro Filter for a single (N, D) signal. =====
    # x_filtered = alpha * x_raw + (1 - alpha) * x_1_filtered
    # alpha → dynamic smoothing
    # tau = 1 / (2 * pi * cutoff_freq)
    # alpha = 1 / (1 + tau / frame_time)
    def __init__(self, n_joints: int, n_dims: int, min_cutoff: float = 1.0, beta: float = 0.05, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self._x_prev: Optional[np.ndarray] = None
        self._dx_prev: Optional[np.ndarray] = None
        self._t_prev: Optional[float] = None

    def reset(self):
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None

    def _smoothing_factor(self, te: float, cutoff: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: np.ndarray, timestamp: Optional[float] = None) -> np.ndarray:
        t = timestamp if timestamp is not None else time.time()

        if self._x_prev is None:
            self._x_prev = x.copy()
            self._dx_prev = np.zeros_like(x)
            self._t_prev = t
            return x.copy()

        te = t - self._t_prev
        if te <= 0:
            te = 1e-5
        self._t_prev = t

        dx = (x - self._x_prev) / te
        a_d = self._smoothing_factor(te, self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        speed = np.linalg.norm(dx_hat, axis=-1, keepdims=True)
        cutoff = self.min_cutoff + self.beta * speed

        a = self._smoothing_factor(te, cutoff)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev = x_hat.copy()
        self._dx_prev = dx_hat.copy()

        return x_hat


class HandTrackingFilter:
    # Pipeline:
    # 1. Hand size normalization (my hand size → robot hand size)
    # 2. Velocity clamp
    # 3. One Euro Filter
    def __init__(self, max_velocity: float = 5.0, min_cutoff: float = 1.0, beta: float = 0.05,
                 user_hand_size: Optional[float] = 0.2, ref_hand_size: float = 0.218):
        self.max_velocity = max_velocity

        if user_hand_size is not None and user_hand_size > 0:
            self._hand_scale = ref_hand_size / user_hand_size
        else:
            self._hand_scale = None

        self._prev_positions: Optional[np.ndarray] = None
        self._one_euro = OneEuroFilter(n_joints=21, n_dims=3, min_cutoff=min_cutoff, beta=beta)

    def reset(self):
        self._prev_positions = None
        self._one_euro.reset()

    def filter(self, hand_positions: np.ndarray) -> np.ndarray:
        assert hand_positions.shape == (21, 3), f"Expected (21,3), got {hand_positions.shape}"

        pos = hand_positions.copy()

        # 1. Hand size normalization
        if self._hand_scale is not None:
            pos = pos * self._hand_scale

        # 2. Velocity clamp
        pos = self._velocity_clamp(pos)

        # 3. One Euro Filter
        pos = self._one_euro.filter(pos)

        self._prev_positions = pos.copy()
        return pos

    def _velocity_clamp(self, pos: np.ndarray) -> np.ndarray:
        if self._prev_positions is None:
            return pos
        delta = pos - self._prev_positions
        distances = np.linalg.norm(delta, axis=1, keepdims=True)
        scale = np.where(distances > self.max_velocity, self.max_velocity / (distances + 1e-8), 1.0)
        return self._prev_positions + delta * scale
