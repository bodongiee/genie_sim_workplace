# ── PICO 26-joint layout ──
# 0: Palm, 1: Wrist
# Thumb: 2(metacarpal), 3(proximal), 4(distal), 5(tip)
# Index: 6(metacarpal), 7(proximal), 8(intermediate), 9(distal), 10(tip)
# Middle: 11(metacarpal), 12(proximal), 13(intermediate), 14(distal), 15(tip)
# Ring: 16(metacarpal), 17(proximal), 18(intermediate), 19(distal), 20(tip)
# Pinky: 21(metacarpal), 22(proximal), 23(intermediate), 24(distal), 25(tip)

from pico_vr_hand_calibration import *
import numpy as np

# ===== Connections =====
HAND_CONNECTIONS = [
    # Thumb
    (1, 2), (2, 3), (3, 4), (4, 5),
    # Index
    (6, 7), (7, 8), (8, 9), (9, 10),
    # Middle
    (1, 11), (11, 12), (12, 13), (13, 14), (14, 15),
    # Ring
    (1, 16), (16, 17), (17, 18), (18, 19), (19, 20),
    # Pinky
    (1, 21), (21, 22), (22, 23), (23, 24), (24, 25),
    # Bridge
    (6, 11), (11, 16), (16, 21),
]

# ===== Finger Chain =====
PICO_FINGER_CHAINS = {
    "index" : [6, 7, 8, 9, 10],
    "middle" : [11, 12, 13, 14, 15],
    "ring" : [16, 17, 18, 19, 20],
    "pinky" : [21, 22, 23, 24, 25],
}

Joint_ANGLE_NAMES = {
    "index" : ["MCP", "PIP", "DIP"],
    "middle" : ["MCP", "PIP", "DIP"],
    "ring" : ["MCP", "PIP", "DIP"],
    "pinky" : ["MCP", "PIP", "DIP"],
}

CALIB_JOINT = {
    # Left
    "left_index_MCP":  (LEFT_INDEX_OPEN_MCP,  LEFT_INDEX_CLOSE_MCP),
    "left_index_PIP":  (LEFT_INDEX_OPEN_PIP,  LEFT_INDEX_CLOSE_PIP),
    "left_index_DIP":  (LEFT_INDEX_OPEN_DIP,  LEFT_INDEX_CLOSE_DIP),
    "left_middle_MCP": (LEFT_MIDDLE_OPEN_MCP, LEFT_MIDDLE_CLOSE_MCP),
    "left_middle_PIP": (LEFT_MIDDLE_OPEN_PIP, LEFT_MIDDLE_CLOSE_PIP),
    "left_middle_DIP": (LEFT_MIDDLE_OPEN_DIP, LEFT_MIDDLE_CLOSE_DIP),
    "left_ring_MCP":   (LEFT_RING_OPEN_MCP,   LEFT_RING_CLOSE_MCP),
    "left_ring_PIP":   (LEFT_RING_OPEN_PIP,   LEFT_RING_CLOSE_PIP),
    "left_ring_DIP":   (LEFT_RING_OPEN_DIP,   LEFT_RING_CLOSE_DIP),
    "left_pinky_MCP":  (LEFT_PINKY_OPEN_MCP,  LEFT_PINKY_CLOSE_MCP),
    "left_pinky_PIP":  (LEFT_PINKY_OPEN_PIP,  LEFT_PINKY_CLOSE_PIP),
    "left_pinky_DIP":  (LEFT_PINKY_OPEN_DIP,  LEFT_PINKY_CLOSE_DIP),
    # Right
    "right_index_MCP":  (RIGHT_INDEX_OPEN_MCP,  RIGHT_INDEX_CLOSE_MCP),
    "right_index_PIP":  (RIGHT_INDEX_OPEN_PIP,  RIGHT_INDEX_CLOSE_PIP),
    "right_index_DIP":  (RIGHT_INDEX_OPEN_DIP,  RIGHT_INDEX_CLOSE_DIP),
    "right_middle_MCP": (RIGHT_MIDDLE_OPEN_MCP, RIGHT_MIDDLE_CLOSE_MCP),
    "right_middle_PIP": (RIGHT_MIDDLE_OPEN_PIP, RIGHT_MIDDLE_CLOSE_PIP),
    "right_middle_DIP": (RIGHT_MIDDLE_OPEN_DIP, RIGHT_MIDDLE_CLOSE_DIP),
    "right_ring_MCP":   (RIGHT_RING_OPEN_MCP,   RIGHT_RING_CLOSE_MCP),
    "right_ring_PIP":   (RIGHT_RING_OPEN_PIP,   RIGHT_RING_CLOSE_PIP),
    "right_ring_DIP":   (RIGHT_RING_OPEN_DIP,   RIGHT_RING_CLOSE_DIP),
    "right_pinky_MCP":  (RIGHT_PINKY_OPEN_MCP,  RIGHT_PINKY_CLOSE_MCP),
    "right_pinky_PIP":  (RIGHT_PINKY_OPEN_PIP,  RIGHT_PINKY_CLOSE_PIP),
    "right_pinky_DIP":  (RIGHT_PINKY_OPEN_DIP,  RIGHT_PINKY_CLOSE_DIP),
}

# ===== Helper functions =====
def angle_between_vecotrs(v1, v2):
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) +1e-8)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

def compute_joint_angles(pico_pos):
    angles = {}
    for finger, chain in PICO_FINGER_CHAINS.items():
        for i, name in enumerate(Joint_ANGLE_NAMES[finger]):
            a, b, c = pico_pos[chain[i]], pico_pos[chain[i + 1]], pico_pos[chain[i + 2]]
            angles[f"{finger}_{name}"] = angle_between_vectors(a - b, c - b)
    return angles

def normalize(current, open_val, close_val):
    #0.0 = closed (fist), 1.0 = open (flat)
    rom = open_val - close_val
    if abs(rom) < 1e-3:
        return 0.5
    return float(np.clip((current - close_val) / rom, 0.0, 1.0))

