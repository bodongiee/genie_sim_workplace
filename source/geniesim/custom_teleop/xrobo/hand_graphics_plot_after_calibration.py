# Bimanual Hand Tracking - Calibrated 3D Skeleton + Normalized Ratios

import numpy as np
import matplotlib.pyplot as plt

from xrobotoolkit_teleop.common.xr_client import XrClient
from hand_calibration import *

# ── PICO → MediaPipe conversion ──
PICO_TO_MEDIAPIPE = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4,
    7: 5, 8: 6, 9: 7, 10: 8,
    12: 9, 13: 10, 14: 11, 15: 12,
    17: 13, 18: 14, 19: 15, 20: 16,
    22: 17, 23: 18, 24: 19, 25: 20,
}


def pico_hand_state_to_mediapipe(hand_state: np.ndarray) -> np.ndarray:
    mediapipe_state = np.zeros((21, 3), dtype=float)
    for pico_idx, mp_idx in PICO_TO_MEDIAPIPE.items():
        mediapipe_state[mp_idx] = hand_state[pico_idx, :3]
    return mediapipe_state - mediapipe_state[0:1, :]


# ── Skeleton connections ──
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# ── Finger definitions ──
FINGER_CHAINS = {
    "thumb":  [0, 1, 2, 3, 4],
    "index":  [0, 5, 6, 7, 8],
    "middle": [0, 9, 10, 11, 12],
    "ring":   [0, 13, 14, 15, 16],
    "pinky":  [0, 17, 18, 19, 20],
}
JOINT_ANGLE_NAMES = {
    "thumb":  ["CMC", "MCP", "IP"],
    "index":  ["MCP", "PIP", "DIP"],
    "middle": ["MCP", "PIP", "DIP"],
    "ring":   ["MCP", "PIP", "DIP"],
    "pinky":  ["MCP", "PIP", "DIP"],
}
FINGER_MCPS = {"thumb": 1, "index": 5, "middle": 9, "ring": 13, "pinky": 17}
FINGER_TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
FINGER_PAIRS = [
    ("thumb", "index"), ("index", "middle"), ("middle", "ring"), ("ring", "pinky"),
]

# ── Calibration lookup tables (open / close) ──
# key format: "{side}_{finger}_{joint}" -> (open_val, close_val)
CALIB_JOINT = {
    # Left
    "left_thumb_CMC":  (LEFT_THUMB_OPEN_CMC,  LEFT_THUMB_CLOSE_CMC),
    "left_thumb_MCP":  (LEFT_THUMB_OPEN_MCP,  LEFT_THUMB_CLOSE_MCP),
    "left_thumb_IP":   (LEFT_THUMB_OPEN_IP,   LEFT_THUMB_CLOSE_IP),
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
    "right_thumb_CMC":  (RIGHT_THUMB_OPEN_CMC,  RIGHT_THUMB_CLOSE_CMC),
    "right_thumb_MCP":  (RIGHT_THUMB_OPEN_MCP,  RIGHT_THUMB_CLOSE_MCP),
    "right_thumb_IP":   (RIGHT_THUMB_OPEN_IP,   RIGHT_THUMB_CLOSE_IP),
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

CALIB_SPREAD = {
    "left_thumb_index":   (LEFT_THUMB_INDEX_OPEN,  LEFT_THUMB_INDEX_CLOSE),
    "left_index_middle":  (LEFT_INDEX_MIDDLE_OPEN, LEFT_INDEX_MIDDLE_CLOSE),
    "left_middle_ring":   (LEFT_MIDDLE_RING_OPEN,  LEFT_MIDDLE_RING_CLOSE),
    "left_ring_pinky":    (LEFT_RING_PINKY_OPEN,   LEFT_RING_PINKY_CLOSE),
    "right_thumb_index":  (RIGHT_THUMB_INDEX_OPEN,  RIGHT_THUMB_INDEX_CLOSE),
    "right_index_middle": (RIGHT_INDEX_MIDDLE_OPEN, RIGHT_INDEX_MIDDLE_CLOSE),
    "right_middle_ring":  (RIGHT_MIDDLE_RING_OPEN,  RIGHT_MIDDLE_RING_CLOSE),
    "right_ring_pinky":   (RIGHT_RING_PINKY_OPEN,   RIGHT_RING_PINKY_CLOSE),
}

# Tip-to-tip distance calibration (open_dist, close_dist)
CALIB_TIP_DIST = {
    "left_thumb_index":  (LEFT_THUMB_INDEX_TIP_DIST_OPEN,  LEFT_THUMB_INDEX_TIP_DIST_CLOSE),
    "right_thumb_index": (RIGHT_THUMB_INDEX_TIP_DIST_OPEN, RIGHT_THUMB_INDEX_TIP_DIST_CLOSE),
}


# ── Angle helpers ──
def angle_between_vectors(v1, v2):
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def compute_joint_angles(kp):
    angles = {}
    for finger, chain in FINGER_CHAINS.items():
        for i, name in enumerate(JOINT_ANGLE_NAMES[finger]):
            a, b, c = kp[chain[i]], kp[chain[i + 1]], kp[chain[i + 2]]
            angles[f"{finger}_{name}"] = angle_between_vectors(a - b, c - b)
    return angles


def compute_inter_finger_angles(kp):
    angles = {}
    wrist = kp[0]
    for f1, f2 in FINGER_PAIRS:
        if f1 == "thumb":
            # Thumb: use MCP→TIP vector (captures actual finger direction)
            v1 = kp[FINGER_TIPS[f1]] - kp[FINGER_MCPS[f1]]
            v2 = kp[FINGER_TIPS[f2]] - kp[FINGER_MCPS[f2]]
        else:
            # Other fingers: wrist→MCP vector
            v1 = kp[FINGER_MCPS[f1]] - wrist
            v2 = kp[FINGER_MCPS[f2]] - wrist
        angles[f"{f1}_{f2}"] = angle_between_vectors(v1, v2)
    return angles


def compute_tip_distances(kp):
    """Compute thumb-index tip-to-tip distance."""
    d = np.linalg.norm(kp[FINGER_TIPS["thumb"]] - kp[FINGER_TIPS["index"]])
    return {"thumb_index": d}


def normalize(current, open_val, close_val):
    """0.0 = closed (fist), 1.0 = open (flat)"""
    rom = open_val - close_val
    if abs(rom) < 1e-3:
        return 0.5
    return float(np.clip((current - close_val) / rom, 0.0, 1.0))


# ── Drawing ──
def draw_hand(ax, keypoints, color, label):
    if keypoints is None:
        return
    ax.scatter(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2],
               c=color, s=30, depthshade=True, label=label)
    for i, j in HAND_CONNECTIONS:
        ax.plot([keypoints[i, 0], keypoints[j, 0]],
                [keypoints[i, 1], keypoints[j, 1]],
                [keypoints[i, 2], keypoints[j, 2]],
                c=color, linewidth=2.0)


def draw_bar(ax_bar, ratios, title, color):
    """Draw horizontal bar chart of normalized ratios (0~1)."""
    ax_bar.cla()
    labels = list(ratios.keys())
    values = list(ratios.values())
    y_pos = np.arange(len(labels))

    bars = ax_bar.barh(y_pos, values, color=color, alpha=0.7, height=0.7)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(labels, fontsize=8, fontfamily="monospace")
    ax_bar.set_xlim(0.0, 1.0)
    ax_bar.set_xlabel("0=Closed  →  1=Open", fontsize=8)
    ax_bar.set_title(title, fontsize=11, fontweight="bold")
    ax_bar.invert_yaxis()

    # Show percentage on bars
    for bar, val in zip(bars, values):
        ax_bar.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{val:.0%}", va="center", fontsize=7)


# ── Main ──
def main():
    xr_client = XrClient()

    plt.ion()
    fig = plt.figure(figsize=(28, 16))
    fig.suptitle("Bimanual Hand Tracking (Calibrated)", fontsize=16, fontweight="bold")

    # Layout: [Left bars | 3D view | Right bars]
    ax_left_bar  = fig.add_subplot(141)
    ax_3d        = fig.add_subplot(142, projection="3d")
    ax_right_bar = fig.add_subplot(143)
    ax_txt       = fig.add_subplot(144)
    ax_txt.axis("off")
    fig.subplots_adjust(wspace=0.35)

    print("Calibrated hand tracking started. Close the window to stop.")

    left_keypoints = None
    right_keypoints = None
    hand_offset = np.array([0.4, 0.0, 0.0])

    while plt.fignum_exists(fig.number):
        # ── Read hands ──
        left_state = xr_client.get_hand_tracking_state("left")
        if left_state is not None:
            left_state = np.array(left_state)
            if not np.all(left_state == 0):
                left_keypoints = pico_hand_state_to_mediapipe(left_state) - hand_offset

        right_state = xr_client.get_hand_tracking_state("right")
        if right_state is not None:
            right_state = np.array(right_state)
            if not np.all(right_state == 0):
                right_keypoints = pico_hand_state_to_mediapipe(right_state) + hand_offset

        # ── 3D skeleton ──
        ax_3d.cla()
        ax_3d.set_xlabel("X")
        ax_3d.set_ylabel("Y")
        ax_3d.set_zlabel("Z")
        ax_3d.set_title("Hand Skeleton", fontsize=12)

        draw_hand(ax_3d, left_keypoints, color="royalblue", label="Left")
        draw_hand(ax_3d, right_keypoints, color="crimson", label="Right")

        # Draw thumb-index tip distance lines
        for kp, clr in [(left_keypoints, "royalblue"), (right_keypoints, "crimson")]:
            if kp is not None:
                t, idx = kp[4], kp[8]  # thumb_tip, index_tip
                ax_3d.plot([t[0], idx[0]], [t[1], idx[1]], [t[2], idx[2]],
                           c=clr, linewidth=1.0, linestyle="--", alpha=0.6)

        all_pts = [p for p in [left_keypoints, right_keypoints] if p is not None]
        if all_pts:
            pts = np.concatenate(all_pts, axis=0)
            center = pts.mean(axis=0)
            max_range = np.abs(pts - center).max() * 1.2
            ax_3d.set_xlim(center[0] - max_range, center[0] + max_range)
            ax_3d.set_ylim(center[1] - max_range, center[1] + max_range)
            ax_3d.set_zlim(center[2] - max_range, center[2] + max_range)
            ax_3d.legend(loc="upper left")

        # ── Compute angles & ratios ──
        angle_lines = []
        for side, kp, ax_bar, color in [
            ("left",  left_keypoints,  ax_left_bar,  "royalblue"),
            ("right", right_keypoints, ax_right_bar, "crimson"),
        ]:
            if kp is None:
                ax_bar.cla()
                ax_bar.set_title(f"{side.capitalize()} Hand", fontsize=11, fontweight="bold")
                ax_bar.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_bar.transAxes)
                ax_bar.set_xlim(0, 1)
                continue

            joint_angles = compute_joint_angles(kp)
            spread_angles = compute_inter_finger_angles(kp)
            tip_dists = compute_tip_distances(kp)

            ratios = {}
            angle_lines.append(f"=== {side.capitalize()} Hand ===")
            angle_lines.append("Joint angles (raw → ratio):")
            for key, val in joint_angles.items():
                calib_key = f"{side}_{key}"
                open_val, close_val = CALIB_JOINT[calib_key]
                r = normalize(val, open_val, close_val)
                ratios[key] = r
                angle_lines.append(f"  {key:16s}: {val:6.1f}° → {r:.0%}")

            angle_lines.append("Inter-finger (raw → ratio):")
            for key, val in spread_angles.items():
                calib_key = f"{side}_{key}"
                open_val, close_val = CALIB_SPREAD[calib_key]
                r = normalize(val, open_val, close_val)
                ratios[f"spread_{key}"] = r
                angle_lines.append(f"  {key:16s}: {val:6.1f}° → {r:.0%}")

            angle_lines.append("Tip distance (raw → pinch ratio):")
            for key, dist in tip_dists.items():
                calib_key = f"{side}_{key}"
                d_open, d_close = CALIB_TIP_DIST[calib_key]
                # pinch_ratio: 0=open, 1=touching (inverted from normalize)
                pinch = 1.0 - normalize(dist, d_open, d_close)
                ratios[f"pinch_{key}"] = pinch
                angle_lines.append(f"  {key:16s}: {dist:.4f}m → pinch {pinch:.0%}")

            draw_bar(ax_bar, ratios, f"{side.capitalize()} Hand", color)

        # ── Text panel ──
        ax_txt.cla()
        ax_txt.axis("off")
        if angle_lines:
            ax_txt.text(0.02, 0.98, "\n".join(angle_lines),
                        transform=ax_txt.transAxes,
                        fontsize=8, fontfamily="monospace",
                        verticalalignment="top")

        plt.draw()
        plt.pause(0.03)

    xr_client.close()
    print("Done.")


if __name__ == "__main__":
    main()
