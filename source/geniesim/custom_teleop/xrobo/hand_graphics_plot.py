# Hand Gesture - 3D Skeleton Visualization
#"leftHand":{"isActive":0,"count":26,"scale":1.0,"timeStampNs":1732613438765715200,"HandJointLocations":[{"p":"0,0,0,0,0,0,0","s":0.0,"r":0.0}, ...]},
#"rightHand":{"isActive":0,"count":26,"scale":1.0,"HandJointLocations":[{"p":"0,0,0,0,0,0,0","s":0.0,"r":0.0}, ...]}

import numpy as np
import matplotlib.pyplot as plt

from xrobotoolkit_teleop.common.xr_client import XrClient
# PICO joint index -> MediaPipe joint index mapping
PICO_TO_MEDIAPIPE = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4,
    7: 5, 8: 6, 9: 7, 10: 8,
    12: 9, 13: 10, 14: 11, 15: 12,
    17: 13, 18: 14, 19: 15, 20: 16,
    22: 17, 23: 18, 24: 19, 25: 20,
}


def pico_hand_state_to_mediapipe(hand_state: np.ndarray) -> np.ndarray:
    """Convert PICO hand state (27, 7) to MediaPipe keypoints (21, 3), centered at wrist."""
    mediapipe_state = np.zeros((21, 3), dtype=float)
    for pico_idx, mp_idx in PICO_TO_MEDIAPIPE.items():
        mediapipe_state[mp_idx] = hand_state[pico_idx, :3]
    return mediapipe_state - mediapipe_state[0:1, :]


# MediaPipe hand skeleton connections (joint index pairs)
# Finger definitions: thumb, index, middle, ring, pinky
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm cross connections
    (5, 9), (9, 13), (13, 17),
]


# --- Finger joint chains (MediaPipe indices) ---
# Each chain: sequence of joints from base to tip
FINGER_CHAINS = {
    "thumb":  [0, 1, 2, 3, 4],       # Wrist→CMC→MCP→IP→TIP
    "index":  [0, 5, 6, 7, 8],       # Wrist→MCP→PIP→DIP→TIP
    "middle": [0, 9, 10, 11, 12],    # Wrist→MCP→PIP→DIP→TIP
    "ring":   [0, 13, 14, 15, 16],   # Wrist→MCP→PIP→DIP→TIP
    "pinky":  [0, 17, 18, 19, 20],   # Wrist→MCP→PIP→DIP→TIP
}

# Joint angle names per finger
JOINT_ANGLE_NAMES = {
    "thumb":  ["CMC", "MCP", "IP"],
    "index":  ["MCP", "PIP", "DIP"],
    "middle": ["MCP", "PIP", "DIP"],
    "ring":   ["MCP", "PIP", "DIP"],
    "pinky":  ["MCP", "PIP", "DIP"],
}

# Finger tip indices for inter-finger angle calculation
FINGER_TIPS = {
    "thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20,
}
FINGER_MCPS = {
    "thumb": 1, "index": 5, "middle": 9, "ring": 13, "pinky": 17,
}

# Adjacent finger pairs for inter-finger (spread) angles
FINGER_PAIRS = [
    ("thumb", "index"),
    ("index", "middle"),
    ("middle", "ring"),
    ("ring", "pinky"),
]


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle (degrees) between two vectors."""
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def compute_joint_angles(kp: np.ndarray) -> dict:
    """
    Compute all finger joint angles (flexion) from MediaPipe (21,3) keypoints.
    For each 3-joint chain A→B→C, the angle at B = angle between vectors BA and BC.
    Returns dict: { "thumb_CMC": deg, "thumb_MCP": deg, ... }
    """
    angles = {}
    for finger, chain in FINGER_CHAINS.items():
        names = JOINT_ANGLE_NAMES[finger]
        # 3 angles from 5 joints: joints[0-1-2], [1-2-3], [2-3-4]
        for i, name in enumerate(names):
            a, b, c = kp[chain[i]], kp[chain[i + 1]], kp[chain[i + 2]]
            angles[f"{finger}_{name}"] = angle_between_vectors(a - b, c - b)
    return angles


def compute_inter_finger_angles(kp: np.ndarray) -> dict:
    """
    Compute spread angles between adjacent fingers.
    Thumb-index: MCP→TIP vectors (captures actual thumb spread).
    Others: Wrist→MCP vectors.
    Returns dict: { "thumb_index": deg, "index_middle": deg, ... }
    """
    angles = {}
    wrist = kp[0]
    for f1, f2 in FINGER_PAIRS:
        if f1 == "thumb":
            v1 = kp[FINGER_TIPS[f1]] - kp[FINGER_MCPS[f1]]
            v2 = kp[FINGER_TIPS[f2]] - kp[FINGER_MCPS[f2]]
        else:
            v1 = kp[FINGER_MCPS[f1]] - wrist
            v2 = kp[FINGER_MCPS[f2]] - wrist
        angles[f"{f1}_{f2}"] = angle_between_vectors(v1, v2)
    return angles


def compute_all_angles(kp: np.ndarray) -> tuple[dict, dict]:
    """Compute both joint angles and inter-finger angles."""
    return compute_joint_angles(kp), compute_inter_finger_angles(kp)


def draw_hand(ax, keypoints, color, label):
    """Draw a hand skeleton on the 3D axes."""
    if keypoints is None:
        return
    # Draw joints
    ax.scatter(
        keypoints[:, 0], keypoints[:, 1], keypoints[:, 2],
        c=color, s=20, depthshade=True, label=label,
    )
    # Draw bones
    for i, j in HAND_CONNECTIONS:
        ax.plot(
            [keypoints[i, 0], keypoints[j, 0]],
            [keypoints[i, 1], keypoints[j, 1]],
            [keypoints[i, 2], keypoints[j, 2]],
            c=color, linewidth=1.5,
        )


def main():
    """
    Bimanual hand tracking with 3D matplotlib skeleton visualization.
    """
    xr_client = XrClient()

    plt.ion()
    fig = plt.figure(figsize=(25, 18))
    ax_3d = fig.add_subplot(121, projection="3d")
    ax_txt = fig.add_subplot(122)
    ax_txt.axis("off")
    fig.suptitle("Bimanual Hand Tracking")

    print("Bimanual hand tracking started. Close the plot window to stop.")

    left_keypoints = None
    right_keypoints = None

    # Offset to separate left/right hands on X axis
    hand_offset = np.array([0.4, 0.0, 0.0])

    while plt.fignum_exists(fig.number):
        # Read left hand
        left_state = xr_client.get_hand_tracking_state("left")
        if left_state is not None:
            left_state = np.array(left_state)
            if not np.all(left_state == 0):
                left_keypoints = pico_hand_state_to_mediapipe(left_state) - hand_offset

        # Read right hand
        right_state = xr_client.get_hand_tracking_state("right")
        if right_state is not None:
            right_state = np.array(right_state)
            if not np.all(right_state == 0):
                right_keypoints = pico_hand_state_to_mediapipe(right_state) + hand_offset

        # Redraw
        ax_3d.cla()
        ax_3d.set_xlabel("X")
        ax_3d.set_ylabel("Y")
        ax_3d.set_zlabel("Z")
        ax_3d.set_title("Hand Skeleton")

        draw_hand(ax_3d, left_keypoints, color="blue", label="Left")
        draw_hand(ax_3d, right_keypoints, color="red", label="Right")

        # Auto-scale axes symmetrically
        all_pts = [p for p in [left_keypoints, right_keypoints] if p is not None]
        if all_pts:
            pts = np.concatenate(all_pts, axis=0)
            center = pts.mean(axis=0)
            max_range = np.abs(pts - center).max() * 1.2
            ax_3d.set_xlim(center[0] - max_range, center[0] + max_range)
            ax_3d.set_ylim(center[1] - max_range, center[1] + max_range)
            ax_3d.set_zlim(center[2] - max_range, center[2] + max_range)
            ax_3d.legend(loc="upper left")

        # Compute and display angles
        angle_lines = []
        for hand_name, kp in [("Left", left_keypoints), ("Right", right_keypoints)]:
            if kp is None:
                continue
            joint_angles, finger_angles = compute_all_angles(kp)
            angle_lines.append(f"=== {hand_name} Hand ===")
            angle_lines.append("Joint angles:")
            for key, val in joint_angles.items():
                angle_lines.append(f"  {key:16s}: {val:6.1f}°")
            angle_lines.append("Inter-finger angles:")
            for key, val in finger_angles.items():
                angle_lines.append(f"  {key:16s}: {val:6.1f}°")

        ax_txt.cla()
        ax_txt.axis("off")
        if angle_lines:
            ax_txt.text(
                0.02, 0.98, "\n".join(angle_lines),
                transform=ax_txt.transAxes,
                fontsize=9, fontfamily="monospace",
                verticalalignment="top",
            )

        plt.draw()
        plt.pause(0.03)

    xr_client.close()
    print("Done.")


if __name__ == "__main__":
    main()
