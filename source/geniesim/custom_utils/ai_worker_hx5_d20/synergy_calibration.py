#  python workspace/synergy_calibration.py
#  python workspace/synergy_calibration.py --hand_type left
#  python workspace/synergy_calibration.py --output workspace/synergy_calibration.yaml
import os
import time

import numpy as np
import tyro
import yaml

from xrobotoolkit_teleop.common.xr_client import XrClient

# PICO finger chains: [metacarpal, proximal, intermediate, distal, tip]
PICO_FINGER_CHAINS = {
    "middle": [11, 12, 13, 14, 15],
    "ring": [16, 17, 18, 19, 20],
    "pinky": [21, 22, 23, 24, 25],
}

PICO_THUMB_TIP = 5
PICO_INDEX_TIP = 10


def angle_between_vectors(v1, v2):
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def compute_grip_mcp_angles(pico_pos):
    angles = {}
    for finger, chain in PICO_FINGER_CHAINS.items():
        # MCP angle: metacarpal → proximal → intermediate
        a = pico_pos[chain[0]]  # metacarpal
        b = pico_pos[chain[1]]  # proximal
        c = pico_pos[chain[2]]  # intermediate
        angles[finger] = angle_between_vectors(a - b, c - b)
    return angles


def compute_pinch_dist(pico_pos):
    # Thumb tip to Index Tip
    return float(np.linalg.norm(pico_pos[PICO_THUMB_TIP] - pico_pos[PICO_INDEX_TIP]))

def collect_samples(xr_client, hand_type, prompt_msg, n_samples=50, interval=0.02):
    # Collect Frames
    print(f"\n>>> {prompt_msg}")
    print(f"    Press Enter if you are ready (After {n_samples} frames will be collected")
    input()

    samples = []
    missed = 0
    while len(samples) < n_samples:
        hand_state = xr_client.get_hand_tracking_state(hand_type)
        if hand_state is not None:
            pico_pos = np.array(hand_state)[:, :3]
            samples.append(pico_pos)
        else:
            missed += 1
            if missed > 200:
                print("Error")
                missed = 0
        time.sleep(interval)

    print(f"    {len(samples)} frames are collected")
    return samples


def compute_stats(values):
    arr = np.array(values)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr)), "min": float(np.min(arr)), "max": float(np.max(arr)), }


def main(
    hand_type: str = "right",
    output: str = os.path.join(os.path.dirname(__file__), "synergy_calibration.yaml"),
    n_samples: int = 50,
):
    xr_client = XrClient()

    # Wait for first valid frame
    while xr_client.get_hand_tracking_state(hand_type) is None:
        time.sleep(0.1)
    print("VR Connected\n")

    # =========================================================================
    # 1. Open hand
    # =========================================================================
    open_samples = collect_samples(
        xr_client, hand_type,
        "[1/3] OPEN your hand",
        n_samples=n_samples,
    )

    open_pinch_dists = [compute_pinch_dist(s) for s in open_samples]
    open_mcp_angles = {f: [] for f in PICO_FINGER_CHAINS}
    for s in open_samples:
        angles = compute_grip_mcp_angles(s)
        for f in PICO_FINGER_CHAINS:
            open_mcp_angles[f].append(angles[f])

    # =========================================================================
    # 2. Pinch
    # =========================================================================
    pinch_samples = collect_samples(
        xr_client, hand_type,
        "[2/3] PINCH your hand (Thumb, Middle)",
        n_samples=n_samples,
    )

    pinch_dists = [compute_pinch_dist(s) for s in pinch_samples]

    # =========================================================================
    # 3. Grip
    # =========================================================================
    grip_samples = collect_samples(
        xr_client, hand_type,
        "[3/3] GRIP your hand",
        n_samples=n_samples,
    )

    grip_mcp_angles = {f: [] for f in PICO_FINGER_CHAINS}
    for s in grip_samples:
        angles = compute_grip_mcp_angles(s)
        for f in PICO_FINGER_CHAINS:
            grip_mcp_angles[f].append(angles[f])

    # =========================================================================
    # Summary
    # =========================================================================
    pinch_open_stats = compute_stats(open_pinch_dists)
    pinch_close_stats = compute_stats(pinch_dists)

    grip_open_mcp_stats = {f: compute_stats(open_mcp_angles[f]) for f in PICO_FINGER_CHAINS}
    grip_close_mcp_stats = {f: compute_stats(grip_mcp_angles[f]) for f in PICO_FINGER_CHAINS}

    # =========================================================================
    # Summary show
    # =========================================================================
    print("\n=== Calibration Results ===")
    print(f"  pinch_open_dist:  {pinch_open_stats['mean']:.4f}m (std={pinch_open_stats['std']:.4f})")
    print(f"  pinch_close_dist: {pinch_close_stats['mean']:.4f}m (std={pinch_close_stats['std']:.4f})")
    for f in PICO_FINGER_CHAINS:
        o = grip_open_mcp_stats[f]["mean"]
        c = grip_close_mcp_stats[f]["mean"]
        print(f"  {f}_mcp: open={o:.1f}° close={c:.1f}°")

    # =========================================================================
    # YAML
    # =========================================================================
    result = {
        "synergy_calibration": {
            "hand_type": hand_type,
            "pinch": {
                "open_dist": round(pinch_open_stats["mean"], 4),
                "close_dist": round(pinch_close_stats["mean"], 4),
                "open_dist_stats": pinch_open_stats,
                "close_dist_stats": pinch_close_stats,
            },
            "grip": {},
        }
    }

    for f in PICO_FINGER_CHAINS:
        result["synergy_calibration"]["grip"][f] = {
            "open_mcp": round(grip_open_mcp_stats[f]["mean"], 1),
            "close_mcp": round(grip_close_mcp_stats[f]["mean"], 1),
            "open_mcp_stats": grip_open_mcp_stats[f],
            "close_mcp_stats": grip_close_mcp_stats[f],
        }

    with open(output, "w") as f:
        yaml.dump(result, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"\nSAVED: {output}")


if __name__ == "__main__":
    tyro.cli(main)
