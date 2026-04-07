#!/usr/bin/env python3
"""Small regression helper to verify that guidance changes the generated video."""

import argparse
import hashlib
import sys

import cv2
import numpy as np


def read_video(path: str) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from video: {path}")
    return np.stack(frames, axis=0)


def sha256_digest(x: np.ndarray) -> str:
    return hashlib.sha256(x.tobytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Check that a guided MAGI run differs from the baseline run.")
    parser.add_argument("--baseline", required=True, help="Path to baseline video (e.g. guidance_scale=0).")
    parser.add_argument("--guided", required=True, help="Path to guided video (e.g. guidance_scale>0).")
    parser.add_argument(
        "--min-mean-diff",
        type=float,
        default=1e-6,
        help="Minimum mean absolute pixel difference required to treat the videos as different.",
    )
    args = parser.parse_args()

    baseline = read_video(args.baseline)
    guided = read_video(args.guided)

    if baseline.shape != guided.shape:
        print(f"FAIL: video shapes differ: baseline={baseline.shape}, guided={guided.shape}")
        return 1

    baseline_hash = sha256_digest(baseline)
    guided_hash = sha256_digest(guided)

    abs_diff = np.abs(baseline.astype(np.float32) - guided.astype(np.float32))
    mean_diff = float(abs_diff.mean())
    max_diff = int(abs_diff.max())
    changed_pixels = int(np.count_nonzero(abs_diff))

    print(f"baseline_sha256={baseline_hash}")
    print(f"guided_sha256={guided_hash}")
    print(f"shape={baseline.shape}")
    print(f"mean_abs_diff={mean_diff:.8f}")
    print(f"max_abs_diff={max_diff}")
    print(f"changed_pixels={changed_pixels}")

    if baseline_hash == guided_hash or mean_diff <= args.min_mean_diff:
        print("FAIL: guidance had no measurable effect on the generated video.")
        return 1

    print("PASS: guided output differs from the baseline output.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
