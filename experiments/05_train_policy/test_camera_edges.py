#!/usr/bin/env python3
# Copyright (c) 2026, Microduck Robotics Project
# Standalone OpenCV 480p Camera Edge Extraction Test Script

"""Standalone runner to verify real-time 480p OpenCV Canny Edge extraction.

Usage:
    DISPLAY=:0 cd ~/robotics/IsaacLab && uv run python /home/shubham/robotics/isaac-sim-microduck/experiments/05_train_policy/test_camera_edges.py
"""

import os
import sys
import time
from pathlib import Path

# Add paths
EXP_04 = Path(__file__).resolve().parent.parent / "04_rl_env"
if str(EXP_04) not in sys.path:
    sys.path.append(str(EXP_04))

import cv2
import numpy as np
from microduck_camera import MicroduckCameraEdgeFeed


def main():
    print("=" * 70)
    print(" MICRODUCK REAL-TIME 480p OPENCV CAMERA EDGE FEED TEST")
    print("=" * 70)
    print(" Initializing 480p camera edge pipeline (Canny detector + 8x8 spatial grid)...")

    feed = MicroduckCameraEdgeFeed(camera_id=0, width=640, height=480, target_fps=30, enable_display=True)

    print("\n Controls in OpenCV Window:")
    print("   [TAB] or [V] : Toggle Views (Side-by-Side <-> Pure Edges <-> Overlay)")
    print("   [+] / [-]    : Adjust Canny Edge Detection Threshold")
    print("   [S]          : Save Edge Snapshot to Disk")
    print("   [Q] / [ESC]  : Exit Test")
    print("=" * 70 + "\n")

    time.sleep(0.5)
    step = 0
    t_start = time.perf_counter()

    try:
        while True:
            key = feed.update_display()
            if key in (ord("q"), ord("Q"), 27):  # 'q' or ESC
                break
            elif key in (ord("s"), ord("S")):
                img = feed.get_display_image()
                save_path = Path(__file__).resolve().parent / "edge_snapshot.jpg"
                cv2.imwrite(str(save_path), img)
                print(f"\r[SNAPSHOT SAVED] -> {save_path.name}                             ", flush=True)

            features = feed.get_edge_features()
            step += 1

            if step % 30 == 0:
                elapsed = time.perf_counter() - t_start
                rate = step / elapsed
                mean_density = float(features.mean())
                max_density = float(features.max())
                print(
                    f"\r[STATUS] Stream: {feed.fps:4.1f} FPS | Display: {rate:4.1f} Hz | "
                    f"Edge Grid (8x8): Mean={mean_density:.3f}, Max={max_density:.3f} | "
                    f"View: '{feed.view_mode}'",
                    end="",
                    flush=True,
                )

            time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        feed.close()
        print("\n\n[INFO]: OpenCV Camera Edge feed cleanly released.")


if __name__ == "__main__":
    main()
