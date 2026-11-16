# Microduck RL — Complete Execution & Deployment Guide

This document contains step-by-step instructions, command references, architecture details, and troubleshooting steps to run, train, evaluate, and deploy the **Microduck Biped Robot Reinforcement Learning Policy** with **multimodal perception (8x8 ToF depth + 480p OpenCV Canny edge vision)**, **explosive hurdle jumping**, **aggressive locomotion (up to 0.75 m/s sprint & 2.5 rad/s turns)**, and **anti-fall emergency braking**.

---

## Table of Contents

1. [Hardware & Software Prerequisites](#1-hardware--software-prerequisites)
2. [Quick Setup & Cloning](#2-quick-setup--cloning)
3. [480p OpenCV Camera Edge Feed (Standalone Test)](#3-480p-opencv-camera-edge-feed-standalone-test)
4. [Reinforcement Learning Training (Workstation & Laptop)](#4-reinforcement-learning-training-workstation--laptop)
5. [Monitoring Training with TensorBoard](#5-monitoring-training-with-tensorboard)
6. [Interactive Playback & Keyboard Teleoperation](#6-interactive-playback--keyboard-teleoperation)
7. [Policy Architecture & Observation Breakdown](#7-policy-architecture--observation-breakdown)
8. [Policy Export & Real-Robot Deployment](#8-policy-export--real-robot-deployment)
9. [Troubleshooting & FAQ](#9-troubleshooting--faq)

---

## 1. Hardware & Software Prerequisites

* **Operating System:** Linux (Ubuntu 22.04 LTS or 24.04 LTS recommended)
* **Primary Target Workstation:**
  * **CPU:** Intel Core i9 (12th/13th/14th Gen or similar)
  * **RAM:** 64 GB
  * **GPU:** NVIDIA GeForce RTX 5070 Ti (16 GB VRAM) or RTX 4080/4090/A5000+
  * **NVIDIA Driver:** >= 550.x (CUDA >= 12.4 or 13.0 support)
* **Webcam:** Any USB / V4L2 480p/720p/1080p camera at `/dev/video0` (automatic synthetic fallback if no camera is plugged in)
* **Python Tooling:** `uv` package manager (ultra-fast virtual environment management)

---

## 2. Quick Setup & Cloning

### Step 2.1: Clone the Repositories
```bash
# Create base robotics directory
mkdir -p ~/robotics && cd ~/robotics

# 1. Clone NVIDIA Isaac Lab (Physics & Simulation Framework)
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# Install Isaac Lab dependencies using uv
uv sync
uv pip install opencv-python

# 2. Clone Microduck Project Repository
cd ~/robotics
git clone https://github.com/ShubhamManglaw/isaac-sim-microduck.git
# If already cloned, ensure latest commit:
cd ~/robotics/isaac-sim-microduck && git pull origin main
```

---

## 3. 480p OpenCV Camera Edge Feed (Standalone Test)

Test the OpenCV Canny edge extraction pipeline directly on your camera before starting RL training.

```bash
DISPLAY=:0 cd ~/robotics/IsaacLab
uv run python /home/shubham/robotics/isaac-sim-microduck/experiments/05_train_policy/test_camera_edges.py
```

### Controls in the OpenCV Window:
| Key | Action | Description |
| :--- | :--- | :--- |
| **`[TAB]` / `[V]`** | **Toggle View Modes** | Cycle between **Side-by-Side** (Raw + Edges), **Pure Glowing Edges**, and **Overlay** |
| **`[+]` / `[=]`** | **Increase Threshold** | Reduce edge sensitivity (filters minor textures) |
| **`[-]` / `[_]`** | **Decrease Threshold** | Increase edge sensitivity (picks up fine silhouettes) |
| **`[S]`** | **Save Snapshot** | Saves high-resolution snapshot `edge_snapshot.jpg` |
| **`[Q]` / `[ESC]`** | **Exit Test** | Cleanly closes camera thread and window |

> [!NOTE]
> If you are on a remote server or PC without a physical webcam, the pipeline automatically detects this and spawns a dynamic **synthetic 480p hallway scene** with obstacles so testing never fails.

---

## 4. Reinforcement Learning Training (Workstation & Laptop)

The training harness utilizes Isaac Lab's GPU Newton Solver (`MJWarpSolverCfg`) with Asymmetric Actor-Critic (PPO).

### Preset A: Workstation (RTX 5070 Ti / i9 / 64 GB RAM)
Runs **16,384 parallel environments** for **100,000 iterations** with 8 mini-batches:
```bash
cd ~/robotics/IsaacLab
uv run python /home/shubham/robotics/isaac-sim-microduck/experiments/05_train_policy/train.py --preset workstation
```

### Preset B: Laptop / Local GPU (RTX 3070/4060/4070)
Runs **4,096 parallel environments** for **30,000 iterations**:
```bash
cd ~/robotics/IsaacLab
uv run python /home/shubham/robotics/isaac-sim-microduck/experiments/05_train_policy/train.py --preset laptop
```

### Preset C: Fast 2-Iteration Smoke Test
Validates observation shapes, GPU buffers, and all 18 reward terms in 5 seconds:
```bash
cd ~/robotics/IsaacLab
uv run python /home/shubham/robotics/isaac-sim-microduck/experiments/05_train_policy/train.py --num_envs 16 --max_iterations 2
```

### Custom Overrides
You can customize any training run with CLI flags:
```bash
uv run python /path/to/train.py \
  --num_envs 8192 \
  --max_iterations 50000 \
  --experiment_name microduck_custom_run
```

---

## 5. Monitoring Training with TensorBoard

Monitor reward curves, forward velocity tracking, air time, jumping rewards, and episode lengths in real time:

```bash
cd ~/robotics/IsaacLab
uv run tensorboard --logdir /home/shubham/robotics/isaac-sim-microduck/experiments/05_train_policy/logs --port 6006
```
Open your browser at `http://localhost:6006`.

### Key Reward Curves to Watch:
* `Episode_Reward/track_lin_vel_xy_exp`: Forward sprint velocity tracking ($0.75\text{ m/s}$).
* `Episode_Reward/obstacle_jump`: Explosive vertical takeoff and landing reward.
* `Episode_Reward/sudden_stop_balance`: Anti-pitch damping during rapid deceleration.
* `Episode_Reward/directional_lin_gain`: Directional alignment gain.
* `Metrics/base_velocity/error_vel_xy`: Tracking error (should drop toward $0.05\text{ m/s}$).

---

## 6. Interactive Playback & Keyboard Teleoperation

Launch real-time 3D physics visualization with the interactive keyboard controller and live OpenCV edge feed window:

```bash
DISPLAY=:0 cd ~/robotics/IsaacLab
uv run python /home/shubham/robotics/isaac-sim-microduck/experiments/05_train_policy/play.py
```

*(By default, this automatically locates the latest trained checkpoint in the `logs/` folder).*

### To specify a specific checkpoint:
```bash
DISPLAY=:0 cd ~/robotics/IsaacLab
uv run python /home/shubham/robotics/isaac-sim-microduck/experiments/05_train_policy/play.py \
  --load_checkpoint /home/shubham/robotics/isaac-sim-microduck/experiments/05_train_policy/logs/latest/model_59900.pt
```

### Interactive Keyboard Controls:
| Key | Command | Action & Speed Range |
| :--- | :--- | :--- |
| **`[W]` / `[Up Arrow]`** | **Walk Forward** | Accelerate forward up to **$+0.75\text{ m/s}$** |
| **`[S]` / `[Down Arrow]`**| **Walk Backward** | Step backward down to **$-0.40\text{ m/s}$** |
| **`[A]` / `[Left Arrow]`**| **Snappy Turn Left** | Spin CCW up to **$+2.50\text{ rad/s}$** |
| **`[D]` / `[Right Arrow]`**| **Snappy Turn Right**| Spin CW down to **$-2.50\text{ rad/s}$** |
| **`[Q]`** | **Strafe Left** | Sidestep left up to $+0.35\text{ m/s}$ |
| **`[E]`** | **Strafe Right** | Sidestep right down to $-0.35\text{ m/s}$ |
| **`[F]`** | **Instant Sprint Boost** | Commands maximum **$+0.75\text{ m/s}$** sprint speed |
| **`[B]` / `[Space]` / `[X]`**| **EMERGENCY HARD BRAKE** | Instantly drops command to zero; robot engages anti-pitch braking reaction |
| **`[Ctrl+C]`** | **Exit** | Automatically exports `policy.pt` and `policy.onnx` |

> [!TIP]
> The controller works in **both** the terminal and the focused 3D OpenGL window.

---

## 7. Policy Architecture & Observation Breakdown

### 179D Policy Observation Space (Actor)
| Indices | Observation Term | Dims | Description |
| :---: | :--- | :---: | :--- |
| `0..3` | `projected_gravity` | 3 | Gravity vector in base frame $[g_x, g_y, g_z]$ |
| `3..6` | `base_ang_vel` | 3 | Base angular velocity $[\omega_x, \omega_y, \omega_z]$ |
| `6..20` | `joint_pos` | 14 | Current joint angles (relative to nominal STAND2) |
| `20..34` | `joint_vel` | 14 | Current joint angular velocities |
| `34..48` | `actions` | 14 | Previous action vector applied to motors |
| `48..51` | `velocity_commands` | 3 | Commanded velocity $[v_x, v_y, \omega_z]$ |
| `51..115`| `tof_depth` | 64 | 8x8 ToF distance matrix normalized in $[0, 1]$ |
| `115..179`| `camera_edges` | 64 | **8x8 OpenCV spatial edge density grid** in $[0, 1]$ |
| **Total**| **Actor Observation** | **179** | Input to MLP `[512, 256, 128]` $\rightarrow$ 14 actions |

### 202D Privileged Critic Observation Space (Critic)
Contains all 179 Actor observations $+$ 23 ground-truth simulation states:
* `base_lin_vel` (3D): True linear velocity in base frame.
* `base_pos_z` (1D): True base elevation above ground plane.
* `joint_effort` (14D): Applied joint torques.
* `hurdle_rel` (3D): Exact 3D vector from robot base to hurdle obstacle.
* `foot_forces` (2D): Net contact force magnitude on feet.
* **Total Critic Input:** **202 dimensions** $\rightarrow$ MLP `[512, 512, 256, 128]` $\rightarrow$ 1 value.

---

## 8. Policy Export & Real-Robot Deployment

When playback exits or training finishes, policies are automatically exported to:
```
<checkpoint_dir>/exported/
├── policy.pt    # TorchScript JIT compiled model
└── policy.onnx  # Universal ONNX runtime model
```

To run the exported policy directly in Python using ONNX Runtime or PyTorch on physical hardware:
```python
import torch
import cv2
import numpy as np

# Load TorchScript model
policy = torch.jit.load("policy.pt")
policy.eval()

# Run real camera edge extractor
# obs_tensor shape: (1, 179)
with torch.no_grad():
    motor_actions = policy(obs_tensor)
```

---

## 9. Troubleshooting & FAQ

### Q1: `ModuleNotFoundError: No module named 'cv2'`
**Fix:** Install OpenCV in the Isaac Lab Python environment:
```bash
cd ~/robotics/IsaacLab && uv pip install opencv-python
```

### Q2: `No display found (DISPLAY is unset)` / Remote SSH
**Fix:** If you are running over SSH with a local desktop session, export the display:
```bash
export DISPLAY=:0
# Or for headless training (no visualizer needed):
uv run python train.py --preset workstation
```

### Q3: `Cannot open /dev/video0: Permission denied`
**Fix:** Add your user account to the Linux `video` group:
```bash
sudo usermod -aG video $USER
# Log out and log back in, or run:
sudo chmod 666 /dev/video0
```

### Q4: `RuntimeError: CUDA out of memory` during training
**Fix:** If training with 16,384 envs exceeds your VRAM, simply decrease the number of environments:
```bash
uv run python train.py --num_envs 8192
# Or 4096:
uv run python train.py --num_envs 4096
```

### Q5: Bot wobbling or twitching at rest (Zero Velocity Command)
**Resolution:**
The system features a dual-layer rest stabilizer:
1. **Interactive Playback (`play.py`):** Uses an exponential stance blend (`rest_alpha * 0.88` per 20ms step) that smoothly and critically damps actions to zero, settling the robot into the nominal STAND2 stance within ~250ms without relay chattering.
2. **RL Training (`microduck_env_cfg.py`):** 15% of parallel environments (`rel_standing_envs=0.15`) are permanently tasked with stationary balance. Multi-objective rewards penalize non-zero actions (`stand_still_action_l2`), enforce bilateral ground contact, and lock joint positions to nominal stance.

---

## Summary of Essential Commands

```bash
# 1. Update repo
cd ~/robotics/isaac-sim-microduck && git pull origin main

# 2. Test 480p OpenCV Camera Edge Window
DISPLAY=:0 cd ~/robotics/IsaacLab && uv run python /home/shubham/robotics/isaac-sim-microduck/experiments/05_train_policy/test_camera_edges.py

# 3. Train from Scratch (Workstation: 16k envs, 100k iters)
cd ~/robotics/IsaacLab && uv run python /home/shubham/robotics/isaac-sim-microduck/experiments/05_train_policy/train.py --preset workstation

# 4. Interactive Keyboard Teleoperation & Real-Time Playback
DISPLAY=:0 cd ~/robotics/IsaacLab && uv run python /home/shubham/robotics/isaac-sim-microduck/experiments/05_train_policy/play.py
```
