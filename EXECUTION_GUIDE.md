# Microduck RL — Complete Execution & Deployment Guide

This document contains step-by-step instructions, command references, architecture details, and troubleshooting steps to run, train, evaluate, and deploy the **Microduck Biped Robot Reinforcement Learning Policy** with **forward 8x8 ToF depth perception**, **locomotion up to 0.75 m/s and 2.5 rad/s turns**, **curved walking, waypoint following and head look commands**, **hurdle clearance**, **push recovery**, and **anti-fall emergency braking**.

---

> [!IMPORTANT]
> To train on the RTX 5070 Ti PC, follow **[AGENT_HANDOFF.md](AGENT_HANDOFF.md)**. It is the authoritative runbook (setup, pilot run, health check, fixes). This guide is the reference for commands and the observation layout.

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
Runs **16,384 parallel environments** for **15,000 iterations** with 8 mini-batches:
```bash
cd ~/robotics/IsaacLab
uv run python /home/shubham/robotics/isaac-sim-microduck/experiments/05_train_policy/train.py --preset workstation
```

### Preset B: Laptop (RTX 5060, 8 GB)
Runs **8,192 parallel environments** for **10,000 iterations**:
```bash
cd ~/robotics/IsaacLab
uv run python /home/shubham/robotics/isaac-sim-microduck/experiments/05_train_policy/train.py --preset laptop
```

### Preset C: Fast 2-Iteration Smoke Test
Validates observation shapes, GPU buffers, and all reward terms:
```bash
cd ~/robotics/IsaacLab
uv run python /home/shubham/robotics/isaac-sim-microduck/experiments/05_train_policy/train.py --num_envs 16 --max_iterations 2
```

### Custom Overrides and Resuming
```bash
# Custom size / length
uv run python /path/to/train.py --num_envs 8192 --max_iterations 5000

# Continue the most recent run of this experiment (new run folder, weights and curriculum level carried over)
uv run python /path/to/train.py --preset workstation --resume

# Continue from a specific checkpoint, forcing the curriculum speed scale
uv run python /path/to/train.py --load_checkpoint /path/to/model_8000.pt --speed_scale 1.0
```

### How Training Progresses
* **Speed curriculum:** commanded speeds start at 50% of the full range. Every 20 s of simulated time the scale grows by 0.05 (up to 1.0) if both hold: linear velocity tracking is above 80% of its maximum, and robots average more than 10 s between falls.
* **Ramps tied to the curriculum:** random pushes and the smoothness penalties start at 30% strength and reach 100% at scale 1.0.
* **Fall recovery** is implemented but **off by default** (`reset_fallen` fraction = 0.0 in `microduck_env_cfg.py`). In testing, robots reset on the floor made the physics diverge and, because they are exempt from fall terminations for 8 s, they took over most of the envs and stalled walking. Only try it to fine-tune a policy that already walks well.
* **Hurdles** (8% of commands place a bar in the robot's path) unlock at scale 0.90 with the 1.5 cm bar only; the 2.5 and 4 cm bars join at scale 1.0.
* The scale is saved to `curriculum_state.json` in the run folder so `--resume` continues at the same level.

---

## 5. Monitoring Training with TensorBoard

Monitor reward curves, forward velocity tracking, air time, jumping rewards, and episode lengths in real time:

```bash
cd ~/robotics/IsaacLab
uv run tensorboard --logdir /home/shubham/robotics/isaac-sim-microduck/experiments/05_train_policy/logs --port 6006
```
Open your browser at `http://localhost:6006`.

### Health Checks (look at these first):
`check_training.py` reads these for you and prints a verdict:
```bash
cd ~/robotics/IsaacLab
uv run python /home/shubham/robotics/isaac-sim-microduck/experiments/05_train_policy/check_training.py
```
* `Policy/mean_std`: exploration noise. Starts at 0.6 and should stay between 0.15 and 0.6. If it climbs to the 1.5 cap, the run is not learning (the first training run failed this way, reaching a std of ~30).
* `Train/mean_episode_length`: should climb toward 1000 steps (20 s). Short episodes mean the robot keeps falling.
* `Curriculum/command_speed/speed_scale`: 0.5 → 1.0. `.../tracking` and `.../seconds_between_falls` show which of the two gates is holding it back.

### Key Reward Curves to Watch:
* `Episode_Reward/track_lin_vel_xy_exp`: Linear velocity tracking.
* `Episode_Reward/gait_contact`: Alternating left / right stepping that follows the gait clock.
* `Episode_Reward/hurdle_clearance`: Hurdle task (zero until hurdles unlock at speed scale 0.90).
* `Episode_Reward/fall_penalty`: Cost of falls; should shrink toward 0.
* `Episode_Termination/nan_state`: robots reset because their physics diverged. Should stay near 0.
* `Episode_Reward/head_look`: Head look command tracking.
* `Episode_Reward/sudden_stop_balance`: Anti-pitch damping during rapid deceleration.
* `Metrics/base_velocity/error_vel_xy`: Tracking error (should drop toward $0.05\text{ m/s}$).

---

## 6. Interactive Playback & Keyboard Teleoperation

Launch real-time 3D physics visualization with the interactive keyboard controller:

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

### Playback options:
```bash
# Walk a list of waypoints (x,y in metres) on its own; [Space] cancels and returns to keyboard control
uv run python /path/to/play.py --waypoints "0.6,0;0.6,0.6;0,0"

# Put a hurdle bar 0.6 m in front of the robot (low = 1.5 cm, mid = 2.5 cm, high = 4 cm)
uv run python /path/to/play.py --hurdle low

# Turn off the playback-side rest stabilizer to see how the policy itself stands still
uv run python /path/to/play.py --no-rest-blend

# Feed live webcam Canny edges into the edge inputs (outside the training distribution; off by default)
uv run python /path/to/play.py --camera
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
| **`[B]` / `[Space]` / `[X]`**| **EMERGENCY HARD BRAKE** | Instantly drops command to zero; also cancels waypoint following |
| **`[I]` / `[K]`** | **Head Look Up / Down** | Head pitch offset up to $\pm 0.35\text{ rad}$ |
| **`[J]` / `[L]`** | **Head Look Left / Right** | Head yaw offset up to $\pm 0.70\text{ rad}$ |
| **`[C]`** | **Center Head** | Resets the head look command |
| **`[Ctrl+C]`** | **Exit** | Automatically exports `policy.pt` and `policy.onnx` |

> [!TIP]
> The controller works in **both** the terminal and the focused 3D OpenGL window.

---

## 7. Policy Architecture & Observation Breakdown

### 285D Policy Observation Space (Actor)
The first six terms keep a **3-step history**: the two previous 20 ms samples followed by the current one (oldest first).

| Indices | Observation Term | Dims | Description |
| :---: | :--- | :---: | :--- |
| `0..9` | `projected_gravity` | 3 × 3 | Gravity vector in base frame $[g_x, g_y, g_z]$ |
| `9..18` | `base_ang_vel` | 3 × 3 | Base angular velocity $[\omega_x, \omega_y, \omega_z]$ |
| `18..60` | `joint_pos` | 3 × 14 | Joint angles (relative to nominal STAND2) |
| `60..102` | `joint_vel` | 3 × 14 | Joint angular velocities |
| `102..144` | `actions` | 3 × 14 | Previous action vectors |
| `144..153` | `velocity_commands` | 3 × 3 | Commanded velocity $[v_x, v_y, \omega_z]$ |
| `153..217`| `tof_depth` | 64 | Forward 8x8 ToF distances (row 0 = top of view), normalized by the 1 m range |
| `217..281`| `camera_edges` | 64 | 8x8 Sobel edge map of the depth image in $[0, 1]$ |
| `281..283`| `head_look` | 2 | Commanded head pitch / yaw offsets [rad] |
| `283..285`| `gait_clock` | 2 | $(\sin, \cos)$ of the gait cycle phase; zero when commanded to stand |
| **Total**| **Actor Observation** | **285** | Input to MLP `[512, 256, 128]` $\rightarrow$ 14 actions |

In code, get these ranges from `observation_term_slices(env, "policy")` in `microduck_env_cfg.py` rather than hardcoding them.

* Gravity, angular velocity, joint position / velocity and ToF inputs carry uniform noise during training.
* Actions are joint position offsets (`action * scale + default pose`), clipped to $\pm 3$ and then clamped to the joint limits.
* The gait clock advances each 20 ms step by `gait_frequency(command) * 0.02`, where the cycle takes 0.50 s near zero speed and shortens linearly to 0.30 s at 0.75 m/s (`gait_frequency` in `microduck_env_cfg.py`).

### 311D Privileged Critic Observation Space (Critic)
Contains all 285 Actor observations $+$ 26 ground-truth simulation states:
* `base_lin_vel` (3D): True linear velocity in base frame.
* `base_pos_z` (1D): True base elevation above ground plane.
* `joint_effort` (14D): Applied joint torques.
* `hurdle_state` (4D): Robot position in the hurdle frame, bar height, hurdle-active flag.
* `foot_forces` (2D): Net contact force magnitude on feet.
* `task_flags` (2D): Recovering from a fall, following a waypoint.
* **Total Critic Input:** **311 dimensions** $\rightarrow$ MLP `[512, 512, 256, 128]` $\rightarrow$ 1 value.

> [!NOTE]
> Checkpoints trained on an older observation layout (115D or 179D) are not compatible. `play.py` refuses them with an explanation.

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
# obs_tensor shape: (1, 285), built in the index order of the table in section 7,
# including the 3-step history and the speed-dependent gait clock described there
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
1. **Interactive Playback (`play.py`):** Uses an exponential stance blend (`rest_alpha * 0.88` per 20ms step) that fades the leg actions toward zero, settling the robot into the nominal STAND2 stance. Disable it with `--no-rest-blend` to check the policy's own standing behavior.
2. **RL Training (`microduck_env_cfg.py`):** 14% of commands (`rel_standing_envs=0.14`) are stand-still commands. Multi-objective rewards penalize non-zero leg actions (`stand_still_action_l2`), enforce bilateral ground contact, and lock joint positions to nominal stance.

### Q7: `ValueError: The ground slab ... is too small for N robots`
**Cause:** robots are placed on a grid `env_spacing` apart, and the grid for this many robots would extend past the ground slab. Robots beyond the edge would fall forever and poison training (this is what happened in the first training run: 81% of 8,192 robots had no ground).
**Fix:** increase `GROUND_SIZE` in `microduck_env_cfg.py`, or lower `scene.env_spacing`.

### Q6: `Policy/mean_std` keeps rising / episodes stay very short
**Cause:** exploration noise is winning over the reward signal. Check that `clip_actions` in `microduck_ppo_cfg.py` is still wide (3.0) and `entropy_coef` is small (0.005). With a tight $\pm 1$ clip the action penalties cannot see large actions and the noise grows without limit. Run `check_training.py` and apply the fix it names from `AGENT_HANDOFF.md`.

---

## Summary of Essential Commands

```bash
# 1. Update repo
cd ~/robotics/isaac-sim-microduck && git pull origin main

# 2. Test 480p OpenCV Camera Edge Window
DISPLAY=:0 cd ~/robotics/IsaacLab && uv run python /home/shubham/robotics/isaac-sim-microduck/experiments/05_train_policy/test_camera_edges.py

# 3. Train from Scratch (Workstation: 16k envs, 15k iters)
cd ~/robotics/IsaacLab && uv run python /home/shubham/robotics/isaac-sim-microduck/experiments/05_train_policy/train.py --preset workstation

# 4. Interactive Keyboard Teleoperation & Real-Time Playback
DISPLAY=:0 cd ~/robotics/IsaacLab && uv run python /home/shubham/robotics/isaac-sim-microduck/experiments/05_train_policy/play.py
```
