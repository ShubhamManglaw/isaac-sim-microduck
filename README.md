# Isaac Sim / Isaac Lab — Microduck Biped Robot

Reconstruction and reinforcement learning control of the **Pollen Robotics Microduck** biped robot using NVIDIA Isaac Lab with GPU Newton physics solver (`isaaclab_newton`), Asymmetric Actor-Critic (PPO), and multimodal perception.

---

## Quick Navigation

* 📖 **[Execution & Deployment Guide](file:///home/shubham/robotics/isaac-sim-microduck/EXECUTION_GUIDE.md)**: Full step-by-step commands for training, OpenCV edge extraction, playback, and workstation setup.
* 🏗️ **[System Architecture](file:///home/shubham/robotics/isaac-sim-microduck/docs/architecture.md)**: Technical breakdown of physics, simulation layers, and robotics framework.

---

## Key Features

* **Multimodal Perception (179D Observation):**
  * **8x8 ToF Metric Depth Matrix** (64 rays on GPU Warp, matching VL53L5CX sensor).
  * **480p OpenCV Canny Edge Feed** (64-dim $8 \times 8$ spatial edge density grid extracting obstacle silhouettes and ground transitions).
* **Aggressive Biped Locomotion:**
  * Forward sprinting up to **$+0.75\text{ m/s}$** with deep crouch-and-stride mechanics.
  * Snappy high-rate turning up to **$\pm 2.50\text{ rad/s}$**.
  * Lateral strafing up to $\pm 0.35\text{ m/s}$.
* **Autonomous Obstacle Jumping:**
  * Dynamic hurdle clearance with explosive vertical takeoff velocity ($V_z = 0.65\text{ m/s}$) clearing hurdles up to $17.5\text{ cm}$.
* **Emergency Hard Braking & Anti-Fall Stabilization:**
  * Active pitch damping and counter-torque balancing when coming to an immediate halt from full sprint without falling forward.
* **Sim-to-Real Domain Randomization:**
  * Startup mass variations ($\pm 15\%$), joint friction/armature variations ($\pm 20\%$), and actuator gain scaling ($\pm 15\%$).
* **Backwards-Compatible Evaluation & Interactive Teleoperation:**
  * Keyboard teleoperation supporting both terminal and 3D window, with instant sprint (`[F]`) and emergency brake (`[B]`).
  * Live OpenCV edge visualization window with glowing cyber-cyan contours.

---

## Quick Start Commands

```bash
# 1. Test 480p OpenCV Camera Edge Feed
DISPLAY=:0 cd ~/robotics/IsaacLab && uv run python /home/shubham/robotics/isaac-sim-microduck/experiments/05_train_policy/test_camera_edges.py

# 2. Train on RTX Workstation (16,384 envs, 100k iterations)
cd ~/robotics/IsaacLab && uv run python /home/shubham/robotics/isaac-sim-microduck/experiments/05_train_policy/train.py --preset workstation

# 3. Interactive Visual Playback & Keyboard Teleoperation
DISPLAY=:0 cd ~/robotics/IsaacLab && uv run python /home/shubham/robotics/isaac-sim-microduck/experiments/05_train_policy/play.py
```

For complete instructions, see the **[EXECUTION_GUIDE.md](file:///home/shubham/robotics/isaac-sim-microduck/EXECUTION_GUIDE.md)**.
