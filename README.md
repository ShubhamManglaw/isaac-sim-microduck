# Isaac Sim / Isaac Lab — Microduck Biped Robot

Reconstruction and reinforcement learning control of the **Pollen Robotics Microduck** biped robot using NVIDIA Isaac Lab with GPU Newton physics solver (`isaaclab_newton`), Asymmetric Actor-Critic (PPO), and multimodal perception.

---

## Quick Navigation

* 🤖 **[Agent Handoff](AGENT_HANDOFF.md)**: The step-by-step runbook for training on the RTX 5070 Ti PC (setup, pilot run, health check, what to change if a check fails). **Start here to train.**
* 📖 **[Execution & Deployment Guide](file:///home/shubham/robotics/isaac-sim-microduck/EXECUTION_GUIDE.md)**: Full step-by-step commands for training, OpenCV edge extraction, playback, and workstation setup.
* 🏗️ **[System Architecture](file:///home/shubham/robotics/isaac-sim-microduck/docs/architecture.md)**: Technical breakdown of physics, simulation layers, and robotics framework.

---

## Key Features

* **Perception and Proprioception (285D Observation):**
  * **3-step history** of gravity, angular velocity, joint positions / velocities, last actions and commands.
  * **Forward 8x8 ToF Depth Matrix** (64 rays in a 45° fan pitched 22° down, VL53L5CX-style, 1 m range).
  * **8x8 Depth Edge Map** (Sobel edges of the depth image). `play.py --camera` can substitute live 480p OpenCV Canny edges, but the policy is not trained on webcam edges.
* **Trained Tasks (what the reward and command setup targets):**
  * Velocity tracking: forward up to $+0.75\text{ m/s}$, backward, strafing up to $\pm 0.35\text{ m/s}$, turning up to $\pm 2.50\text{ rad/s}$, and curved walking (forward + turn together).
  * Standing still and sudden stops without tipping forward.
  * Walking to waypoints.
  * Head look commands (head pitch / yaw) while standing or walking.
  * Clearing hurdle bars of 1.5 / 2.5 / 4 cm placed in the robot's path.
  * Push recovery. (Standing up after a fall is implemented but off by default: it destabilized training.)
* **Smooth Gait Shaping:**
  * Gait clock whose cadence follows the commanded speed, with an alternating left / right contact reward, torque, joint-acceleration, action-rate and action-jerk penalties, and a foot-impact penalty.
* **Speed Curriculum:**
  * Commanded speeds start at 50% and grow to 100% once the robot tracks well and stays on its feet. Pushes and smoothness penalties ramp up with it, and hurdles unlock near the end.
* **Sim-to-Real Domain Randomization:**
  * Sensor noise, trunk mass ($\pm 15\%$), joint friction/armature ($\pm 20\%$), actuator gains ($\pm 15\%$), and random pushes.
* **Backwards-Compatible Evaluation & Interactive Teleoperation:**
  * Keyboard teleoperation supporting both terminal and 3D window, with instant sprint (`[F]`) and emergency brake (`[B]`).
  * Live OpenCV edge visualization window with glowing cyber-cyan contours.

---

## Quick Start Commands

```bash
# 1. Test 480p OpenCV Camera Edge Feed
DISPLAY=:0 cd ~/robotics/IsaacLab && uv run python /home/shubham/robotics/isaac-sim-microduck/experiments/05_train_policy/test_camera_edges.py

# 2. Train on RTX Workstation (16,384 envs): 2,000-iteration pilot first, then follow AGENT_HANDOFF.md
~/robotics/isaac-sim-microduck/experiments/05_train_policy/train_tmux.sh --preset workstation --max_iterations 2000
cd ~/robotics/IsaacLab && uv run python ~/robotics/isaac-sim-microduck/experiments/05_train_policy/check_training.py

# 3. Interactive Visual Playback & Keyboard Teleoperation
DISPLAY=:0 cd ~/robotics/IsaacLab && uv run python /home/shubham/robotics/isaac-sim-microduck/experiments/05_train_policy/play.py
```

For complete instructions, see the **[EXECUTION_GUIDE.md](file:///home/shubham/robotics/isaac-sim-microduck/EXECUTION_GUIDE.md)**.
