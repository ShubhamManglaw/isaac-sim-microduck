# Experiment 03 — Understanding Isaac Lab Articulations Through Microduck

## Overview
This experiment builds a complete end-to-end understanding of how **Isaac Lab 3.0** represents, initializes, controls, and reads a floating-base robot articulation. We take the biped robot **Microduck** from its native MuJoCo MJCF representation (`robot_walk.xml`) and convert it into an OpenUSD articulation (`microduck.usd`), configure it using Isaac Lab's `ArticulationCfg`, and simulate it using the Newton physics engine with GPU tensor readback.

---

## 1. Robot Kinematic Architecture

Microduck is a 14-DOF bipedal robot with an articulated neck/head:
- **Floating Trunk (`trunk_base`)**: 6 uncontrolled degrees of freedom ($x, y, z, q_w, q_x, q_y, q_z$).
- **Left Leg (5 DOFs)**: `left_hip_yaw` $\to$ `left_hip_roll` $\to$ `left_hip_pitch` $\to$ `left_knee` $\to$ `left_ankle`.
- **Right Leg (5 DOFs)**: `right_hip_yaw` $\to$ `right_hip_roll` $\to$ `right_hip_pitch` $\to$ `right_knee` $\to$ `right_ankle`.
- **Neck & Head (4 DOFs)**: `neck_pitch` $\to$ `head_pitch` $\to$ `head_yaw` $\to$ `head_roll`.

In total:
- **15 Rigid Bodies**: `trunk_base`, 5 left leg links, 5 right leg links, 4 head links.
- **14 Revolute Joints**: Actuated by Dynamixel XL330 smart servos.
- **73 Visual Meshes**: Exactly extracted from MuJoCo's preprocessed mesh buffers (`model.mesh_vert` and `model.mesh_face`) for zero CAD offset error.
- **8 Collision Hulls**: Solid colliders covering torso, head/beak, legs, and foot soles so no body falls through the floor.
- **2 Foot Contact Sensors**: `ContactSensorCfg` tracking normal ground reaction forces ($F_z$) on `ankle_left` and `ankle_right`.

---

## 2. Running the Simulation

Execute the scripts from the `IsaacLab` virtual environment:

```bash
cd ~/robotics/IsaacLab

# 1. Inspect the robot mid-air (fixed base) with 360° mouse navigation:
uv run python ~/robotics/isaac-sim-microduck/experiments/03_robot_model/microduck_articulation.py --visualizer newton_gl --fix-base --mode squat

# 2. Balanced standing on the ground with head looking around:
uv run python ~/robotics/isaac-sim-microduck/experiments/03_robot_model/microduck_articulation.py --visualizer newton_gl --mode stand

# 3. Dynamic knee squatting on the ground:
uv run python ~/robotics/isaac-sim-microduck/experiments/03_robot_model/microduck_articulation.py --visualizer newton_gl --mode squat
```

### Viewer Navigation Controls
- **Left Click + Drag**: Orbit camera around the robot.
- **Middle Click (or Shift + Left Click) + Drag**: Pan camera.
- **Scroll Wheel**: Zoom in/out.
