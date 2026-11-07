# Isaac Sim Microduck

**From-scratch reconstruction and reverse engineering of Pollen Robotics Microduck using NVIDIA Isaac Sim, Isaac Lab, PhysX, and ROS 2.**

<p align="center">
  <strong>MuJoCo → Isaac Sim → Isaac Lab → ROS 2</strong>
</p>

---

## 1. Project Overview

This project aims to reverse engineer and independently reconstruct the **Pollen Robotics Microduck** robot in the NVIDIA Isaac ecosystem.

The original Microduck simulation and reinforcement learning implementation is built around MuJoCo and MJLab. This project will use that implementation as a reference to understand how the robot is constructed, how its physics and actuators work, how its sensors are configured, and how its control architecture operates.

Rather than directly converting the existing MJCF model into another format, the robot will be reconstructed incrementally, subsystem by subsystem.

The goal is to understand and reproduce the robot's:

- Mechanical structure and kinematic hierarchy.
- Link geometry and coordinate frames.
- Joint positions, axes, and limits.
- Mass, centre of mass, and inertia.
- Collision geometry and contact behaviour.
- Motor and actuator characteristics.
- Sensors and their mounting frames.
- Robot control architecture.
- Simulation and reinforcement learning pipeline.
- ROS 2 communication interfaces.

Every implementation stage will involve comparing the original MuJoCo implementation with the corresponding Isaac Sim, Isaac Lab, and ROS 2 architecture.

**This is an engineering reconstruction project, not simply a robot model conversion project.**

## 2. Reference Robot: Microduck

**Microduck** is a compact bipedal robot developed by Pollen Robotics.

The original project describes a robot approximately:

- Height: 25 cm
- Weight: 800 g
- Actuated joints: 14 in the reinforcement learning model
- Physical servos: 15

The difference between the physical servo count and the 14 controlled joints will be investigated during reverse engineering.

### Original project repositories

| Repository | Purpose |
|---|---|
| [microduck](https://github.com/pollen-robotics/microduck) | Robot runtime and supporting software |
| [microduck_rl](https://github.com/pollen-robotics/microduck_rl) | Simulation, reinforcement learning, and robot models |

### Original simulation technology

The original reinforcement learning system uses:

- MuJoCo for physics simulation.
- MJLab as the robotics reinforcement learning framework.
- MuJoCo Warp for GPU-accelerated simulation.
- PPO for reinforcement learning.
- Custom actuator modelling for Dynamixel XL330 servos.
- ONNX for policy export and deployment.

The original robot models are described using MJCF (MuJoCo XML).

## 3. Project Objectives

The project has six primary objectives.

### 3.1 Understand the original robot

Reverse engineer the existing Microduck implementation before developing the Isaac version.

Investigate:

- MJCF robot descriptions.
- Link and joint relationships.
- Physical properties.
- Actuator configurations.
- Sensors and coordinate frames.
- Training configurations.
- Control and observation pipelines.

### 3.2 Reconstruct the robot from scratch

Build a new robot implementation using native Isaac Sim technologies.

The reconstruction will include:

- USD robot representation.
- Rigid bodies and articulation joints.
- Visual and collision geometry.
- Physical properties.
- Joint drives.
- Sensors.
- Simulation scenes.

The implementation will be developed incrementally rather than relying entirely on automatic MJCF or URDF conversion.

### 3.3 Understand physics differences

Investigate how MuJoCo and PhysX represent and simulate the same physical system.

Compare:

- Rigid-body dynamics.
- Joint constraints.
- Gravity.
- Collision detection.
- Contact modelling.
- Friction.
- Actuator behaviour.
- Numerical simulation differences.

Matching parameter values does not necessarily produce matching physical behaviour. Validation will therefore focus on both numerical properties and observable behaviour.

### 3.4 Reconstruct the control system

Understand how the original robot receives commands and converts them into joint movements.

Investigate:

- Position control.
- Joint target positions.
- Actuator torque limits.
- Motor dynamics.
- Action scaling.
- Observation construction.
- Control frequency.
- Domain randomization.

Reproduce the appropriate functionality using Isaac Lab.

### 3.5 Integrate ROS 2

Introduce ROS 2 as the communication and integration layer.

Planned functionality includes:

- Joint state communication.
- Joint command interfaces.
- Sensor data publishing.
- TF2 coordinate frames.
- Robot state communication.
- External controller integration.

ROS 2 will remain separate from the simulator's physics responsibilities.

### 3.6 Investigate reinforcement learning and sim-to-real

After the robot model, physics, and control architecture are validated, investigate reproducing the original learning pipeline.

The long-term objective is to understand how a trained policy interacts with the robot and how simulation assumptions affect real-world deployment.

---

## 4. Technology Stack

| Technology | Responsibility |
|---|---|
| NVIDIA Isaac Sim | Main robotics simulation platform |
| NVIDIA PhysX | Rigid-body physics and contact simulation |
| USD | Robot and simulation scene representation |
| NVIDIA Isaac Lab | Robotics environments, control, and reinforcement learning |
| ROS 2 | Robot communication and software integration |
| Python | Simulation scripts and robotics implementation |
| Git and GitHub | Version control and project documentation |

### Understanding the technology boundaries

These technologies have different responsibilities.

**Isaac Sim**

Provides the simulation environment, scene management, robot representation, sensors, and simulation interfaces.

**PhysX**

Provides the physics simulation used for rigid-body dynamics, articulation behaviour, collision detection, and contact.

**USD**

Provides the underlying scene and asset representation used by Isaac Sim.

**Isaac Lab**

Provides robotics-specific abstractions for articulations, actions, observations, environments, and reinforcement learning.

**ROS 2**

Provides communication infrastructure between robot software components.

ROS 2 is not a physics simulator, and Isaac Lab is not a replacement for ROS 2.

---

## 5. MuJoCo vs Isaac Sim Architecture

A major part of this project is understanding how equivalent robotics responsibilities are implemented differently across simulation frameworks.

| Responsibility | Original implementation | New implementation |
|---|---|---|
| Robot description | MJCF | USD |
| Scene representation | MuJoCo MJCF | USD Stage |
| Rigid-body physics | MuJoCo | PhysX |
| Articulated robot | MJCF bodies and joints | USD articulation |
| Joint configuration | MJCF joint definitions | USD articulation joints |
| Actuation | MuJoCo actuators and custom models | PhysX drives and Isaac Lab actuators |
| Collision geometry | MJCF geoms | USD collision APIs |
| Simulation sensors | MuJoCo sensors | Isaac Sim sensors |
| Reinforcement learning | MJLab | Isaac Lab |
| Robot communication | Existing runtime interfaces | ROS 2 integration |

The purpose is not to assume that these systems behave identically.

For each subsystem, we will investigate:

1. How the original implementation works.
2. Which component owns the responsibility.
3. How the equivalent Isaac component works.
4. Which concepts are equivalent.
5. Which concepts have different semantics.
6. How the new implementation can be validated.

---

## 6. Original Microduck Robot Structure

The original `robot_walk.xml` model provides our initial reference for the robot's kinematic structure.

### 6.1 Floating base

The robot has a root body:

`trunk_base`

Its root joint is:

`trunk_base_freejoint`

This represents a floating base with six degrees of freedom:

- Translation along X.
- Translation along Y.
- Translation along Z.
- Rotation around X.
- Rotation around Y.
- Rotation around Z.

The initial root position in the inspected model is:

`0 0 0.12`

### 6.2 Actuated joints

The original model contains 14 actuated joints.

**Left leg**

| Joint | Description |
|---|---|
| left_hip_yaw | Hip yaw |
| left_hip_roll | Hip roll |
| left_hip_pitch | Hip pitch |
| left_knee | Knee |
| left_ankle | Ankle |

**Head and neck**

| Joint | Description |
|---|---|
| neck_pitch | Neck pitch |
| head_pitch | Head pitch |
| head_yaw | Head yaw |
| head_roll | Head roll |

**Right leg**

| Joint | Description |
|---|---|
| right_hip_yaw | Hip yaw |
| right_hip_roll | Hip roll |
| right_hip_pitch | Hip pitch |
| right_knee | Knee |
| right_ankle | Ankle |

### 6.3 Physical properties

The original MJCF model defines:

- Link masses.
- Centre-of-mass positions.
- Full inertia tensors.
- Joint damping.
- Joint friction loss.
- Joint armature.

These properties will be investigated and reconstructed in the Isaac implementation.

### 6.4 Sensors

The inspected model includes:

- IMU orientation.
- Angular velocity.
- Linear velocity.
- Linear acceleration.
- Root angular momentum.
- Head camera.
- ToF sensor frame.
- Head IMU frame.
- Left and right foot frames.

The exact sensor behaviour and equivalent Isaac sensor configuration will be investigated separately.

### 6.5 Actuation

The original `robot_walk.xml` declares position actuators for its 14 controlled joints.

However, the training implementation also includes custom actuator modelling for Dynamixel XL330 servos.

The relationship between the MJCF actuator definitions and the training framework's custom actuator implementation must be understood before attempting to reproduce the motor behaviour.

---

## 7. Development Methodology

The reconstruction will follow a consistent engineering workflow.

### Step 1: Reverse-engineer

Inspect the original source files and understand how the subsystem works.

### Step 2: Understand the Isaac equivalent

Study the relevant Isaac Sim, PhysX, Isaac Lab, or ROS 2 architecture.

### Step 3: Design

Determine which components, configurations, and files are required.

### Step 4: Implement

Build the corresponding subsystem independently.

### Step 5: Validate

Compare the new implementation against the original reference.

### Step 6: Document

Record implementation decisions, differences, problems, and validation results.

No subsystem should be considered complete merely because it loads or appears visually correct.

---

## 8. Development Roadmap

The project will be developed in the following stages.

### Phase 1 — Development Environment

- [ ] Establish a clean Isaac development workspace.
- [ ] Configure the development environment.
- [ ] Verify Isaac Sim compatibility.
- [ ] Verify Isaac Lab functionality.
- [ ] Establish the laptop and desktop development workflow.
- [ ] Establish Git-based project synchronization.

### Phase 2 — Robot Kinematics

- [ ] Reverse engineer the complete MJCF hierarchy.
- [ ] Understand the coordinate conventions.
- [ ] Identify all link origins and orientations.
- [ ] Reconstruct the trunk.
- [ ] Reconstruct the left leg.
- [ ] Reconstruct the right leg.
- [ ] Reconstruct the neck and head.
- [ ] Reconstruct joint axes and limits.
- [ ] Validate articulation behaviour.

### Phase 3 — Robot Geometry

- [ ] Inspect original mesh assets.
- [ ] Understand mesh coordinate frames.
- [ ] Reconstruct visual geometry.
- [ ] Reconstruct collision geometry.
- [ ] Validate mesh alignment.
- [ ] Investigate self-collision.

### Phase 4 — Physical Properties

- [ ] Extract link masses.
- [ ] Extract centre-of-mass positions.
- [ ] Extract inertia tensors.
- [ ] Configure PhysX rigid-body properties.
- [ ] Configure gravity.
- [ ] Validate static equilibrium.
- [ ] Compare dynamic behaviour.

### Phase 5 — Collision and Contact

- [ ] Reconstruct foot collision geometry.
- [ ] Configure ground contact.
- [ ] Investigate collision filtering.
- [ ] Investigate friction.
- [ ] Validate standing contact.
- [ ] Validate falling and recovery behaviour.

### Phase 6 — Actuators and Motors

- [ ] Understand the original position actuator configuration.
- [ ] Understand the Dynamixel XL330 actuator model.
- [ ] Investigate BAM actuator implementation.
- [ ] Reconstruct joint drives.
- [ ] Configure torque and velocity limits.
- [ ] Investigate friction and backlash.
- [ ] Validate position tracking.
- [ ] Compare actuator responses.

### Phase 7 — Sensors

- [ ] Reconstruct IMU mounting.
- [ ] Configure IMU simulation.
- [ ] Reconstruct camera mounting.
- [ ] Configure camera simulation.
- [ ] Reconstruct ToF mounting.
- [ ] Reconstruct foot coordinate frames.
- [ ] Validate sensor transforms and outputs.

### Phase 8 — Control Architecture

- [ ] Understand the original control loop.
- [ ] Understand the action representation.
- [ ] Understand the observation representation.
- [ ] Implement joint control.
- [ ] Implement robot command interfaces.
- [ ] Validate control behaviour.

### Phase 9 — ROS 2 Integration

- [ ] Establish ROS 2 workspace.
- [ ] Configure joint state publishing.
- [ ] Configure joint command interfaces.
- [ ] Integrate TF2.
- [ ] Publish simulated sensor data.
- [ ] Validate communication.
- [ ] Integrate external ROS 2 nodes.

### Phase 10 — Complete Simulation

- [ ] Complete robot assembly.
- [ ] Validate standing.
- [ ] Validate falling.
- [ ] Validate joint trajectories.
- [ ] Validate contact behaviour.
- [ ] Validate balance.
- [ ] Investigate walking behaviour.

### Phase 11 — Reinforcement Learning

- [ ] Understand the original MJLab task configuration.
- [ ] Understand the PPO training setup.
- [ ] Reconstruct observation definitions.
- [ ] Reconstruct action definitions.
- [ ] Reconstruct reward functions.
- [ ] Configure Isaac Lab environments.
- [ ] Train initial policies.
- [ ] Evaluate policy behaviour.

### Phase 12 — Sim-to-Real Investigation

- [ ] Investigate actuator discrepancies.
- [ ] Investigate domain randomization.
- [ ] Investigate sensor noise.
- [ ] Investigate command delays.
- [ ] Evaluate simulation-to-hardware assumptions.

---

## 9. Validation Strategy

Visual similarity is not sufficient to establish a successful reconstruction.

Each subsystem will be validated independently.

| Subsystem | Validation criteria |
|---|---|
| Kinematics | Link transforms, joint axes, origins, and limits |
| Geometry | Mesh alignment and dimensions |
| Inertial properties | Mass, centre of mass, and inertia |
| Dynamics | Gravity response and motion |
| Actuators | Position tracking and force/torque limits |
| Collision | Contact points and collision behaviour |
| Sensors | Coordinate frames and measurements |
| Control | Command response and stability |
| ROS 2 | Message communication and frame consistency |
| Reinforcement learning | Training stability and task performance |

Where practical, equivalent test scenarios will be executed in both MuJoCo and Isaac Sim.

Differences in physics solvers, contact models, actuator semantics, and numerical integration will be documented rather than hidden.

---

## 10. Planned Repository Structure

The repository will evolve as implementation progresses.

The following is the intended structure, not a claim that these components have already been implemented.

```text
isaac-sim-microduck/
│
├── README.md
├── .gitignore
│
├── assets/
│   ├── meshes/
│   └── usd/
│
├── robot_description/
│   ├── urdf/
│   └── config/
│
├── isaac_sim/
│   ├── scenes/
│   └── scripts/
│
├── isaac_lab/
│   ├── robots/
│   ├── actuators/
│   ├── tasks/
│   └── environments/
│
├── ros2_ws/
│   └── src/
│
├── tests/
│
└── docs/
    ├── architecture.md
    └── reverse-engineering/
```

The final architecture and source-of-truth strategy for robot descriptions will be decided after investigating the requirements of each subsystem.

---

## 11. Development Environment

The intended development setup uses two machines.

**Primary development machine**

- Ubuntu 24.04.
- Python-based development.
- Git and GitHub.
- Isaac development tools.
- Lightweight simulation and testing where supported.

**Remote compute machine**

- Windows 11.
- WSL2 Ubuntu 24.04.
- NVIDIA GPU.
- SSH access through Tailscale.
- Larger simulation and training workloads, subject to compatibility and rendering requirements.

The exact Isaac Sim and Isaac Lab version requirements will be verified against the official NVIDIA documentation before installation.

The two machines will use separate development environments and synchronize source code through Git.

Virtual environments, generated simulation caches, and training outputs will not be synchronized through Git.

---

## 12. Engineering Principles

The project follows these principles:

1. Understand before implementing.
2. Reverse engineer before converting.
3. Preserve physical meaning rather than visual appearance alone.
4. Validate every subsystem independently.
5. Separate simulation, control, and communication responsibilities.
6. Document differences between MuJoCo and PhysX.
7. Avoid unnecessary duplication of robot descriptions.
8. Prefer reproducible experiments over assumptions.
9. Keep implementation modular and maintainable.
10. Build engineering understanding alongside the software.

---

## 13. Licensing and Attribution

This project uses the Pollen Robotics Microduck repositories as technical references.

The original project contains software and robot assets that may have different licensing conditions.

In particular, the original repository documentation identifies:

- Source code under Apache License 2.0.
- Mesh assets under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International.

Before redistributing original meshes, modified assets, or other reference materials, their applicable license and attribution requirements must be checked.

This project does not claim ownership of the original Microduck design or assets.

Original reference repositories:

- https://github.com/pollen-robotics/microduck
- https://github.com/pollen-robotics/microduck_rl

---

## 14. Project Status

**Current stage: Initial repository setup and documentation.**

Completed:

- [x] Created public GitHub repository.
- [x] Initialized the `main` branch.
- [x] Added initial `.gitignore`.
- [x] Pushed the initial repository commit.

In progress:

- [ ] Establish project documentation.
- [ ] Define the final development architecture.
- [ ] Begin systematic reverse engineering.
- [ ] Establish the Isaac development workspace.

Not yet implemented:

- Isaac Sim robot model.
- USD robot assets.
- PhysX articulation.
- Isaac Lab environment.
- ROS 2 integration.
- Reinforcement learning environment.

Repository:

**[github.com/ShubhamManglaw/isaac-sim-microduck](https://github.com/ShubhamManglaw/isaac-sim-microduck)**

---

## 15. Long-Term Goal

The long-term goal is to develop a technically validated Microduck reconstruction in the NVIDIA Isaac ecosystem and understand the engineering decisions involved in building a complete robotics simulation system.

The final outcome should demonstrate more than the ability to load a robot into a simulator.

It should demonstrate the ability to:

- Reverse engineer an existing robotic system.
- Understand and reconstruct robot kinematics and dynamics.
- Model realistic actuator behaviour.
- Build modular simulation architectures.
- Integrate ROS 2 with simulation.
- Develop reinforcement learning environments.
- Investigate sim-to-real challenges.

**The objective is not simply to reproduce a robot. It is to understand how to engineer the complete robotic system.**
