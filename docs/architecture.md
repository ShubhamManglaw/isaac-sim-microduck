# Microduck — System Architecture

## 1. Purpose

This document defines the system architecture for reconstructing Pollen Robotics Microduck using NVIDIA Isaac Sim, Isaac Lab, PhysX, and ROS 2.

The original Microduck implementation, based on MuJoCo and MJLab, will serve as the reference system.

The objective is to understand the original robot's mechanical structure, physical properties, actuation, sensing, and control architecture, then independently reconstruct these components in the Isaac ecosystem.

This document describes the intended architecture. It does not imply that the Isaac implementation is already complete.

---

## 2. System Overview

The project consists of four primary technology layers.

### 2.1 Physics Layer — PhysX

Responsibilities:

- Rigid-body dynamics.
- Articulated-body simulation.
- Joint constraints.
- Gravity.
- Collision detection.
- Contact simulation.
- Friction and physical interactions.

PhysX will provide the physics simulation through Isaac Sim.

### 2.2 Simulation Layer — NVIDIA Isaac Sim

Responsibilities:

- USD scene management.
- Robot asset representation.
- Rigid-body configuration.
- Articulation configuration.
- Simulation management.
- Sensor simulation.
- Visualization.

Isaac Sim will serve as the main simulation platform.

### 2.3 Robotics Framework — Isaac Lab

Responsibilities:

- Robot articulation configuration.
- Actuator configuration.
- Joint control.
- Action definitions.
- Observation definitions.
- Environment configuration.
- Reinforcement learning.
- Simulation-based robotics experiments.

Isaac Lab will provide the robotics and learning framework on top of Isaac Sim.

### 2.4 Communication Layer — ROS 2

Responsibilities:

- Joint state communication.
- Joint command interfaces.
- Sensor data communication.
- Coordinate frame management.
- Communication with external robotics software.
- Integration with other ROS 2 nodes.

ROS 2 will provide the robot software communication architecture.

---

## 3. Original Microduck Architecture

The original Microduck system uses:

| Component | Technology |
|---|---|
| Robot description | MJCF |
| Physics simulation | MuJoCo |
| GPU simulation | MuJoCo Warp |
| Robotics learning framework | MJLab |
| Reinforcement learning | PPO |
| Actuator modelling | Custom BAM actuator model |
| Policy deployment | ONNX |

### 3.1 Robot Structure

The inspected `robot_walk.xml` model contains:

- One floating trunk base.
- Two articulated legs.
- One neck and head mechanism.
- Fourteen actuated joints.
- Visual geometry.
- Collision geometry.
- Link inertial properties.
- Joint limits.
- Sensor frames.
- Actuator definitions.

The robot's root body is `trunk_base`.

Its root joint is `trunk_base_freejoint`.

### 3.2 Joint Organization

**Left leg**

- left_hip_yaw
- left_hip_roll
- left_hip_pitch
- left_knee
- left_ankle

**Head and neck**

- neck_pitch
- head_pitch
- head_yaw
- head_roll

**Right leg**

- right_hip_yaw
- right_hip_roll
- right_hip_pitch
- right_knee
- right_ankle

The original training model controls fourteen joints. The difference between this and the physical robot's servo count will be investigated separately.

### 3.3 Physical Properties

The original MJCF defines:

- Link mass.
- Centre-of-mass position.
- Full inertia tensor.
- Joint damping.
- Joint friction loss.
- Joint armature.

These properties are essential to reproducing the robot's physical behaviour.

### 3.4 Actuation

The inspected `robot_walk.xml` declares position actuators.

The original training system additionally uses custom actuator modelling for Dynamixel XL330 motors through the BAM implementation.

The exact interaction between the MJCF actuator definitions and the training actuator implementation must be investigated before reconstruction.

### 3.5 Sensors

The inspected model includes:

- IMU orientation.
- Angular velocity.
- Linear velocity.
- Linear acceleration.
- Root angular momentum.
- Head camera.
- ToF sensor frame.
- Head IMU frame.
- Left foot frame.
- Right foot frame.

---

## 4. Target Isaac Architecture

The intended implementation will use the following architecture.

### 4.1 Robot Representation

Primary representation: USD.

USD will describe:

- Robot links.
- Link transforms.
- Visual geometry.
- Collision geometry.
- Rigid-body properties.
- Articulation joints.
- Joint properties.
- Sensor mounting frames.

URDF may be used as an interchange and validation format.

The exact source-of-truth strategy will be finalized during implementation.

### 4.2 Physics

PhysX will handle:

- Rigid-body dynamics.
- Articulation simulation.
- Joint constraints.
- Gravity.
- Collision detection.
- Contact forces.
- Friction.

### 4.3 Robotics Framework

Isaac Lab will handle:

- Robot articulation configuration.
- Actuator configuration.
- Joint commands.
- Action and observation definitions.
- Environment configuration.
- Reinforcement learning.

### 4.4 Communication

ROS 2 will handle:

- Joint state messages.
- Joint command messages.
- Sensor messages.
- TF2 transforms.
- External controller communication.

---

## 5. Technology Comparison

| Responsibility | MuJoCo / MJLab | Isaac Ecosystem |
|---|---|---|
| Robot description | MJCF | USD |
| Scene representation | MJCF scene | USD Stage |
| Physics | MuJoCo | PhysX |
| Rigid bodies | MJCF bodies | USD rigid bodies |
| Joints | MJCF joints | USD articulation joints |
| Actuation | MuJoCo actuators / custom models | PhysX drives / Isaac Lab actuators |
| Collision | MJCF geoms | USD collision APIs |
| Sensors | MuJoCo sensors | Isaac Sim sensors |
| Learning framework | MJLab | Isaac Lab |
| Robot communication | Existing runtime interfaces | ROS 2 integration |

Important: equivalent parameter names do not guarantee equivalent simulation behaviour.

Physics solver settings, actuator semantics, contact models, and integration methods may differ.

These differences will be documented and validated throughout the project.

---

## 6. Robot Description Architecture

The robot description will be reconstructed in stages.

### 6.1 Link Hierarchy

First, reconstruct the complete kinematic tree.

The process begins with:

1. Trunk base.
2. Left leg.
3. Right leg.
4. Neck.
5. Head.

Each link must have the correct parent-child relationship.

### 6.2 Coordinate Frames

For every link and joint, investigate:

- Parent frame.
- Child frame.
- Joint origin.
- Joint orientation.
- Joint axis.
- Local coordinate system.
- World-space transform.

Coordinate transformations must be validated rather than inferred from visual appearance.

### 6.3 Geometry

Visual and collision geometry will be treated separately.

Visual geometry is responsible for appearance.

Collision geometry is responsible for physical interaction.

The reconstruction must preserve the intended collision behaviour rather than simply reusing all visual meshes as collision meshes.

### 6.4 Inertial Properties

For each rigid body, reconstruct:

- Mass.
- Centre of mass.
- Inertia tensor.

These values must be represented correctly in the Isaac physics model.

---

## 7. Actuator Architecture

The actuator system will be investigated independently from the robot geometry.

The original system includes position actuator definitions and a custom Dynamixel-oriented actuator model.

The reconstruction will investigate:

- Position control.
- Velocity effects.
- Torque limits.
- Motor friction.
- Backlash.
- Command delay.
- Voltage effects.
- Actuator dynamics.

The Isaac implementation will use appropriate PhysX drives and Isaac Lab actuator models.

The exact implementation will be selected after understanding the original BAM actuator model.

---

## 8. Sensor Architecture

The sensor subsystem will include:

### IMU

- Orientation.
- Angular velocity.
- Linear acceleration.
- Sensor mounting frame.

### Camera

- Head camera mounting.
- Camera orientation.
- Image output.

### ToF

- Sensor mounting position.
- Sensor orientation.
- Distance measurement.

### Foot Frames

- Left foot frame.
- Right foot frame.

Sensor coordinate frames will be validated before ROS 2 integration.

---

## 9. Control Architecture

The control architecture will be developed after the robot model and physical properties are validated.

The planned control pipeline is:

1. Receive robot commands.
2. Generate target joint positions or other control references.
3. Apply actuator models.
4. Simulate joint response.
5. Read robot state.
6. Construct observations.
7. Repeat at the configured control frequency.

The original Microduck action and observation definitions will be inspected before implementing their Isaac equivalents.

---

## 10. ROS 2 Integration Architecture

ROS 2 will be integrated after the basic simulation and control systems are functional.

Planned interfaces include:

| Interface | Responsibility |
|---|---|
| Joint states | Publish robot joint positions, velocities and efforts |
| Joint commands | Receive external joint commands |
| IMU | Publish simulated IMU measurements |
| Camera | Publish camera images |
| TF2 | Publish robot coordinate transforms |
| Robot state | Communicate relevant robot status |

ROS 2 communication and simulator physics will remain separate responsibilities.

The exact interfaces and controllers will be determined during implementation.

---

## 11. Validation Architecture

Each subsystem will be validated independently.

| Subsystem | Validation |
|---|---|
| Kinematics | Link transforms, joint origins and axes |
| Geometry | Mesh alignment and dimensions |
| Inertia | Mass, centre of mass and inertia |
| Dynamics | Gravity and motion response |
| Actuation | Position tracking and force limits |
| Collision | Contact and collision behaviour |
| Sensors | Sensor frames and measurements |
| Control | Command response and stability |
| ROS 2 | Communication and frame consistency |

Equivalent experiments will be performed in MuJoCo and Isaac Sim wherever practical.

Numerical differences will be recorded and investigated.

---

## 12. Development Workflow

Every subsystem will follow this workflow:

1. Inspect the original Microduck implementation.
2. Understand the underlying engineering concept.
3. Identify the equivalent Isaac component.
4. Design the new implementation.
5. Implement the subsystem.
6. Run independent validation tests.
7. Compare against the original system.
8. Document differences and engineering decisions.

No subsystem will be considered complete merely because it loads successfully.

---

## 13. Planned Development Phases

| Phase | Subsystem | Status |
|---|---|---|
| 1 | Development workspace | Pending |
| 2 | Robot kinematics | Pending |
| 3 | Geometry | Pending |
| 4 | Inertial properties | Pending |
| 5 | Collision and contact | Pending |
| 6 | Actuators | Pending |
| 7 | Sensors | Pending |
| 8 | Control architecture | Pending |
| 9 | ROS 2 integration | Pending |
| 10 | Complete simulation | Pending |
| 11 | Reinforcement learning | Pending |
| 12 | Sim-to-real investigation | Pending |

---

## 14. Repository Organization

The intended project organization is:

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

This structure is provisional and will evolve as implementation requirements become clear.

---

## 15. Design Decisions and Open Questions

The following decisions require investigation:

- What should be the authoritative robot description format?
- How should MJCF joint frames map to USD articulation joint frames?
- How closely can PhysX reproduce the original MuJoCo contact behaviour?
- How should the BAM actuator model be implemented in Isaac Lab?
- Which sensors require custom simulation implementations?
- How should the ROS 2 control interfaces be organized?
- How should simulation and reinforcement learning environments share robot assets?

These questions will be resolved through source inspection, implementation, and testing.

---

## 16. Current Status

**Current stage: Initial project documentation.**

Completed:

- Public GitHub repository created.
- Initial `.gitignore` committed.
- Project README prepared.

Not yet implemented:

- Isaac Sim robot model.
- USD robot assets.
- PhysX articulation.
- Isaac Lab robot configuration.
- Actuator models.
- ROS 2 interfaces.
- Reinforcement learning environments.

The next engineering milestone is to establish the development workspace and begin reconstructing the robot's kinematic structure.

---

## 17. References

- [Pollen Robotics Microduck](https://github.com/pollen-robotics/microduck)
- [Microduck Reinforcement Learning](https://github.com/pollen-robotics/microduck_rl)
- [NVIDIA Isaac Sim Documentation](https://docs.isaacsim.omniverse.nvidia.com/)
- [NVIDIA Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [ROS 2 Documentation](https://docs.ros.org/)

