# Experiment 01 — Basic Rigid Body Simulation

## Objective

Create a basic rigid-body simulation using Isaac Lab and the Newton physics backend.

The experiment demonstrates:
- Creating a static ground.
- Creating a dynamic rigid cube.
- Configuring mass and collision properties.
- Running a physics simulation using Newton with the MJWarp solver.
- Observing gravity and ground contact.

## Environment

- Ubuntu 24.04
- Isaac Lab 3.0.0
- Newton physics backend
- Newton Viewer
- GPU: NVIDIA RTX 5060 Laptop GPU

## Scene Configuration

### Ground

| Property | Value |
|---|---|
| Prim path | `/World/ground` |
| Geometry | Cuboid |
| Size | 10 × 10 × 0.1 m |
| Position | (0, 0, -0.05) m |
| Physics | Static |
| Collision | Enabled |

The ground's upper surface is located at z = 0.

### Cube

| Property | Value |
|---|---|
| Prim path | `{ENV_REGEX_NS}/Cube` |
| Geometry | Cuboid |
| Size | 0.1 × 0.1 × 0.1 m |
| Mass | 1 kg |
| Initial position | (0, 0, 1) m |
| Rigid body | Dynamic |
| Kinematic | Disabled |
| Collision | Enabled |

The environment namespace ensures the cube is registered within the environment managed by `InteractiveScene`.

## Physics Configuration

The experiment uses:

- Physics backend: Newton
- Solver: `MJWarpSolverCfg`
- Simulation timestep: 1/120 seconds
- Solver substeps: 10
- Collision decimation: 1

Configuration:

```python
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

physics_cfg = NewtonCfg(
    num_substeps=10,
    collision_decimation=1,
    solver_cfg=MJWarpSolverCfg(),
)
```

Gravity:

```python
gravity = (0.0, 0.0, -9.81)
```

## Simulation Loop

The simulation follows the standard Isaac Lab scene-update workflow:

```python
scene.write_data_to_sim()
sim.step()
scene.update(dt)
```

- `scene.write_data_to_sim()` synchronizes scene data with the simulation.
- `sim.step()` advances the physics simulation.
- `scene.update(dt)` updates the scene's data buffers.

## Solver Compatibility Finding

During initial testing, `XPBDSolverCfg` was used.

The cube remained stationary even though the simulation clock advanced and the solver initialized successfully.

Investigation identified a state-representation mismatch in the tested Isaac Lab Newton integration:

- XPBD updates maximal-coordinate state buffers: `body_q` and `body_qd`.
- The Isaac Lab `RigidObject` interface reads generalized-coordinate state: `joint_q` and `joint_qd`.

In this configuration, the state consumed by the `RigidObject` interface remained unchanged.

For this experiment, `MJWarpSolverCfg` was selected because it correctly updates the state consumed by the Isaac Lab rigid-object interface.

This finding is specific to the tested integration and configuration. It should not be interpreted as a universal limitation of XPBD.

## Execution

```bash
cd ~/robotics/IsaacLab

uv run python \
~/robotics/isaac-sim-microduck/experiments/01_rigid_body/rigid_body.py
```

## Expected Behaviour

1. The cube starts at z = 1 m.
2. Gravity accelerates the cube downward.
3. The cube collides with the ground.
4. The cube settles on the ground surface.

The expected resting centre height is approximately 0.05 m.

## Outcome

This experiment establishes the basic Newton rigid-body simulation setup that will be used as the foundation for subsequent robotics experiments.
