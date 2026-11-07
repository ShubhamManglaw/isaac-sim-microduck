# Experiment 02 — Quantitative Physics Validation

## Objective

Validate the physical behaviour of a dynamic rigid body using quantitative measurements rather than relying only on visual observation.

The experiment measures:

- Cube position over simulation time.
- Cube vertical velocity.
- Free-fall behaviour under gravity.
- Ground collision and impact.
- Final resting position.

The results are compared against analytical free-fall equations.

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
| Geometry | Cuboid |
| Size | 10 × 10 × 0.1 m |
| Position | (0, 0, -0.05) m |
| Top surface | z = 0 m |

### Dynamic Cube

| Property | Value |
|---|---|
| Geometry | Cuboid |
| Size | 0.1 × 0.1 × 0.1 m |
| Mass | 1 kg |
| Initial position | (0, 0, 1) m |
| Initial velocity | (0, 0, 0) m/s |
| Prim path | `{ENV_REGEX_NS}/Cube` |

The cube begins 1 metre above the ground.

## Physics Configuration

The experiment uses Newton with the MJWarp solver.

| Parameter | Configuration |
|---|---|
| Physics backend | Newton |
| Solver | `MJWarpSolverCfg` |
| Simulation timestep | 1/120 s |
| Substeps | 10 |
| Collision decimation | 1 |
| Gravity | (0, 0, -9.81) m/s² |

```python
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

physics_cfg = NewtonCfg(
    num_substeps=10,
    collision_decimation=1,
    solver_cfg=MJWarpSolverCfg(),
)
```

## Analytical Model

Before ground contact, the cube undergoes free fall under constant gravitational acceleration.

Initial conditions:

- Initial height: \(z_0 = 1.0\) m
- Initial vertical velocity: \(v_0 = 0\) m/s
- Gravitational acceleration: \(g = 9.81\) m/s²

### Position

\[
z(t)=z_0-\frac{1}{2}gt^2
\]

### Vertical velocity

\[
v_z(t)=-gt
\]

### Expected contact time

The cube's centre reaches z = 0.05 m when its bottom face contacts the ground.

\[
0.05=1-\frac{1}{2}(9.81)t^2
\]

\[
t=\sqrt{\frac{2(1-0.05)}{9.81}}
\]

\[
t\approx0.440\text{ s}
\]

The free-fall equations apply before contact, not during or after collision.

## Simulation Workflow

The simulation runs for 120 steps at 120 Hz, representing one second of simulated time.

The loop:

```python
for step in range(num_steps):
    scene.write_data_to_sim()
    sim.step()
    scene.update(dt)

    z = cube.data.root_link_pos_w.torch[0, 2].item()
    vz = cube.data.root_link_lin_vel_w.torch[0, 2].item()
```

Position and velocity are recorded throughout the experiment.

Measurements are printed every 12 steps.

## Solver Compatibility Finding

The initial experiment used `XPBDSolverCfg`.

Although the solver and Newton backend initialized successfully, the cube remained at its initial position with zero velocity.

Investigation identified a state-representation mismatch in the tested Isaac Lab Newton integration:

- XPBD updates maximal-coordinate buffers (`body_q` and `body_qd`).
- The Isaac Lab `RigidObject` interface reads generalized-coordinate buffers (`joint_q` and `joint_qd`).

The interface therefore continued reporting the original cube state.

The experiment was switched to `MJWarpSolverCfg`, which correctly updates the state consumed by `RigidObject` in this configuration.

The experiment also uses the environment namespace `{ENV_REGEX_NS}/Cube` and calls `scene.write_data_to_sim()` before every physics step.

This compatibility finding is specific to the tested integration; it is not a claim that XPBD is universally unsuitable for rigid-body simulation.

## Experimental Results

<badge color="success">Physics behaviour validated</badge>

The following measurements were obtained from the simulation:

| Time (s) | Height (m) | Vertical velocity (m/s) |
|---:|---:|---:|
| 0.1 | 0.9505 | -0.9810 |
| 0.2 | 0.8030 | -1.9620 |
| 0.3 | 0.5573 | -2.9430 |
| 0.4 | 0.2136 | -3.9240 |
| 0.5 | 0.0388 | 0.3771 |
| 0.6 | 0.0493 | -0.0029 |
| 0.7 | 0.0496 | -0.0206 |
| 0.8 | 0.0495 | -0.0122 |
| 0.9 | 0.0496 | -0.0016 |
| 1.0 | 0.0498 | 0.0029 |

### Observations

**1. Free fall**

From 0.1 to 0.4 seconds, the cube's vertical velocity increases in the negative direction as expected under gravity.

The measured values closely follow:

\[
v_z(t)=-9.81t
\]

**2. Ground impact**

At approximately 0.44 seconds, the cube reaches the ground.

At 0.5 seconds, the cube has already collided and has a positive vertical velocity, indicating a bounce.

**3. Settling**

After the impact, the cube settles close to the expected resting height.

Final measured position:

```text
tensor([[-5.3427e-09, -2.7645e-09, 4.9783e-02]])
```

Final measured velocity:

```text
tensor([[-3.3561e-10, -1.1883e-10, 2.8818e-03]])
```

The final centre height is approximately 0.0498 m, and the residual vertical velocity is close to zero.

## Execution

```bash
cd ~/robotics/IsaacLab

uv run python \
~/robotics/isaac-sim-microduck/experiments/02_physics_validation/physics_validation.py
```

## Conclusion

The experiment successfully demonstrates:

- Gravity-driven free fall.
- Quantitative agreement with the analytical free-fall model before impact.
- Collision with a static ground surface.
- A small bounce after impact.
- Stable settling at the expected resting height.

The experiment validates the basic rigid-body physics configuration and establishes a quantitative baseline for more complex simulation work.

## Engineering Takeaway

A simulation should not be considered validated merely because its viewer displays a scene or its simulation clock advances.

Physical behaviour must be checked through measurable quantities such as position, velocity, acceleration, collision response, and resting equilibrium.
