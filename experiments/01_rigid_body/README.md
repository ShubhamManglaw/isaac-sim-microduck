'EOF'
# Experiment 01 — Rigid Body Simulation with Isaac Lab and Newton

## Objective

Establish a minimal rigid-body simulation using Isaac Lab and the Newton physics backend.

The purpose of this experiment is to understand the basic simulation architecture before introducing the Microduck robot model.

## Environment

- Ubuntu 24.04
- NVIDIA Isaac Lab
- Newton physics backend
- XPBD solver
- Python

## Scene Configuration

The experiment creates a minimal scene containing:

| Object | Configuration |
|---|---|
| Ground | Static cuboid, 10 × 10 × 0.1 m |
| Cube | Dynamic cuboid, 0.1 × 0.1 × 0.1 m |
| Cube mass | 1 kg |
| Initial cube position | (0, 0, 1) m |
| Gravity | (0, 0, -9.81) m/s² |
| Simulation timestep | 1/120 s |

## Physics Configuration

The Newton configuration uses:

- XPBD solver
- 10 physics substeps
- 20 solver iterations
- Collision decimation of 1
- Restitution enabled

## Implementation

The simulation is implemented in `rigid_body.py`.

The script uses:

- `SimulationContext` for simulation management.
- `InteractiveScene` for scene organization.
- `RigidObjectCfg` for the dynamic cube.
- `AssetBaseCfg` for the ground.
- `NewtonCfg` and `XPBDSolverCfg` for physics configuration.

## Execution

Run from the Isaac Lab repository environment:

```bash
cd ~/robotics/IsaacLab
uv run python ~/robotics/isaac-sim-microduck/experiments/01_rigid_body/rigid_body.py