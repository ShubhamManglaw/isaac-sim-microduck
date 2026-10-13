from __future__ import annotations

import argparse
from isaaclab.app import add_launcher_args, launch_simulation
import isaaclab.sim as sim_utils
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass


@configclass
class SingleTrunkSceneCfg(InteractiveSceneCfg):
    """Minimal scene with just the floating trunk_base articulation."""

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/shubham/robotics/isaac-sim-microduck/experiments/03_robot_model/assets/microduck.usda"
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={},  # 0 actuated joints yet
            joint_vel={},
        ),
        actuators={},
    )


def main():
    parser = argparse.ArgumentParser()
    add_launcher_args(parser)
    parser.set_defaults(visualizer="none")
    args_cli = parser.parse_args()

    physics_cfg = NewtonCfg(
        num_substeps=1,
        solver_cfg=MJWarpSolverCfg(),
    )

    with launch_simulation(cfg=physics_cfg, launcher_args=args_cli) as resolved_physics_cfg:
        sim_cfg = SimulationCfg(
            dt=1.0 / 120.0,
            device=args_cli.device,
            physics=resolved_physics_cfg,
        )

        sim = SimulationContext(sim_cfg)
        scene = InteractiveScene(SingleTrunkSceneCfg(num_envs=1, env_spacing=1.0))
        sim.reset()

        robot = scene["robot"]
        print("\n" + "=" * 50)
        print(" [STEP 1 VERIFIED] FLOATING TRUNK ARTICULATION")
        print("=" * 50)
        print(f"Number of bodies: {robot.num_bodies} -> {robot.body_names}")
        print(f"Number of joints: {robot.num_joints}")
        print(f"Is fixed base:    {robot.is_fixed_base}")
        print(f"Base DOFs:        {robot.num_base_dofs} (3 translational + 3 rotational)")
        print(f"Starting height:  {robot.data.root_link_pos_w.torch[0, 2].item():.4f} m")

        print("\nAdvancing 5 steps under gravity:")
        dt = sim.get_physics_dt()
        for step in range(5):
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt)
            z = robot.data.root_link_pos_w.torch[0, 2].item()
            vz = robot.data.root_link_lin_vel_w.torch[0, 2].item()
            print(f"  Step {step + 1}: z = {z:.4f} m, vz = {vz:.4f} m/s")

        print("=" * 50 + "\n")


if __name__ == "__main__":
    main()

