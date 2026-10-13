from __future__ import annotations

import argparse
import torch

from isaaclab.app import add_launcher_args, launch_simulation
import isaaclab.sim as sim_utils
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass


@configclass
class TwoLinksSceneCfg(InteractiveSceneCfg):
    """Scene with trunk_base + yaw2roll connected by left_hip_yaw joint."""

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/shubham/robotics/isaac-sim-microduck/experiments/03_robot_model/assets/microduck.usda"
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={"left_hip_yaw": 0.0},
            joint_vel={"left_hip_yaw": 0.0},
        ),
        actuators={
            "hip_yaw": IdealPDActuatorCfg(
                joint_names_expr=["left_hip_yaw"],
                stiffness=0.55,       # kp from robot_walk.xml (chosen_actuator)
                damping=0.05,         # damping from robot_walk.xml
                effort_limit=0.96,    # forcerange="+-0.96" from robot_walk.xml
            )
        },
    )


def main():
    parser = argparse.ArgumentParser()
    add_launcher_args(parser)
    parser.set_defaults(visualizer="none")
    args_cli = parser.parse_args()

    physics_cfg = NewtonCfg(
        num_substeps=10,
        solver_cfg=MJWarpSolverCfg(),
    )

    with launch_simulation(cfg=physics_cfg, launcher_args=args_cli) as resolved_physics_cfg:
        sim_cfg = SimulationCfg(
            dt=1.0 / 120.0,
            device=args_cli.device,
            physics=resolved_physics_cfg,
        )

        sim = SimulationContext(sim_cfg)
        scene = InteractiveScene(TwoLinksSceneCfg(num_envs=1, env_spacing=1.0))
        sim.reset()

        robot = scene["robot"]
        print("\n" + "=" * 55)
        print(" [STEP 2 VERIFIED] 2 BODIES + 1 JOINT ARTICULATION")
        print("=" * 55)
        print(f"Number of bodies: {robot.num_bodies} -> {robot.body_names}")
        print(f"Number of joints: {robot.num_joints} -> {robot.joint_names}")
        print(f"Is fixed base:    {robot.is_fixed_base}")
        print(f"Base DOFs:        {robot.num_base_dofs} (floating root)")
        print(f"Joint pos shape:  {robot.data.joint_pos.shape}")
        print(f"Initial joint q:  {robot.data.joint_pos.torch[0, 0].item():.4f} rad")

        # Command the joint to rotate to +0.2 rad (within limits [-0.436, +0.524])
        target_q = torch.tensor([[0.2]], device=args_cli.device)
        robot.set_joint_position_target(target_q)
        print(f"\nCommanding left_hip_yaw to target: +0.2000 rad...")

        dt = sim.get_physics_dt()
        for step in range(10):
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt)
            q = robot.data.joint_pos.torch[0, 0].item()
            qd = robot.data.joint_vel.torch[0, 0].item()
            print(f"  Step {step + 1:2d}: angle = {q:+.4f} rad, vel = {qd:+.4f} rad/s")

        print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
