from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add current and parent directories to sys.path
CURRENT_DIR = Path(__file__).resolve().parent
EXP_03_DIR = CURRENT_DIR.parent / "03_robot_model"
for p in [str(CURRENT_DIR), str(EXP_03_DIR)]:
    if p not in sys.path:
        sys.path.append(p)

import torch
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab.envs import ManagerBasedRLEnv

from microduck_env_cfg import MicroduckEnvCfg


def main():
    parser = argparse.ArgumentParser(description="Test Microduck RL Environment in Isaac Lab.")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments to simulate.")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum number of environment steps to run.")
    parser.add_argument("--real_time", action="store_true", default=True, help="Lock simulation speed to 1.0x real time.")
    parser.add_argument("--no_real_time", action="store_false", dest="real_time", help="Run simulation at maximum speed without sleeping.")
    add_launcher_args(parser)
    parser.set_defaults(visualizer="newton_gl")
    args_cli = parser.parse_args()

    # 1. Physics configuration with MJWarp solver (num_substeps=1 for fast real-time stepping)
    physics_cfg = NewtonCfg(
        num_substeps=1,
        solver_cfg=MJWarpSolverCfg(),
        load_visual_shapes=True,
    )

    # 2. Launch simulation context
    with launch_simulation(cfg=physics_cfg, launcher_args=args_cli) as resolved_physics_cfg:
        env_cfg = MicroduckEnvCfg()
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.sim.physics = resolved_physics_cfg

        # 3. Create the parallel RL environment
        print("\n" + "=" * 60)
        print(f" [ISAAC LAB] CREATING PARALLEL RL ENVIRONMENT ({args_cli.num_envs} ENVS)")
        print("=" * 60)
        env = ManagerBasedRLEnv(cfg=env_cfg)

        obs_dict = env.reset()[0]
        policy_obs = obs_dict["policy"]
        action_dim = env.action_manager.action.shape[-1]
        obs_dim = policy_obs.shape[-1]

        print(f"Number of environments: {env.num_envs}")
        print(f"Observation space:      {obs_dim} dimensions (policy group)")
        print(f"Action space:           {action_dim} dimensions (14 joint position deltas)")
        print(f"Control frequency:      {1.0 / env.step_dt:.1f} Hz (decimation={env.cfg.decimation}, dt={env.physics_dt:.4f}s)")
        print("-" * 60)
        print(" Active Reward Terms:")
        for name in env.reward_manager.active_terms:
            cfg = env.reward_manager.get_term_cfg(name)
            print(f"   - {name:25s} (weight: {cfg.weight:+.2f})")
        print("-" * 60)
        print(" Running environment loop... (Press Ctrl+C or close viewer to stop)")
        print("=" * 60)

        step = 0
        total_episodes = 0
        t_last = time.perf_counter()
        steps_since_last = 0

        while env.sim.is_running():
            t_step_start = time.perf_counter()

            if args_cli.max_steps is not None and step >= args_cli.max_steps:
                break

            # Sample random small exploratory actions [-0.5, 0.5]
            actions = 0.20 * torch.randn((env.num_envs, action_dim), device=env.device)

            # Step the environment
            obs, rewards, terminated, truncated, extras = env.step(actions)

            num_resets = torch.sum(terminated | truncated).item()
            total_episodes += num_resets
            steps_since_last += 1

            # Print telemetry every 50 control steps (1 second of sim time)
            if step > 0 and step % 50 == 0:
                t_now = time.perf_counter()
                elapsed_wall = t_now - t_last
                fps = steps_since_last / elapsed_wall
                rtf = (steps_since_last * env.step_dt) / elapsed_wall
                mean_rew = rewards.mean().item()
                mean_z = env.scene["robot"].data.root_link_pos_w.torch[:, 2].mean().item()
                print(
                    f" Step {step:5d} | {fps:5.1f} steps/s ({rtf:4.2f}x Real-Time) | "
                    f"Mean Z: {mean_z:.3f} m | Resets: {num_resets:2d} | Total Resets: {total_episodes:3d}"
                )
                t_last = t_now
                steps_since_last = 0

            # Real-time synchronization (sleep to lock 50 Hz when viewing in GUI)
            if args_cli.real_time:
                step_elapsed = time.perf_counter() - t_step_start
                sleep_sec = env.step_dt - step_elapsed
                if sleep_sec > 0:
                    time.sleep(sleep_sec)

            step += 1

        env.close()
        print("\nEnvironment test ended cleanly.")


if __name__ == "__main__":
    main()
