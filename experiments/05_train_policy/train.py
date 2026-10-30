from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Add paths for Experiment 03, 04, and current directory
CURRENT_DIR = Path(__file__).resolve().parent
EXP_04_DIR = CURRENT_DIR.parent / "04_rl_env"
EXP_03_DIR = CURRENT_DIR.parent / "03_robot_model"
for p in [str(CURRENT_DIR), str(EXP_04_DIR), str(EXP_03_DIR)]:
    if p not in sys.path:
        sys.path.append(p)

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    check_rsl_rl_version,
    create_rsl_rl_runner,
    handle_deprecated_rsl_rl_cfg,
)

from microduck_env_cfg import MicroduckEnvCfg
from microduck_ppo_cfg import MicroduckPPORunnerCfg


def main():
    parser = argparse.ArgumentParser(description="Train Microduck Biped Locomotion Policy with PPO (rsl_rl).")
    parser.add_argument("--num_envs", type=int, default=8192, help="Number of parallel simulation environments.")
    parser.add_argument("--max_iterations", type=int, default=10000, help="Maximum number of PPO learning iterations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for environment and policy.")
    parser.add_argument("--log_dir", type=str, default=None, help="Root directory for experiment logs.")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume training from latest checkpoint.")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to checkpoint (.pt) to resume from.")
    add_launcher_args(parser)
    parser.set_defaults(visualizer="none")  # Headless by default for maximum throughput
    args_cli = parser.parse_args()

    # 1. Physics backend configuration: high-throughput GPU Newton solver
    physics_cfg = NewtonCfg(
        num_substeps=1,
        solver_cfg=MJWarpSolverCfg(),
        load_visual_shapes=(args_cli.visualizer != "none"),
    )

    # 2. Launch simulation context
    with launch_simulation(cfg=physics_cfg, launcher_args=args_cli) as resolved_physics_cfg:
        # Create environment configuration
        env_cfg = MicroduckEnvCfg()
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.sim.physics = resolved_physics_cfg
        env_cfg.seed = args_cli.seed

        # Create PPO Runner configuration
        agent_cfg = MicroduckPPORunnerCfg()
        if args_cli.max_iterations is not None:
            agent_cfg.max_iterations = args_cli.max_iterations

        installed_version = check_rsl_rl_version()
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

        # Determine log directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        root_log_dir = Path(args_cli.log_dir) if args_cli.log_dir else (CURRENT_DIR / "logs")
        run_log_dir = root_log_dir / f"{agent_cfg.experiment_name}_{timestamp}"
        run_log_dir.mkdir(parents=True, exist_ok=True)

        # Update 'latest' symlink immediately so play.py always finds the current run
        latest_symlink = root_log_dir / "latest"
        if latest_symlink.is_symlink() or latest_symlink.exists():
            latest_symlink.unlink()
        latest_symlink.symlink_to(run_log_dir.resolve(), target_is_directory=True)

        print("\n" + "=" * 65)
        print(" [ISAAC LAB] MICRODUCK BIPED PPO POLICY TRAINING")
        print("=" * 65)
        print(f"Parallel Environments: {args_cli.num_envs:,}")
        print(f"Learning Iterations:   {agent_cfg.max_iterations}")
        print(f"Transitions / Step:    {args_cli.num_envs * agent_cfg.num_steps_per_env:,}")
        print(f"Total Step Budget:     {args_cli.num_envs * agent_cfg.num_steps_per_env * agent_cfg.max_iterations:,} steps")
        print(f"Log Directory:         {run_log_dir}")
        print("=" * 65 + "\n")

        # 3. Instantiate Vectorized RL Environment
        env = ManagerBasedRLEnv(cfg=env_cfg)
        wrapped_env = RslRlVecEnvWrapper(env, clip_actions=1.0)

        # 4. Instantiate PPO Runner
        runner = create_rsl_rl_runner(wrapped_env, agent_cfg, log_dir=str(run_log_dir))

        # 4b. Resume or load checkpoint if requested
        ckpt_to_load = None
        if args_cli.load_checkpoint:
            ckpt_to_load = Path(args_cli.load_checkpoint)
        elif args_cli.resume:
            search_dirs = sorted(root_log_dir.glob("microduck_velocity_*"), key=os.path.getmtime, reverse=True)
            for s_dir in search_dirs:
                if s_dir == run_log_dir:
                    continue
                ckpt_files = sorted(
                    s_dir.glob("model_*.pt"),
                    key=lambda f: int(f.stem.split("_")[-1]) if f.stem.split("_")[-1].isdigit() else -1,
                )
                if ckpt_files:
                    ckpt_to_load = ckpt_files[-1]
                    break

        if ckpt_to_load is not None and ckpt_to_load.exists():
            print(f"[INFO]: Resuming training from checkpoint: {ckpt_to_load}\n")
            runner.load(str(ckpt_to_load))

        # 5. Execute Training Loop
        runner.learn(
            num_learning_iterations=agent_cfg.max_iterations,
            init_at_random_ep_len=True,
        )

        # 6. Save and Export Trained Policy (JIT and ONNX)
        export_dir = run_log_dir / "exported"
        export_dir.mkdir(parents=True, exist_ok=True)

        try:
            runner.export_policy_to_jit(path=str(export_dir), filename="policy.pt")
            print(f"[INFO]: Exported JIT policy to: {export_dir / 'policy.pt'}")
        except Exception as e:
            print(f"[WARNING]: JIT export failed: {e}")

        try:
            runner.export_policy_to_onnx(path=str(export_dir), filename="policy.onnx")
            print(f"[INFO]: Exported ONNX policy to: {export_dir / 'policy.onnx'}")
        except Exception as e:
            print(f"[WARNING]: ONNX export skipped: {e}")

        # Update 'latest' symlink for easy playback
        latest_symlink = root_log_dir / "latest"
        if latest_symlink.is_symlink() or latest_symlink.exists():
            latest_symlink.unlink()
        latest_symlink.symlink_to(run_log_dir.resolve(), target_is_directory=True)

        print("\n" + "=" * 65)
        print(" [SUCCESS] TRAINING COMPLETE!")
        print(f" Final Model Saved: {run_log_dir}")
        print(f" Exported Policy:   {export_dir / 'policy.pt'}")
        print(f" Latest Symlink:    {latest_symlink}")
        print("=" * 65 + "\n")

        env.close()


if __name__ == "__main__":
    main()
