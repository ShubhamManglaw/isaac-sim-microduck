"""Interactive policy evaluation and keyboard teleoperation for trained Microduck policies in Isaac Lab.

Usage:
  # Evaluate latest checkpoint interactively with keyboard control:
  cd ~/robotics/IsaacLab
  uv run python ~/robotics/isaac-sim-microduck/experiments/05_train_policy/play.py --num_envs 1

  # Run in smooth 0.5x slow-motion for detailed gait inspection:
  uv run python ~/robotics/isaac-sim-microduck/experiments/05_train_policy/play.py --num_envs 1 --speed 0.5

  # Or specify checkpoint explicitly:
  uv run python ~/robotics/isaac-sim-microduck/experiments/05_train_policy/play.py \
    --load_checkpoint ~/robotics/isaac-sim-microduck/experiments/05_train_policy/logs/latest/model_900.pt \
    --num_envs 1
"""

from __future__ import annotations

import argparse
import atexit
import math
import os
import select
import sys
import termios
import threading
import time
import tty
from pathlib import Path

# Add project roots to path
EXP_DIR = Path(__file__).resolve().parent
EXP_04_DIR = EXP_DIR.parent / "04_rl_env"
EXP_03_DIR = EXP_DIR.parent / "03_robot_model"
for p in [EXP_DIR, EXP_04_DIR, EXP_03_DIR]:
    if str(p) not in sys.path:
        sys.path.append(str(p))

import torch
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    check_rsl_rl_version,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from rsl_rl.runners import OnPolicyRunner

from microduck_env_cfg import MicroduckEnvCfg
from microduck_ppo_cfg import MicroduckPPORunnerCfg


class KeyboardTeleop:
    """Captures interactive velocity commands from either terminal or 3D OpenGL viewer."""

    def __init__(self, env=None):
        self.env = env
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_wz = 0.0

        self.step_lin = 0.05
        self.step_ang = 0.20
        self.max_lin_x_fwd = 0.40
        self.max_lin_x_bwd = -0.30
        self.max_lin_y = 0.25
        self.max_ang_z = 1.00

        self._running = True
        self._stdin_active = False

        if sys.stdin.isatty():
            try:
                self._old_term_settings = termios.tcgetattr(sys.stdin.fileno())
                atexit.register(self.restore_terminal)
                self._thread = threading.Thread(target=self._terminal_loop, daemon=True)
                self._thread.start()
                self._stdin_active = True
            except Exception as e:
                print(f"[WARNING]: Could not hook terminal input: {e}")

    def restore_terminal(self):
        if self._stdin_active:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_term_settings)
                self._stdin_active = False
            except Exception:
                pass

    def _terminal_loop(self):
        fd = sys.stdin.fileno()
        tty.setcbreak(fd)
        while self._running:
            try:
                rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
                if rlist:
                    chunk = os.read(fd, 32).decode("utf-8", errors="ignore")
                    idx = 0
                    while idx < len(chunk):
                        if chunk[idx : idx + 3] == "\x1b[A":  # Up arrow
                            self.cmd_vx = min(self.cmd_vx + self.step_lin, self.max_lin_x_fwd)
                            print(f"\r[Key UP]   Cmd Vx -> {self.cmd_vx:+.2f} m/s", flush=True)
                            idx += 3
                        elif chunk[idx : idx + 3] == "\x1b[B":  # Down arrow
                            self.cmd_vx = max(self.cmd_vx - self.step_lin, self.max_lin_x_bwd)
                            print(f"\r[Key DOWN] Cmd Vx -> {self.cmd_vx:+.2f} m/s", flush=True)
                            idx += 3
                        elif chunk[idx : idx + 3] == "\x1b[C":  # Right arrow
                            self.cmd_wz = max(self.cmd_wz - self.step_ang, -self.max_ang_z)
                            print(f"\r[Key RGHT] Cmd Wz -> {self.cmd_wz:+.2f} rad/s", flush=True)
                            idx += 3
                        elif chunk[idx : idx + 3] == "\x1b[D":  # Left arrow
                            self.cmd_wz = min(self.cmd_wz + self.step_ang, self.max_ang_z)
                            print(f"\r[Key LEFT] Cmd Wz -> {self.cmd_wz:+.2f} rad/s", flush=True)
                            idx += 3
                        else:
                            c = chunk[idx].lower()
                            if c == "w":
                                self.cmd_vx = min(self.cmd_vx + self.step_lin, self.max_lin_x_fwd)
                                print(f"\r[Key 'W']  Cmd Vx -> {self.cmd_vx:+.2f} m/s", flush=True)
                            elif c == "s":
                                self.cmd_vx = max(self.cmd_vx - self.step_lin, self.max_lin_x_bwd)
                                print(f"\r[Key 'S']  Cmd Vx -> {self.cmd_vx:+.2f} m/s", flush=True)
                            elif c == "a":
                                self.cmd_wz = min(self.cmd_wz + self.step_ang, self.max_ang_z)
                                print(f"\r[Key 'A']  Cmd Wz -> {self.cmd_wz:+.2f} rad/s", flush=True)
                            elif c == "d":
                                self.cmd_wz = max(self.cmd_wz - self.step_ang, -self.max_ang_z)
                                print(f"\r[Key 'D']  Cmd Wz -> {self.cmd_wz:+.2f} rad/s", flush=True)
                            elif c == "q":
                                self.cmd_vy = min(self.cmd_vy + self.step_lin, self.max_lin_y)
                                print(f"\r[Key 'Q']  Cmd Vy -> {self.cmd_vy:+.2f} m/s", flush=True)
                            elif c == "e":
                                self.cmd_vy = max(self.cmd_vy - self.step_lin, -self.max_lin_y)
                                print(f"\r[Key 'E']  Cmd Vy -> {self.cmd_vy:+.2f} m/s", flush=True)
                            elif c in (" ", "x", "0"):
                                self.cmd_vx = 0.0
                                self.cmd_vy = 0.0
                                self.cmd_wz = 0.0
                                print(f"\r[REST / STAND] Cmd -> [0.00, 0.00, 0.00] (Bot at Rest)", flush=True)
                            idx += 1
            except Exception:
                break

    def update(self) -> tuple[float, float, float]:
        # Also poll 3D OpenGL viewer keys if window is focused
        if self.env is not None and hasattr(self.env.sim, "visualizers") and self.env.sim.visualizers:
            viz = self.env.sim.visualizers[0]
            viewer = getattr(viz, "_viewer", viz)
            if hasattr(viewer, "_paused") and viewer._paused:
                viewer._paused = False
                self.cmd_vx = 0.0
                self.cmd_vy = 0.0
                self.cmd_wz = 0.0
                print("\r[REST / STAND] Space pressed in 3D Window -> Bot brought to rest!           ", flush=True)

            check_key = None
            if hasattr(viz, "is_key_down"):
                check_key = viz.is_key_down
            elif hasattr(viewer, "is_key_down"):
                check_key = viewer.is_key_down

            if check_key is not None:
                changed = False
                if check_key("w") or check_key("up"):
                    self.cmd_vx = min(self.cmd_vx + 0.02, self.max_lin_x_fwd)
                    changed = True
                elif check_key("s") or check_key("down"):
                    self.cmd_vx = max(self.cmd_vx - 0.02, self.max_lin_x_bwd)
                    changed = True

                if check_key("a") or check_key("left"):
                    self.cmd_wz = min(self.cmd_wz + 0.05, self.max_ang_z)
                    changed = True
                elif check_key("d") or check_key("right"):
                    self.cmd_wz = max(self.cmd_wz - 0.05, -self.max_ang_z)
                    changed = True

                if check_key("q"):
                    self.cmd_vy = min(self.cmd_vy + 0.02, self.max_lin_y)
                    changed = True
                elif check_key("e"):
                    self.cmd_vy = max(self.cmd_vy - 0.02, -self.max_lin_y)
                    changed = True

                if check_key("space") or check_key("x") or check_key("0"):
                    self.cmd_vx = 0.0
                    self.cmd_vy = 0.0
                    self.cmd_wz = 0.0
                    print("\r[REST / STAND] Rest Key pressed -> Bot brought to rest!               ", flush=True)

                if changed:
                    print(
                        f"\r[COMMAND] Vx={self.cmd_vx:+.2f} m/s | Vy={self.cmd_vy:+.2f} m/s | Wz={self.cmd_wz:+.2f} rad/s    ",
                        end="",
                        flush=True,
                    )

        return self.cmd_vx, self.cmd_vy, self.cmd_wz

    def close(self):
        self._running = False
        self.restore_terminal()


def find_latest_checkpoint(root_log_dir: Path) -> Path | None:
    """Finds the most recent checkpoint file from training logs."""
    latest_symlink = root_log_dir / "latest"
    target_dir = latest_symlink.resolve() if latest_symlink.exists() else root_log_dir

    if target_dir.is_dir():
        pts = sorted(
            target_dir.glob("model_*.pt"),
            key=lambda f: int(f.stem.split("_")[-1]) if f.stem.split("_")[-1].isdigit() else -1,
        )
        if pts:
            return pts[-1]

    all_pts = sorted(
        root_log_dir.glob("**/model_*.pt"),
        key=lambda f: int(f.stem.split("_")[-1]) if f.stem.split("_")[-1].isdigit() else -1,
    )
    if all_pts:
        return all_pts[-1]
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Microduck policy with keyboard teleoperation.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel robots in playback (default: 1 for clean focus).")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to policy checkpoint (.pt file).")
    parser.add_argument("--speed", type=float, default=1.0, help="Simulation playback speed multiplier (1.0 = real-time, 0.5 = slow-motion).")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum evaluation steps (default: run forever).")
    parser.add_argument("--export", action="store_true", help="Export policy to ONNX and JIT formats upon load.")
    add_launcher_args(parser)
    parser.set_defaults(visualizer="newton_gl")
    args_cli = parser.parse_args()

    # 1. Locate Checkpoint
    log_dir = EXP_DIR / "logs"
    if args_cli.load_checkpoint is not None:
        checkpoint_path = Path(args_cli.load_checkpoint).resolve()
    else:
        checkpoint_path = find_latest_checkpoint(log_dir)

    if checkpoint_path is None or not checkpoint_path.is_file():
        print(f"[ERROR]: No checkpoint found in {log_dir} and none specified via --load_checkpoint.")
        print("Please train a model first using train.py.")
        sys.exit(1)

    print(f"\n[INFO]: Loading policy checkpoint: {checkpoint_path}")

    # 2. Physics Backend Configuration
    physics_cfg = NewtonCfg(
        num_substeps=1,
        solver_cfg=MJWarpSolverCfg(iterations=100, ls_iterations=50),
        load_visual_shapes=(args_cli.visualizer != "none"),
    )

    with launch_simulation(cfg=physics_cfg, launcher_args=args_cli) as resolved_physics_cfg:
        # Configure Environment for Playback
        env_cfg = MicroduckEnvCfg()
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.scene.env_spacing = 2.0
        env_cfg.sim.physics = resolved_physics_cfg
        env_cfg.sim.render_interval = env_cfg.decimation  # 1 render per policy step (smooth 50 Hz)

        # Disable all random disturbances during interactive evaluation
        env_cfg.events.push_robot = None
        # Pin reset base strictly to origin facing forward (no random yaw or position jumps)
        env_cfg.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        env_cfg.events.reset_base.params["velocity_range"] = {}

        # 3. Instantiate Vectorized RL Environment with STRICT [-1.0, 1.0] Action Clipping
        env = ManagerBasedRLEnv(cfg=env_cfg)
        wrapped_env = RslRlVecEnvWrapper(env, clip_actions=1.0)

        # 4. Load Agent Policy
        agent_cfg: RslRlOnPolicyRunnerCfg = MicroduckPPORunnerCfg()
        installed_version = check_rsl_rl_version()
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

        log_root_path = checkpoint_path.parent
        runner = OnPolicyRunner(
            wrapped_env, agent_cfg.to_dict(), log_dir=str(log_root_path), device=env.unwrapped.device
        )
        runner.load(str(checkpoint_path))
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        # Export if requested
        if args_cli.export:
            export_dir = log_root_path / "exported"
            export_dir.mkdir(exist_ok=True, parents=True)
            export_policy_as_jit(runner.alg.actor_critic, str(export_dir / "policy.pt"))
            try:
                export_policy_as_onnx(
                    runner.alg.actor_critic,
                    normalizer=runner.obs_normalizer,
                    path=str(export_dir / "policy.onnx"),
                )
                print(f"[INFO]: Exported policy to {export_dir}")
            except Exception as e:
                print(f"[WARNING]: ONNX export skipped: {e}")

        # 5. Initialize Interactive Keyboard Teleop
        teleop = KeyboardTeleop(env=env)

        print("\n" + "=" * 70)
        print(" MICRODUCK INTERACTIVE KEYBOARD TELEOPERATION")
        print("=" * 70)
        print(" Controls (usable in BOTH Terminal and 3D Window):")
        print("   [W] / [Up Arrow]    : Walk Forward       (+Vx)")
        print("   [S] / [Down Arrow]  : Walk Backward      (-Vx)")
        print("   [A] / [Left Arrow]  : Turn Left (CCW)    (+Wz)")
        print("   [D] / [Right Arrow] : Turn Right (CW)    (-Wz)")
        print("   [Q]                 : Strafe Left        (+Vy)")
        print("   [E]                 : Strafe Right       (-Vy)")
        print("   [Space] or [X]      : REST / STAND STILL (Brings bot to rest)")
        print("   [Ctrl+C]            : Exit Playback")
        print("=" * 70)
        print(f" Loaded Model:   {checkpoint_path.name}")
        print(f" Robot Envs:     {args_cli.num_envs}")
        print(f" Playback Speed: {args_cli.speed:.2f}x (Real-time locked)")
        print("=" * 70 + "\n")

        # Completely freeze automated command resampling so user teleoperation maintains full control
        cmd_term = env.command_manager.get_term("base_velocity")
        cmd_term.time_left[:] = 1e9
        cmd_term.cfg.resampling_time_range = (1e9, 1e9)
        cmd_term.cfg.rel_forward_envs = 0.0
        cmd_term.cfg.rel_turn_in_place_envs = 0.0
        cmd_term.cfg.rel_heading_envs = 0.0
        cmd_term.cfg.rel_standing_envs = 0.0
        cmd_term.is_heading_env[:] = False
        cmd_term.is_standing_env[:] = False

        obs = wrapped_env.get_observations()
        step = 0
        t_last = time.perf_counter()

        # Target step duration in seconds for real-time lock
        target_step_dt = env.unwrapped.step_dt / max(args_cli.speed, 0.1)

        try:
            while env.sim.is_running():
                t_step_start = time.perf_counter()

                if args_cli.max_steps is not None and step >= args_cli.max_steps:
                    break

                # 1. Update velocity command from keyboard teleop
                vx, vy, wz = teleop.update()
                cmd_tensor = torch.tensor(
                    [[vx, vy, wz]], device=env.unwrapped.device, dtype=torch.float32
                ).repeat(env.num_envs, 1)

                is_at_rest = (vx == 0.0 and vy == 0.0 and wz == 0.0)
                cmd_term.vel_command_b[:] = cmd_tensor
                cmd_term.is_standing_env[:] = is_at_rest
                cmd_term.is_heading_env[:] = False

                # Feed user velocity command directly into policy observation slice (dims 48..51)
                obs["policy"][:, 48:51] = cmd_tensor

                # 2. Forward pass through trained Actor network with strict action clamping
                with torch.no_grad():
                    actions = policy(obs)
                    actions = torch.clamp(actions, -1.0, 1.0)

                # If at rest (Space pressed or zero velocity), lock robot into rock-solid default standing pose
                if is_at_rest:
                    actions = torch.zeros_like(actions)

                # 3. Step environment
                obs, rewards, dones, extras = wrapped_env.step(actions)

                # Keep teleop command persistent for next step
                cmd_term.vel_command_b[:] = cmd_tensor
                cmd_term.is_heading_env[:] = False
                obs["policy"][:, 48:51] = cmd_tensor

                # 4. Telemetry printout every 50 steps (~1.0 second)
                if step > 0 and step % 50 == 0:
                    t_now = time.perf_counter()
                    fps = 50.0 / (t_now - t_last)
                    t_last = t_now
                    robot = env.scene["robot"]
                    mean_z = robot.data.root_link_pos_w.torch[:, 2].mean().item()
                    mean_vx = robot.data.root_link_lin_vel_w.torch[:, 0].mean().item()
                    mean_wz = robot.data.root_link_ang_vel_w.torch[:, 2].mean().item()
                    mode_str = "STAND / REST" if is_at_rest else "LOCOMOTION"
                    print(
                        f"Step {step:5d} | {fps:4.1f} fps | Mode: [{mode_str:12s}] | Height: {mean_z*100:4.1f} cm | "
                        f"Cmd: [Vx: {vx:+.2f}, Vy: {vy:+.2f}, Wz: {wz:+.2f}] | "
                        f"Actual: [Vx: {mean_vx:+.2f} m/s, Wz: {mean_wz:+.2f} rad/s]"
                    )

                # 5. Strict Real-Time Wall-Clock Lock
                step_elapsed = time.perf_counter() - t_step_start
                sleep_sec = target_step_dt - step_elapsed
                if sleep_sec > 0:
                    time.sleep(sleep_sec)

                step += 1

        finally:
            teleop.close()
            env.close()
            print("\nPlayback closed cleanly.")


if __name__ == "__main__":
    main()
