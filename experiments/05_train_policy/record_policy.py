"""Record videos and screenshots of a trained Microduck policy, without opening a window.

Runs the same scenarios as evaluate_policy.py with one robot, renders them with a headless Newton GL
camera that follows the robot, and writes for each scenario:

  <out>/<scenario>.mp4         video (25 fps)
  <out>/<scenario>_sheet.png   contact sheet: 8 frames spread over the scenario, with time stamps

Usage (from ~/robotics/IsaacLab):

  uv run python ~/robotics/isaac-sim-microduck/experiments/05_train_policy/record_policy.py \\
      --load_checkpoint /path/to/model.pt --out ~/microduck_videos
  uv run python .../record_policy.py --scenarios stop_from_0.75,turn_left_1.5,hurdle_low
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
for p in [EXP_DIR, EXP_DIR.parent / "04_rl_env", EXP_DIR.parent / "03_robot_model"]:
    if str(p) not in sys.path:
        sys.path.append(str(p))

import numpy as np
import torch
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_visualizers.newton.newton_visualizer_cfg import NewtonGLVisualizerCfg
from PIL import Image, ImageDraw

from evaluate_policy import HURDLE_DISTANCE, SCENARIOS, find_latest_checkpoint
from microduck_env_cfg import HURDLE_NAMES, MicroduckEnv, MicroduckEnvCfg
from microduck_ppo_cfg import MicroduckPPORunnerCfg, build_runner

WIDTH, HEIGHT = 960, 540
FRAME_EVERY = 2  # policy steps per video frame (50 Hz / 2 = 25 fps)


def follow_camera(viewer, target: np.ndarray, heading: float, distance: float = 0.40, height: float = 0.14,
                  side_angle_deg: float = 55.0) -> None:
    """Place the viewer camera behind-and-beside the robot, looking at it."""
    angle = heading + math.pi + math.radians(side_angle_deg)
    eye = target + np.array([distance * math.cos(angle), distance * math.sin(angle), height])
    look = target - eye
    yaw = math.degrees(math.atan2(look[1], look[0]))
    pitch = math.degrees(math.atan2(look[2], math.hypot(look[0], look[1])))
    import warp as wp

    viewer.set_camera(wp.vec3(*eye.tolist()), pitch, yaw)


def main() -> int:
    parser = argparse.ArgumentParser(description="Record videos and contact sheets of a Microduck policy.")
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--out", type=str, default=str(EXP_DIR / "logs" / "videos"))
    parser.add_argument("--scenarios", type=str, default=None, help="Comma-separated subset (default: all).")
    add_launcher_args(parser)
    parser.set_defaults(visualizer="newton_gl")
    args = parser.parse_args()

    ckpt = Path(args.load_checkpoint).expanduser().resolve() if args.load_checkpoint else find_latest_checkpoint(EXP_DIR / "logs")
    if ckpt is None or not ckpt.is_file():
        print("[ERROR]: No checkpoint found. Pass --load_checkpoint.")
        return 1
    scenarios = SCENARIOS
    if args.scenarios:
        wanted = set(args.scenarios.split(","))
        scenarios = [s for s in SCENARIOS if s.name in wanted]
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    have_ffmpeg = shutil.which("ffmpeg") is not None

    physics_cfg = NewtonCfg(num_substeps=1, solver_cfg=MJWarpSolverCfg(), load_visual_shapes=True)
    with launch_simulation(cfg=physics_cfg, launcher_args=args) as resolved:
        env_cfg = MicroduckEnvCfg()
        env_cfg.scene.num_envs = 1
        env_cfg.sim.physics = resolved
        env_cfg.sim.visualizer_cfgs = NewtonGLVisualizerCfg(
            headless=True, window_width=WIDTH, window_height=HEIGHT, enable_picking=False
        )
        env_cfg.episode_length_s = 60.0
        env_cfg.events.push_robot = None
        env_cfg.events.reset_fallen.params["fraction"] = 0.0
        env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        env_cfg.events.reset_base.params["velocity_range"] = {}
        env_cfg.commands.base_velocity.external_control = True
        env_cfg.commands.base_velocity.initial_speed_scale = 1.0
        env_cfg.curriculum.command_speed = None
        env_cfg.curriculum.smoothness_ramp = None
        env_cfg.observations.policy.enable_corruption = False
        env_cfg.observations.policy.tof_depth.params["noise"] = 0.0

        agent_cfg = MicroduckPPORunnerCfg()
        env = MicroduckEnv(cfg=env_cfg)
        wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = build_runner(wrapped_env, agent_cfg, log_dir=None, device=env.device)
        loaded = torch.load(str(ckpt), map_location=env.device, weights_only=False)
        runner.alg.load(loaded, load_cfg={"actor": True}, strict=True)
        policy = runner.get_inference_policy(device=env.device)

        visualizer = env.sim.visualizers[0]
        viewer = visualizer._viewer
        robot = env.scene["robot"]
        cmd_term = env.command_manager.get_term("base_velocity")
        cmd_term.time_left[:] = 1e9
        all_ids = torch.arange(1, device=env.device)
        clip = float(agent_cfg.clip_actions)

        for scenario in scenarios:
            frame_dir = out_dir / f"{scenario.name}_frames"
            if frame_dir.exists():
                shutil.rmtree(frame_dir)
            frame_dir.mkdir()
            wrapped_env.reset()
            cmd_term.park_hurdle(all_ids)
            if scenario.hurdle is not None:
                cmd_term.place_hurdle_ahead(all_ids, index=HURDLE_NAMES.index(f"hurdle_{scenario.hurdle}"), distance=HURDLE_DISTANCE)
            obs = wrapped_env.get_observations()
            steps = int(round(scenario.duration / env.step_dt))
            frames: list[tuple[float, Path]] = []
            fell_at = None
            for step in range(steps):
                t = step * env.step_dt
                vx, vy, wz = scenario.command(t)
                cmd_term.vel_command_b[:] = torch.tensor([vx, vy, wz], device=env.device)
                cmd_term.head_command[:] = torch.tensor(scenario.head, device=env.device)
                with torch.no_grad():
                    action = torch.clamp(policy(obs), -clip, clip)
                obs, _, dones, _ = wrapped_env.step(action)
                if fell_at is None and bool(env.termination_manager.terminated[0]):
                    fell_at = t
                if step % FRAME_EVERY == 0:
                    target = robot.data.root_link_pos_w.torch[0].cpu().numpy().copy()
                    target[2] = 0.07
                    follow_camera(viewer, target, float(robot.data.heading_w.torch[0]))
                    rgb = visualizer.render_rgb_array()
                    path = frame_dir / f"{len(frames):05d}.png"
                    image = Image.fromarray(np.ascontiguousarray(rgb[..., :3]))
                    draw = ImageDraw.Draw(image)
                    draw.rectangle([0, 0, WIDTH, 26], fill=(0, 0, 0))
                    label = f"{scenario.name}  t={t:4.1f}s  cmd vx={vx:+.2f} vy={vy:+.2f} wz={wz:+.2f}"
                    if fell_at is not None:
                        label += f"  FELL at {fell_at:.1f}s"
                    draw.text((8, 6), label, fill=(255, 255, 255))
                    image.save(path)
                    frames.append((t, path))

            # Contact sheet: 8 frames spread over the scenario, 4 x 2 grid at half size
            picks = [frames[round(i * (len(frames) - 1) / 7)] for i in range(8)]
            tw, th = WIDTH // 2, HEIGHT // 2
            sheet = Image.new("RGB", (tw * 4, th * 2), (0, 0, 0))
            for i, (_, path) in enumerate(picks):
                sheet.paste(Image.open(path).resize((tw, th)), ((i % 4) * tw, (i // 4) * th))
            sheet_path = out_dir / f"{scenario.name}_sheet.png"
            sheet.save(sheet_path)

            video_path = out_dir / f"{scenario.name}.mp4"
            if have_ffmpeg:
                subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(round(1.0 / (env.step_dt * FRAME_EVERY))),
                     "-i", str(frame_dir / "%05d.png"), "-pix_fmt", "yuv420p", "-vcodec", "libx264", str(video_path)],
                    check=False,
                )
                shutil.rmtree(frame_dir)
            print(f"[record] {scenario.name:20s} -> {sheet_path.name}" + (f", {video_path.name}" if have_ffmpeg else "")
                  + (f"  (fell at {fell_at:.1f} s)" if fell_at is not None else ""), flush=True)
        env.close()

    print(f"\nWritten to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
