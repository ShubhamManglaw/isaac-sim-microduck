from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

# Add paths
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

import torch
from isaaclab.app import add_launcher_args, launch_simulation
import isaaclab.sim as sim_utils
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass

from microduck_cfg import MicroduckArticulationCfg
from isaaclab.sensors import ContactSensorCfg


@configclass
class InspectSceneCfg(InteractiveSceneCfg):
    """Scene for 3D visual inspection of Microduck joint constraints."""

    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.CuboidCfg(
            size=(10.0, 10.0, 0.1),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
    )

    robot: MicroduckArticulationCfg = MicroduckArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot"
    )
    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*ankle.*",
        history_length=3,
        track_air_time=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Inspect all Microduck 14-DOF joint angles & constraints visually in 3D.")
    parser.add_argument("--mode", type=str, default="sweep", choices=["sweep", "stand", "crawling", "squat"],
                        help="Inspection mode: 'sweep' (auto sweeps each joint through limits), 'stand', 'crawling', 'squat'.")
    parser.add_argument("--fix-base", action="store_true", default=False, help="Hold root trunk pinned in mid-air (z=0.18m).")
    parser.add_argument("--joint", type=str, default=None, help="Specific joint name to sweep (e.g., 'left_hip_pitch'). If None, cycles through all.")
    parser.add_argument("--speed", type=float, default=1.0, help="Joint sweep oscillation speed multiplier.")
    add_launcher_args(parser)
    parser.set_defaults(visualizer="newton_gl")
    args_cli = parser.parse_args()

    physics_cfg = NewtonCfg(
        num_substeps=1,
        solver_cfg=MJWarpSolverCfg(),
        load_visual_shapes=True,
    )

    with launch_simulation(cfg=physics_cfg, launcher_args=args_cli) as resolved_physics_cfg:
        sim_cfg = sim_utils.SimulationCfg(
            dt=1.0 / 120.0,
            device=args_cli.device,
            physics=resolved_physics_cfg,
        )
        sim = sim_utils.SimulationContext(sim_cfg)
        scene_cfg = InspectSceneCfg(num_envs=1, env_spacing=2.5)
        scene = InteractiveScene(scene_cfg)
        sim.reset()
        robot = scene["robot"]
        contact_sensor = scene["contact_forces"]
        sim_dt = sim.get_physics_dt()

        # Initialize to STAND2
        robot.write_root_pose_to_sim_index(root_pose=robot.data.default_root_pose.torch.clone(), env_ids=slice(None))
        robot.write_joint_state_to_sim_index(position=robot.data.default_joint_pos.torch.clone(), velocity=robot.data.default_joint_vel.torch.clone(), env_ids=slice(None))
        robot.actuators.target_command.set_position_index(value=robot.data.default_joint_pos.torch.clone())
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(sim_dt)

        joint_names = robot.joint_names
        num_joints = robot.num_joints
        lower_limits = robot.data.soft_joint_pos_limits.torch[0, :, 0].cpu().numpy()
        upper_limits = robot.data.soft_joint_pos_limits.torch[0, :, 1].cpu().numpy()
        default_pos = robot.data.default_joint_pos.torch.clone()

        # Camera view setup
        sim.set_camera_view(eye=(0.42, 0.42, 0.20), target=(0.0, 0.0, 0.11))

        print("\n" + "=" * 70)
        print(" [ISAAC LAB] MICRODUCK 3D JOINT CONSTRAINT INSPECTOR")
        print("=" * 70)
        print(f"Mode:            {args_cli.mode.upper()}")
        print(f"Fixed Base:      {args_cli.fix_base}")
        print(f"Total Joints:    {num_joints}")
        print("=" * 70)
        print("All 14 Joints with Soft Limits:")
        for idx, name in enumerate(joint_names):
            q_def = default_pos[0, idx].item()
            print(f"  [{idx:2d}] {name:16s}: def={q_def:+.3f} rad ({q_def*180/math.pi:+5.1f}°) | lim=[{lower_limits[idx]:+.2f}, {upper_limits[idx]:+.2f}] rad")
        print("=" * 70)
        print("Controls: Close viewer window or press Ctrl+C to stop.")
        print("=" * 70 + "\n")

        fixed_root_pose = robot.data.default_root_pose.torch.clone()
        fixed_root_pose[:, 2] = 0.18
        fixed_root_vel = torch.zeros((robot.num_instances, 6), device=robot.device)

        step = 0
        current_joint_idx = 0
        cycle_duration_s = 4.0 / max(args_cli.speed, 0.1)
        last_switch_t = 0.0

        if args_cli.joint is not None:
            matches = [i for i, name in enumerate(joint_names) if args_cli.joint.lower() in name.lower()]
            if matches:
                current_joint_idx = matches[0]
                print(f"[INFO]: Focusing on joint [{current_joint_idx}] '{joint_names[current_joint_idx]}'")

        while sim.is_running():
            t = step * sim_dt
            targets = default_pos.clone()

            if args_cli.fix_base:
                robot.write_root_pose_to_sim_index(root_pose=fixed_root_pose, env_ids=slice(None))
                robot.write_root_velocity_to_sim_index(root_velocity=fixed_root_vel, env_ids=slice(None))

            if args_cli.mode == "sweep":
                # Sweep the selected joint through its soft limit range
                if args_cli.joint is None and (t - last_switch_t) >= cycle_duration_s:
                    current_joint_idx = (current_joint_idx + 1) % num_joints
                    last_switch_t = t
                    print(f"\n--> Now sweeping joint [{current_joint_idx:2d}] '{joint_names[current_joint_idx]}'")

                j_idx = current_joint_idx
                j_min = float(lower_limits[j_idx])
                j_max = float(upper_limits[j_idx])
                # Smooth sine oscillation between min and max
                phase = 0.5 * (1.0 + math.sin(2.0 * math.pi * (t - last_switch_t) / cycle_duration_s - math.pi / 2))
                targets[0, j_idx] = float(j_min + phase * (j_max - j_min))

            elif args_cli.mode == "crawling":
                # Reproduce the observed squatting posture from model_2250.pt
                targets[0, 2] = -1.254  # left_hip_pitch (deep squat)
                targets[0, 3] = 0.978   # left_knee (forward flex)
                targets[0, 4] = 1.347   # left_ankle
                targets[0, 1] = -0.365  # left_hip_roll
                targets[0, 11] = 0.170  # right_hip_pitch
                targets[0, 12] = -1.054 # right_knee
                targets[0, 13] = -1.388 # right_ankle
                targets[0, 10] = -0.365 # right_hip_roll

            elif args_cli.mode == "squat":
                amp = 0.25 * math.sin(2.0 * math.pi * 0.5 * t)
                targets[0, 2] -= amp * 0.5
                targets[0, 3] -= amp
                targets[0, 4] += amp * 0.5
                targets[0, 11] += amp * 0.5
                targets[0, 12] += amp
                targets[0, 13] -= amp * 0.5

            robot.actuators.target_command.set_position_index(value=targets)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(sim_dt)

            if sim.is_rendering:
                sim.render()

            if step % 50 == 0:
                root_z = robot.data.root_link_pos_w.torch[0, 2].item()
                if args_cli.mode == "sweep":
                    q_act = robot.data.joint_pos.torch[0, current_joint_idx].item()
                    q_tgt = targets[0, current_joint_idx].item()
                    print(
                        f"Step {step:5d} | Joint '{joint_names[current_joint_idx]}': "
                        f"tgt={q_tgt:+.3f} rad ({q_tgt*180/math.pi:+5.1f}°) | "
                        f"actual={q_act:+.3f} rad | Root z={root_z*100:.1f} cm"
                    )
                else:
                    print(f"Step {step:5d} | Root Height z={root_z*100:.1f} cm | Mode={args_cli.mode}")

            step += 1

        print("\nInspection closed cleanly.")


if __name__ == "__main__":
    main()
