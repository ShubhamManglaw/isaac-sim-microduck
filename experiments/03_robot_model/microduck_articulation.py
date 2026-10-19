from __future__ import annotations

import argparse
import math
import torch

from isaaclab.app import add_launcher_args, launch_simulation
import isaaclab.sim as sim_utils
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass

from microduck_cfg import MicroduckArticulationCfg
from isaaclab.sensors import ContactSensorCfg

@configclass
class MicroduckSceneCfg(InteractiveSceneCfg):
    """Scene configuration with static ground plane and Microduck articulation."""

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
    

def run_simulator(sim: SimulationContext, scene: InteractiveScene, mode: str = "stand", fix_base: bool = False, max_steps: int | None = None):
    """Executes the simulation loop, telemetry readback, and joint control."""
    robot = scene["robot"]
    contact_sensor = scene["contact_forces"]
    sim_dt = sim.get_physics_dt()

    print("\n" + "=" * 60)
    print(" [ISAAC LAB] MICRODUCK ARTICULATION INITIALIZED")
    print("=" * 60)
    print(f"Number of instances: {robot.num_instances}")
    print(f"Is fixed base:       {robot.is_fixed_base}")
    print(f"Base DOFs:           {robot.num_base_dofs} (3 linear + 3 angular)")
    print(f"Number of bodies:    {robot.num_bodies}")
    print(f"Number of joints:    {robot.num_joints}")
    print("\nJoint Names (0-13):")
    for idx, name in enumerate(robot.joint_names):
        print(f"  [{idx:2d}] {name}")

    # 1. Initialize robot to configured STAND2 posture
    robot.write_root_pose_to_sim(robot.data.default_root_pose)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    robot.actuators.target_command.set_position_index(value=robot.data.default_joint_pos)
    scene.write_data_to_sim()
    sim.step(render=False)
    scene.update(sim_dt)

    print("\n" + "-" * 60)
    print(" INITIAL STATE READBACK")
    print("-" * 60)
    root_pos = robot.data.root_link_pos_w.torch[0].cpu().numpy()
    root_vel = robot.data.root_link_lin_vel_w.torch[0].cpu().numpy()
    q_init = robot.data.joint_pos.torch[0].cpu().numpy()

    print(f"Root link position (x, y, z):  [{root_pos[0]:.4f}, {root_pos[1]:.4f}, {root_pos[2]:.4f}] m")
    print(f"Root link linear velocity:     [{root_vel[0]:.4f}, {root_vel[1]:.4f}, {root_vel[2]:.4f}] m/s")
    print(f"Left knee angle:               {q_init[3]:.4f} rad")
    print(f"Right knee angle:              {q_init[12]:.4f} rad")

    print("\n" + "-" * 60)
    print(f" RUNNING INTERACTIVE SIMULATION LOOP (Mode: {mode.upper()}, Fix Base: {fix_base})")
    print(" (Close viewer window or press Ctrl+C to stop)")
    print("-" * 60)
    print(" Mouse Controls: Left Click = Orbit | Middle / Shift+Left = Pan | Scroll = Zoom")
    print("-" * 60)

    # Set default camera to view Microduck close up
    sim.set_camera_view(eye=(0.45, 0.45, 0.22), target=(0.0, 0.0, 0.12))

    # Base stance targets from configured default
    base_targets = robot.data.default_joint_pos.torch.clone()
    fixed_root_pose = robot.data.default_root_pose.torch.clone()
    fixed_root_pose[:, 2] = 0.18  # Hold in mid-air if fix_base=True
    fixed_root_vel = torch.zeros_like(robot.data.root_link_lin_vel_w.torch)
    step = 0

    while sim.is_running():
        if max_steps is not None and step >= max_steps:
            break
        t = step * sim_dt
        targets = base_targets.clone()

        if fix_base:
            # Hold root link fixed in place
            robot.write_root_pose_to_sim(fixed_root_pose)
            robot.write_root_velocity_to_sim(fixed_root_vel)

        if mode == "squat":
            # Continuous sinusoidal squatting motion
            amp = 0.20 * math.sin(2.0 * math.pi * 0.5 * t)
            targets[:, 2] -= amp * 0.5   # left_hip_pitch
            targets[:, 3] -= amp         # left_knee flex
            targets[:, 4] += amp * 0.5   # left_ankle
            targets[:, 11] += amp * 0.5  # right_hip_pitch
            targets[:, 12] += amp        # right_knee flex
            targets[:, 13] -= amp * 0.5  # right_ankle
        elif mode == "stand":
            # Balanced standing posture with gentle duck head looking around
            head_look_yaw = 0.25 * math.sin(2.0 * math.pi * 0.4 * t)
            head_nod_pitch = 0.15 * math.cos(2.0 * math.pi * 0.6 * t)
            targets[:, 7] = head_look_yaw     # head_yaw
            targets[:, 6] += head_nod_pitch   # head_pitch

        robot.actuators.target_command.set_position_index(value=targets)

        # 1. Write computed torques to GPU solver memory
        scene.write_data_to_sim()

        # 2. Physics integration step
        sim.step(render=False)

        # 3. Read back new state from GPU buffers
        scene.update(sim_dt)

        # 4. Render frame to visualizer window if active
        if sim.is_rendering:
            sim.render()

        if step % 60 == 0:
            z = robot.data.root_link_pos_w.torch[0, 2].item()
            # Read vertical ground reaction force F_z for left (index 0) and right (index 1) foot
            net_forces = contact_sensor.data.net_forces_w.torch[0]
            f_l_z = net_forces[0, 2].item()
            f_r_z = net_forces[1, 2].item()
            lk = robot.data.joint_pos.torch[0, 3].item()
            rk = robot.data.joint_pos.torch[0, 12].item()
            print(f" t = {t:6.2f} s | Root Z: {z:.4f} m | L_Foot Fz: {f_l_z:+5.1f} N | R_Foot Fz: {f_r_z:+5.1f} N | L_Knee: {lk:+.3f} rad | R_Knee: {rk:+.3f} rad")

        step += 1

    print("\nSimulation ended.")



def main():
    parser = argparse.ArgumentParser(description="Microduck Articulation Simulation in Isaac Lab.")
    parser.add_argument("--mode", type=str, default="stand", choices=["stand", "squat"], help="Motion mode: 'stand' (balanced standing) or 'squat'")
    parser.add_argument("--fix-base", action="store_true", default=False, help="Fix the trunk in mid-air to inspect joint articulation freely")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum simulation steps before exiting")
    add_launcher_args(parser)
    parser.set_defaults(visualizer="newton_gl")
    args_cli = parser.parse_args()

    physics_cfg = NewtonCfg(
        num_substeps=10,
        collision_decimation=1,
        solver_cfg=MJWarpSolverCfg(),
        load_visual_shapes=True,
    )

    with launch_simulation(cfg=physics_cfg, launcher_args=args_cli) as resolved_physics_cfg:
        sim_cfg = SimulationCfg(
            dt=1.0 / 120.0,
            device=args_cli.device,
            physics=resolved_physics_cfg,
        )

        sim = SimulationContext(sim_cfg)
        scene = InteractiveScene(MicroduckSceneCfg(num_envs=1, env_spacing=1.0))

        sim.reset()
        run_simulator(sim, scene, mode=args_cli.mode, fix_base=args_cli.fix_base, max_steps=args_cli.max_steps)


if __name__ == "__main__":
    main()

