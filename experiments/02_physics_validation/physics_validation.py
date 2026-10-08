
import argparse

from isaaclab.app import add_launcher_args, launch_simulation
import isaaclab.sim as sim_utils
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.sim import SimulationCfg, SimulationContext


@configclass
class RigidBodyTestCfg(InteractiveSceneCfg):

    # Ground
    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.CuboidCfg(
            size=(10.0, 10.0, 0.1),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, -0.05)
        ),
    )

    # Dynamic cube
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
            ),
            mass_props=sim_utils.MassCfg(mass=1.0),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0)
        ),
    )


# Newton physics configuration (MJWarp is the validated solver for rigid objects and articulations)
physics_cfg = NewtonCfg(
    num_substeps=10,
    collision_decimation=1,
    solver_cfg=MJWarpSolverCfg(),
)


def main():

    parser = argparse.ArgumentParser()
    add_launcher_args(parser)
    parser.set_defaults(visualizer="newton")
    args_cli = parser.parse_args()

    with launch_simulation(
        cfg=physics_cfg,
        launcher_args=args_cli,
    ) as resolved_physics_cfg:

        sim_cfg = SimulationCfg(
            dt=1.0 / 120.0,
            device=args_cli.device,
            physics=resolved_physics_cfg,
        )

        sim = SimulationContext(sim_cfg)

        sim.set_camera_view(
            eye=[3.0, 3.0, 2.0],
            target=[0.0, 0.0, 0.5],
        )

        scene = InteractiveScene(
            RigidBodyTestCfg(
                num_envs=1,
                env_spacing=1.0,
            )
        )

        sim.reset()

        cube = scene["cube"]
        dt = sim_cfg.dt
        num_steps = 120

        measurements = []

        # Record initial state
        z = cube.data.root_link_pos_w.torch[0, 2].item()
        vz = cube.data.root_link_lin_vel_w.torch[0, 2].item()

        measurements.append([0.0, z, vz])

        print(f"Initial | Z: {z:.4f}m | Vz: {vz:.4f}m/s")

        # Simulation loop
        for step in range(num_steps):

            scene.write_data_to_sim()
            sim.step()
            scene.update(dt)

            z = cube.data.root_link_pos_w.torch[0, 2].item()
            vz = cube.data.root_link_lin_vel_w.torch[0, 2].item()

            t = (step + 1) * dt
            measurements.append([t, z, vz])

            if (step + 1) % 12 == 0:
                print(
                    f"Time: {t:.3f}s | "
                    f"Z: {z:.4f}m | "
                    f"Vz: {vz:.4f}m/s"
                )

        print("\nSimulation completed.")
        print("Final cube position:", cube.data.root_link_pos_w.torch)
        print("Final cube velocity:", cube.data.root_link_lin_vel_w.torch)


if __name__ == "__main__":
    main()
