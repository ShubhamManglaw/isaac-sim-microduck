import argparse
from isaaclab.app import add_launcher_args, launch_simulation
import isaaclab.sim as sim_utils
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.sim import SimulationCfg, SimulationContext

# TODO 1: Configure command-line arguments.
@configclass
class RigidBodyTestCfg(InteractiveSceneCfg):
    # Define groung
    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/ground",

        spawn=sim_utils.CuboidCfg(
            size=(10.0, 10.0, 0.1),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        ),

        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, -0.05)
        )
    )
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",

        spawn=sim_utils.CuboidCfg(
            # TODO: Define cube size
            size=(0.1, 0.1, 0.1),
            # TODO: Define rigid body properties
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                
            ),
            # TODO: Define mass properties
            mass_props=sim_utils.MassCfg(mass=1.0),
            # TODO: Define collision properties
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        ),

        init_state=RigidObjectCfg.InitialStateCfg(
            # TODO: Define initial position
            pos=(0.0, 0.0, 1.0)
        )
    )
    # define cube
    # pass

physics_cfg = NewtonCfg(
    num_substeps=10,
    collision_decimation=1,
    solver_cfg=MJWarpSolverCfg(),
)
# TODO 2: Define the rigid-body physics configuration.


# TODO 3: Define the scene configuration.
# The scene should contain:
# - Ground plane
# - One cube


# TODO 4: Create the main simulation function.
# TODO 5: Create the simulation loop.

def main():

    parser = argparse.ArgumentParser()
    add_launcher_args(parser)
    parser.set_defaults(visualizer="newton")
    args_cli = parser.parse_args()

    with launch_simulation(
        cfg=physics_cfg,
        launcher_args=args_cli
    ) as resolved_physics_cfg:

        sim_cfg = SimulationCfg(
            dt=1.0 / 120.0,
            device=args_cli.device,
            physics=resolved_physics_cfg,
        )

        sim = SimulationContext(sim_cfg)
        sim.set_camera_view(
            eye=[3.0, 3.0, 2.0],
            target=[0.0, 0.0, 0.5]
        )
        scene = InteractiveScene(
            RigidBodyTestCfg(
                num_envs=1,
                env_spacing=1.0
            )
        )
        sim.reset()
        while sim.is_running():
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_cfg.dt)


if __name__ == "__main__":
    main()