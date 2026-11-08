from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Sequence

# Add Experiment 03 to sys.path to import MicroduckArticulationCfg
EXP_03_DIR = Path(__file__).resolve().parent.parent / "03_robot_model"
if str(EXP_03_DIR) not in sys.path:
    sys.path.append(str(EXP_03_DIR))

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
import isaaclab.envs.mdp as mdp
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab_tasks.core.velocity.mdp import feet_slide
from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass

from microduck_cfg import MicroduckArticulationCfg


# ==============================================================================
# CUSTOM VELOCITY COMMAND GENERATOR (100% MuJoCo Directional Alignment)
# ==============================================================================
class MicroduckVelocityCommand(UniformVelocityCommand):
    """Replicates Pollen Robotics MuJoCo velocity command distribution:
    - 20% Pure Forward Locomotion (lin_vel_x >= 0.30 m/s, vy=0, wz=0)
    - 15% Pure Turn-in-Place Rotation (lin_vel_x=0, vy=0, |wz| >= 0.40 rad/s)
    - 20% Closed-loop Compass Heading Hold (360-degree target directions)
    - 2% Standing Still (all velocities = 0)
    - 43% Omnidirectional Uniform Exploration
    """

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)

        if isinstance(env_ids, slice):
            all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)[env_ids]
        elif isinstance(env_ids, torch.Tensor):
            all_env_ids = env_ids.to(device=self.device, dtype=torch.long)
        else:
            all_env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        num_envs = len(all_env_ids)
        if num_envs == 0:
            return

        r = torch.empty(num_envs, device=self.device).uniform_(0.0, 1.0)

        rel_forward = getattr(self.cfg, "rel_forward_envs", 0.20)
        rel_turn = getattr(self.cfg, "rel_turn_in_place_envs", 0.15)
        rel_bwd = getattr(self.cfg, "rel_backward_envs", 0.10)
        rel_strafe = getattr(self.cfg, "rel_strafe_envs", 0.10)

        p_fwd = rel_forward
        p_turn = p_fwd + rel_turn
        p_bwd = p_turn + rel_bwd
        p_strafe = p_bwd + rel_strafe

        # 1. Forward-only envs: lin_vel_x in [0.30, 0.40], vy=0, wz=0
        fwd_ids = all_env_ids[(r >= 0.0) & (r < p_fwd)]
        if len(fwd_ids) > 0:
            min_fwd = getattr(self.cfg, "forward_min_vel", 0.30)
            max_fwd = self.cfg.ranges.lin_vel_x[1]
            self.vel_command_b[fwd_ids, 0] = self.vel_command_b[fwd_ids, 0].abs().clamp(min=min_fwd, max=max_fwd)
            self.vel_command_b[fwd_ids, 1] = 0.0
            self.vel_command_b[fwd_ids, 2] = 0.0
            self.is_standing_env[fwd_ids] = False
            if self.cfg.heading_command:
                self.is_heading_env[fwd_ids] = False

        # 2. Turn-in-place envs: lin=0, |wz| in [0.4*max, max]
        turn_ids = all_env_ids[(r >= p_fwd) & (r < p_turn)]
        if len(turn_ids) > 0:
            self.vel_command_b[turn_ids, 0] = 0.0
            self.vel_command_b[turn_ids, 1] = 0.0
            max_ang = max(abs(self.cfg.ranges.ang_vel_z[0]), abs(self.cfg.ranges.ang_vel_z[1]))
            lo_frac = getattr(self.cfg, "turn_in_place_min_frac", 0.40)
            signs = torch.where(
                torch.empty(len(turn_ids), device=self.device).uniform_(0.0, 1.0) < 0.5, -1.0, 1.0
            )
            mags = torch.empty(len(turn_ids), device=self.device).uniform_(lo_frac * max_ang, max_ang)
            self.vel_command_b[turn_ids, 2] = signs * mags
            self.is_standing_env[turn_ids] = False
            if self.cfg.heading_command:
                self.is_heading_env[turn_ids] = False

        # 3. Backward-only envs: lin_vel_x in [-0.30, -0.15], vy=0, wz=0
        bwd_ids = all_env_ids[(r >= p_turn) & (r < p_bwd)]
        if len(bwd_ids) > 0:
            self.vel_command_b[bwd_ids, 0] = torch.empty(len(bwd_ids), device=self.device).uniform_(-0.30, -0.15)
            self.vel_command_b[bwd_ids, 1] = 0.0
            self.vel_command_b[bwd_ids, 2] = 0.0
            self.is_standing_env[bwd_ids] = False
            if self.cfg.heading_command:
                self.is_heading_env[bwd_ids] = False

        # 4. Lateral strafe envs: lin_vel_x=0, |vy| in [0.15, 0.25], wz=0
        strafe_ids = all_env_ids[(r >= p_bwd) & (r < p_strafe)]
        if len(strafe_ids) > 0:
            self.vel_command_b[strafe_ids, 0] = 0.0
            signs_y = torch.where(
                torch.empty(len(strafe_ids), device=self.device).uniform_(0.0, 1.0) < 0.5, -1.0, 1.0
            )
            mags_y = torch.empty(len(strafe_ids), device=self.device).uniform_(0.15, 0.25)
            self.vel_command_b[strafe_ids, 1] = signs_y * mags_y
            self.vel_command_b[strafe_ids, 2] = 0.0
            self.is_standing_env[strafe_ids] = False
            if self.cfg.heading_command:
                self.is_heading_env[strafe_ids] = False


@configclass
class MicroduckVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for Microduck velocity command generator matching Pollen MuJoCo."""

    class_type: type = MicroduckVelocityCommand

    rel_forward_envs: float = 0.20
    """Fraction of envs forced to walk straight forward with lin_vel_x >= 0.30 m/s."""

    forward_min_vel: float = 0.30
    """Minimum linear velocity for forward-only walking environments."""

    rel_turn_in_place_envs: float = 0.15
    """Fraction of envs commanded to spin on the spot (lin=0, |wz| >= 0.4*max)."""

    turn_in_place_min_frac: float = 0.40
    """Lower bound of forced turn-in-place angular rate as fraction of max."""

    rel_backward_envs: float = 0.10
    """Fraction of envs forced to walk backward with lin_vel_x in [-0.30, -0.15] m/s."""

    rel_strafe_envs: float = 0.10
    """Fraction of envs forced to strafe sideways with |lin_vel_y| in [0.15, 0.25] m/s."""


# ==============================================================================
# CUSTOM REWARD FUNCTIONS (100% Pollen Robotics MuJoCo Reference Alignment)
# ==============================================================================
def variable_leg_posture(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std_standing: float = 0.15,
    std_walking: float = 0.40,
    walking_threshold: float = 0.01,
) -> torch.Tensor:
    """Gaussian leg posture reward matching Pollen MuJoCo reference.

    Permits wide joint swings (std=0.40 rad) while walking, but severely penalizes squatting/crawling.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])

    is_walking = (total_speed >= walking_threshold).unsqueeze(1)
    std = torch.where(is_walking, std_walking, std_standing)

    current_joint_pos = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
    desired_joint_pos = asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))


def upright_gaussian(
    env: ManagerBasedRLEnv,
    std: float = 0.2236,  # sqrt(0.05) matching MuJoCo upright std
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Torso upright reward using Gaussian kernel matching MuJoCo reference."""
    asset: Articulation = env.scene[asset_cfg.name]
    xy_squared = torch.sum(torch.square(asset.data.projected_gravity_b.torch[:, :2]), dim=1)
    return torch.exp(-xy_squared / (std**2))


def track_yaw_fine(
    env: ManagerBasedRLEnv,
    std: float = 0.35,  # MuJoCo TRACK_YAW_FINE_STD = 0.35
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Yaw-rate-only Gaussian reward matching MuJoCo track_yaw_fine.

    Provides sharp gradient for turning in place and straight-line heading tracking.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    ang_vel_b = asset.data.root_ang_vel_b.torch
    return torch.exp(-torch.square(cmd[:, 2] - ang_vel_b[:, 2]) / (std**2))


def feet_air_time_mujoco(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    threshold_min: float = 0.125,
    threshold_max: float = 0.300,
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """Reward biped feet air time matching MuJoCo reference.

    Active for both linear and angular locomotion (lin_norm + |wz| > 0.01).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    total_cmd = torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])
    active = (total_cmd > command_threshold).float()

    current_air_time = contact_sensor.data.current_air_time.torch[:, sensor_cfg.body_ids]
    in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
    reward = torch.sum(in_range.float(), dim=1) * active
    return reward


def directional_linear_velocity_gain(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.05,
) -> torch.Tensor:
    """Directional Linear Velocity Gain (Cos-similarity + Forward Progress).
    
    Rewards the policy when the robot moves in the EXACT commanded direction:
    - Moving in commanded direction (+proj_speed and cos > 0) -> High positive reward
    - Moving perpendicular (e.g. commanded to strafe/turn but moving forward) -> 0 reward
    - Moving opposite (e.g. commanded backward but moving forward) -> Negative penalty
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    cmd_xy = cmd[:, :2]
    cmd_speed = torch.norm(cmd_xy, dim=1)

    act_xy = asset.data.root_lin_vel_b.torch[:, :2]
    act_speed = torch.norm(act_xy, dim=1)

    cmd_dir = cmd_xy / (cmd_speed.unsqueeze(1) + 1e-6)
    proj_speed = torch.sum(act_xy * cmd_dir, dim=1)
    cos_sim = torch.sum(act_xy * cmd_xy, dim=1) / (cmd_speed * act_speed + 1e-6)

    is_moving_cmd = (cmd_speed > threshold).float()
    dir_reward = 0.5 * torch.clamp(proj_speed / (cmd_speed + 1e-6), -1.0, 1.0) + 0.5 * cos_sim
    return dir_reward * is_moving_cmd


def directional_angular_velocity_gain(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.10,
) -> torch.Tensor:
    """Directional Angular Velocity Gain (Yaw Turning Alignment).
    
    Directly rewards turning in the commanded direction (CW vs CCW):
    - Turning in commanded direction -> Positive gain
    - Failing to turn (moving straight) -> Zero gain
    - Turning in reverse direction -> Negative penalty
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    wz_cmd = cmd[:, 2]
    wz_act = asset.data.root_ang_vel_b.torch[:, 2]

    is_turning_cmd = (torch.abs(wz_cmd) > threshold).float()
    dir_yaw = torch.clamp((wz_act * torch.sign(wz_cmd)) / (torch.abs(wz_cmd) + 1e-6), -1.0, 1.0)
    return dir_yaw * is_turning_cmd


def stand_still_rest(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lin_threshold: float = 0.05,
    ang_threshold: float = 0.10,
) -> torch.Tensor:
    """Exponential reward for remaining still when commanded velocity is zero."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    cmd_speed = torch.norm(cmd[:, :2], dim=1)
    wz_cmd = torch.abs(cmd[:, 2])

    is_stand_cmd = ((cmd_speed <= lin_threshold) & (wz_cmd <= ang_threshold)).float()
    act_lin_speed_sq = torch.sum(torch.square(asset.data.root_lin_vel_b.torch[:, :2]), dim=1)
    act_yaw_sq = torch.square(asset.data.root_ang_vel_b.torch[:, 2])
    motion = act_lin_speed_sq + act_yaw_sq
    return torch.exp(-motion / 0.02) * is_stand_cmd


def robot_state_is_nan(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate environments where solver encountered non-finite state, matching MuJoCo nan_state."""
    asset: Articulation = env.scene[asset_cfg.name]
    d = asset.data
    bad = ~torch.isfinite(d.joint_pos.torch).all(dim=1)
    bad |= ~torch.isfinite(d.joint_vel.torch).all(dim=1)
    bad |= ~torch.isfinite(d.root_link_pos_w.torch).all(dim=1)
    bad |= ~torch.isfinite(d.root_link_quat_w.torch).all(dim=1)
    bad |= ~torch.isfinite(d.root_link_lin_vel_w.torch).all(dim=1)
    bad |= ~torch.isfinite(d.root_link_ang_vel_w.torch).all(dim=1)
    return bad


# ==============================================================================
# 1. SCENE CONFIGURATION
# ==============================================================================
@configclass
class MicroduckSceneCfg(InteractiveSceneCfg):
    """Scene configuration with large ground plane, Microduck robot, and contact sensors."""

    # Static ground plane
    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.CuboidCfg(
            size=(100.0, 100.0, 0.2),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.1)),
    )

    # Microduck Articulation (15 bodies, 14 revolute joints)
    robot: MicroduckArticulationCfg = MicroduckArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    # Foot Contact Sensors on left and right ankle colliders
    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*ankle.*",
        history_length=3,
        track_air_time=True,
    )


# ==============================================================================
# 2. ACTION CONFIGURATION
# ==============================================================================
@configclass
class ActionsCfg:
    """Action specifications: 14 joint position target offsets added to default standing angles."""

    joint_pos: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale={
            ".*hip_yaw.*": 0.25,
            ".*hip_roll.*": 0.20,
            ".*hip_pitch.*": 0.50,
            ".*knee.*": 0.50,
            ".*ankle.*": 0.50,
            ".*neck.*": 0.30,
            ".*head.*": 0.30,
        },
        use_default_offset=True,
    )


# ==============================================================================
# 3. COMMAND CONFIGURATION
# ==============================================================================
@configclass
class CommandsCfg:
    """Velocity command specifications: 360-degree compass heading directions and twist targets matching MuJoCo."""

    base_velocity: MicroduckVelocityCommandCfg = MicroduckVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=0.02,
        rel_heading_envs=0.20,
        rel_forward_envs=0.20,
        rel_backward_envs=0.10,
        rel_turn_in_place_envs=0.15,
        rel_strafe_envs=0.10,
        heading_command=True,
        heading_control_stiffness=0.5,
        ranges=MicroduckVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.40, 0.40),  # Forward (+vx) and backward (-vx)
            lin_vel_y=(-0.30, 0.30),  # Lateral strafe (+/- vy)
            ang_vel_z=(-1.00, 1.00),  # Yaw turning (+/- wz)
            heading=(-math.pi, math.pi),  # 360-degree compass direction exploration
        ),
    )


# ==============================================================================
# 4. OBSERVATION CONFIGURATION
# ==============================================================================
@configclass
class ObservationsCfg:
    """Observation specifications for the Actor policy network."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations fed to the policy (Total: 48 dimensions)."""

        # 1. Base orientation (projected gravity vector in torso frame: 3 dims)
        projected_gravity = ObservationTermCfg(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # 2. Base angular velocity (roll, pitch, yaw rates: 3 dims)
        base_ang_vel = ObservationTermCfg(
            func=mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # 3. Joint positions relative to default standing pose (14 dims)
        joint_pos = ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # 4. Joint angular velocities (14 dims)
        joint_vel = ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # 5. Last action vector applied to motors (14 dims)
        actions = ObservationTermCfg(func=mdp.last_action)

        # 6. Target velocity command (vx, vy, wz: 3 dims)
        velocity_commands = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ==============================================================================
# 5. EVENT CONFIGURATION (Resets & Disturbances)
# ==============================================================================
@configclass
class EventCfg:
    """Startup and reset events."""

    # Randomize initial robot base spawn position on reset (matching MuJoCo z=12-13cm)
    reset_base = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.10, 0.10),
                "y": (-0.10, 0.10),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (-0.02, 0.02),
                "y": (-0.02, 0.02),
                "z": (-0.02, 0.02),
                "roll": (-0.02, 0.02),
                "pitch": (-0.02, 0.02),
                "yaw": (-0.02, 0.02),
            },
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Clean initial joint positions: spawn strictly in nominal STAND2 posture
    reset_robot_joints = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Periodic velocity push disturbances (every 3 to 6 seconds)
    push_robot = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.20, 0.20), "y": (-0.10, 0.10)},
        },
    )


# ==============================================================================
# 6. REWARDS CONFIGURATION
# ==============================================================================
@configclass
class RewardsCfg:
    """Reward terms shaping the biped locomotion behavior matching MuJoCo reference."""

    # 1. Primary Objectives: Velocity Command Tracking & Directional Gain
    track_lin_vel_xy_exp = RewardTermCfg(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.1), "asset_cfg": SceneEntityCfg("robot")},
    )
    track_ang_vel_z_exp = RewardTermCfg(
        func=mdp.track_ang_vel_z_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.5), "asset_cfg": SceneEntityCfg("robot")},
    )
    track_yaw_fine = RewardTermCfg(
        func=track_yaw_fine,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.35, "asset_cfg": SceneEntityCfg("robot")},
    )

    # Directional Alignment Gains (Guarantees bot gains reward strictly for moving in the commanded direction)
    directional_lin_gain = RewardTermCfg(
        func=directional_linear_velocity_gain,
        weight=2.0,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot"), "threshold": 0.05},
    )
    directional_yaw_gain = RewardTermCfg(
        func=directional_angular_velocity_gain,
        weight=2.0,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot"), "threshold": 0.10},
    )
    stand_still_rest = RewardTermCfg(
        func=stand_still_rest,
        weight=1.5,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
    )

    # 2. Torso Upright Stability (Gaussian kernel: weight 2.0, std = sqrt(0.05) matching MuJoCo)
    upright = RewardTermCfg(
        func=upright_gaussian,
        weight=2.0,
        params={"std": math.sqrt(0.05), "asset_cfg": SceneEntityCfg("robot")},
    )
    ang_vel_xy_l2 = RewardTermCfg(
        func=mdp.ang_vel_xy_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    lin_vel_z_l2 = RewardTermCfg(
        func=mdp.lin_vel_z_l2,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # 3. Leg Posture Anchor (Gaussian variable posture: permits 0.40 rad stepping swing, stops crawling)
    leg_posture = RewardTermCfg(
        func=variable_leg_posture,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip.*", ".*knee.*", ".*ankle.*"]),
            "std_standing": 0.15,
            "std_walking": 0.40,
            "walking_threshold": 0.01,
        },
    )

    # 4. Joint Limit Penalty (MuJoCo dof_pos_limits = -1.0)
    dof_pos_limits = RewardTermCfg(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # 5. Energy & Action Smoothness (action_rate_l2 matching MuJoCo, NO action_l2 to allow natural stepping)
    action_rate_l2 = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.08)
    joint_vel_l2 = RewardTermCfg(
        func=mdp.joint_vel_l2,
        weight=-0.0005,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # 6. Biped Stepping Cadence: Reward Feet Air Time matching MuJoCo (both walking and turning)
    feet_air_time = RewardTermCfg(
        func=feet_air_time_mujoco,
        weight=3.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "threshold_min": 0.125,
            "threshold_max": 0.300,
            "command_threshold": 0.01,
        },
    )

    # 7. Foot Sliding Penalty (MuJoCo foot_slip = -0.1)
    feet_slide = RewardTermCfg(
        func=feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle.*"]),
        },
    )

    # 8. Head/Neck Posture Regularization: Mild centering to maintain upright head position
    head_neck_pose = RewardTermCfg(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*neck.*", ".*head.*"])},
    )


# ==============================================================================
# 7. TERMINATION CONFIGURATION (Episode resets on fall)
# ==============================================================================
@configclass
class TerminationsCfg:
    """Conditions terminating an episode."""

    # Episode length timeout (20.0 seconds)
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)

    # Robot fallen: base height drops below 3.5 cm (severe fall onto floor)
    base_contact = TerminationTermCfg(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.035, "asset_cfg": SceneEntityCfg("robot")},
    )

    # Robot tilted excessively (pitch or roll > 70 degrees, matching MuJoCo limit_angle=1.22 rad)
    bad_orientation = TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.22, "asset_cfg": SceneEntityCfg("robot")},
    )

    # Terminate immediately on any non-finite / NaN physics state to reset cleanly (matching MuJoCo nan_state)
    nan_state = TerminationTermCfg(
        func=robot_state_is_nan,
        time_out=False,
    )


# ==============================================================================
# 8. COMPLETE RL ENVIRONMENT CONFIGURATION
# ==============================================================================
@configclass
class MicroduckEnvCfg(ManagerBasedRLEnvCfg):
    """Complete Manager-Based RL Environment Configuration for Microduck Locomotion."""

    # Scene settings (64 parallel robots, 2.5m spacing)
    scene: MicroduckSceneCfg = MicroduckSceneCfg(num_envs=64, env_spacing=2.5)

    # Core MDP Managers
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post-initialization configurations for simulation timing."""
        # General physics timing
        self.decimation = 4                   # 50 Hz policy control frequency
        self.episode_length_s = 20.0          # 20.0 seconds per episode
        self.sim.dt = 1.0 / 200.0             # 200 Hz physics simulation step (5 ms)
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = False
