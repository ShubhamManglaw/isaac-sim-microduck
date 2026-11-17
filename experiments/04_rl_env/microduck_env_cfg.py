from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable, Sequence

# Add Experiment 03 to sys.path to import MicroduckArticulationCfg
EXP_03_DIR = Path(__file__).resolve().parent.parent / "03_robot_model"
if str(EXP_03_DIR) not in sys.path:
    sys.path.append(str(EXP_03_DIR))

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
import isaaclab.envs.mdp as mdp
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab_tasks.core.velocity.mdp import feet_slide
from isaaclab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, RayCaster, RayCasterCfg
from isaaclab.sensors.ray_caster.patterns.patterns_cfg import PatternBaseCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils
from isaaclab.utils.noise import UniformNoiseCfg

from microduck_cfg import MicroduckArticulationCfg


# ==============================================================================
# ROBOT CONSTANTS (measured in simulation with the robot standing at rest)
# ==============================================================================
STAND_HEIGHT = 0.1167
"""Trunk height [m] in the nominal STAND2 pose."""

FOOT_REST_HEIGHT = 0.0232
"""Height [m] of the ankle body origin when the foot is flat on the ground."""

HURDLE_NAMES = ("hurdle_low", "hurdle_mid", "hurdle_high")
"""Scene names of the hurdle bars. Each env owns one bar of every height; unused bars are parked underground."""

HURDLE_HEIGHTS = (0.015, 0.025, 0.040)
"""Heights [m] of the hurdle bars, in the same order as HURDLE_NAMES."""

HURDLE_FLOOR_GAP = 0.0005
"""Gap [m] left between a placed bar and the floor (an overlapping kinematic bar stalls the contact solver)."""

TOF_MAX_RANGE = 1.0
"""Maximum range [m] of the forward ToF sensor. Misses are reported as this value."""

GROUND_SIZE = 1000.0
"""Side length [m] of the square ground slab. It must reach under every robot: the robots are laid out on
a grid (env_spacing apart) that is 320 m wide for 16,384 envs. MicroduckEnv refuses to start otherwise."""

GROUND_WALK_MARGIN = 20.0
"""Distance [m] a robot can walk away from its grid position within one episode (0.75 m/s for 20 s, rounded up)."""


# ==============================================================================
# SHARED HELPERS
# ==============================================================================
def waypoint_velocity_controller(
    delta_xy_w: torch.Tensor,
    heading_w: torch.Tensor,
    max_lin_vel: float,
    max_ang_vel: float = 1.5,
    kp_lin: float = 1.5,
    kp_ang: float = 2.5,
    arrive_radius: float = 0.06,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Turn-and-go controller that converts a world-frame offset to a waypoint into (vx, wz).

    Used both by the training command generator (waypoint envs) and by play.py, so the policy is
    trained on exactly the command stream it sees when following waypoints at playback.

    Args:
        delta_xy_w: Offset from the robot to the target in world frame, shape (N, 2).
        heading_w: Robot yaw heading in world frame, shape (N,).
        max_lin_vel: Forward speed cap [m/s].
        max_ang_vel: Yaw-rate cap [rad/s].

    Returns:
        Tuple of (vx, wz, arrived), each of shape (N,).
    """
    dist = torch.norm(delta_xy_w, dim=-1)
    bearing = math_utils.wrap_to_pi(torch.atan2(delta_xy_w[:, 1], delta_xy_w[:, 0]) - heading_w)
    wz = torch.clamp(kp_ang * bearing, -max_ang_vel, max_ang_vel)
    # Only drive forward once roughly facing the target
    vx = torch.clamp(kp_lin * dist, max=max_lin_vel) * torch.clamp(torch.cos(bearing), min=0.0)
    arrived = dist < arrive_radius
    vx = torch.where(arrived, torch.zeros_like(vx), vx)
    wz = torch.where(arrived, torch.zeros_like(wz), wz)
    return vx, wz, arrived


def gait_frequency(
    command: torch.Tensor,
    period_slow: float = 0.50,
    period_fast: float = 0.30,
    speed_ref: float = 0.75,
    yaw_weight: float = 0.10,
) -> torch.Tensor:
    """Stepping frequency [Hz] of the gait clock as a function of the velocity command.

    Short legs cannot cover a fast command with long strides, so the cadence rises with the commanded
    speed: one full left+right cycle takes ``period_slow`` seconds at a crawl and ``period_fast`` seconds
    at ``speed_ref`` m/s and above. Yaw rate counts as ``yaw_weight`` m/s per rad/s.

    A deployment must compute the clock with this same function (see AGENT_HANDOFF.md).

    Args:
        command: Velocity command (vx, vy, wz), shape (N, 3).

    Returns:
        Gait cycle frequency [Hz], shape (N,).
    """
    speed = torch.norm(command[:, :2], dim=1) + yaw_weight * torch.abs(command[:, 2])
    blend = torch.clamp(speed / speed_ref, 0.0, 1.0)
    return 1.0 / period_slow + (1.0 / period_fast - 1.0 / period_slow) * blend


def gait_phase(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    """Gait clock in [0, 1), integrated by the command term. Left foot leads, right foot is offset by 0.5."""
    return _cmd_term(env, command_name).gait_phase


def curriculum_progress(cmd_term: "MicroduckVelocityCommand") -> float:
    """How far the speed curriculum has advanced: 0.0 at its starting scale, 1.0 at full speed."""
    start = cmd_term.cfg.curriculum_start_scale
    return min(max((cmd_term.speed_scale - start) / max(1.0 - start, 1e-6), 0.0), 1.0)


def _cmd_term(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> "MicroduckVelocityCommand":
    return env.command_manager.get_term(command_name)


def _not_getup(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    """1.0 for normal envs, 0.0 for envs that are currently recovering from a fall."""
    return (~_cmd_term(env, command_name).getup_mask).float()


# ==============================================================================
# FORWARD-FACING 8x8 ToF RAY PATTERN (VL53L5CX-style square field of view)
# ==============================================================================
def tof_fov_pattern(cfg: "TofFovPatternCfg", device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """8x8 fan of rays looking forward (+x) and pitched down so it sees the ground ahead and obstacles.

    Ray index = row * N + col, with row 0 at the top of the field of view and col 0 on the left.
    """
    n = cfg.rays_per_side
    half = math.radians(cfg.fov_deg) / 2.0
    elevation = torch.linspace(half, -half, n, device=device) - math.radians(cfg.pitch_down_deg)
    azimuth = torch.linspace(half, -half, n, device=device)
    el, az = torch.meshgrid(elevation, azimuth, indexing="ij")
    directions = torch.stack(
        [torch.cos(el) * torch.cos(az), torch.cos(el) * torch.sin(az), torch.sin(el)], dim=-1
    ).reshape(-1, 3)
    starts = torch.zeros_like(directions)
    return starts, directions


@configclass
class TofFovPatternCfg(PatternBaseCfg):
    """Configuration for the forward-facing ToF ray fan."""

    func: Callable = tof_fov_pattern
    rays_per_side: int = 8
    fov_deg: float = 45.0
    """Square field of view [deg]."""
    pitch_down_deg: float = 22.0
    """Downward tilt of the optical axis [deg]."""


# ==============================================================================
# CUSTOM VELOCITY COMMAND GENERATOR
# ==============================================================================
class MicroduckVelocityCommand(UniformVelocityCommand):
    """Velocity command generator with task categories, a speed curriculum and auxiliary task state.

    Each resample assigns every env to one category (fractions are set in the config):

    - forward / sprint / backward / strafe / turn-in-place: single-axis commands
    - curve: forward walking while turning (combined vx + wz)
    - waypoint: (vx, wz) produced every step by :func:`waypoint_velocity_controller` toward a world target
    - hurdle: straight forward walk with a hurdle teleported into the robot's path
    - standing: zero command
    - remainder: uniform omnidirectional sampling (a share of which holds a compass heading)

    Besides the velocity command, this term owns:

    - ``speed_scale``: multiplier on all commanded speeds, raised by the speed curriculum
    - ``head_command``: (head_pitch, head_yaw) look targets relative to the default head pose
    - ``gait_phase``: the gait clock, advanced every step at a cadence that follows the commanded speed
    - ``getup_mask``: envs that were reset lying on the floor and still have to stand up
    - hurdle pose buffers used by the hurdle reward and the privileged critic observation
    """

    cfg: "MicroduckVelocityCommandCfg"

    def __init__(self, cfg: "MicroduckVelocityCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.speed_scale: float = float(cfg.initial_speed_scale)

        self.is_waypoint_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.waypoint_target_w = torch.zeros(self.num_envs, 2, device=self.device)

        self.is_hurdle_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.hurdle_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.hurdle_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.hurdle_yaw = torch.zeros(self.num_envs, device=self.device)
        self.hurdle_top_height = torch.zeros(self.num_envs, device=self.device)

        self.head_command = torch.zeros(self.num_envs, 2, device=self.device)
        self.getup_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.gait_phase = torch.rand(self.num_envs, device=self.device)

        self._hurdles: list[RigidObject] = []
        self._hurdle_heights: list[float] = []
        for name, height in zip(cfg.hurdle_asset_names, cfg.hurdle_heights):
            if name in env.scene.keys():
                self._hurdles.append(env.scene[name])
                self._hurdle_heights.append(height)

    # --------------------------------------------------------------------------
    # Hurdle placement
    # --------------------------------------------------------------------------
    def _write_hurdle_poses(self, env_ids: torch.Tensor):
        """Push the hurdle buffers to the simulation: the selected bar goes to its pose, the others underground."""
        if not self._hurdles or len(env_ids) == 0:
            return
        zeros = torch.zeros(len(env_ids), device=self.device)
        quat = math_utils.quat_from_euler_xyz(zeros, zeros, self.hurdle_yaw[env_ids])
        parked = self._env.scene.env_origins[env_ids].clone()
        parked[:, 2] = -1.0
        for index, hurdle in enumerate(self._hurdles):
            selected = self.is_hurdle_env[env_ids] & (self.hurdle_index[env_ids] == index)
            pos = torch.where(selected.unsqueeze(1), self.hurdle_pos_w[env_ids], parked)
            hurdle.write_root_pose_to_sim_index(root_pose=torch.cat([pos, quat], dim=-1), env_ids=env_ids)

    def park_hurdle(self, env_ids: torch.Tensor):
        """Remove the hurdle from these envs (all bars buried below the floor)."""
        if len(env_ids) == 0:
            return
        self.is_hurdle_env[env_ids] = False
        self.hurdle_top_height[env_ids] = 0.0
        self._write_hurdle_poses(env_ids)

    def place_hurdle_ahead(self, env_ids: torch.Tensor, index: int | None = None, distance: float | None = None):
        """Teleport a hurdle bar across the robot's current heading.

        Args:
            env_ids: Environments to place a hurdle in.
            index: Which bar (see HURDLE_HEIGHTS) to use. Random if None.
            distance: Distance ahead of the robot [m]. Sampled from ``hurdle_distance_range`` if None.
        """
        n = len(env_ids)
        if n == 0 or not self._hurdles:
            return
        root_pos = self.robot.data.root_link_pos_w.torch[env_ids]
        heading = self.robot.data.heading_w.torch[env_ids]
        if distance is None:
            dist = torch.empty(n, device=self.device).uniform_(*self.cfg.hurdle_distance_range)
        else:
            dist = torch.full((n,), float(distance), device=self.device)
        if index is None:
            # Only the lowest bar until the speed curriculum is complete, then all heights
            num_choices = len(self._hurdles) if self.speed_scale >= 1.0 else 1
            choice = torch.randint(0, num_choices, (n,), device=self.device)
        else:
            choice = torch.full((n,), int(index), dtype=torch.long, device=self.device)
        height = torch.tensor(self._hurdle_heights, device=self.device)[choice]

        self.hurdle_index[env_ids] = choice
        self.hurdle_pos_w[env_ids, 0] = root_pos[:, 0] + dist * torch.cos(heading)
        self.hurdle_pos_w[env_ids, 1] = root_pos[:, 1] + dist * torch.sin(heading)
        self.hurdle_pos_w[env_ids, 2] = 0.5 * height + HURDLE_FLOOR_GAP
        self.hurdle_yaw[env_ids] = heading + torch.empty(n, device=self.device).uniform_(-0.15, 0.15)
        self.hurdle_top_height[env_ids] = height + HURDLE_FLOOR_GAP
        self.is_hurdle_env[env_ids] = True
        self._write_hurdle_poses(env_ids)

    # --------------------------------------------------------------------------
    # Command sampling
    # --------------------------------------------------------------------------
    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)

        if isinstance(env_ids, slice):
            all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)[env_ids]
        elif isinstance(env_ids, torch.Tensor):
            all_env_ids = env_ids.to(device=self.device, dtype=torch.long)
        else:
            all_env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        num = len(all_env_ids)
        if num == 0:
            return

        self.is_waypoint_env[all_env_ids] = False

        # Teleoperation / playback: commands are written from outside, nothing to sample
        if self.cfg.external_control:
            self.is_standing_env[all_env_ids] = False
            if self.cfg.heading_command:
                self.is_heading_env[all_env_ids] = False
            return

        cfg = self.cfg
        s = self.speed_scale
        dev = self.device

        def uniform(n: int, lo: float, hi: float) -> torch.Tensor:
            return torch.empty(n, device=dev).uniform_(lo, hi)

        def random_sign(n: int) -> torch.Tensor:
            return torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)

        # Uniformly sampled remainder also follows the speed curriculum
        self.vel_command_b[all_env_ids] *= s

        # Hurdles only appear once the robot can walk reasonably well
        hurdles_on = len(self._hurdles) > 0 and s >= cfg.hurdle_unlock_scale
        rel_hurdle = cfg.rel_hurdle_envs if hurdles_on else 0.0
        rel_forward = cfg.rel_forward_envs + (0.0 if hurdles_on else cfg.rel_hurdle_envs)

        r = torch.rand(num, device=dev)
        edges = torch.cumsum(
            torch.tensor(
                [
                    rel_forward,
                    cfg.rel_sprint_envs,
                    cfg.rel_turn_in_place_envs,
                    cfg.rel_backward_envs,
                    cfg.rel_strafe_envs,
                    cfg.rel_curve_envs,
                    cfg.rel_waypoint_envs,
                    rel_hurdle,
                    cfg.rel_standing_envs,
                ],
                device=dev,
            ),
            dim=0,
        )
        category = torch.bucketize(r, edges, right=True)  # 0..8 = forced categories, 9 = uniform remainder
        max_ang = max(abs(cfg.ranges.ang_vel_z[0]), abs(cfg.ranges.ang_vel_z[1]))

        def select(index: int) -> torch.Tensor:
            ids = all_env_ids[category == index]
            if len(ids) > 0:
                self.is_standing_env[ids] = False
                if cfg.heading_command:
                    self.is_heading_env[ids] = False
            return ids

        def set_cmd(ids: torch.Tensor, vx, vy, wz):
            self.vel_command_b[ids, 0] = vx
            self.vel_command_b[ids, 1] = vy
            self.vel_command_b[ids, 2] = wz

        # 0. Forward walk
        ids = select(0)
        if len(ids) > 0:
            set_cmd(ids, uniform(len(ids), cfg.forward_min_vel, 0.55) * s, 0.0, 0.0)

        # 1. Sprint
        ids = select(1)
        if len(ids) > 0:
            set_cmd(ids, uniform(len(ids), 0.55, 0.75) * s, 0.0, 0.0)

        # 2. Turn in place
        ids = select(2)
        if len(ids) > 0:
            mags = uniform(len(ids), cfg.turn_in_place_min_frac * max_ang, max_ang) * s
            set_cmd(ids, 0.0, 0.0, random_sign(len(ids)) * mags)

        # 3. Backward walk
        ids = select(3)
        if len(ids) > 0:
            set_cmd(ids, uniform(len(ids), -0.40, -0.20) * s, 0.0, 0.0)

        # 4. Lateral strafe
        ids = select(4)
        if len(ids) > 0:
            set_cmd(ids, 0.0, random_sign(len(ids)) * uniform(len(ids), 0.20, 0.35) * s, 0.0)

        # 5. Curved walking: forward speed and yaw rate together
        ids = select(5)
        if len(ids) > 0:
            set_cmd(
                ids,
                uniform(len(ids), 0.20, 0.50) * s,
                0.0,
                random_sign(len(ids)) * uniform(len(ids), 0.30, 1.20) * s,
            )

        # 6. Waypoint following: target is refreshed into (vx, wz) every step in _update_command
        ids = select(6)
        if len(ids) > 0:
            root_xy = self.robot.data.root_link_pos_w.torch[ids, :2]
            radius = uniform(len(ids), *cfg.waypoint_distance_range)
            angle = uniform(len(ids), -math.pi, math.pi)
            self.waypoint_target_w[ids, 0] = root_xy[:, 0] + radius * torch.cos(angle)
            self.waypoint_target_w[ids, 1] = root_xy[:, 1] + radius * torch.sin(angle)
            self.is_waypoint_env[ids] = True
            set_cmd(ids, 0.0, 0.0, 0.0)

        # 7. Hurdle run: walk straight at a hurdle placed across the path
        hurdle_ids = select(7)
        if len(hurdle_ids) > 0:
            set_cmd(hurdle_ids, uniform(len(hurdle_ids), *cfg.hurdle_speed_range), 0.0, 0.0)
            self.place_hurdle_ahead(hurdle_ids)
        self.park_hurdle(all_env_ids[category != 7])

        # 8. Standing still
        ids = all_env_ids[category == 8]
        if len(ids) > 0:
            set_cmd(ids, 0.0, 0.0, 0.0)
            self.is_standing_env[ids] = True
            if cfg.heading_command:
                self.is_heading_env[ids] = False

        # Head look target: straight ahead for a share of envs, otherwise a random pitch / yaw offset
        look = torch.rand(num, device=dev) >= cfg.rel_head_centered_envs
        self.head_command[all_env_ids, 0] = uniform(num, *cfg.head_pitch_range) * look
        self.head_command[all_env_ids, 1] = uniform(num, *cfg.head_yaw_range) * look

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        # Start every episode at a random point of the gait cycle
        self.gait_phase[env_ids] = torch.rand_like(self.gait_phase[env_ids])
        return super().reset(env_ids)

    def _update_command(self):
        if not self.cfg.external_control:
            self._update_sampled_command()
        # Advance the gait clock (also under external control, where the caller writes the command)
        frequency = gait_frequency(
            self.vel_command_b,
            period_slow=self.cfg.gait_period_slow,
            period_fast=self.cfg.gait_period_fast,
            speed_ref=self.cfg.gait_speed_ref,
        )
        self.gait_phase = (self.gait_phase + frequency * self._env.step_dt) % 1.0

    def _update_sampled_command(self):
        super()._update_command()

        # Waypoint envs: steer toward the target
        if self.is_waypoint_env.any():
            delta = self.waypoint_target_w - self.robot.data.root_link_pos_w.torch[:, :2]
            vx, wz, _ = waypoint_velocity_controller(
                delta,
                self.robot.data.heading_w.torch,
                max_lin_vel=self.cfg.waypoint_max_lin_vel * self.speed_scale,
                max_ang_vel=self.cfg.waypoint_max_ang_vel * self.speed_scale,
            )
            m = self.is_waypoint_env
            self.vel_command_b[:, 0] = torch.where(m, vx, self.vel_command_b[:, 0])
            self.vel_command_b[:, 1] = torch.where(m, torch.zeros_like(vx), self.vel_command_b[:, 1])
            self.vel_command_b[:, 2] = torch.where(m, wz, self.vel_command_b[:, 2])

        # Get-up envs: the task is finished once the robot is upright at standing height.
        # Until then the command is held at zero so the only goal is to stand up.
        if self.getup_mask.any():
            upright = self.robot.data.projected_gravity_b.torch[:, 2] < -0.92
            tall = self.robot.data.root_link_pos_w.torch[:, 2] > 0.8 * STAND_HEIGHT
            self.getup_mask &= ~(upright & tall)
            self.vel_command_b[self.getup_mask] = 0.0
            self.head_command[self.getup_mask] = 0.0


@configclass
class MicroduckVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the Microduck command generator."""

    class_type: type = MicroduckVelocityCommand

    external_control: bool = False
    """If True, nothing is sampled and commands are written by the caller (teleop / playback)."""

    initial_speed_scale: float = 0.5
    """Speed multiplier applied to all sampled commands when the env is created (raised by the curriculum).

    train.py overrides this when resuming, so a resumed run continues at the level it had reached.
    """

    curriculum_start_scale: float = 0.5
    """Speed scale at which the curriculum starts. Push strength and smoothness penalties ramp from
    their starting values at this scale to their full values at scale 1.0."""

    gait_period_slow: float = 0.50
    """Duration [s] of one full left+right gait cycle at near-zero commanded speed."""

    gait_period_fast: float = 0.30
    """Duration [s] of one full gait cycle at ``gait_speed_ref`` and above."""

    gait_speed_ref: float = 0.75
    """Commanded speed [m/s] at which the gait cycle reaches ``gait_period_fast``."""

    rel_forward_envs: float = 0.12
    """Fraction of envs walking straight forward at [forward_min_vel, 0.55] m/s."""

    rel_sprint_envs: float = 0.08
    """Fraction of envs sprinting forward at [0.55, 0.75] m/s."""

    forward_min_vel: float = 0.35
    """Minimum linear velocity for forward walking environments."""

    rel_turn_in_place_envs: float = 0.10
    """Fraction of envs spinning on the spot (lin=0, |wz| >= turn_in_place_min_frac * max)."""

    turn_in_place_min_frac: float = 0.40
    """Lower bound of forced turn-in-place angular rate as fraction of max."""

    rel_backward_envs: float = 0.07
    """Fraction of envs walking backward at [-0.40, -0.20] m/s."""

    rel_strafe_envs: float = 0.07
    """Fraction of envs strafing sideways at |vy| in [0.20, 0.35] m/s."""

    rel_curve_envs: float = 0.14
    """Fraction of envs walking forward while turning (combined vx + wz)."""

    rel_waypoint_envs: float = 0.10
    """Fraction of envs walking to a waypoint."""

    waypoint_distance_range: tuple[float, float] = (0.4, 1.5)
    """Distance [m] from the robot at which a new waypoint is sampled."""

    waypoint_max_lin_vel: float = 0.45
    """Forward speed cap [m/s] of the waypoint controller (before the speed scale)."""

    waypoint_max_ang_vel: float = 1.5
    """Yaw-rate cap [rad/s] of the waypoint controller (before the speed scale)."""

    rel_hurdle_envs: float = 0.08
    """Fraction of envs that get a hurdle placed in their path (forward walkers until hurdles unlock)."""

    hurdle_asset_names: tuple[str, ...] = HURDLE_NAMES
    """Scene names of the hurdle bars (bars missing from the scene are ignored)."""

    hurdle_heights: tuple[float, ...] = HURDLE_HEIGHTS
    """Heights [m] of the hurdle bars, in the same order as ``hurdle_asset_names``."""

    hurdle_unlock_scale: float = 0.90
    """Speed scale the curriculum must reach before hurdles are used (lowest bar only until scale 1.0)."""

    hurdle_distance_range: tuple[float, float] = (0.35, 0.60)
    """Distance [m] ahead of the robot at which the hurdle is placed."""

    hurdle_speed_range: tuple[float, float] = (0.25, 0.45)
    """Forward speed [m/s] commanded in hurdle envs (not scaled; hurdles unlock late)."""

    rel_standing_envs: float = 0.14
    """Fraction of envs commanded to stand still."""

    rel_head_centered_envs: float = 0.5
    """Fraction of envs whose head look command is straight ahead."""

    head_pitch_range: tuple[float, float] = (-0.35, 0.35)
    """Head pitch look offset range [rad] relative to the default pose."""

    head_yaw_range: tuple[float, float] = (-0.70, 0.70)
    """Head yaw look offset range [rad] relative to the default pose."""


# ==============================================================================
# CUSTOM REWARD FUNCTIONS
# ==============================================================================
def track_lin_vel_xy(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Linear velocity tracking (exponential kernel). Not paid while a robot is still lying on the floor."""
    return mdp.track_lin_vel_xy_exp(env, std=std, command_name=command_name, asset_cfg=asset_cfg) * _not_getup(
        env, command_name
    )


def track_ang_vel_z(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Yaw-rate tracking (exponential kernel). Not paid while a robot is still lying on the floor."""
    return mdp.track_ang_vel_z_exp(env, std=std, command_name=command_name, asset_cfg=asset_cfg) * _not_getup(
        env, command_name
    )


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

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1)) * _not_getup(env, command_name)


def upright_gaussian(
    env: ManagerBasedRLEnv,
    std: float = 0.2236,  # sqrt(0.05) matching MuJoCo upright std
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Torso upright reward using Gaussian kernel matching MuJoCo reference."""
    asset: Articulation = env.scene[asset_cfg.name]
    xy_squared = torch.sum(torch.square(asset.data.projected_gravity_b.torch[:, :2]), dim=1)
    return torch.exp(-xy_squared / (std**2))


def base_height_reached(
    env: ManagerBasedRLEnv,
    target_height: float = STAND_HEIGHT,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward trunk height up to the standing height (no penalty above it, so jumping stays free).

    Discourages crouched / crawling gaits and gives a gradient for standing up from the floor.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.clamp(asset.data.root_link_pos_w.torch[:, 2] / target_height, 0.0, 1.0)


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
    return torch.exp(-torch.square(cmd[:, 2] - ang_vel_b[:, 2]) / (std**2)) * _not_getup(env, command_name)


def feet_air_time_mujoco(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    threshold_min: float = 0.125,
    threshold_max: float = 0.300,
    command_threshold: float = 0.01,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward a foot's swing time while the other foot is on the ground (single support).

    Active for both linear and angular locomotion. Requiring the other foot to be in contact means
    two-footed hopping is not rewarded as a walking gait.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    total_cmd = torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])
    active = (total_cmd > command_threshold).float()

    current_air_time = contact_sensor.data.current_air_time.torch[:, sensor_cfg.body_ids]
    in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
    forces = torch.norm(contact_sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids], dim=-1)
    other_in_contact = torch.flip(forces > contact_threshold, dims=[1])
    reward = torch.sum((in_range & other_in_contact).float(), dim=1) * active
    return reward


def gait_contact_schedule(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    stance_ratio: float = 0.55,
    command_threshold: float = 0.05,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward foot contacts that follow the alternating left / right gait clock.

    The left foot should be on the ground for the first ``stance_ratio`` of the cycle and the right
    foot half a cycle later. The same clock is given to the policy as an observation, so the result
    is an even, repeatable stepping rhythm instead of shuffling or hopping.

    The value is rescaled so that standing with both feet planted averages 0 over a cycle and a
    perfect gait scores 1 (hopping on both feet and anti-phase stepping are negative). Without the
    rescaling, simply standing still would collect more than half of this reward.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    moving = ((torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])) > command_threshold).float()

    phase = gait_phase(env, command_name)
    foot_phase = torch.stack([phase, (phase + 0.5) % 1.0], dim=1)
    expected_stance = foot_phase < stance_ratio

    forces = torch.norm(contact_sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids], dim=-1)
    in_contact = forces > contact_threshold
    match = (in_contact == expected_stance).float().mean(dim=1)
    return (match - stance_ratio) / (1.0 - stance_ratio) * moving


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
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    lin_threshold: float = 0.05,
    ang_threshold: float = 0.10,
) -> torch.Tensor:
    """Multi-objective stability reward for remaining motionless in nominal STAND2 pose when at rest.

    Rewards:
    1. Base linear and angular motion zero (exp(-motion / 0.01))
    2. Joint velocities motionless (exp(-joint_vel_sq / 0.05))
    3. Joint positions anchored to nominal STAND2 (exp(-joint_pos_err / 0.02))
    4. Bilateral foot ground contact (both feet firmly on floor)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    cmd_speed = torch.norm(cmd[:, :2], dim=1)
    wz_cmd = torch.abs(cmd[:, 2])

    is_stand_cmd = ((cmd_speed <= lin_threshold) & (wz_cmd <= ang_threshold)).float()

    # 1. Base motionless
    act_lin_speed_sq = torch.sum(torch.square(asset.data.root_lin_vel_b.torch[:, :2]), dim=1)
    act_yaw_sq = torch.square(asset.data.root_ang_vel_b.torch[:, 2])
    motion = act_lin_speed_sq + act_yaw_sq
    base_still = torch.exp(-motion / 0.01)

    # 2. Joint velocities motionless
    joint_vel_sq = torch.mean(torch.square(asset.data.joint_vel.torch), dim=1)
    joints_still = torch.exp(-joint_vel_sq / 0.05)

    # 3. Joint posture anchored to default standing configuration (legs only; the head may be looking around)
    joint_pos_err = torch.mean(
        torch.square(
            asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
            - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
        ),
        dim=1,
    )
    pose_anchor = torch.exp(-joint_pos_err / 0.02)

    # 4. Bilateral foot ground contact (both feet firmly on floor)
    forces = torch.norm(contact_sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids], dim=-1)
    both_feet = ((forces[:, 0] > 1.0) & (forces[:, 1] > 1.0)).float()

    total_rest = 0.35 * base_still + 0.25 * joints_still + 0.20 * pose_anchor + 0.20 * both_feet
    return total_rest * is_stand_cmd * _not_getup(env, command_name)


def stand_still_action_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    vel_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize non-zero leg actions when at rest, so the policy itself holds the nominal STAND2 stance.

    Only the joints selected by ``asset_cfg`` are penalized (the head is free to follow look commands).
    """
    cmd = env.command_manager.get_command(command_name)
    cmd_speed = torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])
    is_stand_cmd = (cmd_speed < vel_threshold).float()
    actions = env.action_manager.action[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(actions), dim=1) * is_stand_cmd * _not_getup(env, command_name)


def sudden_stop_anti_fall(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    vel_threshold: float = 0.05,
    pitch_damping_weight: float = 0.15,
) -> torch.Tensor:
    """Deceleration stabilization and anti-fall reward for emergency sudden stops.

    When velocity command drops to zero:
    - Rewards rapid linear speed reduction to zero (|v_xy| -> 0)
    - Heavily penalizes forward pitch angular velocity spikes (bot tipping forward onto beak)
    - Heavily penalizes forward tilt (projected gravity x > 0)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    cmd_speed = torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])

    is_stopped_cmd = (cmd_speed < vel_threshold).float()

    lin_vel_xy = asset.data.root_lin_vel_b.torch[:, :2]
    lin_speed_sq = torch.sum(torch.square(lin_vel_xy), dim=1)
    pitch_rate = asset.data.root_ang_vel_b.torch[:, 1]
    proj_grav = asset.data.projected_gravity_b.torch

    stop_reward = torch.exp(-lin_speed_sq / 0.02)
    # Penalize pitching forward while braking (pitch_rate^2 + positive forward tilt)
    pitch_penalty = torch.square(pitch_rate) + torch.clamp(proj_grav[:, 0], min=0.0) * 5.0

    return (stop_reward - pitch_damping_weight * pitch_penalty) * is_stopped_cmd * _not_getup(env, command_name)


def action_smoothness_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Second-order action smoothness: penalize the change of the action change (jerky targets).

    Complements ``action_rate_l2``: that term limits how fast targets move, this one limits how
    abruptly that motion starts, stops or reverses, which is what makes servos buzz and gaits look twitchy.
    """
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    prev_prev = getattr(env, "_action_two_steps_ago", None)
    if prev_prev is None or prev_prev.shape != action.shape:
        prev_prev = prev_action.clone()
    penalty = torch.sum(torch.square(action - 2.0 * prev_action + prev_prev), dim=1)
    # The action history is not valid for the first two steps after a reset
    penalty = penalty * (env.episode_length_buf > 2).float()
    env._action_two_steps_ago = prev_action.clone()
    return penalty


def head_look_tracking(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.17,
) -> torch.Tensor:
    """Reward the head pitch / yaw joints for following the look command (offsets from the default pose).

    ``asset_cfg`` must select exactly (head_pitch, head_yaw) in that order.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    target = _cmd_term(env, command_name).head_command
    offset = (
        asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    )
    error = torch.sum(torch.square(offset - target), dim=1)
    return torch.exp(-error / (std**2)) * _not_getup(env, command_name)


def getup_progress(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_height: float = STAND_HEIGHT,
) -> torch.Tensor:
    """Dense reward for robots that were reset lying on the floor: get the trunk upright and up to height.

    All standing / tracking rewards are withheld from these envs until they are upright, so lying
    still is never the best option; once upright the normal rewards take over.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    upright = 0.5 * (1.0 - asset.data.projected_gravity_b.torch[:, 2])  # 0 = upside down, 1 = upright
    height = torch.clamp(asset.data.root_link_pos_w.torch[:, 2] / target_height, 0.0, 1.0)
    return (0.5 * upright + 0.5 * height * upright) * _cmd_term(env, command_name).getup_mask.float()


def hurdle_clearance(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    foot_rest_height: float = FOOT_REST_HEIGHT,
    clearance_margin: float = 0.012,
    lane_half_width: float = 0.40,
) -> torch.Tensor:
    """Reward getting over the hurdle that the command generator placed in the robot's path.

    Only active in hurdle envs and only while the robot is in the hurdle's lane. In the hurdle frame
    (x along the approach direction, 0 at the bar):

    - a foot that is above the bar is rewarded for being higher than the bar top (step / jump over)
    - both feet airborne right at the bar adds a bonus (favours hopping over taller bars)
    - being upright just past the bar is rewarded (clean landing instead of a face-plant)

    ``asset_cfg`` must select the two foot (ankle) bodies.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    cmd_term = _cmd_term(env, command_name)

    cos_y = torch.cos(cmd_term.hurdle_yaw)
    sin_y = torch.sin(cmd_term.hurdle_yaw)
    hurdle_xy = cmd_term.hurdle_pos_w[:, :2]

    root_delta = asset.data.root_link_pos_w.torch[:, :2] - hurdle_xy
    root_x = root_delta[:, 0] * cos_y + root_delta[:, 1] * sin_y
    root_y = -root_delta[:, 0] * sin_y + root_delta[:, 1] * cos_y
    active = cmd_term.is_hurdle_env & (torch.abs(root_y) < lane_half_width)

    feet_pos = asset.data.body_pos_w.torch[:, asset_cfg.body_ids]  # (N, 2, 3)
    feet_delta = feet_pos[:, :, :2] - hurdle_xy.unsqueeze(1)
    feet_x = feet_delta[:, :, 0] * cos_y.unsqueeze(1) + feet_delta[:, :, 1] * sin_y.unsqueeze(1)
    feet_height = feet_pos[:, :, 2] - foot_rest_height
    needed = (cmd_term.hurdle_top_height + clearance_margin).unsqueeze(1)
    lift = torch.clamp(feet_height / needed, 0.0, 1.0)
    over_bar = (torch.abs(feet_x) < 0.05).float()
    step_over = (lift * over_bar).mean(dim=1)

    forces = torch.norm(contact_sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids], dim=-1)
    both_airborne = ((forces[:, 0] < 1.0) & (forces[:, 1] < 1.0)).float()
    at_bar = ((root_x > -0.12) & (root_x < 0.04)).float()
    hop = 0.5 * both_airborne * at_bar

    upright = torch.exp(-torch.sum(torch.square(asset.data.projected_gravity_b.torch[:, :2]), dim=1) / 0.05)
    past_bar = ((root_x > 0.06) & (root_x < 0.30)).float()
    landing = past_bar * upright

    return (step_over + hop + landing) * active.float()


# ==============================================================================
# CUSTOM OBSERVATION FUNCTIONS
# ==============================================================================
def tof_depth_observation(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tof_sensor"),
    max_distance: float = TOF_MAX_RANGE,
    noise: float = 0.0,
) -> torch.Tensor:
    """Observation term returning normalized 8x8 forward ToF depth measurements (64 dims).

    Clipped to [0.0, max_distance] meters and scaled to [0.0, 1.0]. Rays that hit nothing read 1.0.

    ``noise`` is the half-width [m] of uniform range noise added to rays that hit something. Rays
    that miss stay at exactly 1.0: they are constant most of the time, and the observation normalizer
    would otherwise blow their noise up into full-scale random inputs.
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    dists = torch.nan_to_num(sensor.data.ray_distances.torch, nan=max_distance, posinf=max_distance, neginf=0.0)
    hit = dists < max_distance
    if noise > 0.0:
        dists = dists + (2.0 * torch.rand_like(dists) - 1.0) * noise * hit
    return torch.clamp(dists, 0.0, max_distance) / max_distance


def camera_edges_observation(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tof_sensor"),
    edge_threshold: float = 0.15,
    max_distance: float = TOF_MAX_RANGE,
) -> torch.Tensor:
    """Observation term providing an 8x8 spatial edge map of the forward depth image (64 dims).

    Computes 2D Sobel gradients on the 8x8 depth matrix, highlighting obstacle silhouettes and step edges.

    Note: play.py can overwrite these inputs with Canny edges from a webcam (``--camera``). The policy is
    only trained on depth edges, so a webcam feed is outside the training distribution.
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    dists = torch.nan_to_num(sensor.data.ray_distances.torch, nan=max_distance, posinf=max_distance, neginf=0.0)
    d = torch.clamp(dists, 0.0, max_distance).view(-1, 1, 8, 8)

    # 3x3 Sobel kernel for horizontal and vertical edge extraction
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=dists.device
    ).view(1, 1, 3, 3) / 8.0
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=dists.device
    ).view(1, 1, 3, 3) / 8.0

    d_pad = torch.nn.functional.pad(d, (1, 1, 1, 1), mode="replicate")
    gx = torch.nn.functional.conv2d(d_pad, sobel_x)
    gy = torch.nn.functional.conv2d(d_pad, sobel_y)
    grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-6)

    edges = torch.clamp(grad_mag / edge_threshold, 0.0, 1.0)
    return edges.view(-1, 64)


def head_look_command(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    """Observation: commanded head (pitch, yaw) look offsets [rad] (2 dims)."""
    return _cmd_term(env, command_name).head_command


def gait_phase_observation(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Observation: gait clock as (sin, cos) of the phase (2 dims). Zero when commanded to stand."""
    cmd = env.command_manager.get_command(command_name)
    moving = ((torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])) > command_threshold).float().unsqueeze(1)
    angle = 2.0 * math.pi * gait_phase(env, command_name)
    return torch.stack([torch.sin(angle), torch.cos(angle)], dim=1) * moving


def hurdle_state_privileged(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Privileged critic observation: robot position in the hurdle frame, bar height and active flag (4 dims)."""
    robot: Articulation = env.scene[asset_cfg.name]
    cmd_term = _cmd_term(env, command_name)
    active = cmd_term.is_hurdle_env.float()
    delta = robot.data.root_link_pos_w.torch[:, :2] - cmd_term.hurdle_pos_w[:, :2]
    cos_y = torch.cos(cmd_term.hurdle_yaw)
    sin_y = torch.sin(cmd_term.hurdle_yaw)
    x_h = delta[:, 0] * cos_y + delta[:, 1] * sin_y
    y_h = -delta[:, 0] * sin_y + delta[:, 1] * cos_y
    top = cmd_term.hurdle_top_height
    return torch.stack([x_h * active, y_h * active, top * active, active], dim=1)


def task_flags_privileged(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    """Privileged critic observation: (is recovering from a fall, is following a waypoint) (2 dims)."""
    cmd_term = _cmd_term(env, command_name)
    return torch.stack([cmd_term.getup_mask.float(), cmd_term.is_waypoint_env.float()], dim=1)


def feet_contact_force_mag(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Privileged Critic Observation: net contact force magnitude on left and right feet (2 dims)."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w.torch
    return torch.norm(forces, dim=-1)


# ==============================================================================
# CUSTOM TERMINATION FUNCTIONS
# ==============================================================================
def _getup_exempt(env: ManagerBasedRLEnv, command_name: str, getup_timeout_s: float) -> torch.Tensor:
    """Envs recovering from a fall are not terminated for being on the floor until the timeout expires."""
    within_time = env.episode_length_buf * env.step_dt < getup_timeout_s
    return _cmd_term(env, command_name).getup_mask & within_time


def fallen_low_height(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    command_name: str = "base_velocity",
    getup_timeout_s: float = 8.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the trunk drops below ``minimum_height`` (except while a get-up attempt is allowed)."""
    asset: Articulation = env.scene[asset_cfg.name]
    fallen = asset.data.root_link_pos_w.torch[:, 2] < minimum_height
    return fallen & ~_getup_exempt(env, command_name, getup_timeout_s)


def fallen_bad_orientation(
    env: ManagerBasedRLEnv,
    limit_angle: float,
    command_name: str = "base_velocity",
    getup_timeout_s: float = 8.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the trunk tilts past ``limit_angle`` (except while a get-up attempt is allowed)."""
    asset: Articulation = env.scene[asset_cfg.name]
    tilt = torch.acos(torch.clamp(-asset.data.projected_gravity_b.torch[:, 2], -1.0, 1.0))
    return (tilt > limit_angle) & ~_getup_exempt(env, command_name, getup_timeout_s)


def physics_diverged(
    env: ManagerBasedRLEnv,
    max_joint_vel: float = 100.0,
    max_root_lin_vel: float = 20.0,
    max_root_ang_vel: float = 100.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate environments whose physics state is non-finite or physically impossible.

    A diverging contact solve does not always produce NaN; it often produces finite but absurd
    velocities first (the motors are limited to 10 rad/s). Those states are caught here too, because
    the rewards computed from them are astronomically large and wreck the value function.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    d = asset.data
    bad = ~torch.isfinite(d.joint_pos.torch).all(dim=1)
    bad |= ~torch.isfinite(d.joint_vel.torch).all(dim=1)
    bad |= ~torch.isfinite(d.root_link_pos_w.torch).all(dim=1)
    bad |= ~torch.isfinite(d.root_link_quat_w.torch).all(dim=1)
    bad |= ~torch.isfinite(d.root_link_lin_vel_w.torch).all(dim=1)
    bad |= ~torch.isfinite(d.root_link_ang_vel_w.torch).all(dim=1)
    bad |= d.joint_vel.torch.abs().amax(dim=1) > max_joint_vel
    bad |= torch.norm(d.root_link_lin_vel_w.torch, dim=1) > max_root_lin_vel
    bad |= torch.norm(d.root_link_ang_vel_w.torch, dim=1) > max_root_ang_vel
    return bad


# ==============================================================================
# CUSTOM EVENT AND CURRICULUM FUNCTIONS
# ==============================================================================
def reset_fallen_robots(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | slice,
    command_name: str = "base_velocity",
    fraction: float = 0.08,
    unlock_scale: float = 0.70,
    drop_height: float = 0.15,
    joint_offset: float = 0.4,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset a fraction of robots tumbled on the floor so the policy learns to stand back up.

    Must run after the normal base / joint reset events. The task only starts once the speed
    curriculum has reached ``unlock_scale``, so basic walking is learned first.
    """
    cmd_term = _cmd_term(env, command_name)
    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    cmd_term.getup_mask[env_ids] = False
    if fraction <= 0.0 or cmd_term.speed_scale < unlock_scale:
        return

    picked = env_ids[torch.rand(len(env_ids), device=env.device) < fraction]
    n = len(picked)
    if n == 0:
        return

    asset: Articulation = env.scene[asset_cfg.name]

    # Drop the robot from a small height with a random orientation (on its back, front or side)
    default_pose = asset.data.default_root_pose.torch[picked]
    pos = default_pose[:, 0:3] + env.scene.env_origins[picked]
    pos[:, 2] = drop_height
    roll = torch.empty(n, device=env.device).uniform_(-math.pi, math.pi)
    pitch = torch.empty(n, device=env.device).uniform_(-0.5 * math.pi, 0.5 * math.pi)
    yaw = torch.empty(n, device=env.device).uniform_(-math.pi, math.pi)
    quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
    asset.write_root_pose_to_sim_index(root_pose=torch.cat([pos, quat], dim=-1), env_ids=picked)
    asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=env.device), env_ids=picked)

    joint_pos = asset.data.default_joint_pos.torch[picked].clone()
    joint_pos += torch.empty_like(joint_pos).uniform_(-joint_offset, joint_offset)
    limits = asset.data.soft_joint_pos_limits.torch[picked]
    joint_pos = joint_pos.clamp_(limits[..., 0], limits[..., 1])
    asset.write_joint_position_to_sim_index(position=joint_pos, joint_ids=slice(None), env_ids=picked)
    asset.write_joint_velocity_to_sim_index(
        velocity=torch.zeros_like(joint_pos), joint_ids=slice(None), env_ids=picked
    )

    cmd_term.getup_mask[picked] = True


def push_robot_scaled(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    command_name: str = "base_velocity",
    start_fraction: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Random velocity push whose strength grows with the speed curriculum.

    Pushes start at ``start_fraction`` of ``velocity_range`` and reach the full range when the speed
    curriculum is complete. Full-strength shoves from the first iteration knock over a robot that has
    not learned to balance yet, which hides the reward signal for everything else.
    """
    factor = start_fraction + (1.0 - start_fraction) * curriculum_progress(_cmd_term(env, command_name))
    scaled = {axis: (lo * factor, hi * factor) for axis, (lo, hi) in velocity_range.items()}
    mdp.push_by_setting_velocity(env, env_ids, velocity_range=scaled, asset_cfg=asset_cfg)


def command_speed_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "base_velocity",
    reward_term_name: str = "track_lin_vel_xy_exp",
    tracking_threshold: float = 0.80,
    min_seconds_between_falls: float = 10.0,
    scale_step: float = 0.05,
    max_scale: float = 1.0,
) -> dict[str, float]:
    """Raise the commanded speed scale when the robot both tracks well and stays on its feet.

    Statistics are collected from all episodes that ended since the last check:

    - tracking: linear-velocity tracking reward collected per second of episode time, as a fraction
      of the maximum possible
    - seconds between falls: total episode time divided by the number of episodes that ended in a
      fall (not a timeout)

    Every ``max_episode_length`` environment steps, if both exceed their thresholds, the scale grows
    by ``scale_step``. Gating on falls matters: tracking alone looks fine for a robot that walks well
    for one second and then falls, and raising the difficulty for such a robot stalls training.

    Returns the values that are logged under ``Curriculum/command_speed/...``.
    """
    cmd_term = _cmd_term(env, command_name)
    state = env.__dict__.setdefault(
        "_speed_curriculum",
        {"reward": 0.0, "time": 0.0, "falls": 0, "last_step": 0, "tracking": 0.0, "seconds_between_falls": 0.0},
    )

    reward_manager = env.reward_manager
    weight = reward_manager.get_term_cfg(reward_term_name).weight
    state["reward"] += float(torch.nan_to_num(reward_manager._episode_sums[reward_term_name][env_ids]).sum())
    state["time"] += float(env.episode_length_buf[env_ids].sum()) * env.step_dt
    state["falls"] += int(env.termination_manager.terminated[env_ids].sum())

    if env.common_step_counter - state["last_step"] >= env.max_episode_length:
        if state["time"] > 0.0:
            state["tracking"] = state["reward"] / (state["time"] * weight)
            state["seconds_between_falls"] = state["time"] / max(state["falls"], 1)
            if state["tracking"] > tracking_threshold and state["seconds_between_falls"] > min_seconds_between_falls:
                cmd_term.speed_scale = min(cmd_term.speed_scale + scale_step, max_scale)
        state["reward"] = 0.0
        state["time"] = 0.0
        state["falls"] = 0
        state["last_step"] = env.common_step_counter

    return {
        "speed_scale": cmd_term.speed_scale,
        "tracking": state["tracking"],
        "seconds_between_falls": state["seconds_between_falls"],
    }


def smoothness_penalty_ramp(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_names: list[str],
    command_name: str = "base_velocity",
    start_fraction: float = 0.3,
) -> float:
    """Scale the smoothness penalties with the speed curriculum.

    The listed reward terms start at ``start_fraction`` of their configured weight and reach the full
    weight when the speed curriculum is complete. At full strength from the first iteration these
    penalties make "do not move" the easiest way to score, before the robot has found a gait.
    """
    base_weights = env.__dict__.setdefault("_smoothness_base_weights", {})
    factor = start_fraction + (1.0 - start_fraction) * curriculum_progress(_cmd_term(env, command_name))
    for name in term_names:
        term_cfg = env.reward_manager.get_term_cfg(name)
        if name not in base_weights:
            base_weights[name] = term_cfg.weight
        term_cfg.weight = base_weights[name] * factor
    return factor


# ==============================================================================
# 1. SCENE CONFIGURATION
# ==============================================================================
def _hurdle_cfg(name: str, height: float) -> RigidObjectCfg:
    """Kinematic hurdle bar of the given height, parked below the floor until it is placed."""
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.90, height),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -1.0)),
    )


@configclass
class MicroduckSceneCfg(InteractiveSceneCfg):
    """Scene configuration with large ground plane, Microduck robot, and contact sensors."""

    # Static ground slab. It has to be large enough for the whole grid of robots: with the previous
    # 100 m slab, 81% of 8,192 robots (90% of 16,384) were spawned beyond its edge and simply fell.
    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.CuboidCfg(
            size=(GROUND_SIZE, GROUND_SIZE, 0.2),
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

    # Obstacle hurdles: 4 cm deep, 90 cm wide bars of three heights (1.5 / 2.5 / 4 cm). They start parked
    # underground; the command generator teleports one of them into the path of robots in "hurdle" envs.
    hurdle_low: RigidObjectCfg = _hurdle_cfg(HURDLE_NAMES[0], HURDLE_HEIGHTS[0])
    hurdle_mid: RigidObjectCfg = _hurdle_cfg(HURDLE_NAMES[1], HURDLE_HEIGHTS[1])
    hurdle_high: RigidObjectCfg = _hurdle_cfg(HURDLE_NAMES[2], HURDLE_HEIGHTS[2])

    # Forward-facing 8x8 ToF depth sensor: a 45 deg fan of 64 rays that moves rigidly with the trunk,
    # tilted down so the lower rows see the floor ahead and the upper rows see obstacles.
    tof_sensor: RayCasterCfg = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/trunk_base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.07, 0.0, 0.05)),
        ray_alignment="base",
        pattern_cfg=TofFovPatternCfg(),
        max_distance=TOF_MAX_RANGE,
        mesh_prim_paths=["/World/ground"] + [f"{{ENV_REGEX_NS}}/{name}" for name in HURDLE_NAMES],
        global_world_only=False,
    )


# ==============================================================================
# 2. ACTION CONFIGURATION
# ==============================================================================
@configclass
class ActionsCfg:
    """Action specifications: 14 joint position target offsets (target = action * scale + default pose).

    Leg scales are 0.40 rad per unit action: with the initial exploration noise (std 0.6) that is about
    0.24 rad of joint noise, and the +/-3 action clip still reaches +/-1.2 rad, i.e. the whole useful range.
    """

    joint_pos: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale={
            ".*hip_yaw.*": 0.35,
            ".*hip_roll.*": 0.28,
            ".*hip_pitch.*": 0.40,
            ".*knee.*": 0.40,
            ".*ankle.*": 0.40,
            ".*neck.*": 0.35,
            ".*head.*": 0.50,
        },
        use_default_offset=True,
        # Joint targets are clamped just inside the joint limits. The policy output itself is only loosely
        # clipped (see clip_actions in the PPO config), and targets far outside the joint range make the
        # contact solver diverge to NaN.
        clip={
            ".*hip_yaw.*": (-0.38, 0.38),
            ".*hip_roll.*": (-0.34, 0.34),
            ".*hip_pitch.*": (-1.40, 1.40),
            ".*knee.*": (-1.40, 1.40),
            ".*ankle.*": (-1.40, 1.40),
            "neck_pitch": (-1.40, 0.90),
            "head_pitch": (-1.40, 1.40),
            "head_yaw": (-2.60, 2.60),
            "head_roll": (-0.39, 0.39),
        },
    )


# ==============================================================================
# 3. COMMAND CONFIGURATION
# ==============================================================================
@configclass
class CommandsCfg:
    """Velocity command specifications: sprint up to 0.75 m/s and turns up to 2.5 rad/s at full speed scale."""

    base_velocity: MicroduckVelocityCommandCfg = MicroduckVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(2.5, 6.0),
        rel_heading_envs=0.20,
        heading_command=True,
        heading_control_stiffness=0.60,
        ranges=MicroduckVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.45, 0.75),  # Aggressive forward sprint up to 0.75 m/s
            lin_vel_y=(-0.35, 0.35),  # Fast lateral strafe
            ang_vel_z=(-2.50, 2.50),  # Snappy, aggressive turning up to 2.5 rad/s
            heading=(-math.pi, math.pi),  # 360-degree compass direction exploration
        ),
    )


# ==============================================================================
# 4. OBSERVATION CONFIGURATION
# ==============================================================================
PROPRIO_HISTORY = 3
"""Number of consecutive 20 ms samples of each proprioceptive term given to the policy."""


def observation_term_slices(env: ManagerBasedRLEnv, group: str = "policy") -> dict[str, slice]:
    """Index range of every observation term inside a concatenated observation group.

    Use this instead of hardcoded indices; the layout changes whenever a term or its history changes.
    """
    slices: dict[str, slice] = {}
    start = 0
    names = env.observation_manager.active_terms[group]
    dims = env.observation_manager.group_obs_term_dim[group]
    for name, dim in zip(names, dims):
        size = int(math.prod(dim))
        slices[name] = slice(start, start + size)
        start += size
    return slices


@configclass
class ObservationsCfg:
    """Observation specifications for the Actor policy network."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations fed to the policy (Total: 285 dimensions).

        The six proprioceptive terms keep a 3-step history (the current and the two previous 20 ms samples,
        oldest first), which lets the network infer accelerations, contact events and actuator lag:

        gravity 3x3 | ang vel 3x3 | joint pos 3x14 | joint vel 3x14 | last action 3x14 | velocity command 3x3
        (153 dims), followed by ToF depth 64 | depth edges 64 | head look command 2 | gait clock 2.

        Do not hardcode indices: use ``observation_term_slices(env, "policy")``. Sensor terms carry uniform
        noise matching what real sensors would add.
        """

        # 1. Base orientation (projected gravity vector in torso frame: 3 dims)
        projected_gravity = ObservationTermCfg(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
            history_length=PROPRIO_HISTORY,
        )

        # 2. Base angular velocity (roll, pitch, yaw rates: 3 dims)
        base_ang_vel = ObservationTermCfg(
            func=mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
            history_length=PROPRIO_HISTORY,
        )

        # 3. Joint positions relative to default standing pose (14 dims)
        joint_pos = ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
            history_length=PROPRIO_HISTORY,
        )

        # 4. Joint angular velocities (14 dims)
        joint_vel = ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=UniformNoiseCfg(n_min=-1.0, n_max=1.0),
            history_length=PROPRIO_HISTORY,
        )

        # 5. Last action vector applied to motors (14 dims)
        actions = ObservationTermCfg(func=mdp.last_action, history_length=PROPRIO_HISTORY)

        # 6. Target velocity command (vx, vy, wz: 3 dims)
        velocity_commands = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            history_length=PROPRIO_HISTORY,
        )

        # 7. Forward 8x8 ToF depth matrix (64 dims, normalized [0, 1] relative to 1.0 m range)
        tof_depth = ObservationTermCfg(
            func=tof_depth_observation,
            params={"sensor_cfg": SceneEntityCfg("tof_sensor"), "max_distance": TOF_MAX_RANGE, "noise": 0.02},
        )

        # 8. 8x8 depth edge map (64 dims, in [0, 1])
        camera_edges = ObservationTermCfg(
            func=camera_edges_observation,
            params={"sensor_cfg": SceneEntityCfg("tof_sensor"), "edge_threshold": 0.15},
        )

        # 9. Head look command (head pitch, head yaw offsets: 2 dims)
        head_look = ObservationTermCfg(func=head_look_command, params={"command_name": "base_velocity"})

        # 10. Gait clock (sin, cos of the stepping phase: 2 dims)
        gait_clock = ObservationTermCfg(
            func=gait_phase_observation,
            params={"command_name": "base_velocity"},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObservationGroupCfg):
        """Privileged observations provided strictly to the Critic during training (26 extra dims).

        Supplies ground-truth quantities not directly measurable by real onboard sensors:
        - True base linear velocity in base frame (3 dims)
        - True base height above ground (1 dim)
        - True motor joint effort / applied torques (14 dims)
        - Robot position in the hurdle frame, bar height, hurdle-active flag (4 dims)
        - Net contact force magnitude on feet (2 dims)
        - Task flags: recovering from a fall, following a waypoint (2 dims)
        """
        # 1. True base linear velocity (3 dims)
        base_lin_vel = ObservationTermCfg(
            func=mdp.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        # 2. True base position z (1 dim)
        base_pos_z = ObservationTermCfg(
            func=mdp.base_pos_z,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        # 3. Motor joint efforts / applied torques (14 dims)
        joint_effort = ObservationTermCfg(
            func=mdp.joint_effort,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        # 4. Hurdle state in the hurdle frame (4 dims)
        hurdle_state = ObservationTermCfg(func=hurdle_state_privileged)
        # 5. Net contact force magnitude on left and right feet (2 dims)
        foot_forces = ObservationTermCfg(func=feet_contact_force_mag)
        # 6. Task flags (2 dims)
        task_flags = ObservationTermCfg(func=task_flags_privileged)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


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

    # Fall recovery (EXPERIMENTAL, off by default): a fraction of robots is reset tumbled on the floor
    # (must come after the two resets above). Not ready: tumbled resets make the contact solver diverge,
    # and because these episodes are exempt from fall terminations for getup_timeout_s they last far
    # longer than normal ones, so they end up occupying most of the envs and stall walking progress.
    # Only enable it to fine-tune a policy that already walks well (long episodes).
    reset_fallen = EventTermCfg(
        func=reset_fallen_robots,
        mode="reset",
        params={"command_name": "base_velocity", "fraction": 0.0, "unlock_scale": 0.70},
    )

    # Periodic velocity push disturbances, including a yaw kick, for push recovery. The range below is the
    # full strength, reached when the speed curriculum completes; pushes start at 30% of it.
    push_robot = EventTermCfg(
        func=push_robot_scaled,
        mode="interval",
        interval_range_s=(2.5, 5.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "base_velocity",
            "start_fraction": 0.3,
            "velocity_range": {"x": (-0.45, 0.45), "y": (-0.35, 0.35), "yaw": (-0.6, 0.6)},
        },
    )

    # --------------------------------------------------------------------------
    # DOMAIN RANDOMIZATION (Sim-to-Real Robustness)
    # --------------------------------------------------------------------------
    # Randomize trunk mass by +/- 15% (simulating payloads, batteries, hardware variations)
    randomize_trunk_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*trunk_base.*"]),
            "mass_distribution_params": (0.85, 1.15),
            "operation": "scale",
        },
    )

    # Randomize joint friction and armature (+/- 20%)
    randomize_joint_params = EventTermCfg(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "friction_distribution_params": (0.80, 1.20),
            "armature_distribution_params": (0.80, 1.20),
            "operation": "scale",
        },
    )

    # Randomize actuator stiffness and damping gains (+/- 15%)
    randomize_gains = EventTermCfg(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "stiffness_distribution_params": (0.85, 1.15),
            "damping_distribution_params": (0.85, 1.15),
            "operation": "scale",
        },
    )


# ==============================================================================
# 6. REWARDS CONFIGURATION
# ==============================================================================
LEG_JOINTS = [".*hip.*", ".*knee.*", ".*ankle.*"]
FOOT_BODIES = [".*ankle.*"]
SMOOTHNESS_TERMS = ["action_rate_l2", "action_smoothness", "joint_acc_l2", "joint_torques_l2"]


@configclass
class RewardsCfg:
    """Reward terms shaping the biped locomotion behavior."""

    # 1. Primary Objectives: Velocity Command Tracking & Directional Gain
    # std = 0.2 m/s: tight enough that standing still does not collect most of the reward for a slow walk command
    track_lin_vel_xy_exp = RewardTermCfg(
        func=track_lin_vel_xy,
        weight=3.0,
        params={"command_name": "base_velocity", "std": 0.2, "asset_cfg": SceneEntityCfg("robot")},
    )
    track_ang_vel_z_exp = RewardTermCfg(
        func=track_ang_vel_z,
        weight=2.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.5), "asset_cfg": SceneEntityCfg("robot")},
    )
    track_yaw_fine = RewardTermCfg(
        func=track_yaw_fine,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.35, "asset_cfg": SceneEntityCfg("robot")},
    )

    # Directional Alignment Gains (Guarantees bot gains reward strictly for moving in the commanded direction)
    # The linear gain is what gives a gradient far from the target, where the tight exponential kernel
    # above is flat (e.g. a robot standing still under a sprint command).
    directional_lin_gain = RewardTermCfg(
        func=directional_linear_velocity_gain,
        weight=2.0,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot"), "threshold": 0.05},
    )
    directional_yaw_gain = RewardTermCfg(
        func=directional_angular_velocity_gain,
        weight=1.0,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot"), "threshold": 0.10},
    )
    stand_still_rest = RewardTermCfg(
        func=stand_still_rest,
        weight=3.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )
    # Heavy penalty for non-zero leg actions when at rest, forcing actions to zero
    stand_still_action_l2 = RewardTermCfg(
        func=stand_still_action_l2,
        weight=-0.2,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)},
    )
    # General action L2 penalty keeping actions small (also what holds the exploration noise in check)
    action_l2 = RewardTermCfg(func=mdp.action_l2, weight=-0.002)
    # Falling costs 2.0 on the step it happens (weight x 0.02 s step). Early in training, when a robot
    # earns little per step anyway, this is the clearest signal that staying up comes first.
    fall_penalty = RewardTermCfg(func=mdp.is_terminated, weight=-100.0)
    # Sudden Stop Deceleration & Anti-Fall Stabilization (prevents tipping forward when stopping abruptly)
    sudden_stop_balance = RewardTermCfg(
        func=sudden_stop_anti_fall,
        weight=3.0,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot"), "pitch_damping_weight": 0.15},
    )

    # 2. Tasks
    # Hurdle clearance: only active for robots that have a hurdle placed in their path
    hurdle_clearance = RewardTermCfg(
        func=hurdle_clearance,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )
    # Fall recovery: only active for robots that were reset lying on the floor
    getup = RewardTermCfg(
        func=getup_progress,
        weight=4.0,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
    )
    # Head look command tracking
    head_look = RewardTermCfg(
        func=head_look_tracking,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=["head_pitch", "head_yaw"], preserve_order=True),
        },
    )

    # 3. Torso Upright Stability and Height
    upright = RewardTermCfg(
        func=upright_gaussian,
        weight=2.0,
        params={"std": math.sqrt(0.05), "asset_cfg": SceneEntityCfg("robot")},
    )
    base_height = RewardTermCfg(
        func=base_height_reached,
        weight=1.0,
        params={"target_height": STAND_HEIGHT, "asset_cfg": SceneEntityCfg("robot")},
    )
    ang_vel_xy_l2 = RewardTermCfg(
        func=mdp.ang_vel_xy_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # Kept small so it does not penalize hopping over a hurdle or sprint bounce
    lin_vel_z_l2 = RewardTermCfg(
        func=mdp.lin_vel_z_l2,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # 4. Leg Posture Anchor (tight std_standing=0.10 rad keeps bot steady at rest; std_walking=0.55 rad allows strides)
    leg_posture = RewardTermCfg(
        func=variable_leg_posture,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS),
            "std_standing": 0.10,
            "std_walking": 0.55,
            "walking_threshold": 0.01,
        },
    )

    # 5. Joint Limit Penalty (MuJoCo dof_pos_limits = -1.0)
    dof_pos_limits = RewardTermCfg(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # 6. Smoothness & Energy. The four terms listed in SMOOTHNESS_TERMS start at 30% of the weights given
    # here and ramp to 100% with the speed curriculum (see CurriculumCfg.smoothness_ramp).
    action_rate_l2 = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.02)
    action_smoothness = RewardTermCfg(func=action_smoothness_l2, weight=-0.005)
    joint_vel_l2 = RewardTermCfg(
        func=mdp.joint_vel_l2,
        weight=-0.0005,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_acc_l2 = RewardTermCfg(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_torques_l2 = RewardTermCfg(
        func=mdp.joint_torques_l2,
        weight=-0.03,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # Soft footfalls: penalize foot contact force above ~2x body weight (robot weighs ~6.4 N)
    feet_impact = RewardTermCfg(
        func=mdp.contact_forces,
        weight=-0.02,
        params={"threshold": 12.0, "sensor_cfg": SceneEntityCfg("contact_forces")},
    )

    # 7. Biped Stepping Cadence
    # Swing time of one foot while the other is planted (both walking and turning)
    feet_air_time = RewardTermCfg(
        func=feet_air_time_mujoco,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "threshold_min": 0.125,
            "threshold_max": 0.300,
            "command_threshold": 0.01,
        },
    )
    # Alternating left / right contacts following the gait clock given to the policy
    gait_contact = RewardTermCfg(
        func=gait_contact_schedule,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "stance_ratio": 0.55,
        },
    )

    # 8. Foot Sliding Penalty (MuJoCo foot_slip = -0.1)
    feet_slide = RewardTermCfg(
        func=feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES),
        },
    )

    # 9. Neck / head-roll regularization (head pitch and yaw are driven by the look command instead)
    head_neck_pose = RewardTermCfg(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["neck_pitch", "head_roll"])},
    )


# ==============================================================================
# 7. TERMINATION CONFIGURATION (Episode resets on fall)
# ==============================================================================
@configclass
class TerminationsCfg:
    """Conditions terminating an episode."""

    # Episode length timeout (20.0 seconds)
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)

    # Robot fallen: base height drops below 3.5 cm (severe fall onto floor).
    # Robots practising fall recovery are exempt for the first 8 s of their episode.
    base_contact = TerminationTermCfg(
        func=fallen_low_height,
        params={"minimum_height": 0.035, "command_name": "base_velocity", "getup_timeout_s": 8.0},
    )

    # Robot tilted excessively (pitch or roll > 70 degrees, matching MuJoCo limit_angle=1.22 rad)
    bad_orientation = TerminationTermCfg(
        func=fallen_bad_orientation,
        params={"limit_angle": 1.22, "command_name": "base_velocity", "getup_timeout_s": 8.0},
    )

    # Terminate immediately on a non-finite or physically impossible state (diverged contact solve)
    nan_state = TerminationTermCfg(
        func=physics_diverged,
        time_out=False,
    )


# ==============================================================================
# 8. CURRICULUM CONFIGURATION
# ==============================================================================
@configclass
class CurriculumCfg:
    """Curriculum terms."""

    # Commanded speeds start at 50% and grow to 100% once the robot tracks well AND stays on its feet.
    # Along the way: pushes and smoothness penalties ramp from 30% to 100%, hurdles unlock at 0.90
    # (lowest bar only until 1.0), and fall recovery (if enabled) unlocks at 0.70.
    command_speed = CurriculumTermCfg(
        func=command_speed_curriculum,
        params={
            "command_name": "base_velocity",
            "reward_term_name": "track_lin_vel_xy_exp",
            "tracking_threshold": 0.80,
            "min_seconds_between_falls": 10.0,
            "scale_step": 0.05,
        },
    )

    smoothness_ramp = CurriculumTermCfg(
        func=smoothness_penalty_ramp,
        params={"command_name": "base_velocity", "term_names": SMOOTHNESS_TERMS, "start_fraction": 0.3},
    )


# ==============================================================================
# 9. COMPLETE RL ENVIRONMENT CONFIGURATION
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
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post-initialization configurations for simulation timing."""
        # General physics timing
        self.decimation = 4                   # 50 Hz policy control frequency
        self.episode_length_s = 20.0          # 20.0 seconds per episode
        self.sim.dt = 1.0 / 200.0             # 200 Hz physics simulation step (5 ms)
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = False


class MicroduckEnv(ManagerBasedRLEnv):
    """Microduck environment. Identical to the base env except that rewards from diverged physics are discarded.

    If the physics of one robot diverges, that env is reset by the ``nan_state`` termination, but its
    reward for that step is NaN or astronomically large, which would abort the PPO runner or destroy
    the value function. Such rewards are zeroed, and all rewards are bounded to a sane range
    (a normal step reward is well inside +/-1).
    """

    max_step_reward: float = 5.0

    def __init__(self, cfg: MicroduckEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        # Every robot must have ground under it for the whole episode. Robots are placed on a grid that
        # grows with num_envs; one that starts beyond the edge of the ground slab falls forever, and
        # such robots silently poison training (this is what ruined the first training runs).
        ground_half_size = 0.5 * min(cfg.scene.ground.spawn.size[0], cfg.scene.ground.spawn.size[1])
        grid_half_size = float(self.scene.env_origins[:, :2].abs().max())
        if grid_half_size + GROUND_WALK_MARGIN > ground_half_size:
            raise ValueError(
                f"The ground slab ({2 * ground_half_size:.0f} m wide) is too small for {self.num_envs} robots at"
                f" {cfg.scene.env_spacing} m spacing: the robot grid reaches {grid_half_size:.0f} m from the centre"
                f" and robots need {GROUND_WALK_MARGIN:.0f} m of walking room beyond that. Increase GROUND_SIZE in"
                " microduck_env_cfg.py or reduce scene.env_spacing."
            )

    def step(self, action: torch.Tensor):
        obs, reward, terminated, truncated, extras = super().step(action)
        reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        reward = torch.where(self.termination_manager.get_term("nan_state"), torch.zeros_like(reward), reward)
        reward = torch.clamp(reward, -self.max_step_reward, self.max_step_reward)
        return obs, reward, terminated, truncated, extras
