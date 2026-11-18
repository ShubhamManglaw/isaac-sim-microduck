"""Objective evaluation of a trained Microduck policy.

Runs fixed test scenarios (standing, walking, sprint-then-stop, turning, strafing, curves, hurdles, head look)
with many robots in parallel, measures what a person would judge by eye (falls, drift, jitter, head wobble,
braking, turning response, left/right symmetry, hurdle clearance) and prints PASS / FAIL against targets.

Usage (from ~/robotics/IsaacLab):

  uv run python ~/robotics/isaac-sim-microduck/experiments/05_train_policy/evaluate_policy.py
  uv run python ~/robotics/isaac-sim-microduck/experiments/05_train_policy/evaluate_policy.py \\
      --load_checkpoint /path/to/model_5000.pt --num_envs 64 --json results.json

The policy is evaluated on its own: no rest stabilizer, no pushes, no sensor noise.
Exit code: 0 = all checks pass, 1 = some checks fail.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

EXP_DIR = Path(__file__).resolve().parent
for p in [EXP_DIR, EXP_DIR.parent / "04_rl_env", EXP_DIR.parent / "03_robot_model"]:
    if str(p) not in sys.path:
        sys.path.append(str(p))

import torch
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from microduck_env_cfg import HURDLE_NAMES, MicroduckEnv, MicroduckEnvCfg
from microduck_ppo_cfg import MicroduckPPORunnerCfg, build_runner

LEG_KEYS = ("hip", "knee", "ankle")
HEAD_JOINTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")


# ------------------------------------------------------------------------------------------------
# Scenario definitions
# ------------------------------------------------------------------------------------------------
@dataclass
class Scenario:
    name: str
    duration: float
    command: Callable[[float], tuple[float, float, float]]
    """Velocity command (vx, vy, wz) as a function of time [s]."""
    head: tuple[float, float] = (0.0, 0.0)
    hurdle: str | None = None
    measure_from: float = 2.0
    """Steady-state metrics are averaged from this time on (after the robot has settled into the command)."""


def constant(vx: float = 0.0, vy: float = 0.0, wz: float = 0.0):
    return lambda t: (vx, vy, wz)


def sprint_then_stop(speed: float, run_time: float):
    return lambda t: (speed, 0.0, 0.0) if t < run_time else (0.0, 0.0, 0.0)


SCENARIOS = [
    Scenario("stand", 10.0, constant()),
    Scenario("stand_head_look", 6.0, constant(), head=(0.25, 0.6)),
    Scenario("walk_0.30", 8.0, constant(vx=0.30)),
    Scenario("walk_0.50", 8.0, constant(vx=0.50)),
    Scenario("sprint_0.75", 8.0, constant(vx=0.75)),
    Scenario("backward_0.30", 8.0, constant(vx=-0.30)),
    Scenario("stop_from_0.50", 7.0, sprint_then_stop(0.50, 3.0)),
    Scenario("stop_from_0.75", 7.0, sprint_then_stop(0.75, 3.0)),
    Scenario("turn_left_1.5", 5.0, constant(wz=1.5), measure_from=2.0),
    Scenario("turn_right_1.5", 5.0, constant(wz=-1.5), measure_from=2.0),
    Scenario("strafe_left_0.25", 6.0, constant(vy=0.25)),
    Scenario("strafe_right_0.25", 6.0, constant(vy=-0.25)),
    Scenario("curve_left", 6.0, constant(vx=0.40, wz=0.8)),
    Scenario("curve_right", 6.0, constant(vx=0.40, wz=-0.8)),
    Scenario("hurdle_low", 7.0, constant(vx=0.35), hurdle="low"),
    Scenario("hurdle_mid", 7.0, constant(vx=0.35), hurdle="mid"),
    Scenario("hurdle_high", 7.0, constant(vx=0.35), hurdle="high"),
]

HURDLE_DISTANCE = 0.5


# ------------------------------------------------------------------------------------------------
# Running a scenario
# ------------------------------------------------------------------------------------------------
def run_scenario(env, wrapped_env, policy, scenario: Scenario) -> dict[str, float]:
    """Run one scenario on all envs and return its metrics (averaged over robots that did not fall)."""
    device = env.device
    n = env.num_envs
    dt = env.step_dt
    robot = env.scene["robot"]
    contact = env.scene.sensors["contact_forces"]
    cmd_term = env.command_manager.get_term("base_velocity")
    all_ids = torch.arange(n, device=device)

    leg_ids = [i for i, name in enumerate(robot.joint_names) if any(k in name for k in LEG_KEYS)]
    head_ids = [robot.joint_names.index(name) for name in HEAD_JOINTS]
    head_pitch_id = robot.joint_names.index("head_pitch")
    head_yaw_id = robot.joint_names.index("head_yaw")

    wrapped_env.reset()
    cmd_term.park_hurdle(all_ids)
    if scenario.hurdle is not None:
        cmd_term.place_hurdle_ahead(all_ids, index=HURDLE_NAMES.index(f"hurdle_{scenario.hurdle}"), distance=HURDLE_DISTANCE)
    start_xy = robot.data.root_link_pos_w.torch[:, :2].clone()
    start_heading = robot.data.heading_w.torch.clone()
    cmd_term.head_command[:] = torch.tensor(scenario.head, device=device)
    obs = wrapped_env.get_observations()

    steps = int(round(scenario.duration / dt))
    measure_start = int(round(scenario.measure_from / dt))
    alive = torch.ones(n, dtype=torch.bool, device=device)

    log = {k: [] for k in ("vx", "vy", "wz", "x_fwd", "pitch", "leg_vel", "head_vel", "action_rate", "head_err",
                           "contact_l", "contact_r", "z")}
    prev_action = torch.zeros(n, env.action_manager.action.shape[-1], device=device)

    for step in range(steps):
        t = step * dt
        vx, vy, wz = scenario.command(t)
        cmd_term.vel_command_b[:] = torch.tensor([vx, vy, wz], device=device)
        cmd_term.head_command[:] = torch.tensor(scenario.head, device=device)
        with torch.no_grad():
            action = torch.clamp(policy(obs), -env_clip(env), env_clip(env))
        obs, _, _, _ = wrapped_env.step(action)
        alive &= ~env.termination_manager.terminated

        rel = robot.data.root_link_pos_w.torch[:, :2] - start_xy
        fwd = rel[:, 0] * torch.cos(start_heading) + rel[:, 1] * torch.sin(start_heading)
        forces = torch.norm(contact.data.net_forces_w.torch, dim=-1)
        head_offset = robot.data.joint_pos.torch - robot.data.default_joint_pos.torch
        head_err = torch.sqrt(
            (head_offset[:, head_pitch_id] - scenario.head[0]) ** 2 + (head_offset[:, head_yaw_id] - scenario.head[1]) ** 2
        )
        log["vx"].append(robot.data.root_lin_vel_b.torch[:, 0].clone())
        log["vy"].append(robot.data.root_lin_vel_b.torch[:, 1].clone())
        log["wz"].append(robot.data.root_ang_vel_b.torch[:, 2].clone())
        log["x_fwd"].append(fwd)
        log["pitch"].append(robot.data.projected_gravity_b.torch[:, 0].clone())
        log["leg_vel"].append(robot.data.joint_vel.torch[:, leg_ids].pow(2).mean(dim=1).sqrt())
        log["head_vel"].append(robot.data.joint_vel.torch[:, head_ids].pow(2).mean(dim=1).sqrt())
        log["action_rate"].append((action - prev_action).pow(2).mean(dim=1).sqrt())
        log["head_err"].append(head_err)
        log["contact_l"].append(forces[:, 0] > 1.0)
        log["contact_r"].append(forces[:, 1] > 1.0)
        log["z"].append(robot.data.root_link_pos_w.torch[:, 2].clone())
        prev_action = action

    data = {k: torch.stack(v, dim=0) for k, v in log.items()}  # (steps, n)
    ok = alive
    m: dict[str, float] = {"fall_rate": float(1.0 - alive.float().mean())}
    if not bool(ok.any()):
        return m

    def steady(key: str, start: int = measure_start, end: int | None = None) -> torch.Tensor:
        return data[key][start:end, ok]

    vx_cmd, vy_cmd, wz_cmd = scenario.command(scenario.duration - dt)
    m["vx"] = float(steady("vx").mean())
    m["vy"] = float(steady("vy").mean())
    m["wz"] = float(steady("wz").mean())
    m["leg_vel_rms"] = float(steady("leg_vel").mean())
    m["head_vel_rms"] = float(steady("head_vel").mean())
    m["action_rate_rms"] = float(steady("action_rate", start=max(measure_start, 2)).mean())
    m["head_err"] = float(steady("head_err").mean())
    m["planar_speed"] = float(torch.sqrt(steady("vx") ** 2 + steady("vy") ** 2).mean())

    # Gait: share of time each foot is on the ground, and number of touchdowns per foot
    cl, cr = steady("contact_l").float(), steady("contact_r").float()
    m["stance_l"] = float(cl.mean())
    m["stance_r"] = float(cr.mean())
    touch_l = ((cl[1:] - cl[:-1]) > 0).float().sum(dim=0).mean()
    touch_r = ((cr[1:] - cr[:-1]) > 0).float().sum(dim=0).mean()
    m["steps_per_s_l"] = float(touch_l) / max(scenario.duration - scenario.measure_from, 1e-6)
    m["steps_per_s_r"] = float(touch_r) / max(scenario.duration - scenario.measure_from, 1e-6)

    # Turning response: time until the yaw rate, averaged over 0.2 s, first reaches 80 % of the command
    if scenario.name.startswith("turn"):
        target = 0.8 * scenario.command(0.0)[2]
        window = max(int(round(0.2 / dt)), 1)
        wz = data["wz"][:, ok] * math.copysign(1.0, target)
        wz_avg = torch.nn.functional.avg_pool1d(wz.T.unsqueeze(1), window, stride=1).squeeze(1).T
        reached = wz_avg >= abs(target)
        first = torch.where(reached.any(dim=0), reached.float().argmax(dim=0).float(), torch.full_like(reached[0].float(), steps))
        m["turn_response_s"] = float((first + window).mean() * dt)

    # Braking: time and distance from the stop command until the robot is below 0.05 m/s
    if scenario.name.startswith("stop"):
        stop_step = next(i for i in range(steps) if scenario.command(i * dt) == (0.0, 0.0, 0.0))
        speed = torch.sqrt(data["vx"][stop_step:, ok] ** 2 + data["vy"][stop_step:, ok] ** 2)
        below = speed < 0.05
        first = torch.where(below.any(dim=0), below.float().argmax(dim=0).float(), torch.full_like(below[0].float(), steps - stop_step))
        m["stop_time_s"] = float(first.mean() * dt)
        dist = data["x_fwd"][-1, ok] - data["x_fwd"][stop_step, ok]
        m["stop_distance_m"] = float(dist.mean())
        m["max_forward_tilt_deg"] = float(torch.rad2deg(torch.asin(torch.clamp(data["pitch"][stop_step:, ok], -1, 1))).max(dim=0).values.mean())
        m["drift_after_stop_mps"] = float(speed[-50:].mean())

    # Hurdles: success = got 15 cm past the bar without falling
    if scenario.hurdle is not None:
        passed = data["x_fwd"][-1] > HURDLE_DISTANCE + 0.15
        m["hurdle_success"] = float((passed & alive).float().mean())
        m["hurdle_stalled"] = float(((~passed) & alive).float().mean())

    return m


def env_clip(env) -> float:
    return env.__dict__.setdefault("_eval_clip", float(MicroduckPPORunnerCfg().clip_actions))


# ------------------------------------------------------------------------------------------------
# Checks against targets
# ------------------------------------------------------------------------------------------------
def asym(a: float, b: float) -> float:
    """Relative difference between two magnitudes (0 = identical)."""
    a, b = abs(a), abs(b)
    return abs(a - b) / max(0.5 * (a + b), 1e-6)


def build_checks(r: dict[str, dict[str, float]]) -> list[tuple[str, str, float | None, str, bool | None]]:
    """(area, check, value, target, passed) rows."""
    rows = []

    def add(area, name, value, target_text, predicate):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            rows.append((area, name, None, target_text, False))
        else:
            rows.append((area, name, value, target_text, bool(predicate(value))))

    g = lambda sc, key: r.get(sc, {}).get(key)  # noqa: E731

    # Falls
    for sc in r:
        if not sc.startswith("hurdle"):
            add("falls", f"{sc}: fall rate", g(sc, "fall_rate"), "= 0", lambda v: v <= 0.0)

    # Standing still
    add("rest", "stand: drift speed [m/s]", g("stand", "planar_speed"), "< 0.02", lambda v: v < 0.02)
    add("rest", "stand: leg joint motion [rad/s]", g("stand", "leg_vel_rms"), "< 0.15", lambda v: v < 0.15)
    add("rest", "stand: head motion [rad/s]", g("stand", "head_vel_rms"), "< 0.10", lambda v: v < 0.10)
    add("rest", "stand: action jitter", g("stand", "action_rate_rms"), "< 0.03", lambda v: v < 0.03)
    add("rest", "stand: both feet on ground", min(g("stand", "stance_l") or 0, g("stand", "stance_r") or 0), "> 0.95", lambda v: v > 0.95)

    # Walking
    for sc, speed in [("walk_0.30", 0.30), ("walk_0.50", 0.50), ("sprint_0.75", 0.75), ("backward_0.30", -0.30)]:
        vx = g(sc, "vx")
        add("walk", f"{sc}: speed error [m/s]", None if vx is None else abs(vx - speed), "< 0.08", lambda v: v < 0.08)
        add("smooth", f"{sc}: action jitter", g(sc, "action_rate_rms"), "< 0.15", lambda v: v < 0.15)
        add("head", f"{sc}: head motion [rad/s]", g(sc, "head_vel_rms"), "< 0.50", lambda v: v < 0.50)
        add("symmetry", f"{sc}: L/R stance difference",
            None if g(sc, "stance_l") is None else asym(g(sc, "stance_l"), g(sc, "stance_r")), "< 0.10", lambda v: v < 0.10)

    # Stopping
    for sc in ("stop_from_0.50", "stop_from_0.75"):
        add("stop", f"{sc}: time to stop [s]", g(sc, "stop_time_s"), "< 1.0", lambda v: v < 1.0)
        add("stop", f"{sc}: stopping distance [m]", g(sc, "stop_distance_m"), "< 0.35", lambda v: v < 0.35)
        add("stop", f"{sc}: drift after stop [m/s]", g(sc, "drift_after_stop_mps"), "< 0.03", lambda v: v < 0.03)

    # Turning
    for sc in ("turn_left_1.5", "turn_right_1.5"):
        wz = g(sc, "wz")
        add("turn", f"{sc}: rate achieved [share of 1.5]", None if wz is None else abs(wz) / 1.5, "> 0.80", lambda v: v > 0.80)
        add("turn", f"{sc}: response time [s]", g(sc, "turn_response_s"), "< 0.60", lambda v: v < 0.60)

    # Symmetry
    pairs = [("turn_left_1.5", "turn_right_1.5", "wz"), ("strafe_left_0.25", "strafe_right_0.25", "vy"),
             ("curve_left", "curve_right", "wz")]
    for a, b, key in pairs:
        va, vb = g(a, key), g(b, key)
        add("symmetry", f"{a.rsplit('_', 1)[0] if 'turn' in a else a.split('_')[0]} left vs right ({key})",
            None if va is None or vb is None else asym(va, vb), "< 0.10", lambda v: v < 0.10)
    for sc, speed in [("strafe_left_0.25", 0.25), ("strafe_right_0.25", -0.25)]:
        vy = g(sc, "vy")
        add("walk", f"{sc}: speed error [m/s]", None if vy is None else abs(vy - speed), "< 0.08", lambda v: v < 0.08)

    # Head look
    add("head", "stand_head_look: tracking error [rad]", g("stand_head_look", "head_err"), "< 0.10", lambda v: v < 0.10)

    # Hurdles
    add("hurdle", "hurdle_low: success", g("hurdle_low", "hurdle_success"), "> 0.80", lambda v: v > 0.80)
    add("hurdle", "hurdle_mid: success", g("hurdle_mid", "hurdle_success"), "> 0.50", lambda v: v > 0.50)
    add("hurdle", "hurdle_high: success", g("hurdle_high", "hurdle_success"), "> 0.30", lambda v: v > 0.30)
    return rows


# ------------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------------
def find_latest_checkpoint(root: Path) -> Path | None:
    latest = root / "latest"
    if latest.exists():
        pts = sorted(latest.resolve().glob("model_*.pt"), key=lambda f: int(f.stem.split("_")[-1]))
        if pts:
            return pts[-1]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained Microduck policy against objective targets.")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Checkpoint (.pt); default: newest in logs/latest.")
    parser.add_argument("--num_envs", type=int, default=64, help="Robots per scenario.")
    parser.add_argument("--scenarios", type=str, default=None, help="Comma-separated subset of scenario names.")
    parser.add_argument("--json", type=str, default=None, help="Also write all metrics to this JSON file.")
    add_launcher_args(parser)
    parser.set_defaults(visualizer="none")
    args = parser.parse_args()

    ckpt = Path(args.load_checkpoint).expanduser().resolve() if args.load_checkpoint else find_latest_checkpoint(EXP_DIR / "logs")
    if ckpt is None or not ckpt.is_file():
        print("[ERROR]: No checkpoint found. Pass --load_checkpoint.")
        return 1
    scenarios = SCENARIOS
    if args.scenarios:
        wanted = set(args.scenarios.split(","))
        scenarios = [s for s in SCENARIOS if s.name in wanted]

    physics_cfg = NewtonCfg(num_substeps=1, solver_cfg=MJWarpSolverCfg(), load_visual_shapes=False)
    with launch_simulation(cfg=physics_cfg, launcher_args=args) as resolved:
        env_cfg = MicroduckEnvCfg()
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.sim.physics = resolved
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
        expected = wrapped_env.get_observations()["policy"].shape[-1]
        got = loaded["actor_state_dict"]["mlp.0.weight"].shape[1]
        if got != expected:
            print(f"[ERROR]: {ckpt.name} expects {got}D observations but the environment provides {expected}D.")
            return 1
        runner.alg.load(loaded, load_cfg={"actor": True}, strict=True)
        policy = runner.get_inference_policy(device=env.device)
        cmd_term = env.command_manager.get_term("base_velocity")
        cmd_term.time_left[:] = 1e9
        cmd_term.cfg.resampling_time_range = (1e9, 1e9)

        results: dict[str, dict[str, float]] = {}
        for scenario in scenarios:
            results[scenario.name] = run_scenario(env, wrapped_env, policy, scenario)
            print(f"[eval] {scenario.name:20s} " + "  ".join(f"{k}={v:.3f}" for k, v in results[scenario.name].items()), flush=True)
        env.close()

    rows = build_checks(results)
    print(f"\nCheckpoint: {ckpt}\n")
    area_w = max(len(r[0]) for r in rows)
    name_w = max(len(r[1]) for r in rows)
    for area, name, value, target, passed in rows:
        status = "PASS" if passed else "FAIL"
        value_text = "n/a" if value is None else f"{value:.3f}"
        print(f"[{status}] {area:{area_w}s}  {name:{name_w}s}  {value_text:>8s}   target {target}")
    n_pass = sum(1 for r in rows if r[4])
    print(f"\n{n_pass}/{len(rows)} checks pass.")
    by_area: dict[str, list[bool]] = {}
    for area, _, _, _, passed in rows:
        by_area.setdefault(area, []).append(bool(passed))
    print("By area: " + ", ".join(f"{a} {sum(v)}/{len(v)}" for a, v in by_area.items()))

    if args.json:
        Path(args.json).write_text(json.dumps({"checkpoint": str(ckpt), "metrics": results,
                                               "checks": [dict(zip(("area", "check", "value", "target", "pass"), r)) for r in rows]}, indent=2))
        print(f"Metrics written to {args.json}")
    return 0 if n_pass == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
