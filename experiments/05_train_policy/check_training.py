"""Health check for a Microduck training run. Reads the TensorBoard log and prints a verdict.

Usage (from ~/robotics/IsaacLab, no GPU needed, safe to run while training is in progress):

  uv run python ~/robotics/isaac-sim-microduck/experiments/05_train_policy/check_training.py
  uv run python ~/robotics/isaac-sim-microduck/experiments/05_train_policy/check_training.py --run /path/to/run_dir

Exit code: 0 = healthy, 1 = warnings (keep watching), 2 = failed (stop and follow the printed action).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

OK, WARN, FAIL = "OK  ", "WARN", "FAIL"

# Noise std bounds configured in microduck_ppo_cfg.py (std_range)
STD_FLOOR = 0.15
STD_CAP = 1.5


def recent_mean(acc: EventAccumulator, tag: str, window: int) -> float | None:
    """Mean of the last ``window`` logged values of a scalar, or None if the tag is absent."""
    if tag not in acc.Tags()["scalars"]:
        return None
    values = [e.value for e in acc.Scalars(tag)[-window:]]
    finite = [v for v in values if math.isfinite(v)]
    if len(finite) < len(values):
        return float("nan")
    return sum(finite) / len(finite) if finite else None


def earlier_mean(acc: EventAccumulator, tag: str, window: int, back: int) -> float | None:
    """Mean of ``window`` logged values ending ``back`` iterations before the newest one."""
    if tag not in acc.Tags()["scalars"]:
        return None
    events = acc.Scalars(tag)
    if len(events) <= back:
        return None
    values = [e.value for e in events[max(0, len(events) - back - window) : len(events) - back]]
    finite = [v for v in values if math.isfinite(v)]
    return sum(finite) / len(finite) if finite else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the health of a Microduck training run.")
    parser.add_argument("--run", type=str, default=None, help="Run directory (default: logs/latest).")
    parser.add_argument("--window", type=int, default=50, help="Number of recent iterations to average over.")
    args = parser.parse_args()

    run_dir = Path(args.run).expanduser() if args.run else Path(__file__).resolve().parent / "logs" / "latest"
    run_dir = run_dir.resolve()
    if not run_dir.is_dir() or not list(run_dir.glob("events.out.tfevents.*")):
        print(f"[ERROR] No TensorBoard log found in {run_dir}. Has training started?")
        return 2

    acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    acc.Reload()
    if "Policy/mean_std" not in acc.Tags()["scalars"]:
        print(f"[ERROR] {run_dir} has no training scalars yet. Wait for the first iterations to finish.")
        return 2

    iteration = acc.Scalars("Policy/mean_std")[-1].step
    get = lambda tag: recent_mean(acc, tag, args.window)  # noqa: E731

    std = get("Policy/mean_std")
    ep_len = get("Train/mean_episode_length")
    reward = get("Train/mean_reward")
    value_loss = get("Loss/value")
    diverged = get("Episode_Termination/nan_state")
    err_xy = get("Metrics/base_velocity/error_vel_xy")
    scale = get("Curriculum/command_speed/speed_scale")
    tracking = get("Curriculum/command_speed/tracking")
    fall_gap = get("Curriculum/command_speed/seconds_between_falls")

    rows: list[tuple[str, str, str, str]] = []  # (status, name, value, note)

    def fmt(value: float | None, digits: int = 3) -> str:
        return "n/a" if value is None else f"{value:.{digits}f}"

    # 1. Exploration noise
    if std is None or not math.isfinite(std):
        rows.append((FAIL, "Policy/mean_std", fmt(std), "Not finite. ACTION: stop; see 'Run is broken' in AGENT_HANDOFF.md."))
    elif std >= 0.93 * STD_CAP:
        rows.append((FAIL, "Policy/mean_std", fmt(std), f"Pinned at the {STD_CAP} cap: noise is winning. ACTION: stop; apply fix A."))
    elif std <= 1.05 * STD_FLOOR and iteration < 3000:
        rows.append((WARN, "Policy/mean_std", fmt(std), f"Already at the {STD_FLOOR} floor: exploration ended early. If episode length is also low, apply fix B."))
    else:
        rows.append((OK, "Policy/mean_std", fmt(std), f"Healthy range is {STD_FLOOR}-0.6."))

    # 2. Value loss
    if value_loss is None or not math.isfinite(value_loss) or value_loss > 100.0:
        rows.append((FAIL, "Loss/value", fmt(value_loss), "Value function blew up. ACTION: stop; see 'Run is broken'."))
    else:
        rows.append((OK, "Loss/value", fmt(value_loss), "Should stay finite and below ~10."))

    # 3. Physics divergence
    if diverged is not None and diverged > 0.01:
        rows.append((FAIL, "Episode_Termination/nan_state", fmt(diverged, 4), "More than 1% of resets are diverged physics. ACTION: stop; see 'Run is broken'."))
    elif diverged is not None and diverged > 0.001:
        rows.append((WARN, "Episode_Termination/nan_state", fmt(diverged, 4), "Some physics divergence. Keep watching; must not grow."))
    else:
        rows.append((OK, "Episode_Termination/nan_state", fmt(diverged, 4), "Diverged-physics resets; should stay near 0."))

    # 4. Balance (episode length, max 1000 steps = 20 s). While robots are still learning to balance,
    # what matters is that this keeps rising, so the comparison is against 500 iterations earlier.
    ep_len_before = earlier_mean(acc, "Train/mean_episode_length", args.window, back=500)
    rising = ep_len is not None and ep_len_before is not None and ep_len > 1.15 * ep_len_before
    trend = "" if ep_len_before is None else f" (was {ep_len_before:.0f} 500 iterations ago)"
    if ep_len is None:
        rows.append((WARN, "Train/mean_episode_length", "n/a", "Not logged yet."))
    elif ep_len >= 500:
        rows.append((OK, "Train/mean_episode_length", fmt(ep_len, 0), "Robots stay up. Target is close to 1000." + trend))
    elif iteration < 1000:
        rows.append((OK, "Train/mean_episode_length", fmt(ep_len, 0), "Too early to judge (judged from iteration 1000). Should trend upward." + trend))
    elif rising:
        rows.append((WARN, "Train/mean_episode_length", fmt(ep_len, 0), "Below 500 but still rising: keep training." + trend))
    elif iteration < 2000:
        rows.append((WARN, "Train/mean_episode_length", fmt(ep_len, 0), "Below 500 and not rising. If the same at iteration 2000, apply fix C." + trend))
    else:
        rows.append((FAIL, "Train/mean_episode_length", fmt(ep_len, 0), "Below 500 and no longer rising: robots keep falling. ACTION: stop; apply fix C." + trend))

    # 5. Curriculum
    if scale is None:
        rows.append((WARN, "Curriculum speed_scale", "n/a", "Not logged yet."))
    else:
        note = f"tracking {fmt(tracking, 2)} (needs > 0.80), seconds between falls {fmt(fall_gap, 1)} (needs > 10)."
        if scale >= 0.999:
            rows.append((OK, "Curriculum speed_scale", fmt(scale, 2), "Full speed reached. " + note))
        elif iteration >= 3000 and scale <= 0.501:
            blocker = "falls" if (fall_gap is not None and fall_gap <= 10.0) else "tracking"
            fix = "C" if blocker == "falls" else "D"
            rows.append((WARN, "Curriculum speed_scale", fmt(scale, 2), f"Has not advanced; blocked by {blocker}. " + note + f" If still stuck at 4000, apply fix {fix}."))
        else:
            rows.append((OK, "Curriculum speed_scale", fmt(scale, 2), "Advances 0.50 -> 1.00. " + note))

    # 6. Information only
    rows.append((OK, "Metrics/error_vel_xy", fmt(err_xy), "Velocity error [m/s]; below ~0.08 is good at the current speed scale."))
    rows.append((OK, "Train/mean_reward", fmt(reward, 2), "Information only; drops when the curriculum raises the difficulty."))

    print(f"\nRun:       {run_dir}")
    print(f"Iteration: {iteration}   (values are means over the last {args.window} iterations)\n")
    for status, name, value, note in rows:
        print(f"[{status}] {name:32s} {value:>9s}   {note}")

    statuses = [r[0] for r in rows]
    if FAIL in statuses:
        print("\nVERDICT: FAILED. Stop the run and follow the ACTION of the first FAIL line only (fixes: AGENT_HANDOFF.md, section 6).")
        return 2
    if WARN in statuses:
        print("\nVERDICT: WARNINGS. Let it run and check again in ~500 iterations.")
        return 1
    print("\nVERDICT: HEALTHY. Let it run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
