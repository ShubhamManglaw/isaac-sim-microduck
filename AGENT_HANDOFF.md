# Microduck RL: Training Runbook for the RTX 5070 Ti PC

This file is written for the agent (or person) who will train the Microduck walking policy on the
workstation. Follow the steps in order. Every decision you need to make is spelled out here; where
the runbook says "stop and report", do that instead of improvising.

---

## 0. Rules

1. **Do not edit any code or parameter unless a step in this file tells you to.** The values in the
   repository were chosen deliberately; section 8 explains the ones that look tempting to change.
2. **Change one thing at a time.** If a health check fails, apply exactly the one fix it names
   (section 6), rerun the pilot, and check again.
3. **Report numbers, not impressions.** When you report to the user, paste the output of
   `check_training.py`.
4. **Never claim the robot "walks well" from reward curves alone.** Only playback (section 7) shows that.
5. If something happens that this file does not cover, stop and report it to the user with the exact
   error text.

---

## 1. Background (read once)

- The robot is a 12 cm, 0.65 kg, 14-joint biped. The policy outputs joint position targets at 50 Hz.
- **The first training run (60,000 iterations planned) failed.** Two causes:
  1. **Most robots had no ground under them.** Robots are laid out on a grid 2.5 m apart, and the
     ground slab was only 100 m wide. With 8,192 robots the grid is 228 m wide, so 81% of the robots
     started beyond the edge and fell into empty space, over and over (with 16,384 robots it would
     have been 90%). Their episodes lasted about 7 steps and carried no information.
  2. **The exploration noise grew from 1.0 to about 30.** Actions were clipped to ±1, so the action
     penalties could not see large actions, and for the falling robots actions had no effect at all,
     so nothing held the entropy bonus back.

  Several other environment bugs were found at the same time (the "forward" depth sensor pointed at
  the floor, the jump reward was always on, the hurdle was not in the robot's path).
- This repository contains the fixed and extended setup. Section 9 lists what has and has not been
  validated.
- **An earlier agent session on this PC made local changes in an older clone** (a `--version v2`
  flag, `microduck_env_v2_cfg.py`, `microduck_ppo_v2_cfg.py`, 100,000-iteration preset). Those were
  built on the broken environment and are superseded. What was useful in them is already in this
  repository: the 3-step observation history, the `--resume` fix, the non-deprecated Isaac Lab
  calls, and a tmux launcher. **Use a fresh clone. Do not copy files from the old clone.**
  Two of its choices are intentionally not used here:
  - *48-step rollouts*: with 16,384 robots a 24-step rollout is already 393,000 samples per update;
    48 steps doubles the time per update for no measured gain.
  - *100,000 iterations*: far more than needed. This runbook uses a 2,000-iteration pilot and then
    up to 15,000 in total.

---

## 2. Setup (once)

The PC runs Ubuntu under WSL2. Run everything in the WSL2 terminal.

```bash
# 2.1 CUDA libraries for WSL2 (skip if this line is already in ~/.bashrc)
grep -q "/usr/lib/wsl/lib" ~/.bashrc || echo 'export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"' >> ~/.bashrc
source ~/.bashrc
nvidia-smi          # must list the RTX 5070 Ti

# 2.2 Tools
command -v uv   || curl -LsSf https://astral.sh/uv/install.sh | sh     # then open a new terminal
command -v tmux || sudo apt install -y tmux

# 2.3 This repository: fresh clone next to any old one
mkdir -p ~/robotics && cd ~/robotics
[ -d isaac-sim-microduck ] && mv isaac-sim-microduck isaac-sim-microduck-old-$(date +%Y%m%d)
git clone https://github.com/ShubhamManglaw/isaac-sim-microduck.git

# 2.4 Isaac Lab at the commit this code was validated on
cd ~/robotics
[ -d IsaacLab ] || git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git rev-parse HEAD      # validated commit: 44044b0963001962abe5701133427e37b003fffe (branch release/3.0.0)
```

If `git rev-parse HEAD` prints a different commit, **try the smoke test in section 3 first**. Only if
the smoke test fails, switch Isaac Lab to the validated commit:

```bash
cd ~/robotics/IsaacLab
git status --short      # if this lists local changes, stop and report them
git fetch origin && git checkout 44044b0963001962abe5701133427e37b003fffe
uv sync
```

Then, in both cases:

```bash
cd ~/robotics/IsaacLab
uv sync
uv pip install opencv-python
uv run python -c "import importlib.metadata as m; print('rsl-rl-lib', m.version('rsl-rl-lib'))"   # validated with 5.5.1
```

---

## 3. Step 1: smoke test (about 2 minutes)

```bash
cd ~/robotics/IsaacLab
uv run python ~/robotics/isaac-sim-microduck/experiments/05_train_policy/train.py \
  --num_envs 64 --max_iterations 3 --log_dir /tmp/microduck_smoke
```

It passes if all of these appear in the output:

- `Linear(in_features=285, out_features=512` (actor) and `Linear(in_features=311, out_features=512` (critic)
- three `Learning iteration` blocks
- `[SUCCESS] TRAINING COMPLETE!`
- no `Traceback`

If it fails with an import or attribute error, use the validated Isaac Lab commit (section 2.4) and
run it again. If it still fails, stop and report the full traceback.

---

## 4. Step 2: pilot run (2,000 iterations)

```bash
~/robotics/isaac-sim-microduck/experiments/05_train_policy/train_tmux.sh --preset workstation --max_iterations 2000
```

This starts training (16,384 robots) and TensorBoard (http://localhost:6006) in a detached tmux
session named `microduck`. It keeps running if the terminal closes.

- Watch it: `tmux attach -t microduck` (leave with `Ctrl+B` then `D`; do not press `Ctrl+C`).
- After the first 20 iterations, read `Iteration time` in the output and estimate the duration:
  `2000 × iteration time`. Reference: the laptop (RTX 5060) needs about 1.8 s per iteration with
  8,192 robots.
- `CUDA out of memory`: stop the session (`tmux kill-session -t microduck`) and start again with
  `--num_envs 12288` added; if that also fails, `--num_envs 8192`.

---

## 5. Step 3: health check

Run this at iterations 500, 1000 and 2000 (it does not disturb training):

```bash
cd ~/robotics/IsaacLab
uv run python ~/robotics/isaac-sim-microduck/experiments/05_train_policy/check_training.py
```

It prints one line per metric with `OK`, `WARN` or `FAIL`, and a verdict.

| Verdict | What to do |
|---|---|
| `HEALTHY` | Let it run. |
| `WARNINGS` | Let it run and check again 500 iterations later. |
| `FAILED` | Stop the run (`tmux kill-session -t microduck`), apply the one fix named in the `ACTION` text (section 6), and start the pilot again from scratch. |

**The pilot passes** when, at iteration 2000, the verdict is not `FAILED` and:

- `Policy/mean_std` is between 0.15 and 0.6
- `Train/mean_episode_length` is above 500
- `Curriculum speed_scale` is above 0.50

If the first two hold but the speed scale is still exactly 0.50, the pilot is a borderline pass:
continue to the full run and apply the curriculum rule in section 6 (fix C or D) if it has not
moved by iteration 4000.

---

## 6. Step 4: full run, and the fixes

### Continue to 15,000 iterations in total

```bash
~/robotics/isaac-sim-microduck/experiments/05_train_policy/train_tmux.sh --preset workstation --resume --max_iterations 13000
```

`--resume` loads the newest checkpoint of the newest run and the curriculum level saved beside it
(`curriculum_state.json`), and writes into a new run folder. `--max_iterations` is the number of
*additional* iterations. Run the health check every 1,000 to 2,000 iterations.

**You can stop early** once all of these have held for 1,000 iterations: speed scale 1.00, mean
episode length above 800, `Metrics/error_vel_xy` no longer improving. Stop with
`tmux attach -t microduck`, then `Ctrl+C` once (this saves the curriculum state), then verify with
playback (section 7).

### Fixes (apply only the one that the check names)

All paths are relative to `~/robotics/isaac-sim-microduck/experiments/`.

| Fix | When | File | Change |
|---|---|---|---|
| **A** | `Policy/mean_std` pinned at the 1.5 cap | `05_train_policy/microduck_ppo_cfg.py` | `entropy_coef=0.005` → `0.002`. Also confirm `clip_actions` is `3.0`; if it is not, restore it. |
| **B** | `Policy/mean_std` at the 0.15 floor before iteration 3000 **and** episode length below 150 | `05_train_policy/microduck_ppo_cfg.py` | `std_range=(0.15, 1.5)` → `(0.25, 1.5)` and `entropy_coef=0.005` → `0.008`. |
| **C** | Episode length below 300 at iteration 2000, or the curriculum is blocked by falls at iteration 4000 | `04_rl_env/microduck_env_cfg.py` | In `EventCfg.push_robot`: `"start_fraction": 0.3` → `0.1`. If that was already tried: in `RewardsCfg.fall_penalty`: `weight=-100.0` → `-200.0`. |
| **D** | Curriculum blocked by tracking at iteration 4000 (seconds between falls is above 10, tracking is below 0.80) | `04_rl_env/microduck_env_cfg.py` | In `CurriculumCfg.command_speed`: `"tracking_threshold": 0.80` → `0.70`. |
| **E** | Playback shows shuffling: feet barely leave the ground at walking commands | `04_rl_env/microduck_env_cfg.py` | In `RewardsCfg`: `gait_contact` `weight=1.5` → `2.5` and `feet_air_time` `weight=1.5` → `2.5`. Then `--resume --max_iterations 2000`. |
| **F** | Playback shows a jerky, buzzing gait once the speed scale is 1.00 | `04_rl_env/microduck_env_cfg.py` | In `RewardsCfg`: `action_rate_l2` `weight=-0.02` → `-0.04` and `action_smoothness` `weight=-0.005` → `-0.01`. Then `--resume --max_iterations 2000`. |

Fixes A to D change the learning problem: restart the pilot from scratch after them (no `--resume`).
Fixes E and F are fine-tuning: continue with `--resume`.

### "Run is broken" (value loss not finite, or more than 1% diverged-physics resets)

1. `cd ~/robotics/isaac-sim-microduck && git status --short` must print nothing except files a fix
   told you to change. If there are other changes, `git stash` them and rerun.
2. In `04_rl_env/microduck_env_cfg.py`, `EventCfg.reset_fallen` must have `"fraction": 0.0`
   (fall recovery is off; turning it on is what broke a validation run).
3. Use the validated Isaac Lab commit (section 2.4).
4. If it still happens, stop and report: the `check_training.py` output and the last 100 lines of
   `experiments/05_train_policy/logs/train_console.log`.

---

## 7. Step 5: look at the result (playback)

```bash
cd ~/robotics/IsaacLab
P=~/robotics/isaac-sim-microduck/experiments/05_train_policy/play.py

uv run python $P                                   # keyboard control, newest checkpoint
uv run python $P --no-rest-blend                   # how the policy itself stands still
uv run python $P --waypoints "0.6,0;0.6,0.6;0,0"   # walks the waypoints on its own
uv run python $P --hurdle low                      # 1.5 cm bar 0.6 m ahead (also: mid, high)
```

Keys: `W/S` forward/back, `A/D` turn, `Q/E` strafe, `F` sprint, `Space` stop, `I/K/J/L` head,
`C` center head. On WSL2 the 3D window needs WSLg (Windows 11); if no window opens, add
`--visualizer none --max_steps 500` to run headless and read the telemetry lines instead.

Checklist to report to the user (yes / no for each):

1. Stands still without drifting or twitching, with and without `--no-rest-blend`.
2. Walks forward at `W` ×3 (0.3 m/s) with feet alternating and clearly leaving the ground.
3. Walks a curve (`W` ×3 then `A` ×2) without falling.
4. Stops from a walk (`Space`) without tipping forward.
5. Reaches the waypoints.
6. Head follows `I/K/J/L` while standing.
7. Steps over the low hurdle.

Items 1 to 4 are the core result. Items 5 to 7 are extra tasks and may need more training.

---

## 8. Do not change these (and why)

| Setting | Value | Why it must stay |
|---|---|---|
| `GROUND_SIZE` (`microduck_env_cfg.py`) | 1000 m | The ground must reach under every robot. If you raise `--num_envs` far above 16,384 or change `env_spacing`, the environment checks this at start-up and stops with an explanation; follow it. |
| `clip_actions` (`microduck_ppo_cfg.py`) | 3.0 | A ±1 clip let the exploration noise grow without limit in the first run. |
| `std_range` upper bound | 1.5 | Safety cap on the noise. |
| `ActionsCfg.joint_pos.clip` (`microduck_env_cfg.py`) | joint limits | Targets outside the joint limits made the physics diverge to NaN. |
| `physics_diverged` termination and `MicroduckEnv.step` reward guard | present | One robot with diverged physics otherwise destroys the value function for all. |
| `EventCfg.reset_fallen` `fraction` | 0.0 | Fall recovery made the physics diverge and stalled walking in validation. |
| Hurdle bars placed 0.5 mm above the floor (`HURDLE_FLOOR_GAP`) | 0.0005 | A bar that overlaps the floor stalls the contact solver. |
| `num_steps_per_env` | 24 | See section 1. |
| ToF sensor pattern (`TofFovPatternCfg`) | forward, 22° down | The old pattern pointed straight down and saw only the robot's own legs. |
| Observation terms and their order | as is | Changing them invalidates every checkpoint. |

---

## 9. What is and is not validated

Validated on the laptop (RTX 5060, Isaac Lab commit `44044b0`, rsl-rl-lib 5.5.1):

- **Smoke test:** actor 285 inputs, critic 311 inputs, no errors; the start-up ground check passes.
- **Training, 2,048 robots, 450 iterations from scratch** (shared the GPU with another run):

  | Iteration | Episode length (max 1000) | Noise std | Velocity error [m/s] | Tracking (gate 0.80) | Value loss |
  |---|---|---|---|---|---|
  | 0 | 23 | 0.60 | 0.26 | 0.00 | 0.53 |
  | 100 | 736 | 0.53 | 0.20 | 0.30 | 0.43 |
  | 200 | 975 | 0.44 | 0.13 | 0.58 | 0.14 |
  | 450 | 996 | 0.32 | 0.094 | 0.77 | 0.10 |

  Zero diverged-physics resets. `check_training.py` verdict: HEALTHY.
  The speed curriculum had **not yet advanced** at iteration 450 (tracking 0.77, gate 0.80, still
  rising), so the push / smoothness ramps and hurdles were not exercised in this test.
- **Playback** (`play.py`, headless): loads a checkpoint, follows `--waypoints`, places `--hurdle`,
  stands at rest, and `--export` writes `policy.pt` and `policy.onnx`.
- **`check_training.py`** on the original failed run reports FAILED (noise pinned, episodes not
  rising) and on the validation run reports HEALTHY.

**Not validated** (this is why the pilot run exists):

- Training with 16,384 robots on the RTX 5070 Ti, including speed and memory.
- The curriculum beyond speed scale 0.50 (pushes, smoothness ramps, hurdles unlocking).
- Anything beyond the iteration counts listed above. Nobody has yet seen this setup produce a
  finished walking policy; the pilot and the health check are how you find out.
- `train_tmux.sh` with a real tmux (the laptop has none; its generated commands were checked with
  `DRY_RUN=1`).
- Hurdle clearance, waypoint following and head look as learned skills (the code paths run; whether
  the policy learns them is unknown).
- Isaac Lab versions other than the validated commit.

Known limitations:

- **Fall recovery (standing up) is off.** It needs separate work before it can be trained.
- **`play.py --camera`** feeds webcam edges into inputs trained on depth edges. Expect worse
  behavior with it; it is off by default.
- **Not yet built:** left/right symmetry augmentation, ground-friction randomization, action delay.
  The last two matter for a real robot, not for simulation.

---

## 10. Parameter reference

**PPO** (`05_train_policy/microduck_ppo_cfg.py`)

| Parameter | Value |
|---|---|
| Robots (`--preset workstation`) | 16,384 |
| `num_steps_per_env` | 24 |
| `num_mini_batches` (workstation) | 8 |
| `num_learning_epochs` | 5 |
| `learning_rate` / `schedule` / `desired_kl` | 1e-3 / adaptive / 0.01 |
| `gamma` / `lam` | 0.99 / 0.95 |
| `entropy_coef` | 0.005 |
| Noise: `init_std` / `std_type` / `std_range` | 0.6 / log / (0.15, 1.5) |
| `clip_actions` | 3.0 |
| Actor / critic MLP | [512, 256, 128] / [512, 512, 256, 128] |
| Actor / critic inputs | 285 / 311 |

**Environment** (`04_rl_env/microduck_env_cfg.py`)

| Item | Value |
|---|---|
| Control rate / physics rate / episode | 50 Hz / 200 Hz / 20 s |
| Action scale (rad per unit action) | hip yaw 0.35, hip roll 0.28, hip pitch / knee / ankle 0.40, neck 0.35, head 0.50 |
| Proprioceptive history | 3 samples (60 ms) |
| Speed curriculum | starts at 0.50; +0.05 every 20 s of sim time if tracking > 0.80 **and** more than 10 s between falls |
| Pushes | every 2.5 to 5 s; x ±0.45 m/s, y ±0.35 m/s, yaw ±0.6 rad/s at full strength; 30% → 100% with the curriculum |
| Smoothness penalties | 30% → 100% with the curriculum |
| Hurdles | from scale 0.90 (1.5 cm bar), all bars (1.5 / 2.5 / 4 cm) at 1.00 |
| Gait clock | cycle 0.50 s near zero speed → 0.30 s at 0.75 m/s |
| Command mix | forward 12%, sprint 8%, turn in place 10%, backward 7%, strafe 7%, curve 14%, waypoint 10%, hurdle 8% (forward until unlocked), stand 14%, uniform random 10% |

**Reward weights** (per second; the environment multiplies by the 0.02 s step)

| Group | Terms |
|---|---|
| Tracking | `track_lin_vel_xy_exp` 3.0 (std 0.2 m/s), `track_ang_vel_z_exp` 2.5, `track_yaw_fine` 2.0, `directional_lin_gain` 2.0, `directional_yaw_gain` 1.0 |
| Standing | `stand_still_rest` 3.0, `stand_still_action_l2` −0.2, `sudden_stop_balance` 3.0 |
| Posture | `upright` 2.0, `base_height` 1.0, `leg_posture` 1.5, `head_neck_pose` −0.1, `dof_pos_limits` −1.0 |
| Gait | `gait_contact` 1.5, `feet_air_time` 1.5, `feet_slide` −0.1, `feet_impact` −0.02 |
| Smoothness (ramped) | `action_rate_l2` −0.02, `action_smoothness` −0.005, `joint_torques_l2` −0.03, `joint_acc_l2` −2.5e-7 |
| Other penalties | `action_l2` −0.002, `joint_vel_l2` −0.0005, `ang_vel_xy_l2` −0.05, `lin_vel_z_l2` −0.02, `fall_penalty` −100 (= −2.0 per fall) |
| Tasks | `hurdle_clearance` 2.0, `head_look` 1.0, `getup` 4.0 (inactive while fall recovery is off) |
