#!/usr/bin/env bash
# Start Microduck training in a detached tmux session, so it keeps running when the terminal,
# SSH connection or IDE closes. TensorBoard is started alongside on port 6006.
#
# Usage (arguments are passed straight to train.py):
#   ./train_tmux.sh --preset workstation --max_iterations 2000     # pilot run
#   ./train_tmux.sh --preset workstation --resume --max_iterations 13000
#
# Afterwards:
#   tmux attach -t microduck      # watch (detach again with Ctrl+B then D)
#   tmux kill-session -t microduck   # stop training and TensorBoard
set -euo pipefail

SESSION="microduck"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_DIR="${ISAACLAB_DIR:-$HOME/robotics/IsaacLab}"
LOG_DIR="$SCRIPT_DIR/logs"
CONSOLE_LOG="$LOG_DIR/train_console.log"

if [ "${DRY_RUN:-0}" != "1" ]; then
    command -v tmux >/dev/null || { echo "tmux is not installed. Install it with: sudo apt install -y tmux"; exit 1; }
fi
command -v uv >/dev/null || { echo "uv is not on PATH. See AGENT_HANDOFF.md, section 2."; exit 1; }
[ -d "$ISAACLAB_DIR" ] || { echo "Isaac Lab not found at $ISAACLAB_DIR. Set ISAACLAB_DIR."; exit 1; }
if [ "${DRY_RUN:-0}" != "1" ] && tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "A tmux session named '$SESSION' is already running. Attach with: tmux attach -t $SESSION"
    echo "Stop it first (tmux kill-session -t $SESSION) if you really want to start a new run."
    exit 1
fi

mkdir -p "$LOG_DIR"

# WSL2 keeps the NVIDIA CUDA libraries in /usr/lib/wsl/lib; without this Warp cannot find the GPU
ENV_SETUP=""
if [ -d /usr/lib/wsl/lib ]; then
    ENV_SETUP="export LD_LIBRARY_PATH=/usr/lib/wsl/lib:\${LD_LIBRARY_PATH:-}; "
fi

TRAIN_ARGS=""
if [ "$#" -gt 0 ]; then
    printf -v TRAIN_ARGS '%q ' "$@"
fi
TRAIN_CMD="${ENV_SETUP}cd '$ISAACLAB_DIR' && uv run python '$SCRIPT_DIR/train.py' $TRAIN_ARGS 2>&1 | tee '$CONSOLE_LOG'"
BOARD_CMD="${ENV_SETUP}cd '$ISAACLAB_DIR' && uv run tensorboard --logdir '$LOG_DIR' --port 6006 --bind_all"

TRAIN_WINDOW="bash -lc \"$TRAIN_CMD; echo; echo '[training process ended]'; exec bash\""
BOARD_WINDOW="bash -lc \"$BOARD_CMD; exec bash\""

# DRY_RUN=1 prints the commands instead of starting anything
if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "$TRAIN_WINDOW"
    echo "$BOARD_WINDOW"
    exit 0
fi

tmux new-session -d -s "$SESSION" -n train "$TRAIN_WINDOW"
tmux new-window -t "$SESSION" -n tensorboard "$BOARD_WINDOW"

echo "Training started in tmux session '$SESSION'."
echo "  Watch:        tmux attach -t $SESSION      (detach: Ctrl+B then D)"
echo "  Console log:  $CONSOLE_LOG"
echo "  TensorBoard:  http://localhost:6006"
echo "  Health check: cd $ISAACLAB_DIR && uv run python $SCRIPT_DIR/check_training.py"
echo "  Stop:         tmux kill-session -t $SESSION"
