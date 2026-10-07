#!/bin/bash
# µNAS search of the MIND byte-level encoder in a MIND-only copy of the ELIOS uNAS fork.
#
# The shared fork ~/uNAS belongs to the Lane-Change-MCU project and is never modified: the first
# run copies it to ~/uNAS_mind (without its artifacts), and only the copy gets the MIND files,
# the trainer patch and the config registrations. The venv ~/dmir_nas is used as it is.
#
# Usage (WSL):
#   bash /mnt/c/Projects/PhD/MIND/unas/run_mind_search.sh setup
#   bash /mnt/c/Projects/PhD/MIND/unas/run_mind_search.sh reference          # June winners, search recipe
#   bash /mnt/c/Projects/PhD/MIND/unas/run_mind_search.sh search mind_h7      # chunked, resumable
# Settings through MIND_ROUNDS, MIND_EPOCHS, MIND_POPULATION, MIND_SAMPLE, MIND_PARALLEL,
# MIND_ERROR_BOUND, MIND_SUFFIX (see unas/mind_config.py). Logs and GPU samples: logs/nas/.
REPO=/mnt/c/Projects/PhD/MIND
FORK_SRC=$HOME/uNAS
FORK=$HOME/uNAS_mind
VENV=$HOME/dmir_nas
source "$VENV/env.sh"          # expands LD_LIBRARY_PATH, which may be unset: source before set -u
set -u
export TF_CPP_MIN_LOG_LEVEL=2 TF_FORCE_GPU_ALLOW_GROWTH=true MIND_UNAS_DATA=$REPO/artifacts/unas/data
mkdir -p "$REPO/logs/nas"
TAG=$(date +%m%d%H%M)

setup() {
  if [ ! -d "$FORK" ]; then
    rsync -a --exclude artifacts --exclude __pycache__ "$FORK_SRC/" "$FORK/"
    echo "copied $FORK_SRC to $FORK"
  fi
  cp "$REPO/unas/mind_dataset.py" "$FORK/dataset/mind_dataset.py"
  cp "$REPO/unas/mind_config.py" "$FORK/configs/mind_config.py"
  cp "$REPO/unas/cnn1d_gap.py" "$FORK/configs/cnn1d_gap.py"
  cp "$REPO/unas/safe_saver.py" "$FORK/configs/safe_saver.py"
  # fresh trainer from the shared fork (read only), then the MIND patch
  cp "$FORK_SRC/uNAS/model_trainer.py" "$FORK/uNAS/model_trainer.py"
  python3 "$REPO/unas/patch_trainer.py" "$FORK/uNAS/model_trainer.py"
  grep -q "from .mind_dataset import MIND_Embedding_Dataset" "$FORK/dataset/__init__.py" \
    || printf '\nfrom .mind_dataset import MIND_Embedding_Dataset\n' >> "$FORK/dataset/__init__.py"
  python3 - "$FORK/driver.py" <<'EOF'
import sys
p = sys.argv[1]
s = open(p).read()
entries = {"mind_h7": "get_mind_h7_setup", "mind_f401": "get_mind_f401_setup"}
new = [f'    "{k}": ("configs.mind_config", "{f}"),' for k, f in entries.items() if f'"{k}"' not in s]
if new:
    s = s.replace("_CONFIGS = {", "_CONFIGS = {" + chr(10) + chr(10).join(new), 1)
    open(p, "w").write(s)
    print("registered", len(new), "MIND configs")
EOF
}

gpu_sampler() {   # $1 = csv path; 30 s samples while the job runs
  ( nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu \
      --format=csv,noheader -l 30 > "$1" 2>/dev/null ) &
  echo $!
}

case "${1:?setup|reference|search}" in
  setup)
    setup ;;
  reference|grid)
    setup
    MODE=$1; ARG=""; [ "$MODE" = grid ] && ARG=grid
    GPU=$(gpu_sampler "$REPO/logs/nas/${MODE}_${TAG}_gpu.csv"); trap "kill $GPU 2>/dev/null" EXIT
    cd "$FORK" && nice -n 10 "$VENV/bin/python" "$REPO/unas/eval_reference.py" $ARG 2>&1 \
      | tee "$REPO/logs/nas/${MODE}_${TAG}.log" | grep -E --line-buffered '^\{|Traceback|Error' ;;
  search)
    CONFIG="${2:?config}"
    setup
    NAME="$CONFIG${MIND_SUFFIX:-}"
    STATE="$FORK/artifacts/$NAME/${NAME}_agingevosearch_state.pickle"
    GPU=$(gpu_sampler "$REPO/logs/nas/${NAME}_${TAG}_gpu.csv"); trap "kill $GPU 2>/dev/null" EXIT
    cd "$FORK"
    for chunk in $(seq 1 "${MAX_CHUNKS:-8}"); do
      LOG="$REPO/logs/nas/${NAME}_${TAG}_chunk${chunk}.log"
      args=(-c "$CONFIG" --seed 42 --save-every 5)
      [ -f "$STATE" ] && args+=(-l "$STATE")
      echo ">>> $NAME chunk $chunk @ $(date '+%F %T') -> $LOG"
      nice -n 10 "$VENV/bin/python" driver.py "${args[@]}" > "$LOG" 2>&1
      done_ok=$(grep -c "Search done" "$LOG")
      hist=$(grep -c "Training complete" "$LOG")
      echo "    chunk $chunk: +$hist candidates, search_done=$done_ok @ $(date '+%F %T')"
      [ "$done_ok" -ge 1 ] && { echo "=== $NAME COMPLETE ==="; break; }
      [ "$hist" -eq 0 ] && { echo "!!! $NAME made no progress; stopping"; break; }
    done
    echo "$NAME models on disk: $(ls "$FORK/artifacts/$NAME/models" 2>/dev/null | grep -c 'h5$')" ;;
esac
