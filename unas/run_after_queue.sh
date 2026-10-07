#!/bin/bash
# Waits for the overnight queue (unas/run_queue.sh) to finish, then runs the seed-based selection
# of both searches (unas/select_by_seeds.py), so the GPU is never shared between jobs.
# Start detached from Windows:
#   Start-Process -WindowStyle Hidden wsl.exe -ArgumentList '-e','bash','/mnt/c/Projects/PhD/MIND/unas/run_after_queue.sh'
R=/mnt/c/Projects/PhD/MIND
QLOG=$(ls -t "$R"/logs/nas/queue_*.log | head -1)
LOG=$R/logs/nas/selection_$(date +%m%d%H%M).log
echo "$(date '+%F %T') waiting for $QLOG" >> "$LOG"
until grep -q "QUEUE DONE" "$QLOG"; do sleep 60; done
source ~/dmir_nas/env.sh
export TF_CPP_MIN_LOG_LEVEL=2 TF_FORCE_GPU_ALLOW_GROWTH=true MIND_UNAS_DATA=$R/artifacts/unas/data
( nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu \
    --format=csv,noheader -l 30 > "$R/logs/nas/selection_gpu.csv" 2>/dev/null ) &
GPU=$!; trap "kill $GPU 2>/dev/null" EXIT
cd ~/uNAS_mind
for c in mind_h7 mind_f401; do
  echo "$(date '+%F %T') $c selection start" >> "$LOG"
  nice -n 10 ~/dmir_nas/bin/python "$R/unas/select_by_seeds.py" "$c" 8 2>&1 \
    | grep -E --line-buffered "candidate|best|Traceback|Error" >> "$LOG"
done
echo "$(date '+%F %T') SELECTION DONE" >> "$LOG"
