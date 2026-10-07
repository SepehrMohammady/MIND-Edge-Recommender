#!/bin/bash
# Overnight queue for step 3, one job at a time on the laptop GPU:
#   1. µNAS search with the STM32H7B3I-DK budget, then its harvest
#   2. µNAS search with the NUCLEO-F401RE budget, then its harvest
#   3. the hand-designed family (6 widths x 5 depths, 384-d output) under the search recipe
# Error bound: validation error of the hand-designed 64-5-384 under the third recipe, mean of
# seeds 42, 1, 2 (paper/results/unas/reference.json): 1 - 0.3431 = 0.657.
# Start detached from Windows so it survives the terminal:
#   Start-Process -WindowStyle Hidden wsl.exe -ArgumentList '-e','bash','/mnt/c/Projects/PhD/MIND/unas/run_queue.sh'
R=/mnt/c/Projects/PhD/MIND
LOG=$R/logs/nas/queue_$(date +%m%d%H%M).log
export MIND_ERROR_BOUND=${MIND_ERROR_BOUND:-0.657}
mkdir -p "$R/logs/nas"
echo "$(date '+%F %T') queue start, error bound $MIND_ERROR_BOUND" >> "$LOG"
for c in mind_h7 mind_f401; do
  echo "$(date '+%F %T') $c search start" >> "$LOG"
  bash "$R/unas/run_mind_search.sh" search "$c" >> "$LOG" 2>&1
  echo "$(date '+%F %T') $c search end" >> "$LOG"
  (cd ~/uNAS_mind && ~/dmir_nas/bin/python "$R/unas/harvest_search.py" "$c") >> "$LOG" 2>&1
done
echo "$(date '+%F %T') grid start" >> "$LOG"
bash "$R/unas/run_mind_search.sh" grid >> "$LOG" 2>&1
echo "$(date '+%F %T') QUEUE DONE" >> "$LOG"
