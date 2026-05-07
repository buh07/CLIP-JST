#!/usr/bin/env bash
set -euo pipefail
ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
OUT="$MM_ROOT/results/stage69_prereg_suite"
LOG_DIR="$MM_ROOT/logs/stage69_prereg_suite"
MARK="$OUT/markers"
mkdir -p "$LOG_DIR" "$MARK"

SESS=(
  mm_stage69_prereg_g0
  mm_stage69_cache_g0
  mm_stage69_g0 mm_stage69_g1 mm_stage69_g2 mm_stage69_g3
  mm_stage69_g4 mm_stage69_g5 mm_stage69_g6 mm_stage69_g7
  mm_stage70to73_g7
)
for s in "${SESS[@]}"; do
  tmux has-session -t "$s" 2>/dev/null && tmux kill-session -t "$s" || true
done

tmux new-session -d -s mm_stage69_prereg_g0 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_stage69_prereg_lock_gpu0.sh'"
tmux new-session -d -s mm_stage69_cache_g0 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_stage69_cache_prebuild_gpu0.sh'"

for g in 0 1 2 3 4 5 6 7; do
  tmux new-session -d -s "mm_stage69_g${g}" "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_stage69_gpu${g}.sh'"
done

tmux new-session -d -s mm_stage70to73_g7 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_stage70to73_gpu7.sh'"

echo "Started stage69-73 campaign sessions."
echo "Results root: $OUT"
echo "Logs: $LOG_DIR"
