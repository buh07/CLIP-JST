#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"

S4="mm_concern_g4_stage8"
S5="mm_concern_g5_stage10"
S6="mm_concern_g6_stage11"
S7="mm_concern_g7_stage12"

for s in "$S4" "$S5" "$S6" "$S7"; do
  if tmux has-session -t "$s" 2>/dev/null; then
    tmux kill-session -t "$s"
  fi
done

tmux new-session -d -s "$S4" "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_concern_gpu4_stage8.sh'"
tmux new-session -d -s "$S5" "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_concern_gpu5_stage10.sh'"
tmux new-session -d -s "$S6" "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_concern_gpu6_stage11.sh'"
tmux new-session -d -s "$S7" "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_concern_gpu7_stage12.sh'"

echo "Started tmux sessions: $S4 $S5 $S6 $S7"
echo "Logs: $MM_ROOT/logs/concern_suite"
