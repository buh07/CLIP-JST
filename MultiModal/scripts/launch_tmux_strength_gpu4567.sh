#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"

S4="mm_strength_g4"
S5="mm_strength_g5"
S6="mm_strength_g6"
S7="mm_strength_g7"

for s in "$S4" "$S5" "$S6" "$S7"; do
  if tmux has-session -t "$s" 2>/dev/null; then
    tmux kill-session -t "$s"
  fi
done

tmux new-session -d -s "$S4" "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_strength_gpu4.sh'"
tmux new-session -d -s "$S5" "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_strength_gpu5.sh'"
tmux new-session -d -s "$S6" "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_strength_gpu6.sh'"
tmux new-session -d -s "$S7" "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_strength_gpu7.sh'"

echo "Started sessions: $S4 $S5 $S6 $S7"
echo "Logs: $MM_ROOT/logs/strengthen_suite"
