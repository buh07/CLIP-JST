#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"

S5H="mm_strength_help5"
S6H="mm_strength_help6"
S7H="mm_strength_help7"

for s in "$S5H" "$S6H" "$S7H"; do
  if tmux has-session -t "$s" 2>/dev/null; then
    tmux kill-session -t "$s"
  fi
done

tmux new-session -d -s "$S5H" "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_strength_helper_gpu5_for_gpu4.sh'"
tmux new-session -d -s "$S6H" "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_strength_helper_gpu6_for_gpu4.sh'"
tmux new-session -d -s "$S7H" "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_strength_helper_gpu7_for_gpu4.sh'"

echo "Started helper sessions: $S5H $S6H $S7H"
