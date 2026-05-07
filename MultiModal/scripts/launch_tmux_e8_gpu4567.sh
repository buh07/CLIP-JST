#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
S4="mm_e8_gpu4"
S5="mm_e8_gpu5"
S6="mm_e8_gpu6"
S7="mm_e8_gpu7"

for s in "$S4" "$S5" "$S6" "$S7"; do
  if tmux has-session -t "$s" 2>/dev/null; then
    tmux kill-session -t "$s"
  fi
done

# Kill prior 2-GPU launcher sessions if present.
for s in "mm_e8_gpu6" "mm_e8_gpu7"; do
  if tmux has-session -t "$s" 2>/dev/null; then
    tmux kill-session -t "$s"
  fi
done

tmux new-session -d -s "$S4" "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_e8_gpu4.sh'"
tmux new-session -d -s "$S5" "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_e8_gpu5.sh'"
tmux new-session -d -s "$S6" "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_e8_gpu6_split.sh'"
tmux new-session -d -s "$S7" "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_e8_gpu7_split.sh'"

echo "Started tmux sessions: $S4 $S5 $S6 $S7"
echo "Attach with: tmux attach -t $S6 (or $S4/$S5/$S7)"
