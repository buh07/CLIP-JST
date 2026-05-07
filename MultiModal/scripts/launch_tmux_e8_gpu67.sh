#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
SESSION6="mm_e8_gpu6"
SESSION7="mm_e8_gpu7"

if tmux has-session -t "$SESSION6" 2>/dev/null; then
  tmux kill-session -t "$SESSION6"
fi
if tmux has-session -t "$SESSION7" 2>/dev/null; then
  tmux kill-session -t "$SESSION7"
fi

tmux new-session -d -s "$SESSION6" "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_e8_gpu6.sh'"
tmux new-session -d -s "$SESSION7" "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_e8_gpu7.sh'"

echo "Started tmux sessions: $SESSION6, $SESSION7"
echo "Attach with: tmux attach -t $SESSION6   or   tmux attach -t $SESSION7"
