#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"

SESSION2="mm_gpu2"
SESSION3="mm_gpu3"

if tmux has-session -t "$SESSION2" 2>/dev/null; then
  tmux kill-session -t "$SESSION2"
fi
if tmux has-session -t "$SESSION3" 2>/dev/null; then
  tmux kill-session -t "$SESSION3"
fi

tmux new-session -d -s "$SESSION2" "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_gpu2.sh'"
tmux new-session -d -s "$SESSION3" "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_gpu3.sh'"

echo "Started tmux sessions: $SESSION2, $SESSION3"
echo "Attach with: tmux attach -t $SESSION2  (or $SESSION3)"
