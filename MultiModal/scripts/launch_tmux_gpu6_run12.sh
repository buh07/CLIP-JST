#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
SESSION="mm_run12_gpu6"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
fi

tmux new-session -d -s "$SESSION" "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_gpu6_run12.sh'"

echo "Started tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
