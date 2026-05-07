#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
SESSION6="mm_run12_gpu6_coco"
SESSION4="mm_run12_gpu4_flickr"

if tmux has-session -t "$SESSION6" 2>/dev/null; then
  tmux kill-session -t "$SESSION6"
fi
if tmux has-session -t "$SESSION4" 2>/dev/null; then
  tmux kill-session -t "$SESSION4"
fi

# Stop the old unsplit session if still running.
if tmux has-session -t "mm_run12_gpu6" 2>/dev/null; then
  tmux kill-session -t "mm_run12_gpu6"
fi

tmux new-session -d -s "$SESSION6" "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_gpu6_run12_coco.sh'"
tmux new-session -d -s "$SESSION4" "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_gpu4_run12_flickr.sh'"

echo "Started tmux sessions: $SESSION6, $SESSION4"
echo "Attach with: tmux attach -t $SESSION6  (or $SESSION4)"
