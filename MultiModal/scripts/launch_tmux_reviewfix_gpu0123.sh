#!/usr/bin/env bash
set -euo pipefail
ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"

for s in mm_reviewfix_g0 mm_reviewfix_g1 mm_reviewfix_g2 mm_reviewfix_g3; do
  tmux kill-session -t "$s" 2>/dev/null || true
done

tmux new-session -d -s mm_reviewfix_g0 "bash '$MM_ROOT/scripts/queue_reviewfix_gpu0.sh'"
tmux new-session -d -s mm_reviewfix_g1 "bash '$MM_ROOT/scripts/queue_reviewfix_gpu1.sh'"
tmux new-session -d -s mm_reviewfix_g2 "bash '$MM_ROOT/scripts/queue_reviewfix_gpu2.sh'"
tmux new-session -d -s mm_reviewfix_g3 "bash '$MM_ROOT/scripts/queue_reviewfix_gpu3.sh'"

echo "launched: mm_reviewfix_g0..g3"
tmux ls | rg 'mm_reviewfix_g'
