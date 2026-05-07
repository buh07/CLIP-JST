#!/usr/bin/env bash
set -euo pipefail
ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"

for s in mm_cleanup_g0 mm_cleanup_g1 mm_cleanup_g2 mm_cleanup_g3; do
  tmux has-session -t "$s" 2>/dev/null && tmux kill-session -t "$s" || true
done

tmux new-session -d -s mm_cleanup_g0 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_cleanup_gpu0.sh'"
tmux new-session -d -s mm_cleanup_g1 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_cleanup_gpu1.sh'"
tmux new-session -d -s mm_cleanup_g2 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_cleanup_gpu2.sh'"
tmux new-session -d -s mm_cleanup_g3 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_cleanup_gpu3.sh'"

echo "Started sessions: mm_cleanup_g0 mm_cleanup_g1 mm_cleanup_g2 mm_cleanup_g3"
echo "Logs: $MM_ROOT/logs/neurips_cleanup"
