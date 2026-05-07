#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"

SESSIONS=(
  mm_semantic_g0
  mm_semantic_g1
  mm_semantic_g2
  mm_semantic_g3
  mm_semantic_g4
  mm_semantic_g5
  mm_semantic_g6
  mm_semantic_g7
)

for s in "${SESSIONS[@]}"; do
  if tmux has-session -t "$s" 2>/dev/null; then
    tmux kill-session -t "$s"
  fi
done

tmux new-session -d -s mm_semantic_g0 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_semantic_gpu0.sh'"
tmux new-session -d -s mm_semantic_g1 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_semantic_gpu1.sh'"
tmux new-session -d -s mm_semantic_g2 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_semantic_gpu2.sh'"
tmux new-session -d -s mm_semantic_g3 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_semantic_gpu3.sh'"
tmux new-session -d -s mm_semantic_g4 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_semantic_gpu4.sh'"
tmux new-session -d -s mm_semantic_g5 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_semantic_gpu5.sh'"
tmux new-session -d -s mm_semantic_g6 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_semantic_gpu6.sh'"
tmux new-session -d -s mm_semantic_g7 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_semantic_gpu7.sh'"

echo "Started sessions: ${SESSIONS[*]}"
echo "Logs: $MM_ROOT/logs/semantic_followup"
