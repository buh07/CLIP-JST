#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"

SESSIONS=(
  mm_semjl_g4
  mm_semjl_g5
  mm_semjl_g6
  mm_semjl_g7
)

for s in "${SESSIONS[@]}"; do
  if tmux has-session -t "$s" 2>/dev/null; then
    tmux kill-session -t "$s"
  fi
done

tmux new-session -d -s mm_semjl_g4 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_semantic_jlablation_gpu4.sh'"
tmux new-session -d -s mm_semjl_g5 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_semantic_jlablation_gpu5.sh'"
tmux new-session -d -s mm_semjl_g6 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_semantic_jlablation_gpu6.sh'"
tmux new-session -d -s mm_semjl_g7 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_semantic_jlablation_gpu7.sh'"

echo "Started sessions: ${SESSIONS[*]}"
echo "Logs: $MM_ROOT/logs/semantic_jlablation"
echo ""
echo "Monitor with:"
echo "  tmux attach -t mm_semjl_g4   # m=64  (Stage25/26 gpu4 shard)"
echo "  tmux attach -t mm_semjl_g5   # m=128 (Stage25/26 gpu5 shard)"
echo "  tmux attach -t mm_semjl_g6   # m=256 (Stage25/26 gpu6 shard)"
echo "  tmux attach -t mm_semjl_g7   # m=512 + Stage35 full aggregate + Stage36"
