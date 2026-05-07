#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"

for s in mm_vitl44_prebuild mm_vitl44_g0 mm_vitl44_g1 mm_vitl44_g2 mm_vitl44_g3 mm_vitl44_g4 mm_vitl44_g5 mm_vitl44_g6 mm_vitl44_g7 mm_vitl44_coord; do
  tmux has-session -t "$s" 2>/dev/null && tmux kill-session -t "$s" || true
done

# Prebuild first
tmux new-session -d -s mm_vitl44_prebuild "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_postw3_vitl14_prebuild_gpu0.sh'"

# Workers
tmux new-session -d -s mm_vitl44_g0 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_postw3_vitl14_rerun_gpu0.sh'"
tmux new-session -d -s mm_vitl44_g1 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_postw3_vitl14_rerun_gpu1.sh'"
tmux new-session -d -s mm_vitl44_g2 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_postw3_vitl14_rerun_gpu2.sh'"
tmux new-session -d -s mm_vitl44_g3 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_postw3_vitl14_rerun_gpu3.sh'"
tmux new-session -d -s mm_vitl44_g4 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_postw3_vitl14_rerun_gpu4.sh'"
tmux new-session -d -s mm_vitl44_g5 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_postw3_vitl14_rerun_gpu5.sh'"
tmux new-session -d -s mm_vitl44_g6 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_postw3_vitl14_rerun_gpu6.sh'"
tmux new-session -d -s mm_vitl44_g7 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_postw3_vitl14_rerun_gpu7.sh'"

# Coordinator
tmux new-session -d -s mm_vitl44_coord "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_postw3_vitl14_rerun_coordinator.sh'"

echo "Launched ViT-L stage44 rerun (prebuild + gpu0..7 + coordinator)."
