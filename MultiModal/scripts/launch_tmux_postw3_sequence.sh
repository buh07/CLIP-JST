#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
SESSION="mm_postw3_sequence"

tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION" || true

chmod +x \
  "$MM_ROOT/scripts/queue_postw3_sequence_controller.sh" \
  "$MM_ROOT/scripts/queue_postw3_stage39_joint_gpu0.sh" \
  "$MM_ROOT/scripts/queue_postw3_stage39_joint_gpu1.sh" \
  "$MM_ROOT/scripts/queue_postw3_stage39_joint_gpu2.sh" \
  "$MM_ROOT/scripts/queue_postw3_stage39_joint_gpu3.sh" \
  "$MM_ROOT/scripts/queue_postw3_stage39_expand_gpu0.sh" \
  "$MM_ROOT/scripts/queue_postw3_stage39_expand_gpu1.sh" \
  "$MM_ROOT/scripts/queue_postw3_stage39_expand_gpu2.sh" \
  "$MM_ROOT/scripts/queue_postw3_stage39_expand_gpu3.sh" \
  "$MM_ROOT/scripts/queue_postw3_vitl14_prebuild_gpu0.sh" \
  "$MM_ROOT/scripts/queue_postw3_vitl14_gpu1.sh" \
  "$MM_ROOT/scripts/queue_postw3_vitl14_gpu2.sh" \
  "$MM_ROOT/scripts/queue_postw3_vitl14_gpu3.sh" \
  "$MM_ROOT/scripts/queue_postw3_vitl14_gpu4.sh" \
  "$MM_ROOT/scripts/queue_postw3_vitl14_stage39_gpu4.sh" \
  "$MM_ROOT/scripts/queue_postw3_vitl14_stage39_gpu5.sh" \
  "$MM_ROOT/scripts/queue_postw3_vitl14_stage39_gpu6.sh" \
  "$MM_ROOT/scripts/queue_postw3_vitl14_stage39_gpu7.sh"

tmux new-session -d -s "$SESSION" "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_postw3_sequence_controller.sh'"

echo "Started session: $SESSION"
echo "Attach: tmux attach -t $SESSION"
echo "Logs: $MM_ROOT/logs/next_run_suite/post_w3_sequence/"
echo "Markers: $MM_ROOT/results/next_run_suite/post_w3_sequence/markers/"

