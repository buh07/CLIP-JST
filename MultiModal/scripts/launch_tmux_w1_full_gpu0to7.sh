#!/usr/bin/env bash
set -euo pipefail
ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
OUT="$MM_ROOT/results/next_run_suite/w1_avcaps_full"
MARK="$OUT/markers"
mkdir -p "$MARK"

SESS=(mm_w1full_cache_g0 mm_w1full_head_g1 mm_w1full_head_g2 mm_w1full_head_g3 mm_w1full_head_g4 mm_w1full_head_g5 mm_w1full_head_g6 mm_w1full_head_g7 mm_w1full_lora_g1 mm_w1full_lora_g2 mm_w1full_lora_g3 mm_w1full_lora_g4 mm_w1full_lora_g5 mm_w1full_lora_g6 mm_w1full_lora_g7 mm_w1full_agg_g7)
for s in "${SESS[@]}"; do
  tmux has-session -t "$s" 2>/dev/null && tmux kill-session -t "$s" || true
done

tmux new-session -d -s mm_w1full_cache_g0 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_w1_full_cache_prebuild_gpu0.sh'"

echo "Waiting for AVCaps cache marker: $MARK/avcaps_cache_ready.done"
for _ in $(seq 1 720); do
  [[ -f "$MARK/avcaps_cache_ready.done" ]] && break
  sleep 30
done
if [[ ! -f "$MARK/avcaps_cache_ready.done" ]]; then
  echo "Timeout waiting for AVCaps cache prebuild marker."
  exit 1
fi

tmux new-session -d -s mm_w1full_head_g1 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_w1_full_gpu1.sh'"
tmux new-session -d -s mm_w1full_head_g2 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_w1_full_gpu2.sh'"
tmux new-session -d -s mm_w1full_head_g3 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_w1_full_gpu3.sh'"
tmux new-session -d -s mm_w1full_head_g4 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_w1_full_gpu4.sh'"
tmux new-session -d -s mm_w1full_head_g5 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_w1_full_gpu5.sh'"
tmux new-session -d -s mm_w1full_head_g6 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_w1_full_gpu6.sh'"
tmux new-session -d -s mm_w1full_head_g7 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_w1_full_gpu7.sh'"

tmux new-session -d -s mm_w1full_lora_g1 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_w1_lora_gpu1.sh'"
tmux new-session -d -s mm_w1full_lora_g2 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_w1_lora_gpu2.sh'"
tmux new-session -d -s mm_w1full_lora_g3 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_w1_lora_gpu3.sh'"
tmux new-session -d -s mm_w1full_lora_g4 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_w1_lora_gpu4.sh'"
tmux new-session -d -s mm_w1full_lora_g5 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_w1_lora_gpu5.sh'"
tmux new-session -d -s mm_w1full_lora_g6 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_w1_lora_gpu6.sh'"
tmux new-session -d -s mm_w1full_lora_g7 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_w1_lora_gpu7.sh'"

tmux new-session -d -s mm_w1full_agg_g7 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_w1_full_aggregate_gpu7.sh'"

echo "Started W1 full campaign sessions (cache+headline+lora+aggregate)."
echo "Logs: $MM_ROOT/logs/next_run_suite/w1_avcaps_full"
