#!/usr/bin/env bash
set -euo pipefail
ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
SESS=(mm_w3ext_g0 mm_w3ext_g1 mm_w3ext_g2 mm_w3ext_g3 mm_w3ext_g4 mm_w3ext_g5 mm_w3ext_g6 mm_w3ext_g7 mm_w3ext_agg_g7)
for s in "${SESS[@]}"; do
  tmux has-session -t "$s" 2>/dev/null && tmux kill-session -t "$s" || true
done

tmux new-session -d -s mm_w3ext_g0 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_w3_ext_gpu0.sh'"
tmux new-session -d -s mm_w3ext_g1 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_w3_ext_gpu1.sh'"
tmux new-session -d -s mm_w3ext_g2 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_w3_ext_gpu2.sh'"
tmux new-session -d -s mm_w3ext_g3 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_w3_ext_gpu3.sh'"
tmux new-session -d -s mm_w3ext_g4 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_w3_ext_gpu4.sh'"
tmux new-session -d -s mm_w3ext_g5 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_w3_ext_gpu5.sh'"
tmux new-session -d -s mm_w3ext_g6 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_w3_ext_gpu6.sh'"
tmux new-session -d -s mm_w3ext_g7 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_w3_ext_gpu7.sh'"
tmux new-session -d -s mm_w3ext_agg_g7 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_w3_ext_aggregate_gpu7.sh'"

echo "Started W3 extension sessions (g0-g7 + aggregate)."
echo "Logs: $MM_ROOT/logs/next_run_suite/w3_holdout_ext"
