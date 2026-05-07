#!/usr/bin/env bash
set -euo pipefail
ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"

SESS=(mm_theorymiss_g0 mm_theorymiss_g1 mm_theorymiss_g2 mm_theorymiss_g3 mm_theorymiss_g4 mm_theorymiss_g5 mm_theorymiss_g6 mm_theorymiss_g7 mm_theorymiss_coord)
for s in "${SESS[@]}"; do
  tmux has-session -t "$s" 2>/dev/null && tmux kill-session -t "$s" || true
done

tmux new-session -d -s mm_theorymiss_g0 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_theory_missing_gpu0.sh'"
tmux new-session -d -s mm_theorymiss_g1 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_theory_missing_gpu1.sh'"
tmux new-session -d -s mm_theorymiss_g2 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_theory_missing_gpu2.sh'"
tmux new-session -d -s mm_theorymiss_g3 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_theory_missing_gpu3.sh'"
tmux new-session -d -s mm_theorymiss_g4 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_theory_missing_gpu4.sh'"
tmux new-session -d -s mm_theorymiss_g5 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_theory_missing_gpu5.sh'"
tmux new-session -d -s mm_theorymiss_g6 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_theory_missing_gpu6.sh'"
tmux new-session -d -s mm_theorymiss_g7 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_theory_missing_gpu7.sh'"
tmux new-session -d -s mm_theorymiss_coord "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_theory_missing_coordinator.sh'"

echo "Started theory-missing suite on tmux sessions: ${SESS[*]}"
echo "Logs: $MM_ROOT/logs/theory_backing_suite"
