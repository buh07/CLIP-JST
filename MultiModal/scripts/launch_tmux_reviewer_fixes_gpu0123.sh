#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"

S0="rf_g0"
S1="rf_g1"
S2="rf_g2"
S3="rf_g3"

for s in "$S0" "$S1" "$S2" "$S3"; do
  tmux has-session -t "$s" 2>/dev/null && tmux kill-session -t "$s" || true
done

# GPU 0: W7 — ImageBind zero-shot baseline on 1K split + full pool
tmux new-session -d -s "$S0" "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_reviewer_fixes_gpu0.sh'"

# GPU 1: W8 — full 1K-split evaluation on Stage 30 + Stage 44 checkpoints (all dims, all seeds)
tmux new-session -d -s "$S1" "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_reviewer_fixes_gpu1.sh'"

# GPU 2: W5 (fast, no GPU) — gap_ia → alpha regression
tmux new-session -d -s "$S2" "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_reviewer_fixes_gpu2.sh'"

# GPU 3: W5 validation + W8 smoke test (subset dims/seeds, separate output file)
tmux new-session -d -s "$S3" "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_reviewer_fixes_gpu3.sh'"

echo "Started sessions: $S0 $S1 $S2 $S3"
echo "Logs: $MM_ROOT/logs/reviewer_fixes/"
echo ""
echo "Monitor with:"
echo "  tmux attach -t $S0   # W7 ImageBind"
echo "  tmux attach -t $S1   # W8 1K split eval"
echo "  tmux attach -t $S2   # W5 regression"
echo "  tmux attach -t $S3   # W5+W8 smoke"
echo ""
echo "Results go to: $MM_ROOT/results/reviewer_fixes_suite/"
