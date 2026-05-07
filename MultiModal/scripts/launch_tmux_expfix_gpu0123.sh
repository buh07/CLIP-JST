#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"

S0="mm_expfix_g0"
S1="mm_expfix_g1"
S2="mm_expfix_g2"
S3="mm_expfix_g3"

for s in "$S0" "$S1" "$S2" "$S3"; do
  tmux has-session -t "$s" 2>/dev/null && tmux kill-session -t "$s" || true
done

tmux new-session -d -s "$S0" "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_expfix_gpu0.sh'"
tmux new-session -d -s "$S1" "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_expfix_gpu1.sh'"
tmux new-session -d -s "$S2" "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_expfix_gpu2.sh'"
tmux new-session -d -s "$S3" "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_expfix_gpu3.sh'"

echo "Started sessions: $S0 $S1 $S2 $S3"
echo "Logs: $MM_ROOT/logs/experimental_fixes_suite"
