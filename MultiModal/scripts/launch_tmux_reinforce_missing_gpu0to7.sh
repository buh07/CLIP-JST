#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
cd "$ROOT"
mkdir -p "$MM_ROOT/logs/reinforce_completion_suite" "$MM_ROOT/results/reinforce_completion_suite/markers"

# Clear only this campaign's marker files.
find "$MM_ROOT/results/reinforce_completion_suite/markers" -maxdepth 1 -type f   \( -name 'w2_*' -o -name 'w9_*' -o -name 'w11_*' -o -name 'w3_*' -o -name 'w7_*' -o -name 'reinforce_missing_gpu*.all_done' \)   -delete || true

for n in mm_reinforce_missing_g0 mm_reinforce_missing_g1 mm_reinforce_missing_g2 mm_reinforce_missing_g3 mm_reinforce_missing_g4 mm_reinforce_missing_g5 mm_reinforce_missing_g6 mm_reinforce_missing_g7; do
  tmux has-session -t "$n" 2>/dev/null && tmux kill-session -t "$n" || true
done

for g in 0 1 2 3 4 5 6 7; do
  tmux new-session -d -s "mm_reinforce_missing_g${g}" "bash '$MM_ROOT/scripts/queue_reinforce_missing_gpu${g}.sh'"
done

echo "launched: mm_reinforce_missing_g0..g7"
