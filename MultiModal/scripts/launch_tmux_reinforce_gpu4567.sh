#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
cd "$ROOT"

mkdir -p "$MM_ROOT/logs/reinforce_suite" "$MM_ROOT/results/reinforce_suite/markers"

for n in mm_reinforce_g4 mm_reinforce_g5 mm_reinforce_g6 mm_reinforce_g7; do
  tmux has-session -t "$n" 2>/dev/null && tmux kill-session -t "$n" || true
done

tmux new-session -d -s mm_reinforce_g4 "bash '$MM_ROOT/scripts/queue_reinforce_gpu4.sh'"
tmux new-session -d -s mm_reinforce_g5 "bash '$MM_ROOT/scripts/queue_reinforce_gpu5.sh'"
tmux new-session -d -s mm_reinforce_g6 "bash '$MM_ROOT/scripts/queue_reinforce_gpu6.sh'"
tmux new-session -d -s mm_reinforce_g7 "bash '$MM_ROOT/scripts/queue_reinforce_gpu7.sh'"

echo "launched: mm_reinforce_g4/g5/g6/g7"
