#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
cd "$ROOT"

# avoid clobbering unrelated sessions
for g in 4 5 6 7; do
  s="mm_sdpi_g${g}"
  tmux kill-session -t "$s" 2>/dev/null || true
done

tmux new-session -d -s mm_sdpi_g4 "bash MultiModal/scripts/queue_sdpi_gpu4.sh"
tmux new-session -d -s mm_sdpi_g5 "bash MultiModal/scripts/queue_sdpi_gpu5.sh"
tmux new-session -d -s mm_sdpi_g6 "bash MultiModal/scripts/queue_sdpi_gpu6.sh"
tmux new-session -d -s mm_sdpi_g7 "bash MultiModal/scripts/queue_sdpi_gpu7.sh"

echo "Launched: mm_sdpi_g4 mm_sdpi_g5 mm_sdpi_g6 mm_sdpi_g7"
