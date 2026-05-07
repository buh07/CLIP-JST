#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
LOG_DIR="$MM_ROOT/logs/reviewer_fixes"
MARK="$MM_ROOT/results/reviewer_fixes_suite/markers"
mkdir -p "$LOG_DIR" "$MARK"

export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2' 2>/dev/null || true
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

# W8: 1K-split evaluation on Stage 30 and Stage 44 checkpoints
# Compares full 4411-item pool vs 883-item 1K split for our trained methods
python -m MultiModal.multimodal.experiments.run_w8_1k_split_eval 2>&1 | tee "$LOG_DIR/w8_1k_split_eval.log"

touch "$MARK/gpu1.w8.done"
echo "[$(date '+%F %T')] queue_reviewer_fixes_gpu1 complete"
