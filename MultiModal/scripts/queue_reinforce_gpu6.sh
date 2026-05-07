#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
LOG_DIR="$MM_ROOT/logs/reinforce_suite"
MARK="$MM_ROOT/results/reinforce_suite/markers"
mkdir -p "$LOG_DIR" "$MARK"

export CUDA_VISIBLE_DEVICES=6
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2' 2>/dev/null || true
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

python -m MultiModal.multimodal.experiments.run_w5_gap_alpha_regression 2>&1 | tee "$LOG_DIR/w5_gap_alpha_regression.gpu6.log"

cp -f "$MM_ROOT/results/reviewer_fixes_suite/w5_gap_alpha_regression/w5_gap_alpha_regression_results.json" \
      "$MM_ROOT/results/reinforce_suite/w5_gap_alpha_regression_results.gpu6.json"

touch "$MARK/gpu6.w5.done"
echo "[$(date '+%F %T')] queue_reinforce_gpu6 complete"
