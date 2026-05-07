#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
LOG_DIR="$MM_ROOT/logs/reviewer_fixes"
MARK="$MM_ROOT/results/reviewer_fixes_suite/markers"
mkdir -p "$LOG_DIR" "$MARK"

export CUDA_VISIBLE_DEVICES=2
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2' 2>/dev/null || true
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

# W5: Gap_ia → alpha regression (pure computation, no GPU required)
# Already ran successfully; re-run for completeness / to confirm reproducibility.
python -m MultiModal.multimodal.experiments.run_w5_gap_alpha_regression 2>&1 | tee "$LOG_DIR/w5_gap_alpha_regression.log"

touch "$MARK/gpu2.w5.done"
echo "[$(date '+%F %T')] queue_reviewer_fixes_gpu2 complete"
