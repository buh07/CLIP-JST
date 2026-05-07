#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
LOG_DIR="$MM_ROOT/logs/reviewer_fixes"
MARK="$MM_ROOT/results/reviewer_fixes_suite/markers"
mkdir -p "$LOG_DIR" "$MARK"

export CUDA_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2' 2>/dev/null || true
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

# GPU 3: W5 (fast, no GPU) + W8 smoke validation (subset dims/seeds, separate output file)
python -m MultiModal.multimodal.experiments.run_w5_gap_alpha_regression 2>&1 | tee "$LOG_DIR/w5_gap_alpha_regression_gpu3.log"

python -m MultiModal.multimodal.experiments.run_w8_1k_split_eval \
  --config MultiModal/configs/w8_smoke_gpu3.yaml 2>&1 | tee "$LOG_DIR/w8_1k_split_eval_smoke_gpu3.log"

touch "$MARK/gpu3.w8_validate.done"
echo "[$(date '+%F %T')] queue_reviewer_fixes_gpu3 complete"
