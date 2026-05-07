#!/usr/bin/env bash
set -euo pipefail
ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
SUITE_OUT="$MM_ROOT/results/theory_backing_suite"
LOG_DIR="$MM_ROOT/logs/theory_backing_suite"
MARK="$SUITE_OUT/markers"
mkdir -p "$LOG_DIR" "$MARK"

export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2' 2>/dev/null || true
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"


python -m MultiModal.multimodal.experiments.run_stage65_isotonic_gap_calibration   --config MultiModal/configs/stage65_isotonic_gap_calibration.yaml 2>&1 | tee "$LOG_DIR/stage65_gpu1.log"
touch "$MARK/stage65.done"

python -m MultiModal.multimodal.experiments.run_stage66_gap_intervention_pilot   --config MultiModal/configs/stage66_gapreg_gpu1.yaml 2>&1 | tee "$LOG_DIR/stage66_gpu1.log"
touch "$MARK/gpu1.stage66.done"
echo "[$(date '+%F %T')] queue_theory_missing_gpu1 complete"
