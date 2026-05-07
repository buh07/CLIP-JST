#!/usr/bin/env bash
set -euo pipefail
ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
OUT="$MM_ROOT/results/next_run_suite/w3_holdout_ext"
LOG_DIR="$MM_ROOT/logs/next_run_suite/w3_holdout_ext"
MARK="$OUT/markers"
mkdir -p "$LOG_DIR" "$MARK"

export CUDA_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2' 2>/dev/null || true
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

python -m MultiModal.multimodal.experiments.run_stage55_wavcaps_holdout_retrain   --config MultiModal/configs/stage55_w3_ext_gpu3.yaml 2>&1 | tee "$LOG_DIR/stage55_w3_ext_gpu3.log"

touch "$MARK/gpu3.stage55.done"
echo "[$(date '+%F %T')] queue_w3_ext_gpu3 complete"
