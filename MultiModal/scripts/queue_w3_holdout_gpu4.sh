#!/usr/bin/env bash
set -euo pipefail
ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
OUT="$MM_ROOT/results/reinforce_suite/w3_wavcaps_holdout"
LOG_DIR="$MM_ROOT/logs/reinforce_suite/w3_wavcaps_holdout"
MARK="$OUT/markers"
mkdir -p "$LOG_DIR" "$MARK"

export CUDA_VISIBLE_DEVICES=4
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2' 2>/dev/null || true
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

python -m py_compile \
  MultiModal/multimodal/experiments/run_stage55_wavcaps_holdout_retrain.py \
  MultiModal/multimodal/experiments/run_stage56_wavcaps_holdout_aggregate.py

python -m MultiModal.multimodal.experiments.run_stage55_wavcaps_holdout_retrain \
  --config MultiModal/configs/stage55_w3_holdout_mixed200k_gpu4.yaml 2>&1 | tee "$LOG_DIR/stage55_mixed200k_gpu4.log"

touch "$MARK/gpu4.stage55.done"
echo "[$(date '+%F %T')] queue_w3_holdout_gpu4 complete"
