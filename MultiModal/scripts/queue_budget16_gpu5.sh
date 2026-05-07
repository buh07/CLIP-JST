#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
LOG_DIR="$MM_ROOT/logs/federated_budget_matched"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=5
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2'
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

python -m MultiModal.multimodal.experiments.run_stage16_budget_matched_frontier \
  --config MultiModal/configs/stage16_budget_matched_gpu5.yaml 2>&1 | tee "$LOG_DIR/stage16_gpu5.log"

echo "[$(date '+%F %T')] stage16 gpu5 shard complete."
