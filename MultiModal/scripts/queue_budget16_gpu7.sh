#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
LOG_DIR="$MM_ROOT/logs/federated_budget_matched"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=7
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2'
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

python -m MultiModal.multimodal.experiments.run_stage16_budget_matched_frontier \
  --config MultiModal/configs/stage16_budget_matched_gpu7.yaml 2>&1 | tee "$LOG_DIR/stage16_gpu7.log"

M4="$MM_ROOT/results/federated_budget_matched_split/gpu4/markers/stage16_budget_matched.done.json"
M5="$MM_ROOT/results/federated_budget_matched_split/gpu5/markers/stage16_budget_matched.done.json"
M6="$MM_ROOT/results/federated_budget_matched_split/gpu6/markers/stage16_budget_matched.done.json"
M7="$MM_ROOT/results/federated_budget_matched_split/gpu7/markers/stage16_budget_matched.done.json"

while [[ ! -f "$M4" || ! -f "$M5" || ! -f "$M6" || ! -f "$M7" ]]; do
  echo "[$(date '+%F %T')] waiting for all stage16 shard markers..."
  sleep 60
done

python -m MultiModal.multimodal.experiments.run_stage16_merge \
  --config MultiModal/configs/stage16_budget_matched_merge.yaml 2>&1 | tee "$LOG_DIR/stage16_merge.log"

python -m MultiModal.multimodal.experiments.run_stage17_budget_matched_aggregate \
  --config MultiModal/configs/stage17_budget_matched_aggregate.yaml 2>&1 | tee "$LOG_DIR/stage17_aggregate.log"

echo "[$(date '+%F %T')] stage16/17 budget-matched pipeline complete."
