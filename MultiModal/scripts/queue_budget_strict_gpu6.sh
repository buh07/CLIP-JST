#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
LOG_DIR="$MM_ROOT/logs/federated_budget_strict"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=6
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2'
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

python -m MultiModal.multimodal.experiments.run_stage13_federated_comparison \
  --config MultiModal/configs/stage13_budget_strict_gpu6.yaml 2>&1 | tee "$LOG_DIR/stage13_gpu6.log"

M="$MM_ROOT/results/federated_budget_strict/markers/stage13_federated.done.json"
while [[ ! -f "$M" ]]; do
  echo "[$(date '+%F %T')] gpu6 waiting for merged stage13 marker..."
  sleep 60
done

python -m MultiModal.multimodal.experiments.run_stage16_budget_matched_frontier \
  --config MultiModal/configs/stage16_budget_strict_gpu6.yaml 2>&1 | tee "$LOG_DIR/stage16_gpu6.log"

echo "[$(date '+%F %T')] strict gpu6 queue complete."
