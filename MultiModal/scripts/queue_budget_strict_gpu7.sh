#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
LOG_DIR="$MM_ROOT/logs/federated_budget_strict"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=7
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2'
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

# Stage13 shard on GPU7
python -m MultiModal.multimodal.experiments.run_stage13_federated_comparison \
  --config MultiModal/configs/stage13_budget_strict_gpu7.yaml 2>&1 | tee "$LOG_DIR/stage13_gpu7.log"

M4="$MM_ROOT/results/federated_budget_strict_stage13_split/gpu4/markers/stage13_federated.done.json"
M5="$MM_ROOT/results/federated_budget_strict_stage13_split/gpu5/markers/stage13_federated.done.json"
M6="$MM_ROOT/results/federated_budget_strict_stage13_split/gpu6/markers/stage13_federated.done.json"
M7="$MM_ROOT/results/federated_budget_strict_stage13_split/gpu7/markers/stage13_federated.done.json"
while [[ ! -f "$M4" || ! -f "$M5" || ! -f "$M6" || ! -f "$M7" ]]; do
  echo "[$(date '+%F %T')] waiting for all strict stage13 shard markers..."
  sleep 60
done

python -m MultiModal.multimodal.experiments.run_stage13_merge \
  --config MultiModal/configs/stage13_budget_strict_merge.yaml 2>&1 | tee "$LOG_DIR/stage13_merge.log"

# Stage16 shard on GPU7
python -m MultiModal.multimodal.experiments.run_stage16_budget_matched_frontier \
  --config MultiModal/configs/stage16_budget_strict_gpu7.yaml 2>&1 | tee "$LOG_DIR/stage16_gpu7.log"

S16_4="$MM_ROOT/results/federated_budget_strict_stage16_split/gpu4/markers/stage16_budget_matched.done.json"
S16_5="$MM_ROOT/results/federated_budget_strict_stage16_split/gpu5/markers/stage16_budget_matched.done.json"
S16_6="$MM_ROOT/results/federated_budget_strict_stage16_split/gpu6/markers/stage16_budget_matched.done.json"
S16_7="$MM_ROOT/results/federated_budget_strict_stage16_split/gpu7/markers/stage16_budget_matched.done.json"
while [[ ! -f "$S16_4" || ! -f "$S16_5" || ! -f "$S16_6" || ! -f "$S16_7" ]]; do
  echo "[$(date '+%F %T')] waiting for all strict stage16 shard markers..."
  sleep 60
done

python -m MultiModal.multimodal.experiments.run_stage16_merge \
  --config MultiModal/configs/stage16_budget_strict_merge.yaml 2>&1 | tee "$LOG_DIR/stage16_merge.log"

python -m MultiModal.multimodal.experiments.run_stage17_budget_matched_aggregate \
  --config MultiModal/configs/stage17_budget_strict_aggregate.yaml 2>&1 | tee "$LOG_DIR/stage17_aggregate.log"

echo "[$(date '+%F %T')] strict budget pipeline complete."
