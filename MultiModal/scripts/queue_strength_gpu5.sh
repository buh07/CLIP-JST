#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
LOG_DIR="$MM_ROOT/logs/strengthen_suite"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=5
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2'
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

python -m MultiModal.multimodal.experiments.run_stage2_e7_karpathy \
  --config MultiModal/configs/stage2_e7_strength_gpu5_flickr.yaml 2>&1 | tee "$LOG_DIR/stage2_strength_gpu5.log"

python -m MultiModal.multimodal.experiments.run_stage13_federated_comparison \
  --config MultiModal/configs/stage13_fedfix_gpu5.yaml 2>&1 | tee "$LOG_DIR/stage13_fedfix_gpu5.log"

M13="$MM_ROOT/results/strengthen_suite/federated_fix/markers/stage13_federated.done.json"
while [[ ! -f "$M13" ]]; do
  echo "[$(date '+%F %T')] gpu5 waiting for merged stage13 marker..."
  sleep 60
done

python -m MultiModal.multimodal.experiments.run_stage14_stronger_attacks \
  --config MultiModal/configs/stage14_fedfix_gpu5.yaml 2>&1 | tee "$LOG_DIR/stage14_fedfix_gpu5.log"

echo "[$(date '+%F %T')] queue_strength_gpu5 complete"
