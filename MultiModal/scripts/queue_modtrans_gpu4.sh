#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
LOG_DIR="$MM_ROOT/logs/modular_transitivity_suite"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=4
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2'
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

python -m MultiModal.multimodal.experiments.run_stage18_spectral_transitivity_prereqs \
  --config MultiModal/configs/stage18_modtrans_gpu4.yaml 2>&1 | tee "$LOG_DIR/stage18_gpu4.log"

python -m MultiModal.multimodal.experiments.run_stage19_pseudomodality_transitivity \
  --config MultiModal/configs/stage19_modtrans_gpu4.yaml 2>&1 | tee "$LOG_DIR/stage19_gpu4.log"

echo "[$(date '+%F %T')] queue_modtrans_gpu4 complete"
