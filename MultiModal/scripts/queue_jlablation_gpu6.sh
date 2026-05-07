#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
LOG_DIR="$MM_ROOT/logs/modular_transitivity_jl_ablation"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=6
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2'
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

python -m MultiModal.multimodal.experiments.run_stage25_modtrans_jl_ablation \
  --config MultiModal/configs/stage25_jlablation_gpu6.yaml 2>&1 | tee "$LOG_DIR/stage25_gpu6.log"

echo "[$(date '+%F %T')] queue_jlablation_gpu6 complete"
