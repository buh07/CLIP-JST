#!/usr/bin/env bash
set -euo pipefail
ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
LOG_DIR="$MM_ROOT/logs/concern_suite"
mkdir -p "$LOG_DIR"
export CUDA_VISIBLE_DEVICES=5
export PYTHONUNBUFFERED=1
bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2'
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"
python -m MultiModal.multimodal.experiments.run_stage12_ood_concat --config MultiModal/configs/stage12_ood_concat_cc3m_mask_gpu5.yaml 2>&1 | tee "$LOG_DIR/stage12_cc3m_mask_gpu5.log"
