#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
OUT_ROOT="$MM_ROOT/results/concern_suite"
LOG_DIR="$MM_ROOT/logs/concern_suite"
mkdir -p "$LOG_DIR" "$OUT_ROOT/markers"

export CUDA_VISIBLE_DEVICES=5
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2'
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

log="$LOG_DIR/stage10_privacy_parity_gpu5.log"
echo "[$(date '+%F %T')] START stage10_privacy_parity"
python -m MultiModal.multimodal.experiments.run_stage10_privacy_parity \
  --config MultiModal/configs/stage10_privacy_parity.yaml 2>&1 | tee "$log"
echo "[$(date '+%F %T')] DONE stage10_privacy_parity"
