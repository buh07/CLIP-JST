#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
OUT_ROOT="$MM_ROOT/results/concern_suite"
LOG_DIR="$MM_ROOT/logs/concern_suite"
MARKERS="$OUT_ROOT/markers"
mkdir -p "$LOG_DIR" "$MARKERS"

export CUDA_VISIBLE_DEVICES=4
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2' opacus
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

run_step () {
  local name="$1"
  local cmd="$2"
  local log="$LOG_DIR/${name}.log"
  echo "[$(date '+%F %T')] START $name"
  echo "[$(date '+%F %T')] CMD   $cmd"
  eval "$cmd" 2>&1 | tee "$log"
  echo "[$(date '+%F %T')] DONE  $name"
}

run_step "stage8_e8d_dpsgd_ckpt_gpu4" \
  "python -m MultiModal.multimodal.experiments.run_stage8_e8d_dpsgd --config MultiModal/configs/stage8_e8d_dpsgd_concern_ckpt.yaml"

echo "[$(date '+%F %T')] concern stage8 queue complete."
