#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
LOG_DIR="$MM_ROOT/logs/run12_gpu6"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=6
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
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

run_step "smoke_run12_gpu6" "python -m MultiModal.multimodal.experiments.run_smoke_tests --config MultiModal/configs/smoke_tests_run12_gpu6.yaml"
run_step "stage2_e7_run12_gpu6" "python -m MultiModal.multimodal.experiments.run_stage2_e7_karpathy --config MultiModal/configs/stage2_e7_run12_gpu6.yaml"

echo "[$(date '+%F %T')] GPU6 run12 queue complete."
