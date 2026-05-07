#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
LOG_DIR="$MM_ROOT/logs"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

wait_for_file () {
  local f="$1"
  echo "Waiting for $f"
  while [[ ! -f "$f" ]]; do
    sleep 60
  done
  echo "Detected $f"
}

run_step () {
  local name="$1"
  local cmd="$2"
  local log="$LOG_DIR/${name}.log"
  echo "[$(date '+%F %T')] START $name"
  echo "[$(date '+%F %T')] CMD   $cmd"
  eval "$cmd" 2>&1 | tee "$log"
  echo "[$(date '+%F %T')] DONE  $name"
}

wait_for_file "$MM_ROOT/results/full_suite/markers/stage1_prepare.done.json"
run_step "multimodal_stage3_trimodal_gpu3" "python -m MultiModal.multimodal.experiments.run_stage3_trimodal --config MultiModal/configs/stage3_trimodal.yaml"

wait_for_file "$MM_ROOT/results/full_suite/stage2_e7/markers/stage2_e7_karpathy.done.json"
run_step "multimodal_stage4_aggregate_gpu3" "python -m MultiModal.multimodal.experiments.run_stage4_aggregate --config MultiModal/configs/stage4_aggregate.yaml"

echo "[$(date '+%F %T')] GPU3 queue complete."
