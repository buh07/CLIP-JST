#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
cd "$ROOT"
mkdir -p logs

export CUDA_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1

run_exp () {
  local name="$1"
  local cmd="$2"
  local log="logs/${name}.log"
  echo "[$(date '+%F %T')] START $name"
  echo "[$(date '+%F %T')] CMD   $cmd"
  eval "$cmd" 2>&1 | tee "$log"
  echo "[$(date '+%F %T')] DONE  $name"
}

run_exp "E2_raw_rigorous_gpu3" "python experiments/run_E2.py --config configs/E2.yaml"
run_exp "E5_raw_rigorous_gpu3" "python experiments/run_E5.py --config configs/E5.yaml"
run_exp "D3_raw_rigorous_gpu3" "python experiments/run_D3.py --config configs/D3.yaml"
run_exp "controls_raw_rigorous_gpu3" "python experiments/run_controls.py --config configs/controls.yaml"
run_exp "E7_jlt_loop_pilot_gpu3" "python experiments/run_E7.py --config configs/E7.yaml"

echo "[$(date '+%F %T')] GPU3 suite completed."
