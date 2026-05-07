#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
cd "$ROOT"
mkdir -p logs

export CUDA_VISIBLE_DEVICES=2
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

run_exp "E1_raw_rigorous_gpu2" "python experiments/run_E1.py --config configs/E1.yaml"
run_exp "E6_raw_rigorous_gpu2" "python experiments/run_E6.py --config configs/E6.yaml"
run_exp "E4_raw_rigorous_gpu2" "python experiments/run_E4.py --config configs/E4.yaml"
run_exp "D1_raw_rigorous_gpu2" "python experiments/run_D1.py --config configs/D1.yaml"

echo "[$(date '+%F %T')] GPU2 suite completed."
