#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
LOG_DIR="$MM_ROOT/logs/modular_transitivity_jl_ablation"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=7
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2'
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

python -m MultiModal.multimodal.experiments.run_stage25_modtrans_jl_ablation \
  --config MultiModal/configs/stage25_jlablation_gpu7.yaml 2>&1 | tee "$LOG_DIR/stage25_gpu7.log"

S25_4="$MM_ROOT/results/modular_transitivity_jl_ablation/split/gpu4/markers/stage25_modtrans_jl_ablation.done.json"
S25_5="$MM_ROOT/results/modular_transitivity_jl_ablation/split/gpu5/markers/stage25_modtrans_jl_ablation.done.json"
S25_6="$MM_ROOT/results/modular_transitivity_jl_ablation/split/gpu6/markers/stage25_modtrans_jl_ablation.done.json"
S25_7="$MM_ROOT/results/modular_transitivity_jl_ablation/split/gpu7/markers/stage25_modtrans_jl_ablation.done.json"

while [[ ! -f "$S25_4" || ! -f "$S25_5" || ! -f "$S25_6" || ! -f "$S25_7" ]]; do
  echo "[$(date '+%F %T')] gpu7 waiting for stage25 shard markers..."
  sleep 120
done

python -m MultiModal.multimodal.experiments.run_stage26_modtrans_jl_aggregate \
  --config MultiModal/configs/stage26_jlablation_aggregate.yaml 2>&1 | tee "$LOG_DIR/stage26_aggregate.log"

echo "[$(date '+%F %T')] queue_jlablation_gpu7 complete"
