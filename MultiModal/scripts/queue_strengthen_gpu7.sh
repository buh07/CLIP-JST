#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
OUT="$MM_ROOT/results/neurips_strengthen_suite"
LOG_DIR="$MM_ROOT/logs/neurips_strengthen_suite"
MARK="$OUT/markers/shards"
mkdir -p "$LOG_DIR" "$MARK"

export CUDA_VISIBLE_DEVICES=7
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2' torchcodec
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

python -m MultiModal.multimodal.experiments.run_stage39_modality_gap_linear_vs_jl \
  --config MultiModal/configs/stage39_modality_gap_gpu7.yaml 2>&1 | tee "$LOG_DIR/stage39_gpu7.log"
touch "$MARK/gpu7.stage39.done"

until [[ -f "$MARK/gpu4.stage37.done" && -f "$MARK/gpu5.stage38.done" && -f "$MARK/gpu4.stage39.done" && -f "$MARK/gpu5.stage39.done" && -f "$MARK/gpu6.stage39.done" ]]; do
  echo "[$(date '+%F %T')] waiting for Stage37/38/39 shards..."
  sleep 60
done

python -m MultiModal.multimodal.experiments.run_stage40_strengthen_suite_aggregate \
  --config MultiModal/configs/stage40_strengthen_suite_aggregate.yaml 2>&1 | tee "$LOG_DIR/stage40_gpu7.log"
touch "$MARK/gpu7.stage40.done"

echo "[$(date '+%F %T')] queue_strengthen_gpu7 complete"
