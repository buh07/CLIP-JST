#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
OUT="$MM_ROOT/results/experimental_fixes_suite"
LOG_DIR="$MM_ROOT/logs/experimental_fixes_suite"
MARK="$OUT/markers/shards"
mkdir -p "$LOG_DIR" "$MARK"

export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2' torchcodec
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

python -m MultiModal.multimodal.experiments.run_stage44_zero_shot_baseline_control \
  --config MultiModal/configs/stage44_expfix_gpu1.yaml 2>&1 | tee "$LOG_DIR/stage44_gpu1.log"
touch "$MARK/gpu1.stage44.done"

while [[ ! -f "$MARK/wavcaps_mixed46k.ready" || ! -f "$MARK/gpu3.stage44.done" ]]; do
  sleep 20
done

python -m MultiModal.multimodal.experiments.run_stage45_quality_quantity_deconfound \
  --config MultiModal/configs/stage45_expfix_mixed46k_gpu1.yaml 2>&1 | tee "$LOG_DIR/stage45_mixed46k_gpu1.log"
python -m MultiModal.multimodal.experiments.run_stage45_quality_quantity_deconfound \
  --config MultiModal/configs/stage45_expfix_mixed200k_gpu1.yaml 2>&1 | tee "$LOG_DIR/stage45_mixed200k_gpu1.log"
python -m MultiModal.multimodal.experiments.run_stage45_quality_quantity_deconfound \
  --config MultiModal/configs/stage45_expfix_clean_source_gpu1.yaml 2>&1 | tee "$LOG_DIR/stage45_clean_source_gpu1.log"

touch "$MARK/gpu1.stage45.done"
echo "[$(date '+%F %T')] queue_expfix_gpu1 complete"
