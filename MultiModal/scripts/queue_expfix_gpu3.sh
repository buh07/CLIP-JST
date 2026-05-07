#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
OUT="$MM_ROOT/results/experimental_fixes_suite"
LOG_DIR="$MM_ROOT/logs/experimental_fixes_suite"
MARK="$OUT/markers/shards"
mkdir -p "$LOG_DIR" "$MARK"

export CUDA_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2' torchcodec
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

python -m MultiModal.multimodal.experiments.run_stage44_zero_shot_baseline_control \
  --config MultiModal/configs/stage44_expfix_gpu3.yaml 2>&1 | tee "$LOG_DIR/stage44_gpu3.log"
touch "$MARK/gpu3.stage44.done"

while [[ ! -f "$MARK/wavcaps_mixed46k.ready" ]]; do
  sleep 20
done

python -m MultiModal.multimodal.experiments.run_stage45_quality_quantity_deconfound \
  --config MultiModal/configs/stage45_expfix_mixed46k_gpu3.yaml 2>&1 | tee "$LOG_DIR/stage45_mixed46k_gpu3.log"
python -m MultiModal.multimodal.experiments.run_stage45_quality_quantity_deconfound \
  --config MultiModal/configs/stage45_expfix_mixed200k_gpu3.yaml 2>&1 | tee "$LOG_DIR/stage45_mixed200k_gpu3.log"
python -m MultiModal.multimodal.experiments.run_stage45_quality_quantity_deconfound \
  --config MultiModal/configs/stage45_expfix_clean_source_gpu3.yaml 2>&1 | tee "$LOG_DIR/stage45_clean_source_gpu3.log"

touch "$MARK/gpu3.stage45.done"

# Final aggregate
while [[ ! -f "$MARK/gpu0.stage44.done" || ! -f "$MARK/gpu1.stage44.done" || ! -f "$MARK/gpu2.stage44.done" || ! -f "$MARK/gpu3.stage44.done" || \
         ! -f "$MARK/gpu0.stage45.done" || ! -f "$MARK/gpu1.stage45.done" || ! -f "$MARK/gpu2.stage45.done" || ! -f "$MARK/gpu3.stage45.done" ]]; do
  sleep 30
done

python -m MultiModal.multimodal.experiments.run_stage46_experimental_fixes_aggregate \
  --config MultiModal/configs/stage46_expfix_aggregate.yaml 2>&1 | tee "$LOG_DIR/stage46_aggregate_gpu3.log"

touch "$MARK/gpu3.stage46.done"
echo "[$(date '+%F %T')] queue_expfix_gpu3 complete"
