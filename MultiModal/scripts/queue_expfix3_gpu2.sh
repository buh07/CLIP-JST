#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
LOG_DIR="$MM_ROOT/logs/experimental_fixes_suite3"
MARK="$MM_ROOT/results/experimental_fixes_suite/markers/shards"
mkdir -p "$LOG_DIR" "$MARK"

export CUDA_VISIBLE_DEVICES=2
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2' torchcodec
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

# Stage 45 multi-dim: seed 3 for all 4 conditions
python -m MultiModal.multimodal.experiments.run_stage45_quality_quantity_deconfound \
  --config MultiModal/configs/stage45_multidim_mixed46k_gpu2.yaml 2>&1 | tee "$LOG_DIR/stage45_multidim_mixed46k_gpu2.log"
python -m MultiModal.multimodal.experiments.run_stage45_quality_quantity_deconfound \
  --config MultiModal/configs/stage45_multidim_mixed200k_gpu2.yaml 2>&1 | tee "$LOG_DIR/stage45_multidim_mixed200k_gpu2.log"
python -m MultiModal.multimodal.experiments.run_stage45_quality_quantity_deconfound \
  --config MultiModal/configs/stage45_multidim_clean_source_gpu2.yaml 2>&1 | tee "$LOG_DIR/stage45_multidim_clean_source_gpu2.log"
python -m MultiModal.multimodal.experiments.run_stage45_quality_quantity_deconfound \
  --config MultiModal/configs/stage45_multidim_clean_source_46k_gpu2.yaml 2>&1 | tee "$LOG_DIR/stage45_multidim_clean_source_46k_gpu2.log"

touch "$MARK/gpu2.stage45_multidim.done"
echo "[$(date '+%F %T')] queue_expfix3_gpu2 complete"
