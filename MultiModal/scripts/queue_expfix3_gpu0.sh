#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
LOG_DIR="$MM_ROOT/logs/experimental_fixes_suite3"
MARK="$MM_ROOT/results/experimental_fixes_suite/markers/shards"
mkdir -p "$LOG_DIR" "$MARK"

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2' torchcodec
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

# Stage 47: identity ablation (epochs_phase_b=0, all dims, all seeds on GPU 0 — fast)
python -m MultiModal.multimodal.experiments.run_stage47_identity_ablation \
  --config MultiModal/configs/stage47_identity_ablation_gpu0.yaml 2>&1 | tee "$LOG_DIR/stage47_identity_ablation_gpu0.log"
touch "$MARK/gpu0.stage47.done"

# Stage 45 multi-dim: seeds 0,1 for all 4 conditions
python -m MultiModal.multimodal.experiments.run_stage45_quality_quantity_deconfound \
  --config MultiModal/configs/stage45_multidim_mixed46k_gpu0.yaml 2>&1 | tee "$LOG_DIR/stage45_multidim_mixed46k_gpu0.log"
python -m MultiModal.multimodal.experiments.run_stage45_quality_quantity_deconfound \
  --config MultiModal/configs/stage45_multidim_mixed200k_gpu0.yaml 2>&1 | tee "$LOG_DIR/stage45_multidim_mixed200k_gpu0.log"
python -m MultiModal.multimodal.experiments.run_stage45_quality_quantity_deconfound \
  --config MultiModal/configs/stage45_multidim_clean_source_gpu0.yaml 2>&1 | tee "$LOG_DIR/stage45_multidim_clean_source_gpu0.log"
python -m MultiModal.multimodal.experiments.run_stage45_quality_quantity_deconfound \
  --config MultiModal/configs/stage45_multidim_clean_source_46k_gpu0.yaml 2>&1 | tee "$LOG_DIR/stage45_multidim_clean_source_46k_gpu0.log"

touch "$MARK/gpu0.stage45_multidim.done"
echo "[$(date '+%F %T')] queue_expfix3_gpu0 complete"
