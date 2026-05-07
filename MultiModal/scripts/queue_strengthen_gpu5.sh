#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
OUT="$MM_ROOT/results/neurips_strengthen_suite"
LOG_DIR="$MM_ROOT/logs/neurips_strengthen_suite"
MARK="$OUT/markers/shards"
mkdir -p "$LOG_DIR" "$MARK"

export CUDA_VISIBLE_DEVICES=5
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2' torchcodec
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

python -m MultiModal.multimodal.experiments.run_stage38_phaseb_quality_analysis \
  --config MultiModal/configs/stage38_phaseb_quality_analysis.yaml 2>&1 | tee "$LOG_DIR/stage38_gpu5.log"
touch "$MARK/gpu5.stage38.done"

python -m MultiModal.multimodal.experiments.run_stage39_modality_gap_linear_vs_jl \
  --config MultiModal/configs/stage39_modality_gap_gpu5.yaml 2>&1 | tee "$LOG_DIR/stage39_gpu5.log"
touch "$MARK/gpu5.stage39.done"

echo "[$(date '+%F %T')] queue_strengthen_gpu5 complete"
