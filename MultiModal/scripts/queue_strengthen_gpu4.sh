#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
OUT="$MM_ROOT/results/neurips_strengthen_suite"
LOG_DIR="$MM_ROOT/logs/neurips_strengthen_suite"
MARK="$OUT/markers/shards"
mkdir -p "$LOG_DIR" "$MARK"

export CUDA_VISIBLE_DEVICES=4
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2' torchcodec
python -m pip install -q --index-url https://download.pytorch.org/whl/cu128 torchaudio==2.7.0 || true
python -m pip install -q git+https://github.com/facebookresearch/ImageBind.git || true
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

python -m py_compile \
  MultiModal/multimodal/experiments/run_stage37_imagebind_comparison.py \
  MultiModal/multimodal/experiments/run_stage38_phaseb_quality_analysis.py \
  MultiModal/multimodal/experiments/run_stage39_modality_gap_linear_vs_jl.py \
  MultiModal/multimodal/experiments/run_stage40_strengthen_suite_aggregate.py

python -m MultiModal.multimodal.experiments.run_stage37_imagebind_comparison \
  --config MultiModal/configs/stage37_imagebind_comparison.yaml 2>&1 | tee "$LOG_DIR/stage37_gpu4.log"
touch "$MARK/gpu4.stage37.done"

python -m MultiModal.multimodal.experiments.run_stage39_modality_gap_linear_vs_jl \
  --config MultiModal/configs/stage39_modality_gap_gpu4.yaml 2>&1 | tee "$LOG_DIR/stage39_gpu4.log"
touch "$MARK/gpu4.stage39.done"

echo "[$(date '+%F %T')] queue_strengthen_gpu4 complete"
