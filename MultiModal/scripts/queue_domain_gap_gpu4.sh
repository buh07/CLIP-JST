#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
OUT="$MM_ROOT/results/domain_gap_closure_suite"
LOG_DIR="$MM_ROOT/logs/domain_gap_closure_suite"
MARK="$OUT/markers/shards"
mkdir -p "$LOG_DIR" "$MARK"

export CUDA_VISIBLE_DEVICES=4
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2' torchcodec
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

# Pre-run integrity checks + smoke
python -m py_compile \
  MultiModal/multimodal/models/trimodal_heads.py \
  MultiModal/multimodal/data/cc3m.py \
  MultiModal/multimodal/data/wavcaps.py \
  MultiModal/multimodal/experiments/run_stage27_bottleneck_decomposition.py \
  MultiModal/multimodal/experiments/run_stage28_category_retrieval.py \
  MultiModal/multimodal/experiments/run_stage29_cc3m_phaseA_modular.py \
  MultiModal/multimodal/experiments/run_stage30_modular_vs_nonmodular.py \
  MultiModal/multimodal/experiments/run_stage31_wavcaps_scaling.py \
  MultiModal/multimodal/experiments/run_stage32_modality_order_ablation.py \
  MultiModal/multimodal/experiments/run_stage33_domain_gap_aggregate.py

python -m MultiModal.multimodal.experiments.run_stage29_cc3m_phaseA_modular \
  --config MultiModal/configs/stage29_domain_gap_smoke.yaml 2>&1 | tee "$LOG_DIR/stage29_smoke_gpu4.log"
python -m MultiModal.multimodal.experiments.run_stage30_modular_vs_nonmodular \
  --config MultiModal/configs/stage30_domain_gap_smoke.yaml 2>&1 | tee "$LOG_DIR/stage30_smoke_gpu4.log"
python -m MultiModal.multimodal.experiments.run_stage32_modality_order_ablation \
  --config MultiModal/configs/stage32_domain_gap_smoke.yaml 2>&1 | tee "$LOG_DIR/stage32_smoke_gpu4.log"

python -m MultiModal.multimodal.experiments.run_stage27_bottleneck_decomposition \
  --config MultiModal/configs/stage27_domain_gap.yaml 2>&1 | tee "$LOG_DIR/stage27_gpu4.log"
touch "$MARK/gpu4.stage27.done"

python -m MultiModal.multimodal.experiments.run_stage29_cc3m_phaseA_modular \
  --config MultiModal/configs/stage29_domain_gap_gpu4.yaml 2>&1 | tee "$LOG_DIR/stage29_gpu4.log"
touch "$MARK/gpu4.stage29.done"

python -m MultiModal.multimodal.experiments.run_stage30_modular_vs_nonmodular \
  --config MultiModal/configs/stage30_domain_gap_gpu4.yaml 2>&1 | tee "$LOG_DIR/stage30_gpu4.log"
touch "$MARK/gpu4.stage30.done"

python -m MultiModal.multimodal.experiments.run_stage32_modality_order_ablation \
  --config MultiModal/configs/stage32_domain_gap_gpu4.yaml 2>&1 | tee "$LOG_DIR/stage32_gpu4.log"
touch "$MARK/gpu4.stage32.done"

echo "[$(date '+%F %T')] queue_domain_gap_gpu4 complete"
