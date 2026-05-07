#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
OUT="$MM_ROOT/results/domain_gap_closure_suite"
LOG_DIR="$MM_ROOT/logs/domain_gap_closure_suite"
MARK="$OUT/markers/shards"
mkdir -p "$LOG_DIR" "$MARK"

export CUDA_VISIBLE_DEVICES=5
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2' torchcodec
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

python -m MultiModal.multimodal.experiments.run_stage28_category_retrieval \
  --config MultiModal/configs/stage28_domain_gap.yaml 2>&1 | tee "$LOG_DIR/stage28_gpu5.log"
touch "$MARK/gpu5.stage28.done"

python -m MultiModal.multimodal.experiments.run_stage29_cc3m_phaseA_modular \
  --config MultiModal/configs/stage29_domain_gap_gpu5.yaml 2>&1 | tee "$LOG_DIR/stage29_gpu5.log"
touch "$MARK/gpu5.stage29.done"

python -m MultiModal.multimodal.experiments.run_stage30_modular_vs_nonmodular \
  --config MultiModal/configs/stage30_domain_gap_gpu5.yaml 2>&1 | tee "$LOG_DIR/stage30_gpu5.log"
touch "$MARK/gpu5.stage30.done"

python -m MultiModal.multimodal.experiments.run_stage32_modality_order_ablation \
  --config MultiModal/configs/stage32_domain_gap_gpu5.yaml 2>&1 | tee "$LOG_DIR/stage32_gpu5.log"
touch "$MARK/gpu5.stage32.done"

echo "[$(date '+%F %T')] queue_domain_gap_gpu5 complete"
