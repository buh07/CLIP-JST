#!/usr/bin/env bash
set -euo pipefail
ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
OUT="$MM_ROOT/results/neurips_strengthen_suite"
LOG_DIR="$MM_ROOT/logs/neurips_cleanup"
MARK="$OUT/markers/cleanup"
mkdir -p "$LOG_DIR" "$MARK"

export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2'
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

python -m py_compile \
  MultiModal/multimodal/experiments/run_stage37_relabel_outputs.py

python -m MultiModal.multimodal.experiments.run_stage37_relabel_outputs \
  --config MultiModal/configs/stage37_relabel_outputs.yaml 2>&1 | tee "$LOG_DIR/stage37_relabel_gpu1_cleanup.log"

touch "$MARK/gpu1.stage37relabel.cleanup.done"
echo "[$(date '+%F %T')] queue_cleanup_gpu1 complete"
