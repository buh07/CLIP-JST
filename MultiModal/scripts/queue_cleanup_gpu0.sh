#!/usr/bin/env bash
set -euo pipefail
ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
OUT="$MM_ROOT/results/neurips_strengthen_suite"
LOG_DIR="$MM_ROOT/logs/neurips_cleanup"
MARK="$OUT/markers/cleanup"
mkdir -p "$LOG_DIR" "$MARK"

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2'
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

python -m py_compile \
  MultiModal/multimodal/experiments/run_stage38_phaseb_quality_analysis.py

rm -f "$OUT/markers/stage38_phaseb_quality_analysis.done.json"
python -m MultiModal.multimodal.experiments.run_stage38_phaseb_quality_analysis \
  --config MultiModal/configs/stage38_phaseb_quality_analysis.yaml 2>&1 | tee "$LOG_DIR/stage38_gpu0_cleanup.log"

touch "$MARK/gpu0.stage38.cleanup.done"
echo "[$(date '+%F %T')] queue_cleanup_gpu0 complete"
