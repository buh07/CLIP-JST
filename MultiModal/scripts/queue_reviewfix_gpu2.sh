#!/usr/bin/env bash
set -euo pipefail
ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
OUT="$MM_ROOT/results/reviewer_fixes_suite"
LOG_DIR="$MM_ROOT/logs/reviewer_fixes"
MARK="$OUT/markers"
mkdir -p "$LOG_DIR" "$MARK"

export CUDA_VISIBLE_DEVICES=2
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2'
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

python -m py_compile MultiModal/multimodal/experiments/run_stage41_semantic_metadata_topk.py

rm -f "$OUT/split/gpu2/markers/stage41_semantic_metadata_topk.done.json"
python -m MultiModal.multimodal.experiments.run_stage41_semantic_metadata_topk \
  --config MultiModal/configs/stage41_semantic_metadata_gpu2.yaml 2>&1 | tee "$LOG_DIR/stage41_gpu2.log"

touch "$MARK/gpu2.stage41.done"
echo "[$(date '+%F %T')] queue_reviewfix_gpu2 complete"
