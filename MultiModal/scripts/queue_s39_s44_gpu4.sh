#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
LOG_DIR="$MM_ROOT/logs/reviewer_fixes"
MARK="$MM_ROOT/results/reviewer_fixes_suite/markers"
mkdir -p "$LOG_DIR" "$MARK"

export CUDA_VISIBLE_DEVICES=4
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2' 2>/dev/null || true
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

# Stage 39 rerun on Stage 44 (COCO Phase A) checkpoints — m=64
python -m MultiModal.multimodal.experiments.run_stage39_modality_gap_linear_vs_jl \
  --config MultiModal/configs/stage39_s44_coco_gpu4.yaml 2>&1 | tee "$LOG_DIR/stage39_s44_gpu4.log"

touch "$MARK/gpu4.stage39_s44.done"
echo "[$(date '+%F %T')] queue_s39_s44_gpu4 complete"
