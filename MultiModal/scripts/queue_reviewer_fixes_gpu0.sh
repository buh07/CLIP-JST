#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
LOG_DIR="$MM_ROOT/logs/reviewer_fixes"
MARK="$MM_ROOT/results/reviewer_fixes_suite/markers"
mkdir -p "$LOG_DIR" "$MARK"

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2' soundfile 2>/dev/null || true
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

# W7: ImageBind zero-shot baseline on 1K split + full pool
# AudioCLIP not installable (git-LFS failure); ImageBind-Huge is the jointly-trained baseline.
# Stage 37 already ran on full pool; this adds the 883-item 1K split.
python -m MultiModal.multimodal.experiments.run_w7_imagebind_1k_split 2>&1 | tee "$LOG_DIR/w7_imagebind_1k_split.log"

touch "$MARK/gpu0.w7.done"
echo "[$(date '+%F %T')] queue_reviewer_fixes_gpu0 complete"
