#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
LOG_DIR="$MM_ROOT/logs/reinforce_suite"
MARK="$MM_ROOT/results/reinforce_suite/markers"
mkdir -p "$LOG_DIR" "$MARK"

export CUDA_VISIBLE_DEVICES=4
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2' soundfile 2>/dev/null || true
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

python -m MultiModal.multimodal.experiments.run_w7_imagebind_1k_split 2>&1 | tee "$LOG_DIR/w7_imagebind_1k_split.gpu4.log"

cp -f "$MM_ROOT/results/reviewer_fixes_suite/w7_imagebind_1k_split/w7_imagebind_1k_split_results.json" \
      "$MM_ROOT/results/reinforce_suite/w7_imagebind_1k_split_results.gpu4.json"

touch "$MARK/gpu4.w7.done"
echo "[$(date '+%F %T')] queue_reinforce_gpu4 complete"
