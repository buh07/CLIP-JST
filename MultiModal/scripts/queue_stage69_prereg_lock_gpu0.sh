#!/usr/bin/env bash
set -euo pipefail
ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
OUT="$MM_ROOT/results/stage69_prereg_suite"
LOG_DIR="$MM_ROOT/logs/stage69_prereg_suite"
MARK="$OUT/markers"
mkdir -p "$LOG_DIR" "$MARK"

TMPDIR="$OUT/tmp"
mkdir -p "$TMPDIR"
export TMPDIR

HF_HOME="$OUT/hf_home"
XDG_CACHE_HOME="$OUT/xdg_cache"
HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"
export HF_HOME XDG_CACHE_HOME HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_XET_HIGH_PERFORMANCE=1

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2' 2>/dev/null || true
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

python -m py_compile \
  MultiModal/multimodal/data/speechcoco_av.py \
  MultiModal/multimodal/experiments/run_stage69_prereg_lock.py \
  MultiModal/multimodal/experiments/run_stage69_third_triple_speechcoco.py \
  MultiModal/multimodal/experiments/run_stage70_third_triple_aggregate.py \
  MultiModal/multimodal/experiments/run_stage71_third_triple_prospective_check.py \
  MultiModal/multimodal/experiments/run_stage72_third_triple_form_compare.py \
  MultiModal/multimodal/experiments/run_stage73_three_triple_meta_update.py

python -m MultiModal.multimodal.experiments.run_stage69_prereg_lock \
  --config MultiModal/configs/stage69_prereg_lock.yaml 2>&1 | tee "$LOG_DIR/stage69_prereg_lock_gpu0.log"

[[ -f "$ROOT/paper/prereg/PREREG_STAGE69.md" ]]
[[ -f "$ROOT/paper/prereg/predictions_stage69.json" ]]
[[ -f "$ROOT/paper/prereg/PREREG_STAGE69_SHA256.txt" ]]
[[ -f "$ROOT/paper/prereg/PREREG_STAGE69_COMMIT.txt" ]]

touch "$MARK/stage69_prereg_locked.done"
echo "[$(date '+%F %T')] queue_stage69_prereg_lock_gpu0 complete"
