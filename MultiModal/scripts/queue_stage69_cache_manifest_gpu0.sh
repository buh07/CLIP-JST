#!/usr/bin/env bash
set -euo pipefail
ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
OUT="$MM_ROOT/results/stage69_prereg_suite"
LOG_DIR="$MM_ROOT/logs/stage69_prereg_suite"
MARK="$OUT/markers"
mkdir -p "$LOG_DIR" "$MARK"

: "${HF_TOKEN:?HF_TOKEN must be set in environment}"

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
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2' 2>/dev/null || true
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES=0

python -m py_compile   MultiModal/multimodal/experiments/run_stage69_speechcoco_cache_manifest.py   MultiModal/multimodal/experiments/run_stage69_speechcoco_cache_encode_shard.py   MultiModal/multimodal/experiments/run_stage69_speechcoco_cache_merge.py

while [[ ! -f "$MARK/stage69_prereg_locked.done" && ! -f "$MARK/stage69_prereg_locked.done.json" ]]; do sleep 15; done
python -m MultiModal.multimodal.experiments.run_stage69_speechcoco_cache_manifest --config MultiModal/configs/stage69_speechcoco_cache_manifest_full.yaml 2>&1 | tee "$LOG_DIR/stage69_cache_manifest_gpu0.log"

[[ -f "$OUT/stage69_speechcoco_cache_manifest/stage69_speechcoco_manifest.json" ]]
touch "$MARK/stage69_cache_manifest.done"
echo "[$(date '+%F %T')] queue_stage69_cache_manifest_gpu0 complete"
