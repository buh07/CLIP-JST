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

export CUDA_VISIBLE_DEVICES=6
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2' 2>/dev/null || true
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

while [[ ! -f "$MARK/stage69_cache_ready.done" && ! -f "$MARK/gpu0.stage69_cache_prebuild.done" ]]; do sleep 30; done

for attempt in 1 2 3; do
  if python -m MultiModal.multimodal.experiments.run_stage69_third_triple_speechcoco --config MultiModal/configs/stage69_speechcoco_gpu6.yaml 2>&1 | tee "$LOG_DIR/stage69_gpu6.attempt${attempt}.log"; then
    touch "$MARK/gpu6.stage69.done"
    echo "[$(date '+%F %T')] queue_stage69_gpu6 complete"
    exit 0
  fi
  echo "[$(date '+%F %T')] queue_stage69_gpu6 attempt $attempt failed"
  if [[ "$attempt" -lt 3 ]]; then sleep 30; fi
done

touch "$MARK/gpu6.stage69.failed"
exit 1
