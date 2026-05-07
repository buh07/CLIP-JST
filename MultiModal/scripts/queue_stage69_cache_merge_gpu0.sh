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

for g in 0 1 2 3 4 5 6 7; do
  while [[ ! -f "$MARK/gpu${g}.stage69_cache_encode.done" ]]; do
    if [[ -f "$MARK/gpu${g}.stage69_cache_encode.failed" ]]; then
      echo "Shard gpu${g} failed; aborting merge."
      exit 1
    fi
    sleep 20
  done
done

python -m MultiModal.multimodal.experiments.run_stage69_speechcoco_cache_merge --config MultiModal/configs/stage69_speechcoco_cache_merge.yaml 2>&1 | tee "$LOG_DIR/stage69_cache_merge_gpu0.log"

[[ -f "$OUT/caches/speechcoco_av/image_feats_clip_raw.pt" ]]
[[ -f "$OUT/caches/speechcoco_av/audio_feats_clap_raw.pt" ]]
[[ -f "$OUT/caches/speechcoco_av/text_feats_clip_raw.pt" ]]
[[ -f "$OUT/caches/speechcoco_av/metadata.json" ]]

# legacy plain marker for existing waiters
: > "$MARK/stage69_cache_ready.done"
touch "$MARK/stage69_cache_merge.done"
echo "[$(date '+%F %T')] queue_stage69_cache_merge_gpu0 complete"
