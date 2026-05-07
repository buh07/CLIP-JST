#!/usr/bin/env bash
set -euo pipefail
ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
OUT="$MM_ROOT/results/reinforce_suite/w1_second_triple_avcaps"
LOG_DIR="$MM_ROOT/logs/reinforce_suite/w1_second_triple_avcaps"
MARK="$OUT/markers"
mkdir -p "$LOG_DIR" "$MARK"

export CUDA_VISIBLE_DEVICES=4
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2' 2>/dev/null || true
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

python -m py_compile \
  MultiModal/multimodal/data/avcaps_av.py \
  MultiModal/multimodal/experiments/run_stage57_second_triple_avcaps.py \
  MultiModal/multimodal/experiments/run_stage58_second_triple_aggregate.py

python -m MultiModal.multimodal.experiments.run_stage57_second_triple_avcaps \
  --config MultiModal/configs/stage57_w1_avcaps_gpu4.yaml 2>&1 | tee "$LOG_DIR/stage57_gpu4.log"

# cache-ready marker once cache exists
CACHE_ROOT="$MM_ROOT/results/reinforce_suite/caches/avcaps_av"
if [[ -f "$CACHE_ROOT/image_feats_clip_raw.pt" && -f "$CACHE_ROOT/audio_feats_clap_raw.pt" && -f "$CACHE_ROOT/text_feats_clip_raw.pt" && -f "$CACHE_ROOT/metadata.json" ]]; then
  touch "$MARK/avcaps_cache_ready.done"
fi

touch "$MARK/gpu4.stage57.done"
echo "[$(date '+%F %T')] queue_w1_avcaps_gpu4 complete"
