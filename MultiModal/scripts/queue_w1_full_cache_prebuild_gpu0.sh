#!/usr/bin/env bash
set -euo pipefail
ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
OUT="$MM_ROOT/results/next_run_suite/w1_avcaps_full"
LOG_DIR="$MM_ROOT/logs/next_run_suite/w1_avcaps_full"
MARK="$OUT/markers"
mkdir -p "$LOG_DIR" "$MARK"

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2' 2>/dev/null || true
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

python -m py_compile \
  MultiModal/multimodal/experiments/run_stage57_second_triple_avcaps.py \
  MultiModal/multimodal/experiments/run_stage58_second_triple_aggregate.py \
  MultiModal/multimodal/experiments/run_stage55_wavcaps_holdout_retrain.py \
  MultiModal/multimodal/experiments/run_stage56_wavcaps_holdout_aggregate.py

python -m MultiModal.multimodal.experiments.run_stage57_second_triple_avcaps \
  --config MultiModal/configs/stage57_w1_full_cache_prebuild_gpu0.yaml 2>&1 | tee "$LOG_DIR/stage57_cache_prebuild_gpu0.log"

CACHE_ROOT="$MM_ROOT/results/reinforce_suite/caches/avcaps_av"
if [[ -f "$CACHE_ROOT/image_feats_clip_raw.pt" && -f "$CACHE_ROOT/audio_feats_clap_raw.pt" && -f "$CACHE_ROOT/text_feats_clip_raw.pt" && -f "$CACHE_ROOT/metadata.json" ]]; then
  touch "$MARK/avcaps_cache_ready.done"
fi

touch "$MARK/gpu0.stage57_cache_prebuild.done"
echo "[$(date '+%F %T')] queue_w1_full_cache_prebuild_gpu0 complete"
