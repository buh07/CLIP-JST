#!/usr/bin/env bash
set -euo pipefail
ROOT='/jumbo/lisp/f004ndc/CLIP JST'
MM_ROOT="$ROOT/MultiModal"
LOG_DIR="$MM_ROOT/logs/domain_gap_closure_suite"
MARK_DIR="$MM_ROOT/results/domain_gap_closure_suite/markers/shards"
SHARD_ROOT="$MM_ROOT/results/domain_gap_closure_suite/caches/wavcaps_humanify_shards"
mkdir -p "$LOG_DIR" "$MARK_DIR" "$SHARD_ROOT"
export CUDA_VISIBLE_DEVICES=7
export PYTHONUNBUFFERED=1
cd "$ROOT"
source "$MM_ROOT/.venv/bin/activate"
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
python MultiModal/scripts/build_wavcaps_cache_shard.py \
  --shard-out-dir "$SHARD_ROOT/shard3" \
  --dataset "humanify/AS-WavCaps" \
  --clap-model "laion/clap-htsat-unfused" \
  --clip-backbone "openai/clip-vit-base-patch32" \
  --target-sr 48000 \
  --max-examples 70000 \
  --sampling-policy "stratified" \
  --device "cuda" \
  --audio-batch-size 64 \
  --text-batch-size 256 \
  --shard-index 3 \
  --shard-count 4 \
  --done-marker "$MARK_DIR/wavcaps_cache.shard3.done.json" \
  --fail-marker "$MARK_DIR/wavcaps_cache.shard3.fail.json" \
  2>&1 | tee "$LOG_DIR/stage31_cache_shard_gpu7.log"
