#!/usr/bin/env bash
set -euo pipefail
ROOT='/jumbo/lisp/f004ndc/CLIP JST'
MM_ROOT="$ROOT/MultiModal"
LOG_DIR="$MM_ROOT/logs/domain_gap_closure_suite"
MARK_DIR="$MM_ROOT/results/domain_gap_closure_suite/markers/shards"
SHARD_ROOT="$MM_ROOT/results/domain_gap_closure_suite/caches/wavcaps_humanify_shards"
FINAL_ROOT="$MM_ROOT/results/domain_gap_closure_suite/caches/wavcaps_humanify"
mkdir -p "$LOG_DIR" "$MARK_DIR" "$SHARD_ROOT" "$FINAL_ROOT"
export PYTHONUNBUFFERED=1
cd "$ROOT"
source "$MM_ROOT/.venv/bin/activate"
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"

rm -f "$MARK_DIR/stage31_wavcaps_cache.done" "$MARK_DIR/stage31_wavcaps_cache.failed.json"

while true; do
  for i in 0 1 2 3; do
    if [[ -f "$MARK_DIR/wavcaps_cache.shard${i}.fail.json" ]]; then
      cp "$MARK_DIR/wavcaps_cache.shard${i}.fail.json" "$MARK_DIR/stage31_wavcaps_cache.failed.json"
      echo "Shard $i failed. Aborting merge."
      exit 1
    fi
  done
  if [[ -f "$MARK_DIR/wavcaps_cache.shard0.done.json" && -f "$MARK_DIR/wavcaps_cache.shard1.done.json" && -f "$MARK_DIR/wavcaps_cache.shard2.done.json" && -f "$MARK_DIR/wavcaps_cache.shard3.done.json" ]]; then
    break
  fi
  sleep 30
done

python MultiModal/scripts/merge_wavcaps_cache_shards.py \
  --shard-root "$SHARD_ROOT" \
  --shard-count 4 \
  --out-dir "$FINAL_ROOT" \
  --dataset "humanify/AS-WavCaps" \
  --clap-model "laion/clap-htsat-unfused" \
  --clip-backbone "openai/clip-vit-base-patch32" \
  --target-sr 48000 \
  --max-examples 200000 \
  --sampling-policy "stratified" \
  --merge-seed 2026 \
  --done-marker "$MARK_DIR/stage31_wavcaps_cache.merge.done.json" \
  --fail-marker "$MARK_DIR/stage31_wavcaps_cache.failed.json" \
  2>&1 | tee "$LOG_DIR/stage31_cache_merge.log"

echo "ok" > "$MARK_DIR/stage31_wavcaps_cache.done"
echo "WavCaps cache merge complete."
