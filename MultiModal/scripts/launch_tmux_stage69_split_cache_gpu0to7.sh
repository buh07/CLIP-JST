#!/usr/bin/env bash
set -euo pipefail
ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
OUT="$MM_ROOT/results/stage69_prereg_suite"
MARK="$OUT/markers"
mkdir -p "$MARK"

SESS=(
  mm_stage69_cache_manifest_g0
  mm_stage69_cache_encode_g0 mm_stage69_cache_encode_g1 mm_stage69_cache_encode_g2 mm_stage69_cache_encode_g3
  mm_stage69_cache_encode_g4 mm_stage69_cache_encode_g5 mm_stage69_cache_encode_g6 mm_stage69_cache_encode_g7
  mm_stage69_cache_merge_g0
)
for s in "${SESS[@]}"; do
  tmux has-session -t "$s" 2>/dev/null && tmux kill-session -t "$s" || true
done

# keep existing prereg marker if present; clear only split-cache markers
rm -f "$MARK/stage69_cache_manifest.done" "$MARK/stage69_cache_merge.done" "$MARK/stage69_cache_ready.done"       "$MARK/stage69_cache_ready.done.json" "$MARK"/gpu*.stage69_cache_encode.done "$MARK"/gpu*.stage69_cache_encode.failed || true

tmux new-session -d -s mm_stage69_cache_manifest_g0 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_stage69_cache_manifest_gpu0.sh'"
for g in 0 1 2 3 4 5 6 7; do
  tmux new-session -d -s "mm_stage69_cache_encode_g${g}" "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_stage69_cache_encode_gpu${g}.sh'"
done
tmux new-session -d -s mm_stage69_cache_merge_g0 "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_stage69_cache_merge_gpu0.sh'"

echo "Started split SpeechCoco cache campaign sessions."
echo "Results root: $OUT"
