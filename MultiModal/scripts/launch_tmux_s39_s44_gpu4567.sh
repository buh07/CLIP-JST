#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"

S4="s39_s44_g4"
S5="s39_s44_g5"
S6="s39_s44_g6"
S7="s39_s44_g7"

for s in "$S4" "$S5" "$S6" "$S7"; do
  tmux has-session -t "$s" 2>/dev/null && tmux kill-session -t "$s" || true
done

chmod +x "$MM_ROOT/scripts/queue_s39_s44_gpu4.sh"
chmod +x "$MM_ROOT/scripts/queue_s39_s44_gpu5.sh"
chmod +x "$MM_ROOT/scripts/queue_s39_s44_gpu6.sh"
chmod +x "$MM_ROOT/scripts/queue_s39_s44_gpu7.sh"

# GPU 4: Stage 39 on Stage 44 checkpoints — m=64
tmux new-session -d -s "$S4" "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_s39_s44_gpu4.sh'"

# GPU 5: Stage 39 on Stage 44 checkpoints — m=128
tmux new-session -d -s "$S5" "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_s39_s44_gpu5.sh'"

# GPU 6: Stage 39 on Stage 44 checkpoints — m=256
tmux new-session -d -s "$S6" "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_s39_s44_gpu6.sh'"

# GPU 7: Stage 39 on Stage 44 checkpoints — m=512
tmux new-session -d -s "$S7" "cd '$ROOT' && bash '$MM_ROOT/scripts/queue_s39_s44_gpu7.sh'"

echo "Started sessions: $S4 $S5 $S6 $S7"
echo "Logs: $MM_ROOT/logs/reviewer_fixes/stage39_s44_gpu{4,5,6,7}.log"
echo ""
echo "Monitor with:"
echo "  tmux attach -t $S4   # m=64"
echo "  tmux attach -t $S5   # m=128"
echo "  tmux attach -t $S6   # m=256"
echo "  tmux attach -t $S7   # m=512"
echo ""
echo "Results go to: $MM_ROOT/results/reviewer_fixes_suite/stage39_s44_coco/"
echo ""
echo "Wait for all 4 markers:"
echo "  $MM_ROOT/results/reviewer_fixes_suite/markers/gpu{4,5,6,7}.stage39_s44.done"
