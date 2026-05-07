#!/usr/bin/env bash
set -euo pipefail
ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
SUITE_OUT="$MM_ROOT/results/theory_backing_suite"
LOG_DIR="$MM_ROOT/logs/theory_backing_suite"
MARK="$SUITE_OUT/markers"
mkdir -p "$LOG_DIR" "$MARK"

export CUDA_VISIBLE_DEVICES=7
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2' 2>/dev/null || true
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

while [[ ! -f "$MARK/gpu0.stage66.done" || ! -f "$MARK/gpu1.stage66.done" || ! -f "$MARK/gpu2.stage66.done" || ! -f "$MARK/gpu3.stage66.done" || ! -f "$MARK/gpu4.stage66.done" || ! -f "$MARK/gpu5.stage66.done" || ! -f "$MARK/gpu6.stage66.done" || ! -f "$MARK/gpu7.stage66.done" ]]; do
  sleep 30
done

python -m MultiModal.multimodal.experiments.run_stage67_gap_intervention_aggregate   --config MultiModal/configs/stage67_gap_intervention_aggregate.yaml 2>&1 | tee "$LOG_DIR/stage67_coordinator.log"
touch "$MARK/stage67.done"
echo "[$(date '+%F %T')] queue_theory_missing_coordinator complete"
