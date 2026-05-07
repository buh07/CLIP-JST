#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
LOG_DIR="$MM_ROOT/logs/next_run_suite/post_w3_sequence"
MARK_DIR="$MM_ROOT/results/next_run_suite/post_w3_sequence/markers"
mkdir -p "$LOG_DIR" "$MARK_DIR"

export CUDA_VISIBLE_DEVICES=2
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

python -m MultiModal.multimodal.experiments.run_stage39_modality_gap_linear_vs_jl \
  --config MultiModal/configs/stage39_joint_gap_gpu2.yaml 2>&1 | tee "$LOG_DIR/stage39_joint_gpu2.log"

touch "$MARK_DIR/gpu2.stage39_joint.done"
echo "[$(date '+%F %T')] queue_postw3_stage39_joint_gpu2 complete"

