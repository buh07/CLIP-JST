#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
SEQ_ROOT="$MM_ROOT/results/next_run_suite/post_w3_sequence"
LOG_DIR="$MM_ROOT/logs/next_run_suite/post_w3_sequence"
MARK_DIR="$SEQ_ROOT/markers"
mkdir -p "$LOG_DIR" "$MARK_DIR"

export PYTHONUNBUFFERED=1

wait_for_file() {
  local f="$1"
  local label="${2:-$1}"
  until [[ -f "$f" ]]; do
    echo "[$(date '+%F %T')] waiting for: $label"
    sleep 20
  done
}

wait_for_success_or_fail() {
  local ok="$1"
  local fail="$2"
  local label="$3"
  until [[ -f "$ok" || -f "$fail" ]]; do
    echo "[$(date '+%F %T')] waiting for: $label"
    sleep 20
  done
  if [[ -f "$fail" ]]; then
    echo "[$(date '+%F %T')] FAILED: $label"
    exit 1
  fi
}

launch_tmux_worker() {
  local session="$1"
  local script="$2"
  tmux has-session -t "$session" 2>/dev/null && tmux kill-session -t "$session" || true
  tmux new-session -d -s "$session" "cd '$ROOT' && bash '$script'"
}

scan_logs_for_errors() {
  local label="$1"
  shift
  local bad=0
  for f in "$@"; do
    [[ -f "$f" ]] || continue
    if rg -n "Traceback|CUDA out of memory|RuntimeError:|Exception:|^ERROR|\\bnan\\b" "$f" >/dev/null; then
      echo "[$(date '+%F %T')] error pattern detected in $f during $label"
      bad=1
    fi
  done
  if [[ "$bad" -ne 0 ]]; then
    exit 1
  fi
}

echo "[$(date '+%F %T')] post-W3 sequence controller starting"

# Gate on completed W3 extension aggregate.
wait_for_file \
  "$MM_ROOT/results/next_run_suite/w3_holdout_ext/aggregate/markers/stage56_wavcaps_holdout_aggregate.done.json" \
  "W3 aggregate completion"

# ---------------------------------------------------------------------------
# Step 1: Joint-method centroid gap + alpha (Stage39 joint + Stage60 analysis)
# ---------------------------------------------------------------------------
launch_tmux_worker "pw3_s39j_g0" "$MM_ROOT/scripts/queue_postw3_stage39_joint_gpu0.sh"
launch_tmux_worker "pw3_s39j_g1" "$MM_ROOT/scripts/queue_postw3_stage39_joint_gpu1.sh"
launch_tmux_worker "pw3_s39j_g2" "$MM_ROOT/scripts/queue_postw3_stage39_joint_gpu2.sh"
launch_tmux_worker "pw3_s39j_g3" "$MM_ROOT/scripts/queue_postw3_stage39_joint_gpu3.sh"

wait_for_file "$MARK_DIR/gpu0.stage39_joint.done" "stage39 joint gpu0"
wait_for_file "$MARK_DIR/gpu1.stage39_joint.done" "stage39 joint gpu1"
wait_for_file "$MARK_DIR/gpu2.stage39_joint.done" "stage39 joint gpu2"
wait_for_file "$MARK_DIR/gpu3.stage39_joint.done" "stage39 joint gpu3"

scan_logs_for_errors "stage39_joint" \
  "$LOG_DIR/stage39_joint_gpu0.log" \
  "$LOG_DIR/stage39_joint_gpu1.log" \
  "$LOG_DIR/stage39_joint_gpu2.log" \
  "$LOG_DIR/stage39_joint_gpu3.log"

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

python -m MultiModal.multimodal.experiments.run_stage60_joint_gap_alpha \
  --config MultiModal/configs/stage60_joint_gap_alpha.yaml 2>&1 | tee "$LOG_DIR/stage60_joint_gap_alpha.log"

scan_logs_for_errors "stage60_joint_gap_alpha" "$LOG_DIR/stage60_joint_gap_alpha.log"
touch "$MARK_DIR/stage60_joint_gap_alpha.done"

# ---------------------------------------------------------------------------
# Step 2: Stage39 S44 method expansion + W5 regression refresh
# ---------------------------------------------------------------------------
launch_tmux_worker "pw3_s39x_g0" "$MM_ROOT/scripts/queue_postw3_stage39_expand_gpu0.sh"
launch_tmux_worker "pw3_s39x_g1" "$MM_ROOT/scripts/queue_postw3_stage39_expand_gpu1.sh"
launch_tmux_worker "pw3_s39x_g2" "$MM_ROOT/scripts/queue_postw3_stage39_expand_gpu2.sh"
launch_tmux_worker "pw3_s39x_g3" "$MM_ROOT/scripts/queue_postw3_stage39_expand_gpu3.sh"

wait_for_file "$MARK_DIR/gpu0.stage39_s44_expand.done" "stage39 expand gpu0"
wait_for_file "$MARK_DIR/gpu1.stage39_s44_expand.done" "stage39 expand gpu1"
wait_for_file "$MARK_DIR/gpu2.stage39_s44_expand.done" "stage39 expand gpu2"
wait_for_file "$MARK_DIR/gpu3.stage39_s44_expand.done" "stage39 expand gpu3"

scan_logs_for_errors "stage39_s44_expand" \
  "$LOG_DIR/stage39_s44_expand_gpu0.log" \
  "$LOG_DIR/stage39_s44_expand_gpu1.log" \
  "$LOG_DIR/stage39_s44_expand_gpu2.log" \
  "$LOG_DIR/stage39_s44_expand_gpu3.log"

python -m MultiModal.multimodal.experiments.run_w5_s44_regression \
  2>&1 | tee "$LOG_DIR/w5_s44_regression_refresh.log"

scan_logs_for_errors "w5_s44_regression_refresh" "$LOG_DIR/w5_s44_regression_refresh.log"
touch "$MARK_DIR/w5_s44_regression_refresh.done"

# ---------------------------------------------------------------------------
# Step 3: Full audio pipeline with alternative image backbone (ViT-L/14)
# ---------------------------------------------------------------------------
launch_tmux_worker "pw3_vitl_pre_g0" "$MM_ROOT/scripts/queue_postw3_vitl14_prebuild_gpu0.sh"
wait_for_file "$MARK_DIR/gpu0.vitl14_prebuild.done" "vitl14 prebuild"
scan_logs_for_errors "vitl14_prebuild" "$LOG_DIR/stage44_vitl14_prebuild_gpu0.log"

launch_tmux_worker "pw3_vitl44_g0" "$MM_ROOT/scripts/queue_postw3_vitl14_rerun_gpu0.sh"
launch_tmux_worker "pw3_vitl44_g1" "$MM_ROOT/scripts/queue_postw3_vitl14_rerun_gpu1.sh"
launch_tmux_worker "pw3_vitl44_g2" "$MM_ROOT/scripts/queue_postw3_vitl14_rerun_gpu2.sh"
launch_tmux_worker "pw3_vitl44_g3" "$MM_ROOT/scripts/queue_postw3_vitl14_rerun_gpu3.sh"
launch_tmux_worker "pw3_vitl44_g4" "$MM_ROOT/scripts/queue_postw3_vitl14_rerun_gpu4.sh"
launch_tmux_worker "pw3_vitl44_g5" "$MM_ROOT/scripts/queue_postw3_vitl14_rerun_gpu5.sh"
launch_tmux_worker "pw3_vitl44_g6" "$MM_ROOT/scripts/queue_postw3_vitl14_rerun_gpu6.sh"
launch_tmux_worker "pw3_vitl44_g7" "$MM_ROOT/scripts/queue_postw3_vitl14_rerun_gpu7.sh"
launch_tmux_worker "pw3_vitl44_coord" "$MM_ROOT/scripts/queue_postw3_vitl14_rerun_coordinator.sh"

wait_for_success_or_fail \
  "$MARK_DIR/vitl14_stage44.ready.done" \
  "$MARK_DIR/vitl14_stage44.ready.fail" \
  "vitl14 stage44 rerun readiness"

scan_logs_for_errors "vitl14_stage44_train_rerun" \
  "$LOG_DIR/stage44_vitl14_rerun_gpu0.log" \
  "$LOG_DIR/stage44_vitl14_rerun_gpu1.log" \
  "$LOG_DIR/stage44_vitl14_rerun_gpu2.log" \
  "$LOG_DIR/stage44_vitl14_rerun_gpu3.log" \
  "$LOG_DIR/stage44_vitl14_rerun_gpu4.log" \
  "$LOG_DIR/stage44_vitl14_rerun_gpu5.log" \
  "$LOG_DIR/stage44_vitl14_rerun_gpu6.log" \
  "$LOG_DIR/stage44_vitl14_rerun_gpu7.log" \
  "$LOG_DIR/stage44_vitl14_rerun_coordinator.log"

launch_tmux_worker "pw3_vitl_s39_g4" "$MM_ROOT/scripts/queue_postw3_vitl14_stage39_gpu4.sh"
launch_tmux_worker "pw3_vitl_s39_g5" "$MM_ROOT/scripts/queue_postw3_vitl14_stage39_gpu5.sh"
launch_tmux_worker "pw3_vitl_s39_g6" "$MM_ROOT/scripts/queue_postw3_vitl14_stage39_gpu6.sh"
launch_tmux_worker "pw3_vitl_s39_g7" "$MM_ROOT/scripts/queue_postw3_vitl14_stage39_gpu7.sh"

wait_for_file "$MARK_DIR/gpu4.vitl14_stage39.done" "vitl14 stage39 gpu4"
wait_for_file "$MARK_DIR/gpu5.vitl14_stage39.done" "vitl14 stage39 gpu5"
wait_for_file "$MARK_DIR/gpu6.vitl14_stage39.done" "vitl14 stage39 gpu6"
wait_for_file "$MARK_DIR/gpu7.vitl14_stage39.done" "vitl14 stage39 gpu7"

scan_logs_for_errors "vitl14_stage39_gap" \
  "$LOG_DIR/stage39_vitl14_gpu4.log" \
  "$LOG_DIR/stage39_vitl14_gpu5.log" \
  "$LOG_DIR/stage39_vitl14_gpu6.log" \
  "$LOG_DIR/stage39_vitl14_gpu7.log"

python -m MultiModal.multimodal.experiments.run_stage36_bottleneck_decomposition \
  --config MultiModal/configs/stage36_vitl14.yaml 2>&1 | tee "$LOG_DIR/stage36_vitl14.log"

scan_logs_for_errors "stage36_vitl14" "$LOG_DIR/stage36_vitl14.log"
touch "$MARK_DIR/stage36_vitl14.done"

touch "$MARK_DIR/postw3_sequence.done"
echo "[$(date '+%F %T')] post-W3 sequence complete"
