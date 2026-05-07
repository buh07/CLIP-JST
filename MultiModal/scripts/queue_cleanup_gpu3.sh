#!/usr/bin/env bash
set -euo pipefail
ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
OUT="$MM_ROOT/results/neurips_strengthen_suite"
LOG_DIR="$MM_ROOT/logs/neurips_cleanup"
MARK="$OUT/markers/cleanup"
mkdir -p "$LOG_DIR" "$MARK"

export CUDA_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
python -m pip install -q 'numpy<2'
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

until [[ -f "$MARK/gpu0.stage38.cleanup.done" && -f "$MARK/gpu2.stage40.cleanup.done" ]]; do
  echo "[$(date '+%F %T')] waiting for stage38 + stage40..."
  sleep 30
done

python - <<'PY' | tee "$LOG_DIR/cleanup_validation_gpu3.log"
import json, pathlib
root=pathlib.Path('/jumbo/lisp/f004ndc/CLIP JST/MultiModal/results/neurips_strengthen_suite')

s37=json.load(open(root/'stage37_imagebind_comparison'/'stage37_imagebind_comparison_results.json'))
s38=json.load(open(root/'stage38_phaseb_quality_analysis'/'stage38_phaseb_quality_analysis_results.json'))
s40=json.load(open(root/'stage40_strengthen_suite_aggregate'/'stage40_strengthen_suite_aggregate.json'))

summary={
 'stage37_relabeled_present': all(k in s37 for k in ['image_audio_relabeled','audio_text_relabeled','reporting_caveats']),
 'stage38_clean_present': 'wavcaps_clean_source' in s38 and 'delta_wav_clean_minus_audio' in s38,
 'stage38_mixed_delta_margin': s38.get('delta_wav_minus_audio',{}).get('margin_mean'),
 'stage38_clean_delta_margin': s38.get('delta_wav_clean_minus_audio',{}).get('margin_mean'),
 'stage40_true_joint_rows': len(s40.get('true_joint_reference_rows',[])),
}
print(json.dumps(summary, indent=2))
with open(root/'cleanup_validation_summary.json','w') as f:
 json.dump(summary,f,indent=2)
PY

touch "$MARK/gpu3.validation.cleanup.done"
echo "[$(date '+%F %T')] queue_cleanup_gpu3 complete"
