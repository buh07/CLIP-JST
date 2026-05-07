#!/usr/bin/env bash
set -euo pipefail
cd "/jumbo/lisp/f004ndc/CLIP JST"
mkdir -p logs results/rerun_fix_20260429/markers
log="logs/rerun_fix_D4_gpu6.log"
start_ts=$(date -Iseconds)
echo "[$start_ts] START D4 rerun (GPU6)" | tee "$log"
CUDA_VISIBLE_DEVICES=6 python -u experiments/run_D4.py --config configs/_rerun_fix_D4.yaml 2>&1 | tee -a "$log"
end_ts=$(date -Iseconds)
python - <<'PY'
import json, time, pathlib
out=pathlib.Path('results/rerun_fix_20260429/markers/D4.done.json')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({
  'stage':'D4',
  'status':'done',
  'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')
}, indent=2))
print('WROTE', out)
PY
echo "[$end_ts] DONE D4 rerun" | tee -a "$log"
