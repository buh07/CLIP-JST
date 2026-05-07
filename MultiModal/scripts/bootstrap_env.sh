#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
VENV="$MM_ROOT/.venv"

if [[ ! -d "$VENV" ]]; then
  python -m venv --system-site-packages "$VENV"
fi

source "$VENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel >/dev/null

# Keep installs minimal; rely on system-site packages for torch.
python - <<'PY'
import importlib.util, subprocess, sys
pkgs = {
    "yaml": "PyYAML",
    "scipy": "scipy",
    "datasets": "datasets",
    "transformers": "transformers",
}
missing = [pip_name for mod, pip_name in pkgs.items() if importlib.util.find_spec(mod) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
print("Bootstrap complete. Missing installed:", missing)
PY
