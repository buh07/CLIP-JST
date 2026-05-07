from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def save_json(obj: Any, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path | str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def git_hash(root: Path | str) -> str:
    root = Path(root)
    try:
        out = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        return out
    except Exception:
        return "unknown"


def pip_freeze(python_exe: str = sys.executable) -> list[str]:
    try:
        out = subprocess.check_output([python_exe, "-m", "pip", "freeze"], text=True)
        return [line for line in out.splitlines() if line.strip()]
    except Exception:
        return []


def env_snapshot(project_root: Path | str, seeds: list[int], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    snap = {
        "timestamp": now_iso(),
        "project_root": str(Path(project_root).resolve()),
        "git_hash": git_hash(project_root),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "seeds": seeds,
        "pip_freeze": pip_freeze(),
    }
    if extra:
        snap.update(extra)
    return snap


def mark_done(path: Path | str, payload: dict[str, Any] | None = None) -> None:
    path = Path(path)
    data = {"done": True, "timestamp": now_iso()}
    if payload:
        data.update(payload)
    save_json(data, path)


def elapsed(start_time: float) -> float:
    return time.time() - start_time
