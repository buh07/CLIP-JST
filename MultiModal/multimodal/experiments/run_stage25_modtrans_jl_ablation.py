from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

from ..common import env_snapshot, mark_done, save_json
from .run_stage20_modular_audio_transitivity import run as run_stage20


def run(cfg: dict) -> None:
    start = time.time()
    run_stage20(cfg)

    output_root = Path(cfg["output_root"]).resolve()
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    stage_summary = {
        "stage": "stage25_modtrans_jl_ablation",
        "timestamp": time.time(),
        "output_root": str(output_root),
        "elapsed_sec": time.time() - start,
        "methods": list(cfg.get("methods", [])),
        "embed_dims": list(cfg.get("embed_dims", [])),
        "seeds": list(cfg.get("seeds", [])),
        "clap_model": cfg.get("clap_model"),
    }
    save_json(stage_summary, output_root / "stage25_summary.json")

    provenance = env_snapshot(
        Path(cfg["project_root"]),
        seeds=[int(s) for s in cfg.get("seeds", [])],
        extra={
            "stage": "stage25_modtrans_jl_ablation",
            "elapsed_sec": time.time() - start,
        },
    )
    save_json(provenance, output_root / "provenance_stage25.json")
    mark_done(markers / "stage25_modtrans_jl_ablation.done.json", {"elapsed_sec": time.time() - start})
    print("Stage 25 complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)
