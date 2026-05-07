from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

from ..common import env_snapshot, mark_done, save_json
from .sdpi_common import canonical_json_hash, collect_source_rows, manifest_summary


def run(cfg: dict) -> None:
    start = time.time()
    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stage_root = output_root / "stage47_sdpi_manifest"
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    dims = {int(x) for x in cfg.get("embed_dims", [])}
    methods = set(cfg.get("methods", []))
    seeds = {int(s) for s in cfg.get("seeds", [])}

    rows = collect_source_rows(
        cfg.get("source_roots", []),
        embed_dims=dims or None,
        methods=methods or None,
        seeds=seeds or None,
    )
    summary = manifest_summary(rows)

    schema = {
        "embed_dims": sorted(dims),
        "methods": sorted(methods),
        "seeds": sorted(seeds),
        "source_ids": sorted({r["source_id"] for r in rows}),
        "n": len(rows),
    }
    schema_hash = canonical_json_hash(schema)

    missing_rows = [r for r in rows if not r.get("has_checkpoint", False)]

    manifest = {
        "stage": "stage47_sdpi_manifest",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "schema_hash": schema_hash,
        "summary": summary,
        "filters": {
            "embed_dims": sorted(dims),
            "methods": sorted(methods),
            "seeds": sorted(seeds),
        },
        "conditions": rows,
    }

    save_json(manifest, stage_root / "stage47_manifest.json")
    save_json({"stage": "stage47_sdpi_manifest", "missing_conditions": missing_rows}, stage_root / "stage47_missing_conditions.json")

    provenance = env_snapshot(
        Path(cfg["project_root"]),
        seeds=sorted(seeds),
        extra={
            "stage": "stage47_sdpi_manifest",
            "elapsed_sec": float(time.time() - start),
            "schema_hash": schema_hash,
            "n_conditions": len(rows),
            "n_missing_checkpoints": len(missing_rows),
        },
    )
    save_json(provenance, stage_root / "provenance_stage47.json")

    mark_done(markers / "stage47_sdpi_manifest.done.json", {
        "elapsed_sec": float(time.time() - start),
        "schema_hash": schema_hash,
        "n_conditions": len(rows),
        "n_missing_checkpoints": len(missing_rows),
    })

    print(
        f"[stage47] manifest complete: n_conditions={len(rows)} "
        f"missing_ckpt={len(missing_rows)} schema={schema_hash}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    run(cfg)


if __name__ == "__main__":
    main()
