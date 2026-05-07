from __future__ import annotations

import argparse
import fcntl
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from ..common import env_snapshot, load_json, mark_done, save_json, set_seed
from ..data import AudioCapsAVCache
from .sdpi_common import (
    canonical_json_hash,
    cond_output_relpath,
    export_embeddings_for_condition,
)


def _merge_by_condition(old_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(r["condition_id"]): r for r in old_rows if "condition_id" in r}
    for r in new_rows:
        by_id[str(r["condition_id"])] = r
    return [by_id[k] for k in sorted(by_id.keys())]


def _save_shard_safe(out_path: Path, incoming: dict[str, Any]) -> None:
    lock_path = out_path.with_suffix(out_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        if out_path.exists():
            try:
                cur = load_json(out_path)
            except Exception:
                cur = {}
        else:
            cur = {}

        merged = dict(cur) if isinstance(cur, dict) else {}
        merged["stage"] = "stage48_sdpi_embed_export"
        merged["manifest_schema_hash"] = incoming.get("manifest_schema_hash")
        merged["filters"] = incoming.get("filters", {})
        merged["rows"] = _merge_by_condition(
            list(cur.get("rows", [])) if isinstance(cur, dict) else [],
            list(incoming.get("rows", [])),
        )
        save_json(merged, out_path)
        fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)


def run(cfg: dict) -> None:
    start = time.time()
    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stage_root = output_root / "stage48_sdpi_embed_export"
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    manifest = load_json(Path(cfg["manifest_path"]).resolve())
    manifest_schema = str(manifest.get("schema_hash", "unknown"))
    conditions = list(manifest.get("conditions", []))

    dims_filter = {int(x) for x in cfg.get("embed_dims", [])}
    methods_filter = set(cfg.get("methods", []))
    seeds_filter = {int(s) for s in cfg.get("seeds", [])}

    def _keep(c: dict[str, Any]) -> bool:
        if dims_filter and int(c["embed_dim"]) not in dims_filter:
            return False
        if methods_filter and str(c["method"]) not in methods_filter:
            return False
        if seeds_filter and int(c["seed"]) not in seeds_filter:
            return False
        if not bool(c.get("has_checkpoint", False)):
            return False
        return True

    conditions = [c for c in conditions if _keep(c)]

    av_dir = Path(cfg["av_cache_root"]).resolve()
    av = AudioCapsAVCache.from_paths(
        av_dir / "image_feats_clip_raw.pt",
        av_dir / "audio_feats_clap_raw.pt",
        av_dir / "text_feats_clip_raw.pt",
        av_dir / "metadata.json",
    )

    device = str(cfg.get("device", "cuda"))
    batch_size = int(cfg.get("eval_batch_size", 2048))
    parity_tol = float(cfg.get("parity_tol", 1e-4))
    store_dtype = str(cfg.get("store_dtype", "float16"))

    rows_out: list[dict[str, Any]] = []

    for cond in conditions:
        set_seed(int(cond.get("seed", 0)))
        rel = cond_output_relpath(cond)
        out_dir = stage_root / rel
        emb_p = out_dir / "embeds.pt"
        meta_p = out_dir / "meta.json"

        if emb_p.exists() and meta_p.exists():
            meta = load_json(meta_p)
            rows_out.append(meta)
            continue

        pack = export_embeddings_for_condition(
            cond,
            cfg,
            av,
            device=device,
            batch_size=batch_size,
            store_dtype=store_dtype,
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "zi": pack["zi"],
                "za": pack["za"],
                "zt": pack["zt"],
            },
            emb_p,
        )

        parity = pack["parity"]
        meta = {
            "condition_id": cond["condition_id"],
            "source_id": cond["source_id"],
            "source_group": cond.get("source_group", cond["source_id"]),
            "stage_name": cond["stage_name"],
            "embed_dim": int(cond["embed_dim"]),
            "method": cond["method"],
            "seed": int(cond["seed"]),
            "phase_a_source": cond.get("phase_a_source"),
            "phase_b_source": cond.get("phase_b_source"),
            "centroid_gap_ia_l2": cond.get("centroid_gap_ia_l2"),
            "schema_hash": canonical_json_hash({
                "condition": cond["condition_id"],
                "store_dtype": store_dtype,
                "parity_src": cond.get("metrics", {}),
            }),
            "embedding_path": str(emb_p),
            "n_samples": int(pack["zi"].shape[0]),
            "shape": {
                "zi": list(pack["zi"].shape),
                "za": list(pack["za"].shape),
                "zt": list(pack["zt"].shape),
            },
            "dtype": str(pack["zi"].dtype).replace("torch.", ""),
            "recomputed_recall": pack["recall"],
            "source_recall": {
                "av_it_avg_R": float(cond.get("metrics", {}).get("av_it_avg_R", 0.0)),
                "av_at_avg_R": float(cond.get("metrics", {}).get("av_at_avg_R", 0.0)),
                "av_ia_avg_R": float(cond.get("metrics", {}).get("av_ia_avg_R", 0.0)),
            },
            "parity_abs_delta": parity,
            "parity_pass": bool(
                parity["delta_av_it"] <= parity_tol
                and parity["delta_av_at"] <= parity_tol
                and parity["delta_av_ia"] <= parity_tol
            ),
        }
        save_json(meta, meta_p)
        rows_out.append(meta)

        print(
            f"[stage48] {cond['condition_id']} n={meta['n_samples']} "
            f"deltas=({parity['delta_av_it']:.2e},{parity['delta_av_at']:.2e},{parity['delta_av_ia']:.2e})"
        )

    incoming = {
        "manifest_schema_hash": manifest_schema,
        "filters": {
            "embed_dims": sorted(dims_filter),
            "methods": sorted(methods_filter),
            "seeds": sorted(seeds_filter),
        },
        "rows": rows_out,
    }

    _save_shard_safe(stage_root / "stage48_sdpi_embed_export_results.json", incoming)

    n_fail = sum(1 for r in rows_out if not bool(r.get("parity_pass", False)))

    provenance = env_snapshot(
        Path(cfg["project_root"]),
        seeds=sorted(seeds_filter) if seeds_filter else [],
        extra={
            "stage": "stage48_sdpi_embed_export",
            "elapsed_sec": float(time.time() - start),
            "manifest_schema_hash": manifest_schema,
            "n_rows": len(rows_out),
            "n_parity_fail": n_fail,
        },
    )
    save_json(provenance, stage_root / "provenance_stage48.json")
    mark_done(markers / "stage48_sdpi_embed_export.done.json", {
        "elapsed_sec": float(time.time() - start),
        "manifest_schema_hash": manifest_schema,
        "n_rows": len(rows_out),
        "n_parity_fail": n_fail,
    })

    print(f"[stage48] complete rows={len(rows_out)} parity_fail={n_fail}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    run(cfg)


if __name__ == "__main__":
    main()
