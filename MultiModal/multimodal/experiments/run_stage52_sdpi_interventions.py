from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..common import env_snapshot, load_json, mark_done, save_json


def _row_key(r: dict[str, Any]) -> tuple[int, str, int]:
    return int(r["embed_dim"]), str(r["method"]), int(r["seed"])


def _pair_delta(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    metric_fn,
) -> dict[str, Any]:
    a = {_row_key(r): r for r in rows_a}
    b = {_row_key(r): r for r in rows_b}
    keys = sorted(set(a.keys()) & set(b.keys()))
    vals = []
    for k in keys:
        vals.append(float(metric_fn(b[k]) - metric_fn(a[k])))
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "n_paired": int(len(arr)),
        "delta_mean": float(np.mean(arr)) if len(arr) else 0.0,
        "delta_std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "delta_values": vals,
    }


def run(cfg: dict) -> None:
    start = time.time()
    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stage_root = output_root / "stage52_sdpi_interventions"
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    s50 = load_json(Path(cfg["stage50_results_path"]).resolve())
    rows = list(s50.get("rows", []))
    if not rows:
        raise RuntimeError("Stage52: no rows in stage50")

    # Intervention A: data quality (AudioCaps vs WavCaps)
    stage30_rows = [r for r in rows if "stage30" in str(r.get("source_id", ""))]
    stage31_rows = [r for r in rows if "stage31" in str(r.get("source_id", ""))]

    quality = {
        "delta_recall_av_at": _pair_delta(stage30_rows, stage31_rows, lambda r: r["recall"]["av_at_avg_R"]),
        "delta_recall_av_ia": _pair_delta(stage30_rows, stage31_rows, lambda r: r["recall"]["av_ia_avg_R"]),
        "delta_mi_at_ksg": _pair_delta(stage30_rows, stage31_rows, lambda r: r["mi"]["ksg"]["at"]),
        "delta_mi_ia_ksg": _pair_delta(stage30_rows, stage31_rows, lambda r: r["mi"]["ksg"]["ia"]),
        "expected_direction": "negative (wavcaps should reduce bridge quality)",
    }

    # Intervention B: projection type (linear probe vs JL) within stage30
    s30_by_dim_seed = defaultdict(dict)
    for r in stage30_rows:
        key = (int(r["embed_dim"]), int(r["seed"]))
        s30_by_dim_seed[key][str(r["method"])] = r

    proj_pairs = []
    for key, mm in sorted(s30_by_dim_seed.items()):
        if "audio_linear_probe" in mm and "modular_shared_jl" in mm:
            lp = mm["audio_linear_probe"]
            jl = mm["modular_shared_jl"]
            proj_pairs.append({
                "embed_dim": key[0],
                "seed": key[1],
                "delta_recall_ia": float(lp["recall"]["av_ia_avg_R"] - jl["recall"]["av_ia_avg_R"]),
                "delta_alphaI_ksg": float(lp["alpha_i"]["ksg"] - jl["alpha_i"]["ksg"]),
                "delta_alphaI_unit_ksg": float(
                    lp.get("alpha_i_unit", {}).get("ksg", lp["alpha_i"]["ksg"])
                    - jl.get("alpha_i_unit", {}).get("ksg", jl["alpha_i"]["ksg"])
                ),
                "delta_mi_ia_ksg": float(lp["mi"]["ksg"]["ia"] - jl["mi"]["ksg"]["ia"]),
            })
    arr_recall = np.asarray([p["delta_recall_ia"] for p in proj_pairs], dtype=np.float64)
    arr_alpha = np.asarray([p["delta_alphaI_ksg"] for p in proj_pairs], dtype=np.float64)
    arr_alpha_u = np.asarray([p["delta_alphaI_unit_ksg"] for p in proj_pairs], dtype=np.float64)
    arr_mi = np.asarray([p["delta_mi_ia_ksg"] for p in proj_pairs], dtype=np.float64)
    projection = {
        "n_pairs": int(len(proj_pairs)),
        "delta_recall_ia_mean": float(np.mean(arr_recall)) if len(arr_recall) else 0.0,
        "delta_alphaI_ksg_mean": float(np.mean(arr_alpha)) if len(arr_alpha) else 0.0,
        "delta_alphaI_unit_ksg_mean": float(np.mean(arr_alpha_u)) if len(arr_alpha_u) else 0.0,
        "delta_mi_ia_ksg_mean": float(np.mean(arr_mi)) if len(arr_mi) else 0.0,
        "expected_direction": "positive (linear probe over JL)",
        "pairs": proj_pairs,
    }

    # Intervention C: sharing variants (stage25 family only)
    s25_rows = [r for r in rows if "stage25" in str(r.get("source_id", ""))]
    by_dim_seed = defaultdict(dict)
    for r in s25_rows:
        by_dim_seed[(int(r["embed_dim"]), int(r["seed"]))][str(r["method"])] = r

    sharing_rows = []
    for (m, seed), mm in sorted(by_dim_seed.items()):
        if "modular_shared_jl" in mm and "modular_separate_jl" in mm:
            sh = mm["modular_shared_jl"]
            sep = mm["modular_separate_jl"]
            sharing_rows.append({
                "embed_dim": m,
                "seed": seed,
                "delta_recall_ia_shared_minus_sep": float(sh["recall"]["av_ia_avg_R"] - sep["recall"]["av_ia_avg_R"]),
                "delta_alphaI_ksg_shared_minus_sep": float(sh["alpha_i"]["ksg"] - sep["alpha_i"]["ksg"]),
                "delta_alphaI_unit_ksg_shared_minus_sep": float(
                    sh.get("alpha_i_unit", {}).get("ksg", sh["alpha_i"]["ksg"])
                    - sep.get("alpha_i_unit", {}).get("ksg", sep["alpha_i"]["ksg"])
                ),
            })

    dim_stats = {}
    for m in sorted({r["embed_dim"] for r in sharing_rows}):
        vals_r = np.asarray([r["delta_recall_ia_shared_minus_sep"] for r in sharing_rows if r["embed_dim"] == m], dtype=np.float64)
        vals_a = np.asarray([r["delta_alphaI_ksg_shared_minus_sep"] for r in sharing_rows if r["embed_dim"] == m], dtype=np.float64)
        vals_au = np.asarray(
            [r["delta_alphaI_unit_ksg_shared_minus_sep"] for r in sharing_rows if r["embed_dim"] == m], dtype=np.float64
        )
        dim_stats[str(m)] = {
            "n": int(len(vals_r)),
            "delta_recall_ia_mean": float(np.mean(vals_r)) if len(vals_r) else 0.0,
            "delta_alphaI_ksg_mean": float(np.mean(vals_a)) if len(vals_a) else 0.0,
            "delta_alphaI_unit_ksg_mean": float(np.mean(vals_au)) if len(vals_au) else 0.0,
        }

    sharing = {
        "n_pairs": int(len(sharing_rows)),
        "dimension_conditioned_effects": dim_stats,
        "pairs": sharing_rows,
        "note": "expected to be dimension-dependent (non-uniform sign).",
    }

    out = {
        "stage": "stage52_sdpi_interventions",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_rows_input": len(rows),
        "quality_intervention": quality,
        "projection_intervention": projection,
        "sharing_intervention": sharing,
        "elapsed_sec": float(time.time() - start),
    }

    save_json(out, stage_root / "stage52_sdpi_interventions.json")

    md = [
        "# Stage52 SDPI Intervention Sanity",
        "",
        f"Input rows: {len(rows)}",
        "",
        "## A) WavCaps vs AudioCaps",
        f"- delta av_at mean: {quality['delta_recall_av_at']['delta_mean']:.6f}",
        f"- delta av_ia mean: {quality['delta_recall_av_ia']['delta_mean']:.6f}",
        f"- delta MI_at(ksg) mean: {quality['delta_mi_at_ksg']['delta_mean']:.6f}",
        f"- delta MI_ia(ksg) mean: {quality['delta_mi_ia_ksg']['delta_mean']:.6f}",
        "",
        "## B) Linear Probe vs JL (Stage30)",
        f"- delta av_ia mean: {projection['delta_recall_ia_mean']:.6f}",
        f"- delta alphaI_ksg mean: {projection['delta_alphaI_ksg_mean']:.6f}",
        f"- delta alphaI_unit_ksg mean: {projection['delta_alphaI_unit_ksg_mean']:.6f}",
        "",
        "## C) Sharing Effects (Stage25)",
        "| m | n | delta av_ia (shared-sep) | delta alphaI_ksg (shared-sep) | delta alphaI_unit_ksg (shared-sep) |",
        "|---|---:|---:|---:|---:|",
    ]
    for m, d in dim_stats.items():
        md.append(
            f"| {m} | {d['n']} | {d['delta_recall_ia_mean']:.6f} | {d['delta_alphaI_ksg_mean']:.6f} | "
            f"{d['delta_alphaI_unit_ksg_mean']:.6f} |"
        )
    (stage_root / "stage52_sdpi_interventions.md").write_text("\n".join(md), encoding="utf-8")

    provenance = env_snapshot(Path(cfg["project_root"]), seeds=[], extra={"stage": "stage52_sdpi_interventions", "elapsed_sec": float(time.time() - start)})
    save_json(provenance, stage_root / "provenance_stage52.json")
    mark_done(markers / "stage52_sdpi_interventions.done.json", {"elapsed_sec": float(time.time() - start), "n_rows": len(rows)})
    print("[stage52] complete")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    run(cfg)


if __name__ == "__main__":
    main()
