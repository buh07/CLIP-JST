from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.stats import linregress

from ..common import env_snapshot, load_json, mark_done, save_json
from ..eval.stats import build_metric_report, mean_std_ci


METRICS = ["combined_avg_R", "coco_avg_R", "av_it_avg_R", "av_at_avg_R", "av_ia_avg_R", "theoretical_ceiling", "efficiency"]


def _merge_seed_rows(dst_rows: list[dict[str, Any]], src_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_seed = {int(r["seed"]): r for r in dst_rows if "seed" in r}
    for r in src_rows:
        by_seed[int(r["seed"])] = r
    return [by_seed[s] for s in sorted(by_seed)]


def _merge_stage20_from_roots(roots: list[Path]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    merged: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for root in roots:
        p = root / "stage20_modular_audio_transitivity" / "stage20_results.json"
        if not p.exists():
            continue
        obj = load_json(p)
        if obj.get("skipped"):
            continue
        for m_key, methods in obj.get("raw", {}).items():
            merged.setdefault(m_key, {})
            for method, rows in methods.items():
                cur = merged[m_key].get(method, [])
                merged[m_key][method] = _merge_seed_rows(cur, rows)
    return merged


def _attach_decomposition(raw: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for m_key, methods in raw.items():
        out[m_key] = {}
        for method, rows in methods.items():
            new_rows = []
            for r in rows:
                av_it = float(r["av_it_avg_R"])
                av_at = float(r["av_at_avg_R"])
                av_ia = float(r["av_ia_avg_R"])
                ceiling = math.sqrt(max(0.0, av_it * av_at))
                eff = av_ia / max(1e-12, ceiling)
                rr = dict(r)
                rr["theoretical_ceiling"] = float(ceiling)
                rr["efficiency"] = float(eff)
                new_rows.append(rr)
            out[m_key][method] = new_rows
    return out


def _regression_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) < 2:
        return {"n": len(rows), "status": "insufficient"}
    x = np.asarray([float(r["theoretical_ceiling"]) for r in rows], dtype=float)
    y = np.asarray([float(r["av_ia_avg_R"]) for r in rows], dtype=float)
    if np.allclose(x, x[0]):
        return {"n": len(rows), "status": "degenerate_x"}
    lr = linregress(x, y)
    return {
        "n": int(len(rows)),
        "slope": float(lr.slope),
        "intercept": float(lr.intercept),
        "r_value": float(lr.rvalue),
        "r2": float(lr.rvalue ** 2),
        "p_value": float(lr.pvalue),
        "stderr": float(lr.stderr),
    }


def run(cfg: dict) -> None:
    start = time.time()
    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stage_root = output_root / "stage27_bottleneck_decomposition"
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    roots = [Path(p).resolve() for p in cfg.get("stage20_source_roots", [])]
    raw_merged = _merge_stage20_from_roots(roots)
    raw = _attach_decomposition(raw_merged)

    baseline = str(cfg.get("baseline_method", "modular_shared_jl"))
    stats: dict[str, Any] = {}
    for m_key, methods in raw.items():
        stats[m_key] = build_metric_report(methods, metrics=METRICS, baseline_method=baseline if baseline in methods else next(iter(methods)))

    # regressions by (dim, method) and global by method.
    regressions: dict[str, Any] = {"by_dim_method": {}, "global_by_method": {}}
    all_by_method: dict[str, list[dict[str, Any]]] = {}
    for m_key, methods in raw.items():
        regressions["by_dim_method"][m_key] = {}
        for method, rows in methods.items():
            regressions["by_dim_method"][m_key][method] = _regression_block(rows)
            all_by_method.setdefault(method, []).extend(rows)

    for method, rows in all_by_method.items():
        regressions["global_by_method"][method] = _regression_block(rows)

    # Additional aggregate over efficiency by method across dims.
    efficiency_global: dict[str, Any] = {}
    for method, rows in all_by_method.items():
        eff_vals = [float(r["efficiency"]) for r in rows]
        ceiling_vals = [float(r["theoretical_ceiling"]) for r in rows]
        actual_vals = [float(r["av_ia_avg_R"]) for r in rows]
        efficiency_global[method] = {
            "efficiency": mean_std_ci(eff_vals),
            "ceiling": mean_std_ci(ceiling_vals),
            "actual": mean_std_ci(actual_vals),
        }

    out = {
        "stage": "stage27_bottleneck_decomposition",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_roots": [str(p) for p in roots],
        "raw": raw,
        "stats": stats,
        "regressions": regressions,
        "efficiency_global": efficiency_global,
        "elapsed_sec": time.time() - start,
    }
    save_json(out, stage_root / "stage27_bottleneck_decomposition_results.json")

    md = [
        "# Stage27 Bottleneck Decomposition",
        "",
        f"- Generated: `{out['generated_at']}`",
        "",
        "## Global Efficiency",
    ]
    for method, blk in sorted(efficiency_global.items()):
        em = blk["efficiency"]["mean"]
        ec0 = blk["efficiency"]["ci95_low"]
        ec1 = blk["efficiency"]["ci95_high"]
        am = blk["actual"]["mean"]
        cm = blk["ceiling"]["mean"]
        md.append(f"- {method}: efficiency={em:.4f} CI[{ec0:.4f},{ec1:.4f}] actual={am:.4f} ceiling={cm:.4f}")
    md.append("")
    md.append("## Global Regressions (actual vs ceiling)")
    for method, rg in sorted(regressions["global_by_method"].items()):
        if rg.get("status"):
            md.append(f"- {method}: {rg['status']} (n={rg['n']})")
        else:
            md.append(f"- {method}: slope={rg['slope']:.3f}, intercept={rg['intercept']:.3f}, R^2={rg['r2']:.3f}, p={rg['p_value']:.3g}, n={rg['n']}")
    (stage_root / "stage27_bottleneck_decomposition_results.md").write_text("\n".join(md), encoding="utf-8")

    provenance = env_snapshot(
        Path(cfg["project_root"]),
        seeds=[],
        extra={"stage": "stage27_bottleneck_decomposition", "elapsed_sec": time.time() - start},
    )
    save_json(provenance, stage_root / "provenance_stage27.json")
    mark_done(markers / "stage27_bottleneck_decomposition.done.json", {"elapsed_sec": time.time() - start})
    print("Stage27 complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)
