from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import yaml
from scipy.stats import pearsonr

from ..common import env_snapshot, load_json, mark_done, save_json


def _collect_stage69(root: Path) -> dict[str, Any]:
    p = root / "stage69_third_triple_speechcoco" / "stage69_third_triple_speechcoco_results.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing stage69 results: {p}")
    return load_json(p)


def _merge_raw(stage69_blocks: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    merged: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    for b in stage69_blocks:
        for m_key, methods in b.get("raw", {}).items():
            dst_m = merged.setdefault(m_key, {})
            for method, rows in methods.items():
                dst_r = dst_m.setdefault(method, {})
                for r in rows:
                    dst_r[int(r["seed"])] = r
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for m_key, methods in merged.items():
        out[m_key] = {}
        for method, by_seed in methods.items():
            out[m_key][method] = [by_seed[s] for s in sorted(by_seed)]
    return out


def _fit_alpha(rows: list[dict[str, Any]]) -> dict[str, float]:
    x = []
    y = []
    for r in rows:
        it = float(r["av_it_avg_R"])
        at = float(r["av_at_avg_R"])
        ia = float(r["av_ia_avg_R"])
        pred = math.sqrt(max(0.0, it * at))
        if pred <= 0:
            continue
        x.append(pred)
        y.append(ia)

    if len(x) < 3:
        return {"n": len(x), "alpha": 0.0, "pearson_r": 0.0, "pearson_p": 1.0, "r2": 0.0, "mae": 0.0}

    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    alpha = float(np.dot(xv, yv) / max(1e-12, np.dot(xv, xv)))
    yhat = alpha * xv
    rr, pp = pearsonr(yv, yhat)
    ss_res = float(np.sum((yv - yhat) ** 2))
    ss_tot = float(np.sum((yv - np.mean(yv)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else 0.0)
    mae = float(np.mean(np.abs(yv - yhat)))
    return {
        "n": int(len(x)),
        "alpha": alpha,
        "pearson_r": float(rr),
        "pearson_p": float(pp),
        "r2": float(r2),
        "mae": float(mae),
    }


def run(cfg: dict[str, Any]) -> None:
    start = time.time()

    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stage_root = output_root / "stage70_third_triple_aggregate"
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    roots = [Path(x).resolve() for x in cfg["stage69_shard_roots"]]
    blocks = [_collect_stage69(r) for r in roots]
    raw = _merge_raw(blocks)

    summary_rows = []
    all_rows = []
    disjoint_rows = []

    for m_key, methods in raw.items():
        for method, rows in methods.items():
            if not rows:
                continue
            vals = {
                "av_it_avg_R": [float(r["av_it_avg_R"]) for r in rows],
                "av_at_avg_R": [float(r["av_at_avg_R"]) for r in rows],
                "av_ia_avg_R": [float(r["av_ia_avg_R"]) for r in rows],
                "combined_avg_R": [float(r["combined_avg_R"]) for r in rows],
            }
            row = {
                "embed_dim": int(m_key[1:]),
                "method": method,
                "n": len(rows),
            }
            for k, arr in vals.items():
                row[f"{k}_mean"] = mean(arr)
                row[f"{k}_std"] = pstdev(arr) if len(arr) > 1 else 0.0
            summary_rows.append(row)
            all_rows.extend(rows)
            disjoint_rows.extend([r.get("disjoint_diagnostics", {}) for r in rows])

    law_global = _fit_alpha(all_rows)
    law_by_method = {}
    for method in sorted({r["method"] for r in all_rows}):
        law_by_method[method] = _fit_alpha([r for r in all_rows if r["method"] == method])

    out = {
        "stage": "stage70_third_triple_aggregate",
        "stage69_shard_roots": [str(r) for r in roots],
        "summary_rows": sorted(summary_rows, key=lambda x: (x["embed_dim"], x["method"])),
        "law_global": law_global,
        "law_by_method": law_by_method,
        "n_seed_rows": int(len(all_rows)),
        "disjoint_diagnostics_any": disjoint_rows[0] if disjoint_rows else {},
        "elapsed_sec": time.time() - start,
    }
    save_json(out, stage_root / "stage70_third_triple_aggregate.json")

    md = [
        "# Stage70 Third Triple Aggregate",
        "",
        "## Law Fit (Global)",
        "",
        f"- n: {law_global['n']}",
        f"- alpha: {law_global['alpha']:.4f}",
        f"- pearson_r: {law_global['pearson_r']:.4f}",
        f"- r2: {law_global['r2']:.4f}",
        f"- mae: {law_global['mae']:.4f}",
        "",
        "## Method/Dim Summary",
        "",
        "| m | method | n | av_it | av_at | av_ia | combined |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in out["summary_rows"]:
        md.append(
            f"| {r['embed_dim']} | {r['method']} | {r['n']} | "
            f"{r['av_it_avg_R_mean']:.4f}±{r['av_it_avg_R_std']:.4f} | "
            f"{r['av_at_avg_R_mean']:.4f}±{r['av_at_avg_R_std']:.4f} | "
            f"{r['av_ia_avg_R_mean']:.4f}±{r['av_ia_avg_R_std']:.4f} | "
            f"{r['combined_avg_R_mean']:.4f}±{r['combined_avg_R_std']:.4f} |"
        )

    (stage_root / "stage70_third_triple_aggregate.md").write_text("\n".join(md), encoding="utf-8")

    provenance = env_snapshot(
        Path(cfg["project_root"]),
        seeds=[],
        extra={"stage": "stage70_third_triple_aggregate", "elapsed_sec": time.time() - start},
    )
    save_json(provenance, stage_root / "provenance_stage70.json")
    mark_done(markers / "stage70_third_triple_aggregate.done.json", {"elapsed_sec": time.time() - start})
    print("stage70_third_triple_aggregate complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)
