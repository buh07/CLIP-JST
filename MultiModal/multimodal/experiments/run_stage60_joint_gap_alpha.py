from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.stats import pearsonr

from ..common import env_snapshot, load_json, mark_done, save_json


def _fit_alpha(rows: list[dict[str, Any]]) -> dict[str, float]:
    x = []
    y = []
    for r in rows:
        it = float(r["av_it_avg_R"])
        at = float(r["av_at_avg_R"])
        ia = float(r["av_ia_avg_R"])
        x.append(math.sqrt(max(0.0, it * at)))
        y.append(ia)
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    if len(xv) == 0:
        return {"n": 0, "alpha": 0.0, "pearson_r": 0.0, "pearson_p": 1.0, "r2": 0.0, "mae": 0.0}
    alpha = float(np.dot(xv, yv) / max(1e-12, np.dot(xv, xv)))
    yhat = alpha * xv
    if len(xv) > 1:
        r, p = pearsonr(yv, yhat)
        ss_res = float(np.sum((yv - yhat) ** 2))
        ss_tot = float(np.sum((yv - np.mean(yv)) ** 2))
        r2 = float(1.0 - (ss_res / max(1e-12, ss_tot)))
    else:
        r, p, r2 = 0.0, 1.0, 0.0
    mae = float(np.mean(np.abs(yv - yhat)))
    return {
        "n": int(len(xv)),
        "alpha": alpha,
        "pearson_r": float(r),
        "pearson_p": float(p),
        "r2": r2,
        "mae": mae,
    }


def _flatten_stage39(stage39: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    raw = stage39.get("raw", {})
    for m_key, methods in raw.items():
        dim = int(str(m_key).lstrip("m"))
        for method, recs in methods.items():
            for r in recs:
                out.append(
                    {
                        "method": method,
                        "embed_dim": dim,
                        "seed": int(r["seed"]),
                        "av_it_avg_R": float(r["av_it_avg_R"]),
                        "av_at_avg_R": float(r["av_at_avg_R"]),
                        "av_ia_avg_R": float(r["av_ia_avg_R"]),
                        "gap_ia_l2": float(r["gap_ia_l2"]),
                    }
                )
    return out


def run(cfg: dict[str, Any]) -> None:
    start = time.time()
    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stage_root = output_root / "stage60_joint_gap_alpha"
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    stage39_path = Path(cfg["joint_stage39_results"]).resolve()
    stage39 = load_json(stage39_path)
    rows = _flatten_stage39(stage39)
    methods = list(cfg.get("methods", ["joint_shared_jl", "joint_clip_head"]))
    rows = [r for r in rows if r["method"] in methods]

    law_global = _fit_alpha(rows)
    by_method: dict[str, Any] = {}
    by_dim: dict[str, Any] = {}

    for m in methods:
        r_m = [r for r in rows if r["method"] == m]
        if not r_m:
            continue
        fit = _fit_alpha(r_m)
        gaps = np.asarray([float(r["gap_ia_l2"]) for r in r_m], dtype=np.float64)
        fit["gap_ia_mean"] = float(np.mean(gaps))
        fit["gap_ia_std"] = float(np.std(gaps))
        by_method[m] = fit

    for d in sorted({int(r["embed_dim"]) for r in rows}):
        r_d = [r for r in rows if int(r["embed_dim"]) == d]
        by_dim[f"m{d}"] = _fit_alpha(r_d)

    # Gap↔alpha relationship (method-level alpha assigned back to each row).
    alpha_map = {k: float(v["alpha"]) for k, v in by_method.items()}
    g = np.asarray([float(r["gap_ia_l2"]) for r in rows if r["method"] in alpha_map], dtype=np.float64)
    a = np.asarray([alpha_map[r["method"]] for r in rows if r["method"] in alpha_map], dtype=np.float64)
    if len(g) > 1 and np.std(g) > 0 and np.std(a) > 0:
        rg, pg = pearsonr(g, a)
        gap_alpha = {"pearson_r": float(rg), "pearson_p": float(pg), "n": int(len(g))}
    else:
        gap_alpha = {"pearson_r": 0.0, "pearson_p": 1.0, "n": int(len(g))}

    result = {
        "stage": "stage60_joint_gap_alpha",
        "joint_stage39_results": str(stage39_path),
        "methods": methods,
        "n_rows": int(len(rows)),
        "law_global": law_global,
        "by_method": by_method,
        "by_embed_dim": by_dim,
        "gap_to_alpha": gap_alpha,
        "rows": rows,
        "elapsed_sec": float(time.time() - start),
    }
    save_json(result, stage_root / "stage60_joint_gap_alpha.json")

    md = [
        "# Stage 60 Joint Gap+Alpha",
        "",
        f"- n rows: {result['n_rows']}",
        f"- global alpha: {law_global['alpha']:.4f}",
        f"- global r: {law_global['pearson_r']:.4f}",
        f"- global r2: {law_global['r2']:.4f}",
        f"- gap↔alpha r: {gap_alpha['pearson_r']:.4f}",
        "",
        "| method | alpha | r | r2 | gap_ia mean | n |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m in methods:
        if m not in by_method:
            continue
        b = by_method[m]
        md.append(
            f"| {m} | {b['alpha']:.4f} | {b['pearson_r']:.4f} | {b['r2']:.4f} | {b['gap_ia_mean']:.4f} | {b['n']} |"
        )
    (stage_root / "stage60_joint_gap_alpha.md").write_text("\n".join(md), encoding="utf-8")

    provenance = env_snapshot(
        Path(cfg["project_root"]).resolve(),
        seeds=sorted({int(r["seed"]) for r in rows}),
        extra={"stage": "stage60_joint_gap_alpha", "elapsed_sec": float(time.time() - start)},
    )
    save_json(provenance, stage_root / "provenance_stage60.json")
    mark_done(markers / "stage60_joint_gap_alpha.done.json", {"elapsed_sec": float(time.time() - start)})
    print("stage60_joint_gap_alpha complete")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

