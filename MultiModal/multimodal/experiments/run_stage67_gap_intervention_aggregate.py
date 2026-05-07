from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.stats import ttest_rel

from ..common import env_snapshot, load_json, mark_done, save_json


def _collect_rows(d: dict[str, Any]) -> dict[tuple[int, str, int], dict[str, Any]]:
    out: dict[tuple[int, str, int], dict[str, Any]] = {}
    for m_key, methods in d.get("raw", {}).items():
        embed_dim = int(str(m_key).lstrip("m"))
        for method, rows in methods.items():
            for r in rows:
                seed = int(r["seed"])
                out[(embed_dim, str(method), seed)] = r
    return out


def _holm_correction(pvals: list[float]) -> list[float]:
    n = len(pvals)
    order = np.argsort(np.asarray(pvals))
    adjusted = np.zeros(n, dtype=np.float64)
    running = 0.0
    for i, idx in enumerate(order):
        val = (n - i) * pvals[idx]
        running = max(running, val)
        adjusted[idx] = min(1.0, running)
    return [float(x) for x in adjusted]


def run(cfg: dict[str, Any]) -> None:
    start = time.time()
    out_root = Path(cfg["output_root"]).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    stage_root = out_root / "stage67_gap_intervention_aggregate"
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = out_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    stage66 = load_json(Path(cfg["stage66_results"]).resolve())
    baseline = load_json(Path(cfg["baseline_stage44_results"]).resolve())

    rows66 = _collect_rows(stage66)
    rows44 = _collect_rows(baseline)

    target_method = str(cfg.get("method", "modular_shared_jl"))
    dims = [int(x) for x in cfg.get("embed_dims", [256, 512])]
    metrics = [str(x) for x in cfg.get("metrics", ["av_ia_avg_R", "av_at_avg_R"])]

    out_rows = []
    tests = []
    for d in dims:
        keys = sorted(k for k in rows66.keys() if k[0] == d and k[1] == target_method)
        if not keys:
            continue
        seeds = [k[2] for k in keys]
        for metric in metrics:
            base_vals = []
            gap_vals = []
            for _, _, s in keys:
                k = (d, target_method, s)
                if k not in rows44:
                    continue
                base_vals.append(float(rows44[k][metric]))
                gap_vals.append(float(rows66[k][metric]))
            if len(base_vals) < 2:
                continue
            b = np.asarray(base_vals, dtype=np.float64)
            g = np.asarray(gap_vals, dtype=np.float64)
            delta = g - b
            t = ttest_rel(g, b)
            tests.append(
                {
                    "embed_dim": d,
                    "method": target_method,
                    "metric": metric,
                    "n": int(len(delta)),
                    "seeds": seeds,
                    "baseline_mean": float(np.mean(b)),
                    "gapreg_mean": float(np.mean(g)),
                    "delta_mean": float(np.mean(delta)),
                    "delta_std": float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.0,
                    "p_value": float(t.pvalue) if np.isfinite(float(t.pvalue)) else 1.0,
                    "t_stat": float(t.statistic) if np.isfinite(float(t.statistic)) else 0.0,
                }
            )
            out_rows.append(
                {
                    "embed_dim": d,
                    "method": target_method,
                    "metric": metric,
                    "baseline_mean": float(np.mean(b)),
                    "gapreg_mean": float(np.mean(g)),
                    "delta_mean": float(np.mean(delta)),
                }
            )

    pvals = [float(x["p_value"]) for x in tests]
    holm = _holm_correction(pvals) if pvals else []
    for rec, ph in zip(tests, holm):
        rec["p_holm"] = float(ph)
        rec["reject_h0"] = bool(ph < 0.05)

    out = {
        "stage": "stage67_gap_intervention_aggregate",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage66_results": str(Path(cfg["stage66_results"]).resolve()),
        "baseline_stage44_results": str(Path(cfg["baseline_stage44_results"]).resolve()),
        "method": target_method,
        "rows": out_rows,
        "paired_tests": tests,
        "elapsed_sec": float(time.time() - start),
    }
    save_json(out, stage_root / "stage67_gap_intervention_aggregate.json")

    lines = [
        "# Stage67 Gap Intervention Aggregate",
        "",
        f"- method: {target_method}",
        "",
        "| m | metric | baseline_mean | gapreg_mean | delta |",
        "|---:|---|---:|---:|---:|",
    ]
    for r in out_rows:
        lines.append(
            f"| {r['embed_dim']} | {r['metric']} | {r['baseline_mean']:.5f} | "
            f"{r['gapreg_mean']:.5f} | {r['delta_mean']:.5f} |"
        )
    lines += [
        "",
        "## Paired Tests",
        "",
        "| m | metric | n | delta_mean | p | p_holm | reject |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for t in tests:
        lines.append(
            f"| {t['embed_dim']} | {t['metric']} | {t['n']} | {t['delta_mean']:.5f} | "
            f"{t['p_value']:.5g} | {t['p_holm']:.5g} | {int(t['reject_h0'])} |"
        )
    (stage_root / "stage67_gap_intervention_aggregate.md").write_text("\n".join(lines), encoding="utf-8")

    provenance = env_snapshot(
        Path(cfg["project_root"]),
        seeds=[],
        extra={
            "stage": "stage67_gap_intervention_aggregate",
            "elapsed_sec": float(time.time() - start),
            "stage66_results": str(Path(cfg["stage66_results"]).resolve()),
            "baseline_stage44_results": str(Path(cfg["baseline_stage44_results"]).resolve()),
        },
    )
    save_json(provenance, stage_root / "provenance_stage67.json")
    mark_done(markers / "stage67_gap_intervention_aggregate.done.json", {"elapsed_sec": float(time.time() - start)})
    print("stage67_gap_intervention_aggregate complete")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)
