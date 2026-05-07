from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats


def mean_std_ci(values: list[float], ci: float = 0.95) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    mean = float(arr.mean()) if n else 0.0
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    if n <= 1:
        return {
            "n": n,
            "mean": mean,
            "std": std,
            "ci95_low": mean,
            "ci95_high": mean,
        }
    alpha = 1.0 - ci
    tcrit = float(stats.t.ppf(1 - alpha / 2.0, df=n - 1))
    half = tcrit * std / math.sqrt(n)
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci95_low": float(mean - half),
        "ci95_high": float(mean + half),
    }


def paired_ttest(a: list[float], b: list[float]) -> dict[str, float]:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.size != bb.size:
        raise ValueError("paired_ttest requires equal-length arrays")
    if aa.size <= 1:
        return {"t_stat": 0.0, "p_value": 1.0}
    t_stat, p_val = stats.ttest_rel(aa, bb)
    if np.isnan(t_stat):
        t_stat = 0.0
    if np.isnan(p_val):
        p_val = 1.0
    return {"t_stat": float(t_stat), "p_value": float(p_val)}


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict[str, float | bool]]:
    """
    Returns adjusted p-values and rejection decisions per key.
    """
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict[str, float | bool]] = {}
    prev_adj = 0.0
    stopped = False
    for rank, (name, p) in enumerate(items, start=1):
        mult = m - rank + 1
        adj = min(1.0, max(prev_adj, mult * p))
        prev_adj = adj
        threshold = alpha / mult
        if not stopped and p <= threshold:
            reject = True
        else:
            stopped = True
            reject = False
        out[name] = {
            "p_raw": float(p),
            "p_holm": float(adj),
            "reject_h0": bool(reject),
            "threshold": float(threshold),
        }
    return out


def build_metric_report(
    per_method_seed_metrics: dict[str, list[dict[str, Any]]],
    metrics: list[str],
    baseline_method: str = "clip_head",
) -> dict[str, Any]:
    report: dict[str, Any] = {"methods": {}, "comparisons": {}}

    # Per-method summary.
    for method, rows in per_method_seed_metrics.items():
        report["methods"][method] = {}
        for metric in metrics:
            vals = [float(r[metric]) for r in rows if metric in r]
            report["methods"][method][metric] = mean_std_ci(vals)

    # Baseline paired deltas and p-values.
    base_rows = per_method_seed_metrics.get(baseline_method, [])
    base_by_seed = {int(r["seed"]): r for r in base_rows if "seed" in r}

    for metric in metrics:
        pvals: dict[str, float] = {}
        comp_rows: dict[str, Any] = {}

        for method, rows in per_method_seed_metrics.items():
            if method == baseline_method:
                continue
            method_by_seed = {int(r["seed"]): r for r in rows if "seed" in r}
            common = sorted(set(base_by_seed.keys()) & set(method_by_seed.keys()))
            if not common:
                continue

            base_vals = [float(base_by_seed[s][metric]) for s in common]
            meth_vals = [float(method_by_seed[s][metric]) for s in common]
            deltas = [m - b for m, b in zip(meth_vals, base_vals)]
            tt = paired_ttest(meth_vals, base_vals)
            pvals[method] = tt["p_value"]

            dsum = mean_std_ci(deltas)
            comp_rows[method] = {
                "n_paired": len(common),
                "seeds": common,
                "delta_mean": dsum["mean"],
                "delta_std": dsum["std"],
                "delta_ci95_low": dsum["ci95_low"],
                "delta_ci95_high": dsum["ci95_high"],
                "t_stat": tt["t_stat"],
                "p_value": tt["p_value"],
            }

        holm = holm_bonferroni(pvals) if pvals else {}
        for method, h in holm.items():
            if method in comp_rows:
                comp_rows[method]["p_holm"] = h["p_holm"]
                comp_rows[method]["reject_h0"] = h["reject_h0"]
                comp_rows[method]["holm_threshold"] = h["threshold"]

        report["comparisons"][metric] = comp_rows

    return report
