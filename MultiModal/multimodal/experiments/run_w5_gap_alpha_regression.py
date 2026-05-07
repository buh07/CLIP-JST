"""W5: Gap_ia → α regression.

Loads per-method, per-dim, per-seed (gap_ia_l2, av_ia_avg_R) from Stage 39 and
per-method α from Stage 36.  Runs OLS linear regression of gap_ia_l2 → α across
methods and dims (pooled seeds), and saves results to results/reviewer_fixes_suite/.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from ..common import env_snapshot, load_json, save_json


_MM_ROOT = Path(__file__).resolve().parents[2]  # MultiModal/

_STAGE36_PATH = (
    _MM_ROOT
    / "results/semantic_full_aggregate/stage36_bottleneck_decomposition"
    / "stage36_bottleneck_decomposition.json"
)

_STAGE39_PATH = (
    _MM_ROOT
    / "results/neurips_strengthen_suite/stage39_modality_gap_linear_vs_jl"
    / "stage39_modality_gap_linear_vs_jl_results.json"
)

_OUTPUT_DIR = (
    _MM_ROOT
    / "results/reviewer_fixes_suite/w5_gap_alpha_regression"
)


def _ols_nointercept(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """OLS regression y = β*x (no intercept)."""
    beta = float(np.dot(x, y) / np.dot(x, x))
    y_pred = beta * x
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"beta": beta, "r2": r2, "ss_res": ss_res, "ss_tot": ss_tot}


def _ols_intercept(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """OLS regression y = a + b*x."""
    X = np.column_stack([np.ones_like(x), x])
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    a, b = float(coeffs[0]), float(coeffs[1])
    y_pred = a + b * x
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else float("nan")
    return {"intercept": a, "slope": b, "r2": r2, "pearson_r": corr, "ss_res": ss_res}


def run() -> None:
    t0 = time.time()
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stage36 = load_json(_STAGE36_PATH)
    stage39 = load_json(_STAGE39_PATH)

    # Per-method α from Stage 36
    method_alpha: dict[str, float] = {
        method: float(vals["alpha"])
        for method, vals in stage36["by_method"].items()
    }
    print("Stage 36 per-method α:", method_alpha)

    # Build flat table: one row per (method, dim, seed) from Stage 39
    raw39 = stage39["raw"]
    rows: list[dict[str, Any]] = []
    for dim_key, method_dict in raw39.items():
        dim = int(dim_key.lstrip("m"))
        for method, seed_list in method_dict.items():
            if method not in method_alpha:
                continue
            alpha = method_alpha[method]
            for rec in seed_list:
                rows.append({
                    "method": method,
                    "embed_dim": dim,
                    "seed": int(rec["seed"]),
                    "gap_ia_l2": float(rec["gap_ia_l2"]),
                    "av_ia_avg_R": float(rec["av_ia_avg_R"]),
                    "alpha": alpha,
                })

    print(f"Collected {len(rows)} (method, dim, seed) rows")

    # Pooled regression across all methods and dims
    gap_all = np.array([r["gap_ia_l2"] for r in rows])
    alpha_all = np.array([r["alpha"] for r in rows])
    avR_all = np.array([r["av_ia_avg_R"] for r in rows])

    pooled_gap_to_alpha = _ols_intercept(gap_all, alpha_all)
    pooled_gap_to_avR = _ols_intercept(gap_all, avR_all)
    pooled_alpha_to_avR = _ols_intercept(alpha_all, avR_all)

    print(f"\nPooled gap_ia → α regression (n={len(rows)}):")
    print(f"  slope={pooled_gap_to_alpha['slope']:.4f}, intercept={pooled_gap_to_alpha['intercept']:.4f}, R²={pooled_gap_to_alpha['r2']:.4f}, r={pooled_gap_to_alpha['pearson_r']:.4f}")
    print(f"\nPooled gap_ia → av_ia_avg_R (n={len(rows)}):")
    print(f"  slope={pooled_gap_to_avR['slope']:.4f}, intercept={pooled_gap_to_avR['intercept']:.4f}, R²={pooled_gap_to_avR['r2']:.4f}, r={pooled_gap_to_avR['pearson_r']:.4f}")
    print(f"\nPooled α → av_ia_avg_R (n={len(rows)}):")
    print(f"  slope={pooled_alpha_to_avR['slope']:.4f}, intercept={pooled_alpha_to_avR['intercept']:.4f}, R²={pooled_alpha_to_avR['r2']:.4f}, r={pooled_alpha_to_avR['pearson_r']:.4f}")

    # Per-method regression: gap_ia → α (only one α per method so this is across dims)
    per_method_rows: dict[str, dict[str, Any]] = {}
    for method in method_alpha:
        method_data = [r for r in rows if r["method"] == method]
        if len(method_data) < 3:
            continue
        g = np.array([r["gap_ia_l2"] for r in method_data])
        avR = np.array([r["av_ia_avg_R"] for r in method_data])
        reg = _ols_intercept(g, avR)
        per_method_rows[method] = {
            "alpha": method_alpha[method],
            "n": len(method_data),
            "gap_ia_mean": float(np.mean(g)),
            "gap_ia_std": float(np.std(g)),
            "av_ia_mean": float(np.mean(avR)),
            "av_ia_std": float(np.std(avR)),
            "gap_to_avR": reg,
        }
        print(f"\n  {method}: α={method_alpha[method]:.4f}, gap_ia_mean={np.mean(g):.4f}, av_ia_mean={np.mean(avR):.4f}")

    # Per-dim regression: gap_ia → α across methods
    per_dim_rows: dict[str, dict[str, Any]] = {}
    for dim_key in raw39:
        dim = int(dim_key.lstrip("m"))
        dim_data = [r for r in rows if r["embed_dim"] == dim]
        if len(dim_data) < 3:
            continue
        g = np.array([r["gap_ia_l2"] for r in dim_data])
        a = np.array([r["alpha"] for r in dim_data])
        avR = np.array([r["av_ia_avg_R"] for r in dim_data])
        reg_ga = _ols_intercept(g, a)
        reg_gavR = _ols_intercept(g, avR)
        per_dim_rows[dim_key] = {
            "dim": dim,
            "n": len(dim_data),
            "gap_to_alpha": reg_ga,
            "gap_to_avR": reg_gavR,
        }

    result = {
        "stage": "w5_gap_alpha_regression",
        "description": "OLS regression: gap_ia_l2 → transmission_efficiency_alpha and → av_ia_avg_R",
        "method_alpha": method_alpha,
        "n_rows": len(rows),
        "pooled": {
            "gap_to_alpha": pooled_gap_to_alpha,
            "gap_to_avR": pooled_gap_to_avR,
            "alpha_to_avR": pooled_alpha_to_avR,
        },
        "per_method": per_method_rows,
        "per_dim": per_dim_rows,
        "rows": rows,
        "elapsed_sec": time.time() - t0,
    }

    out_path = _OUTPUT_DIR / "w5_gap_alpha_regression_results.json"
    save_json(result, out_path)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    run()
