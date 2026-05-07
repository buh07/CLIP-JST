from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..common import env_snapshot, load_json, mark_done, save_json


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _mae(y: np.ndarray, yhat: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    return float(np.mean(np.abs(y - yhat)))


def _fit_linear(x: np.ndarray, y: np.ndarray, x_eval: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    pred = a + b * x_eval
    return {"intercept": a, "slope": b}, pred


def _pav_increasing(y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """
    Pair-adjacent violators for monotone increasing regression over ordered samples.
    """
    n = len(y)
    if w is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = w.astype(np.float64)
    y = y.astype(np.float64)

    # Block representation
    val = y.copy()
    wt = w.copy()
    start = np.arange(n, dtype=np.int64)
    end = np.arange(n, dtype=np.int64)

    m = n
    i = 0
    while i < m - 1:
        if val[i] <= val[i + 1] + 1e-12:
            i += 1
            continue
        # merge i and i+1
        new_w = wt[i] + wt[i + 1]
        new_v = (wt[i] * val[i] + wt[i + 1] * val[i + 1]) / max(1e-12, new_w)
        wt[i] = new_w
        val[i] = new_v
        end[i] = end[i + 1]

        # shift left remaining blocks
        if i + 2 < m:
            wt[i + 1 : m - 1] = wt[i + 2 : m]
            val[i + 1 : m - 1] = val[i + 2 : m]
            start[i + 1 : m - 1] = start[i + 2 : m]
            end[i + 1 : m - 1] = end[i + 2 : m]
        m -= 1
        i = max(0, i - 1)

    fitted = np.zeros(n, dtype=np.float64)
    for j in range(m):
        fitted[start[j] : end[j] + 1] = val[j]
    return fitted


def _fit_isotonic(
    x: np.ndarray,
    y: np.ndarray,
    x_eval: np.ndarray,
    *,
    decreasing: bool,
) -> tuple[dict[str, Any], np.ndarray]:
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    if decreasing:
        fit_sorted = -_pav_increasing(-y_sorted)
    else:
        fit_sorted = _pav_increasing(y_sorted)

    # Step-function prediction with right-continuous bins
    idx = np.searchsorted(x_sorted, x_eval, side="right") - 1
    idx = np.clip(idx, 0, len(x_sorted) - 1)
    pred = fit_sorted[idx]

    model = {
        "decreasing": bool(decreasing),
        "n_points": int(len(x_sorted)),
        "x_sorted": x_sorted.tolist(),
        "y_fit_sorted": fit_sorted.tolist(),
    }
    return model, pred


def _kfold_indices(n: int, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n, dtype=np.int64)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    return [f.astype(np.int64) for f in folds if len(f) > 0]


def _collect_rows(stage39_path: Path, methods_allow: list[str] | None) -> list[dict[str, Any]]:
    d = load_json(stage39_path)
    rows = []
    for m_key, methods in d.get("raw", {}).items():
        for method, recs in methods.items():
            if methods_allow and method not in methods_allow:
                continue
            for r in recs:
                it = float(r["av_it_avg_R"])
                at = float(r["av_at_avg_R"])
                ia = float(r["av_ia_avg_R"])
                ceiling = float(np.sqrt(max(0.0, it * at)))
                alpha_local = ia / max(1e-12, ceiling)
                rows.append(
                    {
                        "method": str(method),
                        "embed_dim": int(r["embed_dim"]),
                        "seed": int(r["seed"]),
                        "gap_ia_l2": float(r["gap_ia_l2"]),
                        "av_ia": ia,
                        "alpha_local": float(alpha_local),
                        "ceiling": ceiling,
                    }
                )
    return rows


def _evaluate_target(
    x: np.ndarray,
    y: np.ndarray,
    dims: np.ndarray,
    *,
    kfold: int,
    cv_seed: int,
    decreasing: bool,
) -> dict[str, Any]:
    # In-sample fits
    lin_params, lin_pred = _fit_linear(x, y, x)
    iso_model, iso_pred = _fit_isotonic(x, y, x, decreasing=decreasing)

    # K-fold CV
    lin_r2, lin_mae = [], []
    iso_r2, iso_mae = [], []
    for fold in _kfold_indices(len(x), kfold, cv_seed):
        tr = np.ones(len(x), dtype=bool)
        tr[fold] = False
        _, p_lin = _fit_linear(x[tr], y[tr], x[fold])
        _, p_iso = _fit_isotonic(x[tr], y[tr], x[fold], decreasing=decreasing)
        lin_r2.append(_r2(y[fold], p_lin))
        lin_mae.append(_mae(y[fold], p_lin))
        iso_r2.append(_r2(y[fold], p_iso))
        iso_mae.append(_mae(y[fold], p_iso))

    # Leave-one-dim-out
    pred_lin_lodo = np.zeros_like(y)
    pred_iso_lodo = np.zeros_like(y)
    for d in sorted(set(dims.tolist())):
        te = dims == d
        tr = ~te
        _, p_lin = _fit_linear(x[tr], y[tr], x[te])
        _, p_iso = _fit_isotonic(x[tr], y[tr], x[te], decreasing=decreasing)
        pred_lin_lodo[te] = p_lin
        pred_iso_lodo[te] = p_iso

    return {
        "linear": {
            "params": lin_params,
            "in_sample_r2": _r2(y, lin_pred),
            "in_sample_mae": _mae(y, lin_pred),
            "cv_r2_mean": float(np.mean(np.asarray(lin_r2, dtype=np.float64))),
            "cv_r2_std": float(np.std(np.asarray(lin_r2, dtype=np.float64), ddof=1)) if len(lin_r2) > 1 else 0.0,
            "cv_mae_mean": float(np.mean(np.asarray(lin_mae, dtype=np.float64))),
            "cv_mae_std": float(np.std(np.asarray(lin_mae, dtype=np.float64), ddof=1)) if len(lin_mae) > 1 else 0.0,
            "lodo_r2": _r2(y, pred_lin_lodo),
            "lodo_mae": _mae(y, pred_lin_lodo),
        },
        "isotonic": {
            "model": iso_model,
            "in_sample_r2": _r2(y, iso_pred),
            "in_sample_mae": _mae(y, iso_pred),
            "cv_r2_mean": float(np.mean(np.asarray(iso_r2, dtype=np.float64))),
            "cv_r2_std": float(np.std(np.asarray(iso_r2, dtype=np.float64), ddof=1)) if len(iso_r2) > 1 else 0.0,
            "cv_mae_mean": float(np.mean(np.asarray(iso_mae, dtype=np.float64))),
            "cv_mae_std": float(np.std(np.asarray(iso_mae, dtype=np.float64), ddof=1)) if len(iso_mae) > 1 else 0.0,
            "lodo_r2": _r2(y, pred_iso_lodo),
            "lodo_mae": _mae(y, pred_iso_lodo),
        },
    }


def run(cfg: dict[str, Any]) -> None:
    start = time.time()
    out_root = Path(cfg["output_root"]).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    stage_root = out_root / "stage65_isotonic_gap_calibration"
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = out_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    stage39_path = Path(cfg["stage39_results"]).resolve()
    methods_allow = [str(m) for m in cfg.get("methods", [])] or None
    rows = _collect_rows(stage39_path, methods_allow)
    if len(rows) < 20:
        raise RuntimeError(f"Too few rows for isotonic calibration: n={len(rows)}")

    x = np.asarray([float(r["gap_ia_l2"]) for r in rows], dtype=np.float64)
    y_alpha = np.asarray([float(r["alpha_local"]) for r in rows], dtype=np.float64)
    y_ia = np.asarray([float(r["av_ia"]) for r in rows], dtype=np.float64)
    dims = np.asarray([int(r["embed_dim"]) for r in rows], dtype=np.int64)

    analysis_alpha = _evaluate_target(
        x,
        y_alpha,
        dims,
        kfold=int(cfg.get("kfold", 5)),
        cv_seed=int(cfg.get("cv_seed", 2026)),
        decreasing=True,
    )
    analysis_ia = _evaluate_target(
        x,
        y_ia,
        dims,
        kfold=int(cfg.get("kfold", 5)),
        cv_seed=int(cfg.get("cv_seed", 2026)),
        decreasing=True,
    )

    out = {
        "stage": "stage65_isotonic_gap_calibration",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage39_results": str(stage39_path),
        "n_rows": int(len(rows)),
        "methods_included": sorted(set(str(r["method"]) for r in rows)),
        "dims_included": sorted(set(int(r["embed_dim"]) for r in rows)),
        "targets": {
            "alpha_local": analysis_alpha,
            "av_ia": analysis_ia,
        },
        "elapsed_sec": float(time.time() - start),
    }
    save_json(out, stage_root / "stage65_isotonic_gap_calibration.json")

    def _line_for(target: str, key: str, m: dict[str, Any]) -> str:
        return (
            f"| {target} | {key} | {m['in_sample_r2']:.4f} | {m['in_sample_mae']:.5f} | "
            f"{m['cv_r2_mean']:.4f}±{m['cv_r2_std']:.4f} | {m['cv_mae_mean']:.5f}±{m['cv_mae_std']:.5f} | "
            f"{m['lodo_r2']:.4f} | {m['lodo_mae']:.5f} |"
        )

    lines = [
        "# Stage65 Isotonic Gap Calibration",
        "",
        f"- n_rows: {len(rows)}",
        f"- methods: {', '.join(out['methods_included'])}",
        f"- dims: {', '.join(str(x) for x in out['dims_included'])}",
        "",
        "| Target | Model | In-sample R² | In-sample MAE | 5-fold CV R² (mean±std) | 5-fold CV MAE (mean±std) | LODO R² | LODO MAE |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        _line_for("alpha_local", "linear", analysis_alpha["linear"]),
        _line_for("alpha_local", "isotonic", analysis_alpha["isotonic"]),
        _line_for("av_ia", "linear", analysis_ia["linear"]),
        _line_for("av_ia", "isotonic", analysis_ia["isotonic"]),
    ]
    (stage_root / "stage65_isotonic_gap_calibration.md").write_text("\n".join(lines), encoding="utf-8")

    provenance = env_snapshot(
        Path(cfg["project_root"]),
        seeds=[],
        extra={
            "stage": "stage65_isotonic_gap_calibration",
            "elapsed_sec": float(time.time() - start),
            "stage39_results": str(stage39_path),
        },
    )
    save_json(provenance, stage_root / "provenance_stage65.json")
    mark_done(markers / "stage65_isotonic_gap_calibration.done.json", {"elapsed_sec": float(time.time() - start)})
    print("stage65_isotonic_gap_calibration complete")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)
