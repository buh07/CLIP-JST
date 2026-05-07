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


def _merge_stage69_rows(roots: list[Path]) -> list[dict[str, Any]]:
    merged: dict[tuple[int, str, int], dict[str, Any]] = {}
    for root in roots:
        p = root / "stage69_third_triple_speechcoco" / "stage69_third_triple_speechcoco_results.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing stage69 results block: {p}")
        block = load_json(p)
        for m_key, methods in block.get("raw", {}).items():
            m = int(str(m_key).lstrip("m"))
            for method, rows in methods.items():
                for r in rows:
                    key = (m, str(method), int(r["seed"]))
                    merged[key] = r
    out = [merged[k] for k in sorted(merged.keys())]
    return out


def _metric_pack(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    if len(y) < 2:
        return {"n": int(len(y)), "pearson_r": 0.0, "r2": 0.0, "mae": 0.0, "calibration_slope": 0.0}
    rr, _ = pearsonr(y, yhat)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else 0.0)
    mae = float(np.mean(np.abs(y - yhat)))
    den = float(np.dot(yhat, yhat))
    slope = float(np.dot(yhat, y) / den) if den > 1e-12 else 0.0
    return {
        "n": int(len(y)),
        "pearson_r": float(rr),
        "r2": float(r2),
        "mae": float(mae),
        "calibration_slope": float(slope),
    }


def _cell_mean_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    acc: dict[tuple[int, str], dict[str, list[float] | int | str]] = {}
    for r in rows:
        key = (int(r["embed_dim"]), str(r["method"]))
        slot = acc.setdefault(key, {"embed_dim": key[0], "method": key[1], "it": [], "at": [], "ia": []})
        slot["it"].append(float(r["av_it_avg_R"]))  # type: ignore[index]
        slot["at"].append(float(r["av_at_avg_R"]))  # type: ignore[index]
        slot["ia"].append(float(r["av_ia_avg_R"]))  # type: ignore[index]

    out = []
    for k in sorted(acc.keys()):
        s = acc[k]
        out.append(
            {
                "embed_dim": int(s["embed_dim"]),
                "method": str(s["method"]),
                "av_it_avg_R": float(np.mean(np.asarray(s["it"], dtype=np.float64))),  # type: ignore[arg-type]
                "av_at_avg_R": float(np.mean(np.asarray(s["at"], dtype=np.float64))),  # type: ignore[arg-type]
                "av_ia_avg_R": float(np.mean(np.asarray(s["ia"], dtype=np.float64))),  # type: ignore[arg-type]
            }
        )
    return out


def _predict(alpha: float, rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray([float(r["av_ia_avg_R"]) for r in rows], dtype=np.float64)
    x = np.sqrt(
        np.clip(
            np.asarray([float(r["av_it_avg_R"]) for r in rows], dtype=np.float64)
            * np.asarray([float(r["av_at_avg_R"]) for r in rows], dtype=np.float64),
            0.0,
            None,
        )
    )
    yhat = alpha * x
    return y, yhat


def run(cfg: dict[str, Any]) -> None:
    start = time.time()
    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stage_root = output_root / "stage71_third_triple_prospective_check"
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    pred_lock = load_json(Path(cfg["predictions_lock_json"]).resolve())
    alpha_locked = float(pred_lock["alpha_locked"])
    criteria = pred_lock.get("success_criteria", {})

    roots = [Path(x).resolve() for x in cfg["stage69_shard_roots"]]
    seed_rows = _merge_stage69_rows(roots)
    cell_rows = _cell_mean_rows(seed_rows)

    y_seed, yhat_seed = _predict(alpha_locked, seed_rows)
    y_cell, yhat_cell = _predict(alpha_locked, cell_rows)

    seed_metrics = _metric_pack(y_seed, yhat_seed)
    cell_metrics = _metric_pack(y_cell, yhat_cell)

    r_thresh = float(criteria.get("cell_mean_r_min", 0.85))
    mae_thresh = float(criteria.get("cell_mean_mae_max", 0.01))

    pass_r = bool(cell_metrics["pearson_r"] >= r_thresh)
    pass_mae = bool(cell_metrics["mae"] <= mae_thresh)

    stage72_check = None
    stage72_path = cfg.get("stage72_form_compare_json")
    if stage72_path:
        p72 = Path(str(stage72_path)).resolve()
        if p72.exists():
            s72 = load_json(p72)
            models = sorted(s72.get("analysis", {}).get("models", []), key=lambda m: m.get("cv_r2_mean", -1e9), reverse=True)
            top2 = [m.get("name") for m in models[:2]]
            pass_geom_top2 = "geometric_mean" in top2
            stage72_check = {
                "top2_by_cv_r2": top2,
                "geometric_mean_top2": bool(pass_geom_top2),
            }

    out = {
        "stage": "stage71_third_triple_prospective_check",
        "alpha_locked": alpha_locked,
        "predictions_lock_json": str(Path(cfg["predictions_lock_json"]).resolve()),
        "seed_level": seed_metrics,
        "cell_mean_level": cell_metrics,
        "criteria": {
            "cell_mean_r_min": r_thresh,
            "cell_mean_mae_max": mae_thresh,
        },
        "pass": {
            "cell_mean_r": pass_r,
            "cell_mean_mae": pass_mae,
            "overall_predictive": bool(pass_r and pass_mae),
        },
        "stage72_form_check": stage72_check,
        "n_seed_rows": int(len(seed_rows)),
        "n_cell_rows": int(len(cell_rows)),
        "elapsed_sec": float(time.time() - start),
    }
    save_json(out, stage_root / "stage71_third_triple_prospective_check.json")

    md = [
        "# Stage71 Third Triple Prospective Check",
        "",
        f"- alpha_locked: {alpha_locked:.6f}",
        f"- n_seed_rows: {len(seed_rows)}",
        f"- n_cell_rows: {len(cell_rows)}",
        "",
        "## Seed-Level",
        f"- r: {seed_metrics['pearson_r']:.4f}",
        f"- R2: {seed_metrics['r2']:.4f}",
        f"- MAE: {seed_metrics['mae']:.5f}",
        f"- calibration_slope: {seed_metrics['calibration_slope']:.4f}",
        "",
        "## Cell-Mean",
        f"- r: {cell_metrics['pearson_r']:.4f} (threshold {r_thresh:.3f})",
        f"- R2: {cell_metrics['r2']:.4f}",
        f"- MAE: {cell_metrics['mae']:.5f} (threshold {mae_thresh:.3f})",
        f"- calibration_slope: {cell_metrics['calibration_slope']:.4f}",
        "",
        "## Pass/Fail",
        f"- pass_cell_mean_r: {pass_r}",
        f"- pass_cell_mean_mae: {pass_mae}",
        f"- overall_predictive_pass: {bool(pass_r and pass_mae)}",
    ]

    if stage72_check is not None:
        md += [
            "",
            "## Stage72 Form Check",
            f"- top2_by_cv_r2: {stage72_check['top2_by_cv_r2']}",
            f"- geometric_mean_top2: {stage72_check['geometric_mean_top2']}",
        ]

    (stage_root / "stage71_third_triple_prospective_check.md").write_text("\n".join(md), encoding="utf-8")

    provenance = env_snapshot(
        Path(cfg["project_root"]),
        seeds=[],
        extra={"stage": "stage71_third_triple_prospective_check", "elapsed_sec": float(time.time() - start)},
    )
    save_json(provenance, stage_root / "provenance_stage71.json")
    mark_done(markers / "stage71_third_triple_prospective_check.done.json", {"elapsed_sec": float(time.time() - start)})
    print("stage71_third_triple_prospective_check complete")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)
