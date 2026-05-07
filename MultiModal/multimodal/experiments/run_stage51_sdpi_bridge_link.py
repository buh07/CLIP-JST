from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import yaml

from ..common import env_snapshot, load_json, mark_done, save_json
from .sdpi_common import corrcoef_safe, permutation_pvalue_for_corr


def _fit_alpha_no_intercept(x: np.ndarray, y: np.ndarray) -> float:
    den = float(np.sum(x * x))
    if den <= 0:
        return 0.0
    return float(np.sum(x * y) / den)


def run(cfg: dict) -> None:
    start = time.time()
    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stage_root = output_root / "stage51_sdpi_bridge_link"
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    s50 = load_json(Path(cfg["stage50_results_path"]).resolve())
    rows = list(s50.get("rows", []))
    if not rows:
        raise RuntimeError("Stage51: no rows in stage50")

    n_perm = int(cfg.get("perm_n", 1000))
    seed = int(cfg.get("analysis_seed", 2026))

    arr_it = np.asarray([float(r.get("recall", {}).get("av_it_avg_R", 0.0)) for r in rows], dtype=np.float64)
    arr_at = np.asarray([float(r.get("recall", {}).get("av_at_avg_R", 0.0)) for r in rows], dtype=np.float64)
    arr_ia = np.asarray([float(r.get("recall", {}).get("av_ia_avg_R", 0.0)) for r in rows], dtype=np.float64)
    arr_ceiling = np.sqrt(np.maximum(arr_it, 0.0) * np.maximum(arr_at, 0.0))
    alpha_hat_recall = _fit_alpha_no_intercept(arr_ceiling, arr_ia)
    pred_recall = alpha_hat_recall * arr_ceiling
    eps_recall = arr_ia - pred_recall

    gaps = np.asarray([
        float(r.get("centroid_gap_ia_l2", np.nan)) if r.get("centroid_gap_ia_l2", None) is not None else np.nan
        for r in rows
    ], dtype=np.float64)
    mask_gap = np.isfinite(gaps)

    out = {
        "stage": "stage51_sdpi_bridge_link",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_rows": len(rows),
        "recall_tightness": {
            "alpha_hat": float(alpha_hat_recall),
            "mae": float(np.mean(np.abs(eps_recall))),
            "rmse": float(np.sqrt(np.mean(eps_recall ** 2))),
            "max_abs": float(np.max(np.abs(eps_recall))),
        },
        "correlations": {},
        "mi_tightness": {},
        "rows": rows,
        "elapsed_sec": float(time.time() - start),
    }

    # E4: alpha_I vs alpha_recall, MI vs recall, and geometry links
    alpha_recall = np.asarray([float(r.get("alpha_recall", 0.0)) for r in rows], dtype=np.float64)

    corr_blocks = {}
    for est in ["ksg", "cancor", "gaussian"]:
        alpha_i = np.asarray([float(r.get("alpha_i", {}).get(est, 0.0)) for r in rows], dtype=np.float64)
        alpha_i_unit = np.asarray([float(r.get("alpha_i_unit", {}).get(est, max(0.0, min(1.0, a)))) for r, a in zip(rows, alpha_i)], dtype=np.float64)
        alpha_in_range = np.asarray(
            [bool(r.get("inequality", {}).get(est, {}).get("alpha_in_0_1", True)) for r in rows], dtype=bool
        )
        mi_ia = np.asarray([float(r.get("mi", {}).get(est, {}).get("ia", 0.0)) for r in rows], dtype=np.float64)
        mi_it = np.asarray([float(r.get("mi", {}).get(est, {}).get("it", 0.0)) for r in rows], dtype=np.float64)
        mi_at = np.asarray([float(r.get("mi", {}).get(est, {}).get("at", 0.0)) for r in rows], dtype=np.float64)
        mi_ceiling = np.sqrt(np.maximum(mi_it, 0.0) * np.maximum(mi_at, 0.0))

        alpha_hat_mi = _fit_alpha_no_intercept(mi_ceiling, mi_ia)
        eps_mi = mi_ia - alpha_hat_mi * mi_ceiling

        block = {
            "alpha_i_vs_alpha_recall": {
                "r": float(corrcoef_safe(alpha_i, alpha_recall)),
                "perm": permutation_pvalue_for_corr(alpha_i, alpha_recall, n_perm=n_perm, seed=seed + hash(est) % 1000 + 11),
            },
            "alpha_i_unit_vs_alpha_recall": {
                "r": float(corrcoef_safe(alpha_i_unit, alpha_recall)),
                "perm": permutation_pvalue_for_corr(alpha_i_unit, alpha_recall, n_perm=n_perm, seed=seed + hash(est) % 1000 + 12),
            },
            "alpha_i_in_range_vs_alpha_recall": None,
            "mi_ia_vs_recall_ia": {
                "r": float(corrcoef_safe(mi_ia, arr_ia)),
                "perm": permutation_pvalue_for_corr(mi_ia, arr_ia, n_perm=n_perm, seed=seed + hash(est) % 1000 + 13),
            },
            "alpha_i_vs_gap": None,
            "mi_tightness": {
                "alpha_hat": float(alpha_hat_mi),
                "mae": float(np.mean(np.abs(eps_mi))),
                "rmse": float(np.sqrt(np.mean(eps_mi ** 2))),
                "max_abs": float(np.max(np.abs(eps_mi))),
            },
        }
        if np.any(mask_gap):
            block["alpha_i_vs_gap"] = {
                "r": float(corrcoef_safe(alpha_i[mask_gap], gaps[mask_gap])),
                "perm": permutation_pvalue_for_corr(alpha_i[mask_gap], gaps[mask_gap], n_perm=n_perm, seed=seed + hash(est) % 1000 + 17),
            }
        if int(np.sum(alpha_in_range)) >= 3:
            ai = alpha_i[alpha_in_range]
            ar = alpha_recall[alpha_in_range]
            block["alpha_i_in_range_vs_alpha_recall"] = {
                "n": int(len(ai)),
                "r": float(corrcoef_safe(ai, ar)),
                "perm": permutation_pvalue_for_corr(ai, ar, n_perm=n_perm, seed=seed + hash(est) % 1000 + 19),
            }
        corr_blocks[est] = block
        out["mi_tightness"][est] = block["mi_tightness"]

    out["correlations"] = corr_blocks

    save_json(out, stage_root / "stage51_sdpi_bridge_link.json")

    lines = [
        "# Stage51 SDPI Bridge Link",
        "",
        f"Rows: {len(rows)}",
        "",
        "## Recall-space Tightness",
        f"- alpha_hat = {out['recall_tightness']['alpha_hat']:.4f}",
        f"- MAE = {out['recall_tightness']['mae']:.6f}",
        f"- RMSE = {out['recall_tightness']['rmse']:.6f}",
        "",
        "## Correlations",
        "| estimator | corr(alphaI, alphaR) | corr(alphaI_unit, alphaR) | corr(MIia, Ria) | corr(alphaI, gap) |",
        "|---|---:|---:|---:|---:|",
    ]
    for est in ["ksg", "cancor", "gaussian"]:
        b = corr_blocks[est]
        rg = b.get("alpha_i_vs_gap", {}).get("r") if b.get("alpha_i_vs_gap") else float("nan")
        lines.append(
            f"| {est} | {b['alpha_i_vs_alpha_recall']['r']:.4f} | {b['alpha_i_unit_vs_alpha_recall']['r']:.4f} | "
            f"{b['mi_ia_vs_recall_ia']['r']:.4f} | {rg:.4f} |"
        )
    (stage_root / "stage51_sdpi_bridge_link.md").write_text("\n".join(lines), encoding="utf-8")

    provenance = env_snapshot(
        Path(cfg["project_root"]),
        seeds=[],
        extra={"stage": "stage51_sdpi_bridge_link", "elapsed_sec": float(time.time() - start), "n_rows": len(rows)},
    )
    save_json(provenance, stage_root / "provenance_stage51.json")
    mark_done(markers / "stage51_sdpi_bridge_link.done.json", {"elapsed_sec": float(time.time() - start), "n_rows": len(rows)})
    print(f"[stage51] complete n_rows={len(rows)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    run(cfg)


if __name__ == "__main__":
    main()
