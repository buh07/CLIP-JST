from __future__ import annotations

import argparse
import glob
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from ..common import load_json, save_json


_JL_ABLATION_BASE = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "modular_transitivity_jl_ablation"
)

_METHODS_OF_INTEREST = ["modular_shared_jl", "modular_separate_jl"]


def _collect_rows(ablation_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(ablation_root.rglob("eval.json")):
        try:
            d = load_json(p)
        except Exception:
            continue
        diag = d.get("diagnostics", {})
        av_diag = diag.get("av", diag)
        cdm = av_diag.get("centroid_distance_matrix", {})
        ia_gap = cdm.get("image", {}).get("audio")
        if ia_gap is None:
            continue
        method = str(d.get("method", ""))
        if method not in _METHODS_OF_INTEREST:
            continue
        rows.append(
            {
                "embed_dim": int(d.get("embed_dim", 0)),
                "method": method,
                "seed": int(d.get("seed", -1)),
                "ia_gap": float(ia_gap),
                "av_ia_avg_R": float(d.get("av_ia_avg_R", 0.0)),
                "av_at_avg_R": float(d.get("av_at_avg_R", 0.0)),
                "source_path": str(p),
            }
        )
    return rows


def _stats(vals: list[float]) -> dict[str, float]:
    n = len(vals)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / max(1, n - 1)
    return {"mean": mean, "std": math.sqrt(var), "n": n}


def run(cfg: dict) -> None:
    ablation_root = Path(cfg.get("ablation_root", str(_JL_ABLATION_BASE))).resolve()
    output_dir = Path(cfg["output_root"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _collect_rows(ablation_root)
    if not rows:
        raise RuntimeError(f"No eval.json files with CDM found under {ablation_root}")

    by_dim_method: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_dim_method[(r["embed_dim"], r["method"])].append(r)

    dims = sorted({r["embed_dim"] for r in rows})
    table: list[dict[str, Any]] = []
    for dim in dims:
        entry: dict[str, Any] = {"embed_dim": dim}
        for method in _METHODS_OF_INTEREST:
            cell = by_dim_method[(dim, method)]
            entry[f"{method}_ia_gap"] = _stats([c["ia_gap"] for c in cell])
            entry[f"{method}_av_ia"] = _stats([c["av_ia_avg_R"] for c in cell])
            entry[f"{method}_n_seeds"] = len(cell)
        # Δgap = separate − shared (positive → separate has larger gap)
        shared_mean = entry.get("modular_shared_jl_ia_gap", {}).get("mean", float("nan"))
        sep_mean = entry.get("modular_separate_jl_ia_gap", {}).get("mean", float("nan"))
        entry["delta_ia_gap_sep_minus_shared"] = sep_mean - shared_mean
        table.append(entry)

    out = {
        "stage": "w5_gap_analysis",
        "ablation_root": str(ablation_root),
        "n_rows_total": len(rows),
        "dims": dims,
        "methods": _METHODS_OF_INTEREST,
        "table": table,
        "raw_rows": rows,
    }
    out_path = output_dir / "w5_gap_analysis_results.json"
    save_json(out, out_path)

    print(f"\n{'dim':>6}  {'shared ia_gap':>16}  {'separate ia_gap':>16}  {'Δ(sep−shr)':>12}  {'shared av_ia':>14}  {'sep av_ia':>12}")
    for entry in table:
        dim = entry["embed_dim"]
        shr = entry["modular_shared_jl_ia_gap"]
        sep = entry["modular_separate_jl_ia_gap"]
        shr_ia = entry["modular_shared_jl_av_ia"]
        sep_ia = entry["modular_separate_jl_av_ia"]
        delta = entry["delta_ia_gap_sep_minus_shared"]
        print(
            f"  m={dim:<4}  {shr['mean']:.4f}±{shr['std']:.4f}  {sep['mean']:.4f}±{sep['std']:.4f}  "
            f"{delta:+.4f}        {shr_ia['mean']:.4f}±{shr_ia['std']:.4f}  {sep_ia['mean']:.4f}±{sep_ia['std']:.4f}"
        )
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument(
        "--output-root",
        default=str(
            Path(__file__).resolve().parents[2]
            / "results"
            / "reviewer_fixes_suite"
            / "w5_gap_analysis"
        ),
    )
    ap.add_argument("--ablation-root", default=str(_JL_ABLATION_BASE))
    args = ap.parse_args()
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {"output_root": args.output_root, "ablation_root": args.ablation_root}
    run(cfg)
