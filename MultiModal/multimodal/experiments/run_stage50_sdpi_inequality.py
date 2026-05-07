from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..common import env_snapshot, load_json, mark_done, save_json
from .sdpi_common import bootstrap_ci


def _group_mean(rows: list[dict[str, Any]], key: str, field_path: list[str], *, n_boot: int, seed: int) -> dict[str, Any]:
    groups: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        g = str(r.get(key, "unknown"))
        v = r
        for p in field_path:
            v = v.get(p, {}) if isinstance(v, dict) else {}
        try:
            groups[g].append(float(v))
        except Exception:
            pass

    out = {}
    for i, (g, vals) in enumerate(sorted(groups.items())):
        arr = np.asarray(vals, dtype=np.float64)
        out[g] = {
            "n": int(len(arr)),
            "mean": float(np.mean(arr)) if len(arr) else 0.0,
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "bootstrap": bootstrap_ci(arr, fn=np.mean, n_boot=n_boot, seed=seed + i) if len(arr) else {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0},
        }
    return out


def run(cfg: dict) -> None:
    start = time.time()
    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stage_root = output_root / "stage50_sdpi_inequality"
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    src = load_json(Path(cfg["stage49_results_path"]).resolve())
    rows = list(src.get("rows", []))
    if not rows:
        raise RuntimeError("Stage50: no rows in stage49 results")

    tol = float(cfg.get("bound_tol", 1e-6))
    n_boot = int(cfg.get("bootstrap_n", 1000))
    seed = int(cfg.get("analysis_seed", 2026))

    enriched = []
    for r in rows:
        rr = dict(r)
        mi = rr.get("mi", {})
        for est in ["ksg", "cancor", "gaussian"]:
            it = float(mi.get(est, {}).get("it", 0.0))
            at = float(mi.get(est, {}).get("at", 0.0))
            ia = float(mi.get(est, {}).get("ia", 0.0))
            bound = math.sqrt(max(it, 0.0) * max(at, 0.0))
            alpha_i = float(rr.get("alpha_i", {}).get(est, 0.0))
            alpha_i_unit = float(rr.get("alpha_i_unit", {}).get(est, max(0.0, min(1.0, alpha_i))))
            if "inequality" not in rr:
                rr["inequality"] = {}
            rr["inequality"][est] = {
                "bound": bound,
                "violation_upper": bool(ia > bound + tol),
                "violation_ratio": float(ia / (bound + 1e-12)),
                "alpha_i": alpha_i,
                "alpha_i_unit": alpha_i_unit,
                "alpha_in_0_1": bool((alpha_i >= -tol) and (alpha_i <= 1.0 + tol)),
            }
        enriched.append(rr)

    def _violation_stats(est: str) -> dict[str, Any]:
        n = len(enriched)
        v_up = sum(1 for r in enriched if r.get("inequality", {}).get(est, {}).get("violation_upper", False))
        v_rng = sum(1 for r in enriched if not r.get("inequality", {}).get(est, {}).get("alpha_in_0_1", True))
        alpha_arr = np.asarray([float(r.get("inequality", {}).get(est, {}).get("alpha_i", 0.0)) for r in enriched], dtype=np.float64)
        alpha_u_arr = np.asarray([float(r.get("inequality", {}).get(est, {}).get("alpha_i_unit", 0.0)) for r in enriched], dtype=np.float64)
        return {
            "n": n,
            "upper_bound_violations": int(v_up),
            "upper_bound_violation_rate": float(v_up / n if n else 0.0),
            "alpha_out_of_range_count": int(v_rng),
            "alpha_out_of_range_rate": float(v_rng / n if n else 0.0),
            "alpha_mean": float(np.mean(alpha_arr)),
            "alpha_std": float(np.std(alpha_arr, ddof=1)) if n > 1 else 0.0,
            "alpha_unit_mean": float(np.mean(alpha_u_arr)),
            "alpha_unit_std": float(np.std(alpha_u_arr, ddof=1)) if n > 1 else 0.0,
            "alpha_bootstrap": bootstrap_ci(alpha_arr, fn=np.mean, n_boot=n_boot, seed=seed + hash(est) % 1000),
            "alpha_unit_bootstrap": bootstrap_ci(alpha_u_arr, fn=np.mean, n_boot=n_boot, seed=seed + hash(est) % 1000 + 13),
        }

    by_estimator = {est: _violation_stats(est) for est in ["ksg", "cancor", "gaussian"]}

    grouped = {
        "by_method_alpha_ksg": _group_mean(enriched, "method", ["inequality", "ksg", "alpha_i"], n_boot=n_boot, seed=seed + 200),
        "by_embed_dim_alpha_ksg": _group_mean(enriched, "embed_dim", ["inequality", "ksg", "alpha_i"], n_boot=n_boot, seed=seed + 300),
        "by_source_alpha_ksg": _group_mean(enriched, "source_id", ["inequality", "ksg", "alpha_i"], n_boot=n_boot, seed=seed + 400),
    }

    out = {
        "stage": "stage50_sdpi_inequality",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_rows": len(enriched),
        "bound_tol": tol,
        "by_estimator": by_estimator,
        "grouped": grouped,
        "rows": enriched,
        "elapsed_sec": float(time.time() - start),
    }

    save_json(out, stage_root / "stage50_sdpi_inequality.json")

    lines = [
        "# Stage50 SDPI Inequality",
        "",
        f"Rows: {len(enriched)}",
        "",
        "## Estimator-level Bound Checks",
        "| estimator | upper violations | violation rate | alpha mean | alpha(unit) mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for est, s in by_estimator.items():
        lines.append(
            f"| {est} | {s['upper_bound_violations']} | {s['upper_bound_violation_rate']:.4f} | {s['alpha_mean']:.4f} | {s['alpha_unit_mean']:.4f} |"
        )
    (stage_root / "stage50_sdpi_inequality.md").write_text("\n".join(lines), encoding="utf-8")

    provenance = env_snapshot(
        Path(cfg["project_root"]),
        seeds=[],
        extra={
            "stage": "stage50_sdpi_inequality",
            "elapsed_sec": float(time.time() - start),
            "n_rows": len(enriched),
        },
    )
    save_json(provenance, stage_root / "provenance_stage50.json")
    mark_done(markers / "stage50_sdpi_inequality.done.json", {"elapsed_sec": float(time.time() - start), "n_rows": len(enriched)})
    print(f"[stage50] complete n_rows={len(enriched)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    run(cfg)


if __name__ == "__main__":
    main()
