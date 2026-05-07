from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from ..common import env_snapshot, load_json, mark_done, save_json
from .sdpi_common import corrcoef_safe, deterministic_subsample_indices, pca_reduce, pca_reduce_joint_blocks


def run(cfg: dict) -> None:
    start = time.time()
    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stage_root = output_root / "stage53_sdpi_constrained_channel"
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    s48 = load_json(Path(cfg["stage48_results_path"]).resolve())
    s50 = load_json(Path(cfg["stage50_results_path"]).resolve())
    rows48 = list(s48.get("rows", []))
    rows50 = {str(r["condition_id"]): r for r in s50.get("rows", [])}

    dims_filter = {int(x) for x in cfg.get("embed_dims", [])}
    methods_filter = set(cfg.get("methods", []))
    seeds_filter = {int(s) for s in cfg.get("seeds", [])}

    mi_subsample_n = int(cfg.get("mi_subsample_n", 2000))
    pca_dim = int(cfg.get("mi_pca_dim", 32))
    pca_mode = str(cfg.get("pca_mode", "joint_all"))
    ridge_lambda = float(cfg.get("ridge_lambda", 1e-3))
    seed0 = int(cfg.get("analysis_seed", 2026))

    out_rows = []
    for m in rows48:
        if dims_filter and int(m["embed_dim"]) not in dims_filter:
            continue
        if methods_filter and str(m["method"]) not in methods_filter:
            continue
        if seeds_filter and int(m["seed"]) not in seeds_filter:
            continue

        cid = str(m["condition_id"])
        if cid not in rows50:
            continue
        p = Path(m["embedding_path"])
        if not p.exists():
            continue

        pack = torch.load(p, map_location="cpu", weights_only=True)
        zi = np.asarray(pack["zi"], dtype=np.float64)
        za = np.asarray(pack["za"], dtype=np.float64)
        zt = np.asarray(pack["zt"], dtype=np.float64)
        n = zi.shape[0]

        seed = seed0 + (abs(hash(cid)) % 10_000_000)
        idx = deterministic_subsample_indices(n, mi_subsample_n, seed)
        zi = zi[idx]
        za = za[idx]
        zt = zt[idx]
        if pca_mode == "per_modality":
            zi = pca_reduce(zi, pca_dim, seed + 101)
            za = pca_reduce(za, pca_dim, seed + 103)
            zt = pca_reduce(zt, pca_dim, seed + 107)
        else:
            zi, za, zt = pca_reduce_joint_blocks([zi, za, zt], pca_dim, seed + 109)

        reg_i = Ridge(alpha=ridge_lambda, fit_intercept=True)
        reg_a = Ridge(alpha=ridge_lambda, fit_intercept=True)
        reg_i.fit(zt, zi)
        reg_a.fit(zt, za)
        pred_i = reg_i.predict(zt)
        pred_a = reg_a.predict(zt)

        r2_it = float(r2_score(zi, pred_i, multioutput="uniform_average"))
        r2_at = float(r2_score(za, pred_a, multioutput="uniform_average"))
        eta1 = float(min(1.0, max(0.0, r2_it)))
        eta2 = float(min(1.0, max(0.0, r2_at)))
        eta_geom = float(math.sqrt(eta1 * eta2))

        r50 = rows50[cid]
        alpha_i_ksg = float(r50.get("alpha_i", {}).get("ksg", 0.0))
        alpha_i_unit_ksg = float(r50.get("alpha_i_unit", {}).get("ksg", max(0.0, min(1.0, alpha_i_ksg))))
        alpha_recall = float(r50.get("alpha_recall", 0.0))

        out_rows.append({
            "condition_id": cid,
            "source_id": m["source_id"],
            "embed_dim": int(m["embed_dim"]),
            "method": m["method"],
            "seed": int(m["seed"]),
            "eta1_proxy": eta1,
            "eta2_proxy": eta2,
            "eta_geom_proxy": eta_geom,
            "alpha_i_ksg": alpha_i_ksg,
            "alpha_i_unit_ksg": alpha_i_unit_ksg,
            "alpha_recall": alpha_recall,
            "n_samples_used": int(len(idx)),
            "fit_diag": {
                "r2_it_raw": r2_it,
                "r2_at_raw": r2_at,
            },
        })

        print(
            f"[stage53] {cid} eta=({eta1:.4f},{eta2:.4f}) sqrt={eta_geom:.4f} "
            f"alphaI={alpha_i_ksg:.4f} alphaI_unit={alpha_i_unit_ksg:.4f} alphaR={alpha_recall:.4f}"
        )

    eta = np.asarray([r["eta_geom_proxy"] for r in out_rows], dtype=np.float64)
    ai = np.asarray([r["alpha_i_ksg"] for r in out_rows], dtype=np.float64)
    ai_u = np.asarray([r["alpha_i_unit_ksg"] for r in out_rows], dtype=np.float64)
    ar = np.asarray([r["alpha_recall"] for r in out_rows], dtype=np.float64)

    out = {
        "stage": "stage53_sdpi_constrained_channel",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_rows": len(out_rows),
        "summary": {
            "corr_eta_vs_alphaI_ksg": float(corrcoef_safe(eta, ai)),
            "corr_eta_vs_alphaI_unit_ksg": float(corrcoef_safe(eta, ai_u)),
            "corr_eta_vs_alpha_recall": float(corrcoef_safe(eta, ar)),
            "eta_mean": float(np.mean(eta)) if len(eta) else 0.0,
            "eta_std": float(np.std(eta, ddof=1)) if len(eta) > 1 else 0.0,
        },
        "rows": out_rows,
        "elapsed_sec": float(time.time() - start),
    }

    save_json(out, stage_root / "stage53_sdpi_constrained_channel.json")

    lines = [
        "# Stage53 Constrained Channel Proxy",
        "",
        f"Rows: {len(out_rows)}",
        f"- corr(eta_geom, alphaI_ksg) = {out['summary']['corr_eta_vs_alphaI_ksg']:.4f}",
        f"- corr(eta_geom, alphaI_unit_ksg) = {out['summary']['corr_eta_vs_alphaI_unit_ksg']:.4f}",
        f"- corr(eta_geom, alpha_recall) = {out['summary']['corr_eta_vs_alpha_recall']:.4f}",
        f"- eta mean/std = {out['summary']['eta_mean']:.4f}/{out['summary']['eta_std']:.4f}",
    ]
    (stage_root / "stage53_sdpi_constrained_channel.md").write_text("\n".join(lines), encoding="utf-8")

    provenance = env_snapshot(
        Path(cfg["project_root"]),
        seeds=sorted(seeds_filter) if seeds_filter else [],
        extra={"stage": "stage53_sdpi_constrained_channel", "elapsed_sec": float(time.time() - start), "n_rows": len(out_rows)},
    )
    save_json(provenance, stage_root / "provenance_stage53.json")
    mark_done(markers / "stage53_sdpi_constrained_channel.done.json", {"elapsed_sec": float(time.time() - start), "n_rows": len(out_rows)})
    print(f"[stage53] complete n_rows={len(out_rows)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    run(cfg)


if __name__ == "__main__":
    main()
