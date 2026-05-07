from __future__ import annotations

import argparse
import fcntl
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from ..common import env_snapshot, load_json, mark_done, save_json
from .sdpi_common import (
    alpha_like,
    alpha_like_unit,
    bootstrap_ci,
    cancor_mi_proxy,
    corrcoef_safe,
    deterministic_subsample_indices,
    gaussian_cmi_from_cov,
    gaussian_mi,
    ksg_mi,
    permutation_pvalue_for_corr,
    pca_reduce,
    pca_reduce_joint_blocks,
    residualize,
)


def _merge_by_condition(old_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(r["condition_id"]): r for r in old_rows if "condition_id" in r}
    for r in new_rows:
        by_id[str(r["condition_id"])] = r
    return [by_id[k] for k in sorted(by_id.keys())]


def _save_shard_safe(out_path: Path, incoming: dict[str, Any]) -> None:
    lock_path = out_path.with_suffix(out_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        if out_path.exists():
            try:
                cur = load_json(out_path)
            except Exception:
                cur = {}
        else:
            cur = {}
        merged = dict(cur) if isinstance(cur, dict) else {}
        merged["stage"] = "stage49_sdpi_mi_diagnostics"
        merged["filters"] = incoming.get("filters", {})
        merged["rows"] = _merge_by_condition(cur.get("rows", []) if isinstance(cur, dict) else [], incoming.get("rows", []))
        prior = list(cur.get("shard_summaries", [])) if isinstance(cur, dict) else []
        new_summary = incoming.get("shard_summary", {})
        # Deduplicate shard summaries by source_groups signature, keeping latest.
        by_sig = {}
        for s in prior:
            sig = tuple(s.get("source_groups", []))
            by_sig[sig] = s
        sig_new = tuple(new_summary.get("source_groups", []))
        by_sig[sig_new] = new_summary
        merged["shard_summaries"] = [by_sig[k] for k in sorted(by_sig.keys())]
        save_json(merged, out_path)
        fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)


def _bootstrap_corr_ci(x: np.ndarray, y: np.ndarray, *, n_boot: int, seed: int) -> dict[str, float]:
    if len(x) < 3:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals.append(corrcoef_safe(x[idx], y[idx]))
    vals = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(np.mean(vals)),
        "ci_low": float(np.quantile(vals, 0.025)),
        "ci_high": float(np.quantile(vals, 0.975)),
    }


def run(cfg: dict) -> None:
    start = time.time()
    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stage_root = output_root / "stage49_sdpi_mi_diagnostics"
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    stage48 = load_json(Path(cfg["stage48_results_path"]).resolve())
    rows = list(stage48.get("rows", []))

    source_groups = set(cfg.get("source_groups", []))
    dims_filter = {int(x) for x in cfg.get("embed_dims", [])}
    methods_filter = set(cfg.get("methods", []))
    seeds_filter = {int(s) for s in cfg.get("seeds", [])}

    def _keep(r: dict[str, Any]) -> bool:
        if source_groups and str(r.get("source_group", "")) not in source_groups:
            return False
        if dims_filter and int(r["embed_dim"]) not in dims_filter:
            return False
        if methods_filter and str(r["method"]) not in methods_filter:
            return False
        if seeds_filter and int(r["seed"]) not in seeds_filter:
            return False
        return True

    rows = [r for r in rows if _keep(r)]

    mi_pca_dim = int(cfg.get("mi_pca_dim", 32))
    mi_subsample_n = int(cfg.get("mi_subsample_n", 2000))
    ksg_k = int(cfg.get("ksg_k", 5))
    ridge_lambda = float(cfg.get("ridge_lambda", 1e-3))
    cov_shrinkage = str(cfg.get("cov_shrinkage", "ledoit_wolf"))
    analysis_seed = int(cfg.get("analysis_seed", 2026))
    pca_mode = str(cfg.get("pca_mode", "joint_pair"))

    out_rows: list[dict[str, Any]] = []

    for r in rows:
        cond_id = str(r["condition_id"])
        emb_path = Path(r["embedding_path"])
        if not emb_path.exists():
            continue
        pack = torch.load(emb_path, map_location="cpu", weights_only=True)
        zi = np.asarray(pack["zi"], dtype=np.float64)
        za = np.asarray(pack["za"], dtype=np.float64)
        zt = np.asarray(pack["zt"], dtype=np.float64)

        n = zi.shape[0]
        seed = analysis_seed + (abs(hash(cond_id)) % 10_000_000)
        idx = deterministic_subsample_indices(n, mi_subsample_n, seed)

        zi = zi[idx]
        za = za[idx]
        zt = zt[idx]

        # MI estimation is sensitive to dimensionality reduction strategy.
        # Default to shared/joint PCA bases to reduce estimator noise from
        # modality-wise marginal PCA truncation.
        if pca_mode == "per_modality":
            zi_it = pca_reduce(zi, mi_pca_dim, seed + 11)
            za_at = pca_reduce(za, mi_pca_dim, seed + 13)
            zt_it = pca_reduce(zt, mi_pca_dim, seed + 17)
            zi_ia = zi_it
            za_ia = za_at
            zi_all = zi_it
            za_all = za_at
            zt_all = zt_it
        elif pca_mode == "joint_all":
            zi_all, za_all, zt_all = pca_reduce_joint_blocks([zi, za, zt], mi_pca_dim, seed + 19)
            zi_it, zt_it = zi_all, zt_all
            za_at, zt_at = za_all, zt_all
            zi_ia, za_ia = zi_all, za_all
        else:
            # joint_pair (default): pair-specific shared bases for MI terms,
            # and an all-modality shared basis for conditional diagnostics.
            zi_it, zt_it = pca_reduce_joint_blocks([zi, zt], mi_pca_dim, seed + 11)
            za_at, zt_at = pca_reduce_joint_blocks([za, zt], mi_pca_dim, seed + 13)
            zi_ia, za_ia = pca_reduce_joint_blocks([zi, za], mi_pca_dim, seed + 17)
            zi_all, za_all, zt_all = pca_reduce_joint_blocks([zi, za, zt], mi_pca_dim, seed + 19)

        mi_ksg_it = ksg_mi(zi_it, zt_it, k=ksg_k)
        mi_ksg_at = ksg_mi(za_at, zt_at, k=ksg_k)
        mi_ksg_ia = ksg_mi(zi_ia, za_ia, k=ksg_k)

        mi_can_it = cancor_mi_proxy(zi_it, zt_it, max_components=min(mi_pca_dim, 16))
        mi_can_at = cancor_mi_proxy(za_at, zt_at, max_components=min(mi_pca_dim, 16))
        mi_can_ia = cancor_mi_proxy(zi_ia, za_ia, max_components=min(mi_pca_dim, 16))

        mi_gau_it = gaussian_mi(zi_it, zt_it, shrinkage=cov_shrinkage, ridge_lambda=ridge_lambda)
        mi_gau_at = gaussian_mi(za_at, zt_at, shrinkage=cov_shrinkage, ridge_lambda=ridge_lambda)
        mi_gau_ia = gaussian_mi(zi_ia, za_ia, shrinkage=cov_shrinkage, ridge_lambda=ridge_lambda)

        cmi_gauss = gaussian_cmi_from_cov(
            zi_all, za_all, zt_all, shrinkage=cov_shrinkage, ridge_lambda=ridge_lambda
        )

        zi_r = residualize(zi_all, zt_all, ridge_lambda=ridge_lambda)
        za_r = residualize(za_all, zt_all, ridge_lambda=ridge_lambda)
        dep_resid_ksg = ksg_mi(zi_r, za_r, k=max(3, min(ksg_k, 8)))
        dep_resid_can = cancor_mi_proxy(zi_r, za_r, max_components=min(mi_pca_dim, 16))

        av_it = float(r.get("source_recall", {}).get("av_it_avg_R", r.get("recomputed_recall", {}).get("av_it_avg_R", 0.0)))
        av_at = float(r.get("source_recall", {}).get("av_at_avg_R", r.get("recomputed_recall", {}).get("av_at_avg_R", 0.0)))
        av_ia = float(r.get("source_recall", {}).get("av_ia_avg_R", r.get("recomputed_recall", {}).get("av_ia_avg_R", 0.0)))

        out = {
            "condition_id": cond_id,
            "source_id": r["source_id"],
            "source_group": r.get("source_group", r["source_id"]),
            "stage_name": r.get("stage_name"),
            "embed_dim": int(r["embed_dim"]),
            "method": r["method"],
            "seed": int(r["seed"]),
            "n_samples_total": int(n),
            "n_samples_used": int(len(idx)),
            "centroid_gap_ia_l2": r.get("centroid_gap_ia_l2"),
            "recall": {
                "av_it_avg_R": av_it,
                "av_at_avg_R": av_at,
                "av_ia_avg_R": av_ia,
            },
            "mi": {
                "ksg": {"it": mi_ksg_it, "at": mi_ksg_at, "ia": mi_ksg_ia},
                "cancor": {"it": mi_can_it, "at": mi_can_at, "ia": mi_can_ia},
                "gaussian": {"it": mi_gau_it, "at": mi_gau_at, "ia": mi_gau_ia},
            },
            "conditional_dependence": {
                "gaussian_cmi_proxy": cmi_gauss,
                "residualized_ksg_proxy": dep_resid_ksg,
                "residualized_cancor_proxy": dep_resid_can,
            },
            "alpha_i": {
                "ksg": alpha_like(mi_ksg_ia, mi_ksg_it, mi_ksg_at),
                "cancor": alpha_like(mi_can_ia, mi_can_it, mi_can_at),
                "gaussian": alpha_like(mi_gau_ia, mi_gau_it, mi_gau_at),
            },
            "alpha_i_unit": {
                "ksg": alpha_like_unit(mi_ksg_ia, mi_ksg_it, mi_ksg_at),
                "cancor": alpha_like_unit(mi_can_ia, mi_can_it, mi_can_at),
                "gaussian": alpha_like_unit(mi_gau_ia, mi_gau_it, mi_gau_at),
            },
            "alpha_recall": alpha_like(av_ia, av_it, av_at),
            "sanity": {
                "all_finite": bool(np.isfinite(np.array([
                    mi_ksg_it, mi_ksg_at, mi_ksg_ia,
                    mi_can_it, mi_can_at, mi_can_ia,
                    mi_gau_it, mi_gau_at, mi_gau_ia,
                    cmi_gauss, dep_resid_ksg, dep_resid_can,
                ])).all()),
                "all_nonnegative": bool(np.all(np.array([
                    mi_ksg_it, mi_ksg_at, mi_ksg_ia,
                    mi_can_it, mi_can_at, mi_can_ia,
                    mi_gau_it, mi_gau_at, mi_gau_ia,
                    cmi_gauss, dep_resid_ksg, dep_resid_can,
                ]) >= -1e-8)),
            },
        }
        out_rows.append(out)

        print(
            f"[stage49] {cond_id} n={len(idx)} "
            f"mi_ia(ksg/can/gau)=({mi_ksg_ia:.4f}/{mi_can_ia:.4f}/{mi_gau_ia:.4f}) "
            f"alphaI_ksg={out['alpha_i']['ksg']:.4f}"
        )

    # shard summary with aggregate-level inference
    xk = np.asarray([r["mi"]["ksg"]["ia"] for r in out_rows], dtype=np.float64)
    xc = np.asarray([r["mi"]["cancor"]["ia"] for r in out_rows], dtype=np.float64)
    yg = np.asarray([r["recall"]["av_ia_avg_R"] for r in out_rows], dtype=np.float64)

    n_boot = int(cfg.get("bootstrap_n", 1000))
    n_perm = int(cfg.get("perm_n", 1000))

    corr_ksg = corrcoef_safe(xk, yg)
    corr_can = corrcoef_safe(xc, yg)
    n_corr = int(len(yg))
    corr_ksg_out = float(corr_ksg) if n_corr >= 3 else None
    corr_can_out = float(corr_can) if n_corr >= 3 else None

    shard_summary = {
        "source_groups": sorted(source_groups),
        "n_rows": len(out_rows),
        "n_rows_for_corr": n_corr,
        "corr_valid": bool(n_corr >= 3),
        "corr_miia_vs_recall": {
            "ksg": {
                "r": corr_ksg_out,
                "bootstrap": _bootstrap_corr_ci(xk, yg, n_boot=n_boot, seed=analysis_seed + 1),
                "permutation": permutation_pvalue_for_corr(xk, yg, n_perm=n_perm, seed=analysis_seed + 2),
            },
            "cancor": {
                "r": corr_can_out,
                "bootstrap": _bootstrap_corr_ci(xc, yg, n_boot=n_boot, seed=analysis_seed + 3),
                "permutation": permutation_pvalue_for_corr(xc, yg, n_perm=n_perm, seed=analysis_seed + 4),
            },
        },
        "alpha_i_ksg": bootstrap_ci(
            np.asarray([r["alpha_i"]["ksg"] for r in out_rows], dtype=np.float64),
            fn=np.mean,
            n_boot=n_boot,
            seed=analysis_seed + 5,
        ),
    }

    incoming = {
        "filters": {
            "source_groups": sorted(source_groups),
            "embed_dims": sorted(dims_filter),
            "methods": sorted(methods_filter),
            "seeds": sorted(seeds_filter),
        },
        "rows": out_rows,
        "shard_summary": shard_summary,
    }
    _save_shard_safe(stage_root / "stage49_sdpi_mi_diagnostics_results.json", incoming)

    provenance = env_snapshot(
        Path(cfg["project_root"]),
        seeds=sorted(seeds_filter) if seeds_filter else [],
        extra={
            "stage": "stage49_sdpi_mi_diagnostics",
            "elapsed_sec": float(time.time() - start),
            "n_rows": len(out_rows),
            "source_groups": sorted(source_groups),
            "pca_mode": pca_mode,
        },
    )
    save_json(provenance, stage_root / "provenance_stage49.json")
    mark_done(markers / "stage49_sdpi_mi_diagnostics.done.json", {
        "elapsed_sec": float(time.time() - start),
        "n_rows": len(out_rows),
        "source_groups": sorted(source_groups),
        "pca_mode": pca_mode,
    })

    print(f"[stage49] complete rows={len(out_rows)} corr_ksg={corr_ksg:.4f} corr_can={corr_can:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    run(cfg)


if __name__ == "__main__":
    main()
