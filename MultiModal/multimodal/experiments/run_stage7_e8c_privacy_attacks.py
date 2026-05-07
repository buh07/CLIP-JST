from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import yaml

from ..common import env_snapshot, mark_done, save_json, set_seed
from ..data.datasets import KarpathyCache
from ..eval.privacy import (
    mlp_inversion_attack,
    pseudoinverse_reconstruction,
    split_coordinate_error,
)
from ..eval.stats import mean_std_ci
from ..models.jl_ops import kane_nelson_jl


def _build_concat_transform(
    d: int,
    m: int,
    *,
    p: float,
    alpha: float,
    beta: float,
    jl_eps: float,
    jl_seed: int,
    mask_seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      - transform A with shape (m + d, d), where z = x @ A.T
      - visible_mask over raw coordinates (length d) for split metrics
    """
    phi = torch.tensor(
        kane_nelson_jl(d, m, eps=jl_eps, seed=jl_seed).toarray(),
        dtype=torch.float32,
    )
    g = torch.Generator().manual_seed(mask_seed)
    visible = (torch.rand(d, generator=g) < p).float()
    mask_mat = torch.diag(visible)
    top = alpha * phi
    bottom = beta * mask_mat
    a = torch.cat([top, bottom], dim=0)
    return a, visible


def _aggregate(records: list[dict], keys: list[str]) -> dict[str, dict[str, float]]:
    out = {}
    for k in keys:
        vals = [float(r[k]) for r in records if k in r and r[k] is not None and not torch.isnan(torch.tensor(r[k]))]
        if vals:
            out[k] = mean_std_ci(vals)
    return out


def run(cfg: dict) -> None:
    start = time.time()
    output_root = Path(cfg["output_root"]).resolve()
    stage_root = output_root / "stage7_e8c"
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    # COCO-only by plan.
    cache_dir = Path(cfg["cache_root"]) / "coco"
    cache = KarpathyCache.from_paths(
        cache_dir / cfg["image_cache_file"],
        cache_dir / cfg["text_cache_file"],
        cache_dir / "metadata.json",
    )
    split_name = cfg.get("privacy_split", "train_restval")
    idx = cache.split_indices(split_name)
    x = cache.image_feats[idx].float()

    max_samples = int(cfg.get("max_samples", 0))
    if max_samples > 0 and len(x) > max_samples:
        x = x[:max_samples]

    seeds = list(cfg["seeds"])
    ps = [float(p) for p in cfg["privacy_ps"]]
    embed_dims = list(cfg.get("privacy_embed_dims", [256]))

    results = {
        "stage": "stage7_e8c_privacy_attacks",
        "config": cfg,
        "raw": {},
        "stats": {},
    }

    d = int(x.shape[1])
    for m in embed_dims:
        m_key = f"m{m}"
        results["raw"][m_key] = {}
        for p in ps:
            p_key = f"p{str(p).replace('.', 'p')}"
            rows = []
            for seed in seeds:
                set_seed(seed)
                a, visible_mask = _build_concat_transform(
                    d=d,
                    m=m,
                    p=p,
                    alpha=float(cfg.get("alpha", 1.0)),
                    beta=float(cfg.get("beta", 1.0)),
                    jl_eps=float(cfg["jl_eps"]),
                    jl_seed=int(cfg["jl_seed"]) + seed,
                    mask_seed=int(cfg.get("mask_seed_base", 1000)) + seed,
                )
                z = x @ a.T
                pseudo = pseudoinverse_reconstruction(z=z, x=x, transform=a)
                mlp, x_hat, x_true = mlp_inversion_attack(
                    z=z,
                    x=x,
                    train_frac=float(cfg.get("attack_train_frac", 0.8)),
                    hidden_dim=int(cfg.get("attack_hidden_dim", 1024)),
                    epochs=int(cfg.get("attack_epochs", 80)),
                    lr=float(cfg.get("attack_lr", 1e-3)),
                    batch_size=int(cfg.get("attack_batch_size", 1024)),
                    seed=seed,
                    device=cfg["device"],
                )
                split_mlp = split_coordinate_error(x_hat, x_true, visible_mask)
                # Pseudoinverse split is computed on full set using full projection formula.
                # Build a deterministic pseudo reconstruction for split reporting.
                c = a.T.float()
                pinv = torch.linalg.pinv(c)
                x_hat_pseudo = z.float() @ pinv
                split_pseudo = split_coordinate_error(x_hat_pseudo, x.float(), visible_mask)

                rec = {
                    "seed": seed,
                    "embed_dim": m,
                    "p": p,
                    "n_samples": int(len(x)),
                    "visible_fraction": float(visible_mask.mean().item()),
                    "mlp_inverter": {**mlp, **split_mlp},
                    "pseudoinverse": {**pseudo, **split_pseudo},
                }
                rows.append(rec)
                print(
                    f"E8c m={m} p={p} seed={seed} "
                    f"mlp_rel={mlp['mean_relative_reconstruction_error']:.4f} "
                    f"pseudo_rel={pseudo['mean_relative_reconstruction_error']:.4f}"
                )
            results["raw"][m_key][p_key] = rows

        # Aggregate by p across seeds.
        results["stats"][m_key] = {}
        for p in ps:
            p_key = f"p{str(p).replace('.', 'p')}"
            rows = results["raw"][m_key][p_key]
            mlp_rows = [r["mlp_inverter"] for r in rows]
            pseudo_rows = [r["pseudoinverse"] for r in rows]
            results["stats"][m_key][p_key] = {
                "mlp_inverter": _aggregate(
                    mlp_rows,
                    keys=[
                        "mean_relative_reconstruction_error",
                        "mean_abs_reconstruction_error",
                        "rmse",
                        "visible_coord_mse",
                        "hidden_coord_mse",
                    ],
                ),
                "pseudoinverse": _aggregate(
                    pseudo_rows,
                    keys=[
                        "mean_relative_reconstruction_error",
                        "mean_abs_reconstruction_error",
                        "rmse",
                        "visible_coord_mse",
                        "hidden_coord_mse",
                    ],
                ),
                "visible_fraction_mean": mean_std_ci([float(r["visible_fraction"]) for r in rows]),
            }

    provenance = env_snapshot(
        Path(cfg["project_root"]),
        seeds=seeds,
        extra={
            "stage": "stage7_e8c_privacy_attacks",
            "elapsed_sec": time.time() - start,
            "n_samples": int(len(x)),
        },
    )
    save_json(provenance, stage_root / "provenance_stage7_e8c.json")
    save_json(results, stage_root / "E8c_privacy_results.json")
    mark_done(markers / "stage7_e8c.done.json", {"elapsed_sec": time.time() - start})
    print("Stage 7 (E8c) complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)
