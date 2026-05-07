from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from ..common import env_snapshot, load_json, mark_done, save_json, set_seed
from ..data.datasets import KarpathyCache
from ..eval.privacy import mlp_inversion_attack, pseudoinverse_reconstruction, split_coordinate_error
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
    phi = torch.tensor(
        kane_nelson_jl(d, m, eps=jl_eps, seed=jl_seed).toarray(),
        dtype=torch.float32,
    )
    g = torch.Generator().manual_seed(mask_seed)
    visible = (torch.rand(d, generator=g) < p).float()
    mask_mat = torch.diag(visible)
    a = torch.cat([alpha * phi, beta * mask_mat], dim=0)
    return a, visible


def _build_jl_only_transform(
    d: int,
    m: int,
    *,
    jl_eps: float,
    jl_seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    phi = torch.tensor(
        kane_nelson_jl(d, m, eps=jl_eps, seed=jl_seed).toarray(),
        dtype=torch.float32,
    )
    visible = torch.zeros(d, dtype=torch.float32)
    return phi, visible


def _aggregate_rows(rows: list[dict]) -> dict:
    mlp = [float(r["mlp_inverter"]["mean_relative_reconstruction_error"]) for r in rows]
    pseudo = [float(r["pseudoinverse"]["mean_relative_reconstruction_error"]) for r in rows]
    vis = [float(r["visible_fraction"]) for r in rows]
    return {
        "n": len(rows),
        "mlp_rel_error": mean_std_ci(mlp),
        "pseudo_rel_error": mean_std_ci(pseudo),
        "visible_fraction": mean_std_ci(vis),
    }


def _run_attack_block(
    *,
    x: torch.Tensor,
    d: int,
    m: int,
    seeds: list[int],
    ps: list[float],
    alpha: float,
    beta: float,
    jl_eps: float,
    jl_seed: int,
    mask_seed_base: int,
    attack_cfg: dict,
    device: str,
) -> tuple[dict, dict]:
    raw: dict[str, list[dict]] = {"jl_only": []}
    for p in ps:
        raw[f"concat_p{str(p).replace('.', 'p')}"] = []

    for seed in seeds:
        set_seed(seed)
        a_jl, vis_jl = _build_jl_only_transform(d=d, m=m, jl_eps=jl_eps, jl_seed=jl_seed + seed)
        z_jl = x @ a_jl.T
        pseudo_jl = pseudoinverse_reconstruction(z=z_jl, x=x, transform=a_jl)
        mlp_jl, xh_jl, xt_jl = mlp_inversion_attack(
            z=z_jl,
            x=x,
            train_frac=float(attack_cfg["train_frac"]),
            hidden_dim=int(attack_cfg["hidden_dim"]),
            epochs=int(attack_cfg["epochs"]),
            lr=float(attack_cfg["lr"]),
            batch_size=int(attack_cfg["batch_size"]),
            seed=seed,
            device=device,
        )
        split_jl = split_coordinate_error(xh_jl, xt_jl, vis_jl)
        raw["jl_only"].append(
            {
                "seed": seed,
                "method": "jl_only",
                "embed_dim": m,
                "n_samples": int(len(x)),
                "visible_fraction": float(vis_jl.mean().item()),
                "mlp_inverter": {**mlp_jl, **split_jl},
                "pseudoinverse": {**pseudo_jl, **split_jl},
            }
        )

        for p in ps:
            key = f"concat_p{str(p).replace('.', 'p')}"
            a, visible = _build_concat_transform(
                d=d,
                m=m,
                p=p,
                alpha=alpha,
                beta=beta,
                jl_eps=jl_eps,
                jl_seed=jl_seed + seed,
                mask_seed=mask_seed_base + seed,
            )
            z = x @ a.T
            pseudo = pseudoinverse_reconstruction(z=z, x=x, transform=a)
            mlp, x_hat, x_true = mlp_inversion_attack(
                z=z,
                x=x,
                train_frac=float(attack_cfg["train_frac"]),
                hidden_dim=int(attack_cfg["hidden_dim"]),
                epochs=int(attack_cfg["epochs"]),
                lr=float(attack_cfg["lr"]),
                batch_size=int(attack_cfg["batch_size"]),
                seed=seed,
                device=device,
            )
            split_mlp = split_coordinate_error(x_hat, x_true, visible)
            c = a.T.float()
            pinv = torch.linalg.pinv(c)
            xh_pseudo = z.float() @ pinv
            split_pseudo = split_coordinate_error(xh_pseudo, x.float(), visible)
            rec = {
                "seed": seed,
                "method": key,
                "p": p,
                "embed_dim": m,
                "n_samples": int(len(x)),
                "visible_fraction": float(visible.mean().item()),
                "mlp_inverter": {**mlp, **split_mlp},
                "pseudoinverse": {**pseudo, **split_pseudo},
            }
            raw[key].append(rec)
            print(
                f"privacy-parity method={key} seed={seed} "
                f"mlp_rel={mlp['mean_relative_reconstruction_error']:.4f} "
                f"pseudo_rel={pseudo['mean_relative_reconstruction_error']:.4f}"
            )

    stats = {k: _aggregate_rows(v) for k, v in raw.items()}
    return raw, stats


def run(cfg: dict) -> None:
    start = time.time()
    output_root = Path(cfg["output_root"]).resolve()
    stage_root = output_root / "stage10_privacy_parity"
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(cfg["cache_root"]) / "coco"
    cache = KarpathyCache.from_paths(
        cache_dir / cfg["image_cache_file"],
        cache_dir / cfg["text_cache_file"],
        cache_dir / "metadata.json",
    )
    split_name = cfg.get("privacy_split", "train_restval")
    idx = cache.split_indices(split_name)
    x_full = cache.image_feats[idx].float()

    max_samples = int(cfg.get("max_samples", 0))
    if max_samples > 0 and len(x_full) > max_samples:
        x_full = x_full[:max_samples]

    legacy_max_samples = int(cfg.get("legacy_max_samples", 1000))
    x_legacy = x_full[:legacy_max_samples] if legacy_max_samples > 0 else x_full

    seeds = list(cfg["seeds"])
    m = int(cfg.get("privacy_embed_dim", 256))
    ps = [float(p) for p in cfg.get("privacy_ps", [0.0, 0.1, 0.5, 1.0])]
    alpha = float(cfg.get("alpha", 1.0))
    beta = float(cfg.get("beta", 1.0))
    jl_eps = float(cfg["jl_eps"])
    jl_seed = int(cfg["jl_seed"])
    mask_seed_base = int(cfg.get("mask_seed_base", 1000))

    parity_cfg = {
        "train_frac": float(cfg.get("attack_train_frac", 0.8)),
        "hidden_dim": int(cfg.get("attack_hidden_dim", 1024)),
        "epochs": int(cfg.get("attack_epochs", 80)),
        "lr": float(cfg.get("attack_lr", 1e-3)),
        "batch_size": int(cfg.get("attack_batch_size", 1024)),
    }
    legacy_cfg = {
        "train_frac": float(cfg.get("legacy_attack_train_frac", 0.8)),
        "hidden_dim": int(cfg.get("legacy_attack_hidden_dim", 512)),
        "epochs": int(cfg.get("legacy_attack_epochs", 100)),
        "lr": float(cfg.get("legacy_attack_lr", 1e-3)),
        "batch_size": int(cfg.get("legacy_attack_batch_size", 512)),
    }

    raw_parity, stats_parity = _run_attack_block(
        x=x_full,
        d=int(x_full.shape[1]),
        m=m,
        seeds=seeds,
        ps=ps,
        alpha=alpha,
        beta=beta,
        jl_eps=jl_eps,
        jl_seed=jl_seed,
        mask_seed_base=mask_seed_base,
        attack_cfg=parity_cfg,
        device=cfg["device"],
    )

    raw_legacy, stats_legacy = _run_attack_block(
        x=x_legacy,
        d=int(x_legacy.shape[1]),
        m=m,
        seeds=seeds,
        ps=ps,
        alpha=alpha,
        beta=beta,
        jl_eps=jl_eps,
        jl_seed=jl_seed,
        mask_seed_base=mask_seed_base,
        attack_cfg=legacy_cfg,
        device=cfg["device"],
    )

    e5_file = Path(cfg.get("e5_results_file", "")).resolve() if cfg.get("e5_results_file") else None
    e5_reference = {}
    if e5_file and e5_file.exists():
        e5_reference = load_json(e5_file)

    stage7_file = Path(cfg.get("stage7_results_file", "")).resolve() if cfg.get("stage7_results_file") else None
    stage7_reference = {}
    if stage7_file and stage7_file.exists():
        stage7_reference = load_json(stage7_file)

    stage8_file = Path(cfg.get("stage8_results_file", "")).resolve() if cfg.get("stage8_results_file") else None
    stage8_reference = {}
    if stage8_file and stage8_file.exists():
        stage8_reference = load_json(stage8_file)

    e5_m256 = (
        e5_reference.get("privacy_curve", {})
        .get(str(m), {})
        .get("neural_inverter", {})
        .get("mean_relative_reconstruction_error")
    )
    stage7_p0 = (
        stage7_reference.get("stats", {})
        .get(f"m{m}", {})
        .get("p0p0", {})
        .get("mlp_inverter", {})
        .get("mean_relative_reconstruction_error", {})
        .get("mean")
    )

    dp_stats = {}
    for eps_key, rows in stage8_reference.get("raw", {}).items():
        mlp_vals = [float(r["mlp_inverter"]["mean_relative_reconstruction_error"]) for r in rows]
        pseudo_vals = [float(r["pseudoinverse"]["mean_relative_reconstruction_error"]) for r in rows]
        avg_r = [float(r["avg_R"]) for r in rows]
        dp_stats[eps_key] = {
            "n": len(rows),
            "avg_R": mean_std_ci(avg_r),
            "mlp_rel_error": mean_std_ci(mlp_vals),
            "pseudo_rel_error": mean_std_ci(pseudo_vals),
        }

    parity_jl_mean = stats_parity["jl_only"]["mlp_rel_error"]["mean"]
    legacy_jl_mean = stats_legacy["jl_only"]["mlp_rel_error"]["mean"]

    summary = {
        "stage": "stage10_privacy_parity",
        "config": cfg,
        "raw": {
            "parity_attack": raw_parity,
            "legacy_attack": raw_legacy,
        },
        "stats": {
            "parity_attack": stats_parity,
            "legacy_attack": stats_legacy,
            "dpsgd_reference": dp_stats,
        },
        "references": {
            "e5_m256_mlp_rel_error": e5_m256,
            "stage7_m256_p0_mlp_rel_error": stage7_p0,
        },
        "discrepancy_analysis": {
            "parity_vs_legacy_jl_only_delta": float(parity_jl_mean - legacy_jl_mean),
            "parity_vs_e5_reported_delta": (float(parity_jl_mean - e5_m256) if e5_m256 is not None else None),
            "legacy_vs_e5_reported_delta": (float(legacy_jl_mean - e5_m256) if e5_m256 is not None else None),
            "parity_vs_stage7_p0_delta": (float(parity_jl_mean - stage7_p0) if stage7_p0 is not None else None),
            "notes": [
                "parity_attack uses Stage7/Stage8 attacker settings (50k samples, hidden=1024, epochs=80).",
                "legacy_attack emulates E5 attacker capacity/sample budget (1k samples, hidden=512, epochs=100).",
            ],
        },
    }

    provenance = env_snapshot(
        Path(cfg["project_root"]),
        seeds=seeds,
        extra={
            "stage": "stage10_privacy_parity",
            "elapsed_sec": time.time() - start,
            "n_samples_parity": int(len(x_full)),
            "n_samples_legacy": int(len(x_legacy)),
        },
    )
    save_json(provenance, stage_root / "provenance_stage10_privacy_parity.json")
    save_json(summary, stage_root / "E10_privacy_parity_results.json")
    mark_done(markers / "stage10_privacy_parity.done.json", {"elapsed_sec": time.time() - start})
    print("Stage 10 (privacy parity) complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)
