from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import yaml

from ..common import env_snapshot, load_json, mark_done, save_json
from ..data.datasets import KarpathyCache
from ..eval.privacy import iterative_feature_inversion_attack, mlp_inversion_attack, reconstruction_metrics
from ..eval.stats import build_metric_report
from .run_stage13_federated_comparison import build_model_from_spec, checkpoint_path


def _encode_images(model, x: torch.Tensor, *, device: str, batch_size: int) -> torch.Tensor:
    model = model.to(device).eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            outs.append(model.encode_image(x[i:i + batch_size].to(device)).cpu())
    return torch.cat(outs, dim=0)


def _linear_probe_reconstruction(z: torch.Tensor, x: torch.Tensor, *, train_frac: float, seed: int) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    z = z.float().cpu()
    x = x.float().cpu()
    n = len(z)
    n_train = max(1, int(train_frac * n))
    n_test = n - n_train
    if n_test <= 0:
        n_train = n - 1
        n_test = 1
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    tr = perm[:n_train]
    te = perm[n_train:]
    b = torch.linalg.pinv(z[tr]) @ x[tr]
    x_hat = z[te] @ b
    return reconstruction_metrics(x_hat, x[te]), x_hat, x[te]


def run(cfg: dict) -> None:
    start = time.time()
    output_root = Path(cfg["output_root"]).resolve()
    stage_root = output_root / "stage14_stronger_attacks"
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    stage13_base = Path(cfg.get("stage13_root", output_root)).resolve()
    stage13_root = stage13_base / "stage13_federated"
    if not stage13_root.exists():
        raise FileNotFoundError(f"stage13 root missing: {stage13_root}")

    cache_dir = Path(cfg["cache_root"]) / "coco"
    cache = KarpathyCache.from_paths(
        cache_dir / cfg["image_cache_file"],
        cache_dir / cfg["text_cache_file"],
        cache_dir / "metadata.json",
    )

    probe_split = cfg.get("privacy_split", "train_restval")
    probe_idx = cache.split_indices(probe_split)
    x_all = cache.image_feats[probe_idx].float()
    max_probe = int(cfg.get("max_probe_samples", 0))

    seeds = list(cfg["seeds"])
    methods = list(cfg["methods"])
    partitions = list(cfg["partitions"])
    method_filter = set(cfg.get("method_filter", []))
    partition_filter = set(cfg.get("partition_filter", []))

    if method_filter:
        methods = [m for m in methods if m["id"] in method_filter]
    if partition_filter:
        partitions = [p for p in partitions if p in partition_filter]

    probe_by_seed: dict[int, torch.Tensor] = {}
    for seed in seeds:
        if max_probe > 0 and len(x_all) > max_probe:
            g = torch.Generator().manual_seed(int(seed) + int(cfg.get("probe_sample_seed_offset", 15485863)))
            sel = torch.randperm(len(x_all), generator=g)[:max_probe]
            probe_by_seed[int(seed)] = x_all[sel]
        else:
            probe_by_seed[int(seed)] = x_all

    result_path = stage_root / "E14_stronger_attacks_results.json"
    if result_path.exists():
        results = load_json(result_path)
        results["stage"] = "stage14_stronger_attacks"
        results["config"] = cfg
        results.setdefault("raw", {})
        results.setdefault("stats", {})
    else:
        results = {
            "stage": "stage14_stronger_attacks",
            "config": cfg,
            "raw": {},
            "stats": {},
        }

    for partition_id in partitions:
        results["raw"].setdefault(partition_id, {})
        for spec in methods:
            method_id = spec["id"]
            rows = []
            for seed in seeds:
                run_dir = stage_root / partition_id / method_id / f"seed{seed}"
                eval_file = run_dir / "eval.json"
                if eval_file.exists():
                    rows.append(load_json(eval_file))
                    continue

                model = build_model_from_spec(spec, cfg)
                ckpt = checkpoint_path(stage13_root, partition_id, method_id, seed)
                if not ckpt.exists():
                    raise FileNotFoundError(f"missing stage13 checkpoint: {ckpt}")
                state = torch.load(ckpt, map_location=cfg["device"], weights_only=True)
                model.load_state_dict(state, strict=True)
                model = model.to(cfg["device"])
                x_seed = probe_by_seed[int(seed)]
                with torch.no_grad():
                    probe_feat = x_seed[:1].to(cfg["device"])
                    out_dim = int(model.encode_image(probe_feat).shape[1])

                z = _encode_images(model, x_seed, device=cfg["device"], batch_size=int(cfg.get("eval_batch_size", 4096)))
                lin, _lin_hat, _lin_true = _linear_probe_reconstruction(
                    z,
                    x_seed,
                    train_frac=float(cfg.get("attack_train_frac", 0.8)),
                    seed=seed,
                )
                mlp, _mlp_hat, _mlp_true = mlp_inversion_attack(
                    z,
                    x_seed,
                    train_frac=float(cfg.get("attack_train_frac", 0.8)),
                    hidden_dim=int(cfg.get("attack_hidden_dim", 1024)),
                    epochs=int(cfg.get("attack_epochs", 80)),
                    lr=float(cfg.get("attack_lr", 1e-3)),
                    batch_size=int(cfg.get("attack_batch_size", 1024)),
                    seed=seed,
                    device=cfg["device"],
                )
                iterative, _it_hat, _it_true = iterative_feature_inversion_attack(
                    model,
                    x_seed,
                    train_frac=float(cfg.get("attack_train_frac", 0.8)),
                    steps=int(cfg.get("iterative_steps", 250)),
                    lr=float(cfg.get("iterative_lr", 0.05)),
                    prior_weight=float(cfg.get("iterative_prior_weight", 1e-3)),
                    batch_size=int(cfg.get("iterative_batch_size", 256)),
                    seed=seed,
                    device=cfg["device"],
                )

                rec = {
                    "seed": int(seed),
                    "partition": partition_id,
                    "method": method_id,
                    "embedding_output_dim": int(out_dim),
                    "embed_dim_target": int(cfg.get("embed_dim", out_dim)),
                    "embed_dim_matched": bool(out_dim == int(cfg.get("embed_dim", out_dim))),
                    "n_probe": int(len(x_seed)),
                    "linear_probe": lin,
                    "mlp_inverter": mlp,
                    "iterative_inverter": iterative,
                    "linear_rel_error": float(lin["mean_relative_reconstruction_error"]),
                    "mlp_rel_error": float(mlp["mean_relative_reconstruction_error"]),
                    "iterative_rel_error": float(iterative["mean_relative_reconstruction_error"]),
                }
                save_json(rec, eval_file)
                rows.append(rec)
                print(
                    f"stage14 partition={partition_id} method={method_id} seed={seed} "
                    f"lin={rec['linear_rel_error']:.4f} mlp={rec['mlp_rel_error']:.4f} "
                    f"iter={rec['iterative_rel_error']:.4f}"
                )

            results["raw"][partition_id][method_id] = rows

        results["stats"][partition_id] = build_metric_report(
            results["raw"][partition_id],
            metrics=["linear_rel_error", "mlp_rel_error", "iterative_rel_error"],
            baseline_method=cfg.get("baseline_method", "mask_concat"),
        )

    provenance = env_snapshot(
        Path(cfg["project_root"]),
        seeds=seeds,
        extra={
            "stage": "stage14_stronger_attacks",
            "elapsed_sec": time.time() - start,
            "n_probe": int(max(len(v) for v in probe_by_seed.values()) if probe_by_seed else 0),
        },
    )
    save_json(provenance, stage_root / "provenance_stage14.json")
    save_json(results, result_path)
    mark_done(markers / "stage14_stronger_attacks.done.json", {"elapsed_sec": time.time() - start})
    print("Stage 14 (stronger attacks) complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)
