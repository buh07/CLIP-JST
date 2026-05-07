from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from ..common import env_snapshot, load_json, mark_done, save_json, set_seed
from ..data.datasets import KarpathyCache
from ..eval.diagnostics import pair_diagnostics
from ..eval.retrieval import recall_at_k
from ..eval.stats import build_metric_report
from ..models import (
    CLIPProjectionHead,
    DirectCLRProxyHead,
    FrozenPretrainedCLIPHead,
    FrozenPretrainedCLIPMahalHead,
    LearnedJLSparseHead,
    MRLProjectionHead,
    OrthogonalProjectionHead,
    OrthogonalPlusMahalanobisHead,
    RandomJLMahalanobisHead,
    RandomJLOnlyHead,
    SparseJLL1Head,
    SparseJLProjectedHead,
    SpectralAlignedProjectionHead,
)
from ..training import train_bimodal


def _load_best(model, ckpt_dir: Path, device: str):
    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=True))
    return model.to(device)


@torch.no_grad()
def _eval_karpathy(
    model,
    cache: KarpathyCache,
    split_name: str,
    *,
    device: str,
    batch_size: int,
    mode: str = "standard",
    mrl_dim: int | None = None,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    model.eval()
    img, txt, gt_i2t, gt_t2i = cache.eval_tensors(split_name)
    all_i, all_t = [], []

    for i in range(0, len(img), batch_size):
        x = img[i:i + batch_size].to(device)
        if mode == "mrl":
            all_i.append(model.encode_image(x, dim=mrl_dim).cpu())
        else:
            all_i.append(model.encode_image(x).cpu())

    for i in range(0, len(txt), batch_size):
        x = txt[i:i + batch_size].to(device)
        if mode == "mrl":
            all_t.append(model.encode_text(x, dim=mrl_dim).cpu())
        else:
            all_t.append(model.encode_text(x).cpu())

    zi = torch.cat(all_i)
    zt = torch.cat(all_t)
    metrics = recall_at_k(zi, zt, gt_i2t=gt_i2t, gt_t2i=gt_t2i)
    return metrics, zi, zt


def _train_or_skip(
    model,
    train_loader,
    val_loader,
    *,
    seed: int,
    out_dir: Path,
    cfg: dict,
    mode: str = "standard",
    mrl_dims: list[int] | None = None,
    skip_training: bool = False,
):
    result_file = out_dir / "train_result.json"
    if result_file.exists():
        return load_json(result_file)

    if skip_training:
        result = {"skipped_training": True, "seed": seed}
        save_json(result, result_file)
        return result

    res = train_bimodal(
        model,
        train_loader,
        val_loader,
        epochs=cfg["epochs"],
        lr=cfg["lr"],
        device=cfg["device"],
        ckpt_dir=out_dir,
        patience=cfg.get("patience", 10),
        warmup_epochs=cfg.get("warmup_epochs", 0),
        mode=mode,
        mrl_dims=mrl_dims,
    )
    res["seed"] = seed
    save_json(res, result_file)
    return res


def _format_lambda_name(lam: float) -> str:
    base_str = f"{lam:.0e}"
    base, exp = base_str.split("e")
    return f"{base}e{int(exp)}"


def _dim_regime(embed_dim: int, vision_dim: int, text_dim: int) -> dict[str, str | bool]:
    img_exp = embed_dim > vision_dim
    txt_exp = embed_dim > text_dim
    if not img_exp and not txt_exp:
        regime = "compression"
    elif img_exp and txt_exp:
        regime = "expansion"
    else:
        regime = "mixed"
    return {
        "embed_dim": int(embed_dim),
        "vision_dim": int(vision_dim),
        "text_dim": int(text_dim),
        "image_expansion": bool(img_exp),
        "text_expansion": bool(txt_exp),
        "projection_regime": regime,
    }


def run(cfg: dict) -> None:
    start = time.time()
    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)
    write_provenance = bool(cfg.get("write_provenance", True))
    write_summary = bool(cfg.get("write_summary", True))
    write_done_marker = bool(cfg.get("write_done_marker", True))

    seeds = list(cfg["seeds"])
    embed_dims = list(cfg["embed_dims"])
    mrl_max_dim = max(embed_dims)
    method_filter = set(cfg.get("method_filter", []))
    allow_dim_expansion = bool(cfg.get("allow_dim_expansion", True))

    run_summary: dict = {
        "stage": "stage2_e7_karpathy",
        "datasets": {},
        "config": cfg,
    }

    for dataset in cfg["datasets"]:
        print(f"\n=== Dataset: {dataset} ===")
        cache_dir = Path(cfg["cache_root"]) / dataset
        cache = KarpathyCache.from_paths(
            cache_dir / cfg["image_cache_file"],
            cache_dir / cfg["text_cache_file"],
            cache_dir / "metadata.json",
        )

        if dataset == "coco":
            train_split = "train_restval"
            val_split = "val"
            test_split = "test"
        else:
            train_split = "train"
            val_split = "val"
            test_split = "test"

        train_ds = cache.make_train_dataset(train_split, training=True)
        val_ds = cache.make_train_dataset(val_split, training=False)

        loader_kw = {
            "batch_size": cfg["batch_size"],
            "num_workers": cfg.get("num_workers", 4),
            "pin_memory": True,
        }
        train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
        val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)

        dataset_out = output_root / dataset
        dataset_out.mkdir(parents=True, exist_ok=True)

        # Per m/method/seed raw records for statistical aggregation.
        by_m_method: dict[str, dict[str, list[dict]]] = {f"m{m}": {} for m in embed_dims}

        for seed in seeds:
            print(f"\n-- seed={seed} --")
            set_seed(seed)

            run_mrl = (not method_filter) or ("mrl" in method_filter)
            if run_mrl:
                # MRL: single training per seed at max dim, evaluated across all m.
                mrl_seed_dir = dataset_out / "mrl" / f"seed{seed}"
                mrl_model = MRLProjectionHead(
                    vision_dim=cfg["vision_dim"],
                    text_dim=cfg["text_dim"],
                    max_dim=mrl_max_dim,
                    nested_dims=embed_dims,
                )
                _train_or_skip(
                    mrl_model,
                    train_loader,
                    val_loader,
                    seed=seed,
                    out_dir=mrl_seed_dir,
                    cfg=cfg,
                    mode="mrl",
                    mrl_dims=embed_dims,
                )
                mrl_model = _load_best(mrl_model, mrl_seed_dir, cfg["device"])

                for m in embed_dims:
                    m_key = f"m{m}"
                    by_m_method[m_key].setdefault("mrl", [])
                    eval_file = dataset_out / m_key / "mrl" / f"seed{seed}" / "eval.json"
                    if eval_file.exists():
                        rec = load_json(eval_file)
                        by_m_method[m_key]["mrl"].append(rec)
                        continue

                    metrics, zi, zt = _eval_karpathy(
                        mrl_model,
                        cache,
                        test_split,
                        device=cfg["device"],
                        batch_size=cfg["eval_batch_size"],
                        mode="mrl",
                        mrl_dim=m,
                    )
                    diag = pair_diagnostics(zi, zt, prefix="it")
                    rec = {"seed": seed, **metrics, **diag, "n_params": mrl_model.n_trainable_params()}
                    save_json(rec, eval_file)
                    by_m_method[m_key]["mrl"].append(rec)

            for m in embed_dims:
                m_key = f"m{m}"
                regime_meta = _dim_regime(int(m), int(cfg["vision_dim"]), int(cfg["text_dim"]))
                if (regime_meta["image_expansion"] or regime_meta["text_expansion"]) and not allow_dim_expansion:
                    raise ValueError(
                        f"embed_dim={m} enters {regime_meta['projection_regime']} regime "
                        f"(image_expansion={regime_meta['image_expansion']}, text_expansion={regime_meta['text_expansion']}) "
                        "but allow_dim_expansion=False"
                    )
                if regime_meta["projection_regime"] != "compression":
                    print(
                        f"[dim-regime] {dataset} {m_key}: {regime_meta['projection_regime']} "
                        f"(img_expand={regime_meta['image_expansion']}, txt_expand={regime_meta['text_expansion']})"
                    )
                builders = {
                    "clip_head": lambda: CLIPProjectionHead(
                        vision_dim=cfg["vision_dim"],
                        text_dim=cfg["text_dim"],
                        embed_dim=m,
                    ),
                    "random_jl_mahal": lambda: RandomJLMahalanobisHead(
                        vision_dim=cfg["vision_dim"],
                        text_dim=cfg["text_dim"],
                        embed_dim=m,
                        jl_eps=cfg["jl_eps"],
                        jl_seed=cfg["jl_seed"],
                    ),
                    "orth_jl_trainable": lambda: OrthogonalProjectionHead(
                        vision_dim=cfg["vision_dim"],
                        text_dim=cfg["text_dim"],
                        embed_dim=m,
                        orth_reg=cfg.get("orth_reg", 1e-3),
                    ),
                    "learned_jl_sparse": lambda: LearnedJLSparseHead(
                        vision_dim=cfg["vision_dim"],
                        text_dim=cfg["text_dim"],
                        embed_dim=m,
                        jl_eps=cfg["jl_eps"],
                        jl_seed=cfg["jl_seed"],
                        sparsity_reg=cfg.get("sparsity_reg", 1e-4),
                    ),
                    "directclr_proxy": lambda: DirectCLRProxyHead(
                        vision_dim=cfg["vision_dim"],
                        text_dim=cfg["text_dim"],
                        full_dim=cfg.get("directclr_full_dim", mrl_max_dim),
                        train_subdim=m,
                        seed=cfg["jl_seed"] + seed,
                    ),
                    "random_jl_only": lambda: RandomJLOnlyHead(
                        vision_dim=cfg["vision_dim"],
                        text_dim=cfg["text_dim"],
                        embed_dim=m,
                        jl_eps=cfg["jl_eps"],
                        jl_seed=cfg["jl_seed"],
                    ),
                    "sparse_jl_projected": lambda: SparseJLProjectedHead(
                        vision_dim=cfg["vision_dim"],
                        text_dim=cfg["text_dim"],
                        embed_dim=m,
                        jl_eps=cfg["jl_eps"],
                        jl_seed=cfg["jl_seed"],
                    ),
                    "orth_jl_plus_mahal": lambda: OrthogonalPlusMahalanobisHead(
                        vision_dim=cfg["vision_dim"],
                        text_dim=cfg["text_dim"],
                        embed_dim=m,
                        orth_reg=cfg.get("orth_reg", 1e-3),
                    ),
                    "frozen_clip_pretrained": lambda: FrozenPretrainedCLIPHead(
                        vision_dim=cfg["vision_dim"],
                        text_dim=cfg["text_dim"],
                        embed_dim=m,
                        pretrained_model_name=cfg.get("pretrained_clip_model", "openai/clip-vit-base-patch32"),
                        jl_eps=cfg["jl_eps"],
                        jl_seed=int(cfg.get("pretrained_proj_seed", cfg["jl_seed"])),
                        train_logit_scale=bool(cfg.get("train_logit_scale", True)),
                    ),
                    "frozen_clip_pretrained_mahal": lambda: FrozenPretrainedCLIPMahalHead(
                        vision_dim=cfg["vision_dim"],
                        text_dim=cfg["text_dim"],
                        embed_dim=m,
                        pretrained_model_name=cfg.get("pretrained_clip_model", "openai/clip-vit-base-patch32"),
                        jl_eps=cfg["jl_eps"],
                        jl_seed=int(cfg.get("pretrained_proj_seed", cfg["jl_seed"])),
                        train_logit_scale=bool(cfg.get("train_logit_scale", True)),
                    ),
                    "spectral_align_trainable": lambda: SpectralAlignedProjectionHead(
                        vision_dim=cfg["vision_dim"],
                        text_dim=cfg["text_dim"],
                        embed_dim=m,
                        pretrained_model_name=cfg.get("pretrained_clip_model", "openai/clip-vit-base-patch32"),
                        spectral_reg=float(cfg.get("spectral_reg", 1e-2)),
                    ),
                }
                for reg in cfg.get("spectral_reg_values", []):
                    reg = float(reg)
                    reg_name = str(reg).replace(".", "p")
                    builders[f"spectral_align_trainable_reg{reg_name}"] = lambda reg=reg: SpectralAlignedProjectionHead(
                        vision_dim=cfg["vision_dim"],
                        text_dim=cfg["text_dim"],
                        embed_dim=m,
                        pretrained_model_name=cfg.get("pretrained_clip_model", "openai/clip-vit-base-patch32"),
                        spectral_reg=reg,
                    )
                for lam in cfg.get("run1_l1_lambdas", [1e-6, 1e-5, 1e-4]):
                    lam_name = _format_lambda_name(float(lam))
                    builders[f"sparse_jl_l1_lambda{lam_name}"] = lambda lam=float(lam): SparseJLL1Head(
                        vision_dim=cfg["vision_dim"],
                        text_dim=cfg["text_dim"],
                        embed_dim=m,
                        jl_eps=cfg["jl_eps"],
                        jl_seed=cfg["jl_seed"],
                        lambda_l1=lam,
                    )

                selected_method_names = sorted(builders.keys())
                if method_filter:
                    selected_method_names = [n for n in selected_method_names if n in method_filter]

                for method_name in selected_method_names:
                    by_m_method[m_key].setdefault(method_name, [])
                    seed_dir = dataset_out / m_key / method_name / f"seed{seed}"
                    eval_file = seed_dir / "eval.json"
                    if eval_file.exists():
                        rec = load_json(eval_file)
                        by_m_method[m_key][method_name].append(rec)
                        continue

                    model = builders[method_name]()

                    mode = "standard"
                    skip_training = False
                    if method_name == "directclr_proxy":
                        mode = "directclr_proxy"
                    if method_name == "random_jl_only":
                        skip_training = True

                    _train_or_skip(
                        model,
                        train_loader,
                        val_loader,
                        seed=seed,
                        out_dir=seed_dir,
                        cfg=cfg,
                        mode=mode,
                        skip_training=skip_training,
                    )
                    if not skip_training:
                        model = _load_best(model, seed_dir, cfg["device"])
                    else:
                        model = model.to(cfg["device"])

                    metrics, zi, zt = _eval_karpathy(
                        model,
                        cache,
                        test_split,
                        device=cfg["device"],
                        batch_size=cfg["eval_batch_size"],
                    )
                    diag = pair_diagnostics(zi, zt, prefix="it")
                    rec = {"seed": seed, **metrics, **diag, **regime_meta, "n_params": model.n_trainable_params()}
                    if hasattr(model, "spectral_alignment_metrics"):
                        rec.update(model.spectral_alignment_metrics())
                    save_json(rec, eval_file)
                    by_m_method[m_key][method_name].append(rec)
                    print(f"{dataset} {m_key} {method_name} seed={seed} avg_R={metrics['avg_R']:.4f}")

        metrics_for_stats = [
            "i2t_R@1", "i2t_R@5", "i2t_R@10",
            "t2i_R@1", "t2i_R@5", "t2i_R@10",
            "avg_R",
        ]
        dataset_report = {"raw": by_m_method, "stats": {}}
        for m_key, per_method in by_m_method.items():
            dataset_report["stats"][m_key] = build_metric_report(
                per_method,
                metrics=metrics_for_stats,
                baseline_method="clip_head",
            )
        run_summary["datasets"][dataset] = dataset_report

    if write_provenance:
        provenance = env_snapshot(Path(cfg["project_root"]), seeds=seeds, extra={
            "stage": "stage2_e7_karpathy",
            "elapsed_sec": time.time() - start,
        })
        save_json(provenance, output_root / "provenance_stage2.json")
    if write_summary:
        save_json(run_summary, output_root / "E7_karpathy_full_results.json")
    if write_done_marker:
        mark_done(markers / "stage2_e7_karpathy.done.json", {"elapsed_sec": time.time() - start})
    print("Stage 2 complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)
