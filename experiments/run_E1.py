"""
E1. Performance vs. embedding dimension.

Trains JL+Mahalanobis and the CLIP-head baseline for each m in {64,128,256,512}
on MS-COCO and Flickr30K.  Reports Recall@{1,5,10} (i2t and t2i) on a strict
held-out test split (10% of data, never seen during training or validation).

Evaluation uses the full multi-caption ground truth (1:N retrieval) via
MultiCaptionDataset.get_eval_tensors(), which is the standard COCO/Flickr protocol.

Results are aggregated across seeds and reported as mean ± std.
"""

from __future__ import annotations

import argparse
import sys
import statistics
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.cache import PairedFeatureDataset, MultiCaptionDataset
from eval.retrieval import recall_at_k
from models.baselines import CLIPProjectionHead
from models.pipeline import CLIPJSTPipeline
from training.trainer import train
from utils.common import set_seed, save_json, load_best_checkpoint


def _make_splits(cfg: dict, ds):
    """80/10/10 train/val/test split with a fixed generator for reproducibility."""
    n = len(ds)
    n_test  = int(n * 0.10)
    n_val   = int(n * 0.10)
    n_train = n - n_val - n_test
    gen = torch.Generator().manual_seed(cfg.get("seed", 0))
    if isinstance(ds, MultiCaptionDataset):
        ds.train(True)
    train_ds, val_ds, test_ds = random_split(
        ds, [n_train, n_val, n_test], generator=gen
    )
    if isinstance(ds, MultiCaptionDataset):
        val_ds.dataset.train(False)
    return train_ds, val_ds, test_ds


def _make_loaders(cfg: dict, train_ds, val_ds):
    kw = dict(batch_size=cfg["batch_size"], num_workers=4, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
    )


def _eval_on_test(model, ds, test_ds, device: str, batch_size: int) -> dict:
    """
    Evaluate on the held-out test split with proper multi-caption ground truth.

    For MultiCaptionDataset: extracts all 5 captions per test image and builds
    the gt_i2t dict.  For PairedFeatureDataset: 1:1 evaluation.
    """
    model.eval()
    test_indices = list(test_ds.indices)

    if isinstance(ds, MultiCaptionDataset):
        img_feats, txt_feats, gt_i2t = ds.get_eval_tensors()
        # Restrict to test images only.
        n_cap = ds.n_captions
        test_img  = img_feats[test_indices]                              # (N_test, d)
        # text rows for test image i are at [i*n_cap : (i+1)*n_cap] in global order.
        txt_rows  = [idx * n_cap + k for idx in test_indices for k in range(n_cap)]
        test_txt  = txt_feats[txt_rows]                                  # (N_test*n_cap, d)
        # Rebuild gt_i2t relative to the test-local text index space.
        test_gt_i2t = {
            local_i: list(range(local_i * n_cap, (local_i + 1) * n_cap))
            for local_i in range(len(test_indices))
        }
        # Encode in batches.
        all_img_emb, all_txt_emb = [], []
        with torch.no_grad():
            for start in range(0, len(test_img), batch_size):
                all_img_emb.append(
                    model.encode_image(test_img[start:start+batch_size].to(device)).cpu()
                )
            for start in range(0, len(test_txt), batch_size):
                all_txt_emb.append(
                    model.encode_text(test_txt[start:start+batch_size].to(device)).cpu()
                )
        img_emb = torch.cat(all_img_emb)
        txt_emb = torch.cat(all_txt_emb)
        return recall_at_k(img_emb, txt_emb, gt_i2t=test_gt_i2t)
    else:
        # 1:1 paired dataset — simple diagonal GT.
        img_feats = ds.img[test_indices]
        txt_feats = ds.txt[test_indices]
        all_img_emb, all_txt_emb = [], []
        with torch.no_grad():
            for start in range(0, len(img_feats), batch_size):
                all_img_emb.append(
                    model.encode_image(img_feats[start:start+batch_size].to(device)).cpu()
                )
            for start in range(0, len(txt_feats), batch_size):
                all_txt_emb.append(
                    model.encode_text(txt_feats[start:start+batch_size].to(device)).cpu()
                )
        return recall_at_k(torch.cat(all_img_emb), torch.cat(all_txt_emb))


def _load_dataset(cfg: dict, dataset_name: str):
    cache_dir = Path(cfg["cache_dir"]) / dataset_name
    n_cap = cfg.get("n_captions", {}).get(dataset_name, 1)
    if n_cap > 1:
        return MultiCaptionDataset(
            cache_dir / cfg["image_cache_file"],
            cache_dir / cfg["text_cache_file"],
            n_captions=n_cap,
            training=True,
        )
    return PairedFeatureDataset(
        cache_dir / cfg["image_cache_file"],
        cache_dir / cfg["text_cache_file"],
    )


def _agg_seeds(seed_metrics: list[dict]) -> dict:
    """Compute mean ± std across seed runs for each metric key."""
    if len(seed_metrics) == 1:
        return {k: {"mean": v, "std": 0.0} for k, v in seed_metrics[0].items()}
    agg = {}
    for key in seed_metrics[0]:
        vals = [m[key] for m in seed_metrics if isinstance(m.get(key), (int, float))]
        if vals:
            mean = sum(vals) / len(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            agg[key] = {"mean": mean, "std": std}
        else:
            agg[key] = seed_metrics[0][key]
    return agg


def run(cfg: dict) -> None:
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    seeds  = cfg.get("seeds", [cfg.get("seed", 0)])
    results: dict = {
        "eval_protocol": "multi_caption_1N_10pct_held_out_train_split",
    }

    for dataset_name in cfg["datasets"]:
        ds = _load_dataset(cfg, dataset_name)
        train_ds, val_ds, test_ds = _make_splits(cfg, ds)
        results[dataset_name] = {}

        # ---- CLIP projection head (upper-bound baseline) ----
        for m in cfg["embed_dims"]:
            label = f"clip_head_m{m}"
            print(f"\n=== {dataset_name} | {label} ===")
            seed_metrics: list[dict] = []

            for seed in seeds:
                set_seed(seed)
                model = CLIPProjectionHead(cfg["vision_dim"], cfg["text_dim"], m)
                print(f"  seed={seed}  params={model.n_trainable_params():,}")
                ckpt_dir = Path(cfg["output_dir"]) / dataset_name / label / f"seed{seed}"
                train(model, *_make_loaders(cfg, train_ds, val_ds),
                      epochs=cfg["epochs"], lr=cfg["lr"],
                      temperature=cfg.get("temperature", 0.07), device=device,
                      ckpt_dir=ckpt_dir, patience=cfg.get("patience", 10),
                      warmup_epochs=cfg.get("warmup_epochs", 0))
                model = load_best_checkpoint(model, ckpt_dir, device)
                metrics = _eval_on_test(model, ds, test_ds, device, cfg["batch_size"])
                metrics["n_params"] = model.n_trainable_params()
                seed_metrics.append(metrics)
                print(f"  seed={seed}: {metrics}")

            agg = _agg_seeds(seed_metrics)
            agg["n_params"] = seed_metrics[0]["n_params"]
            results[dataset_name][label] = agg
            print(f"  {label} mean: { {k: round(v['mean'],4) for k,v in agg.items() if isinstance(v, dict)} }")

        # ---- JL + Mahalanobis (main method) ----
        for m in cfg["embed_dims"]:
            for rank in cfg["mahal_ranks"]:
                rank_tag = "full" if rank is None else str(rank)
                label = f"jl_mahal_m{m}_r{rank_tag}"
                print(f"\n=== {dataset_name} | {label} ===")
                seed_metrics = []

                for seed in seeds:
                    set_seed(seed)
                    model = CLIPJSTPipeline(
                        vision_dim=cfg["vision_dim"],
                        text_dim=cfg["text_dim"],
                        embed_dim=m,
                        mahal_rank=rank,
                        jl_eps=cfg["jl_eps"],
                        jl_seed=cfg["jl_seed"],
                    )
                    print(f"  seed={seed}  params={model.n_trainable_params():,}")
                    ckpt_dir = Path(cfg["output_dir"]) / dataset_name / label / f"seed{seed}"
                    train(model, *_make_loaders(cfg, train_ds, val_ds),
                          epochs=cfg["epochs"], lr=cfg["lr"],
                          temperature=cfg.get("temperature", 0.07), device=device,
                          ckpt_dir=ckpt_dir, patience=cfg.get("patience", 10),
                          warmup_epochs=cfg.get("warmup_epochs", 0))
                    model = load_best_checkpoint(model, ckpt_dir, device)
                    metrics = _eval_on_test(model, ds, test_ds, device, cfg["batch_size"])
                    metrics["n_params"] = model.n_trainable_params()
                    seed_metrics.append(metrics)
                    print(f"  seed={seed}: {metrics}")

                agg = _agg_seeds(seed_metrics)
                agg["n_params"] = seed_metrics[0]["n_params"]
                results[dataset_name][label] = agg
                print(f"  {label} mean: { {k: round(v['mean'],4) for k,v in agg.items() if isinstance(v, dict)} }")

    save_json(results, Path(cfg["output_dir"]) / "E1_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run(cfg)
