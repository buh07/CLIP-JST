"""
E1. Performance vs. embedding dimension.

Trains JL+Mahalanobis and the CLIP-head baseline for each m in {64,128,256,512}
on MS-COCO and Flickr30K.  Reports Recall@{1,5,10} (i2t and t2i).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.cache import PairedFeatureDataset, MultiCaptionDataset
from models.baselines import CLIPProjectionHead
from models.pipeline import CLIPJSTPipeline
from training.trainer import train
from utils.common import set_seed, save_json, load_best_checkpoint, eval_dataset


def _make_loaders(cfg: dict, dataset_name: str, ds):
    """Build train/val DataLoaders.  Val is used only for early stopping."""
    n_val = int(len(ds) * 0.1)
    if isinstance(ds, MultiCaptionDataset):
        ds.train(True)
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val])
    if isinstance(ds, MultiCaptionDataset):
        # Deterministic caption selection in val loader (caption 0 per image).
        val_ds.dataset.train(False)
    kw = dict(batch_size=cfg["batch_size"], num_workers=4, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
    )


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


def run(cfg: dict) -> None:
    set_seed(cfg.get("seed", 0))
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    results: dict = {}

    for dataset_name in cfg["datasets"]:
        ds = _load_dataset(cfg, dataset_name)
        train_loader, val_loader = _make_loaders(cfg, dataset_name, ds)
        results[dataset_name] = {}

        # ---- CLIP projection head (upper-bound baseline) ----
        for m in cfg["embed_dims"]:
            label = f"clip_head_m{m}"
            print(f"\n=== {dataset_name} | {label} ===")
            model = CLIPProjectionHead(cfg["vision_dim"], cfg["text_dim"], m)
            print(f"  params: {model.n_trainable_params():,}")
            ckpt_dir = Path(cfg["output_dir"]) / dataset_name / label
            train(model, train_loader, val_loader,
                  epochs=cfg["epochs"], lr=cfg["lr"],
                  temperature=cfg["temperature"], device=device,
                  ckpt_dir=ckpt_dir, patience=cfg.get("patience", 5))
            model = load_best_checkpoint(model, ckpt_dir, device)
            # Final eval on FULL dataset — correct GT indices.
            metrics = eval_dataset(model, ds, device, cfg["batch_size"])
            metrics["n_params"] = model.n_trainable_params()
            results[dataset_name][label] = metrics
            print(f"  {metrics}")

        # ---- JL + Mahalanobis (main method) ----
        for m in cfg["embed_dims"]:
            for rank in cfg["mahal_ranks"]:
                rank_tag = "full" if rank is None else str(rank)
                label = f"jl_mahal_m{m}_r{rank_tag}"
                print(f"\n=== {dataset_name} | {label} ===")
                model = CLIPJSTPipeline(
                    vision_dim=cfg["vision_dim"],
                    text_dim=cfg["text_dim"],
                    embed_dim=m,
                    mahal_rank=rank,
                    jl_eps=cfg["jl_eps"],
                    jl_seed=cfg["jl_seed"],
                )
                print(f"  params: {model.n_trainable_params():,}")
                ckpt_dir = Path(cfg["output_dir"]) / dataset_name / label
                train(model, train_loader, val_loader,
                      epochs=cfg["epochs"], lr=cfg["lr"],
                      temperature=cfg["temperature"], device=device,
                      ckpt_dir=ckpt_dir, patience=cfg.get("patience", 5))
                model = load_best_checkpoint(model, ckpt_dir, device)
                metrics = eval_dataset(model, ds, device, cfg["batch_size"])
                metrics["n_params"] = model.n_trainable_params()
                results[dataset_name][label] = metrics
                print(f"  {metrics}")

    save_json(results, Path(cfg["output_dir"]) / "E1_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run(cfg)
