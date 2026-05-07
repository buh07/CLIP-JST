"""
Negative controls and sanity checks (Section 5.5).

1. Shuffle-label control: randomly permute image–text pairings.
   Performance should collapse to chance, confirming genuine learning.

2. Zero-Mahalanobis control: freeze Mahalanobis at identity (pure JL).
   Should be strictly worse than trained Mahalanobis.

3. Random-seed variability: 2D grid over JL seeds × training seeds.
   - JL seed variability: validates obliviousness (any random draw works).
   - Training seed variability: measures noise from training dynamics.
   Low variance on both axes validates the JL + Mahalanobis pipeline.

All experiments use the full multi-caption pair set (5 captions × N images)
rather than collapsing to a single caption per image.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.baselines import RandomProjectionPipeline
from models.pipeline import CLIPJSTPipeline
from training.trainer import train, extract_embeddings
from eval.retrieval import recall_at_k
from utils.common import set_seed, save_json, load_best_checkpoint


def _make_loaders(img_feats, txt_feats, batch_size: int, seed: int = 0):
    n = len(img_feats)
    n_val = max(1, int(n * 0.1))
    ds = TensorDataset(img_feats, txt_feats)
    train_ds, val_ds = random_split(
        ds, [n - n_val, n_val],
        generator=torch.Generator().manual_seed(seed)
    )
    kw = dict(batch_size=min(batch_size, len(train_ds)), num_workers=0)
    return (
        DataLoader(train_ds, shuffle=True, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
        val_ds,
    )


def run(cfg: dict) -> None:
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg["output_dir"])
    results: dict = {}

    cache_dir = Path(cfg["cache_dir"]) / cfg["dataset"]
    img_feats = torch.load(cache_dir / cfg["image_cache_file"],
                           map_location="cpu", weights_only=True)
    txt_feats = torch.load(cache_dir / cfg["text_cache_file"],
                           map_location="cpu", weights_only=True)

    # Expand image features to match multi-caption text features.
    if len(txt_feats) != len(img_feats):
        n_cap = len(txt_feats) // len(img_feats)
        img_feats = img_feats.repeat_interleave(n_cap, dim=0)

    m = cfg["embed_dim"]
    train_seed_0 = cfg.get("seed", 0)

    # ------------------------------------------------------------------ #
    # 1. Shuffle-label control
    # ------------------------------------------------------------------ #
    print("\n=== Control 1: Shuffle-label ===")
    set_seed(train_seed_0)
    perm = torch.randperm(len(txt_feats))
    shuffled_txt = txt_feats[perm]
    train_loader, val_loader, val_ds = _make_loaders(
        img_feats, shuffled_txt, cfg["batch_size"], seed=train_seed_0
    )
    model_shuffle = CLIPJSTPipeline(
        vision_dim=cfg["vision_dim"], text_dim=cfg["text_dim"], embed_dim=m,
        jl_eps=cfg["jl_eps"], jl_seed=cfg["jl_seed"],
    )
    ckpt_shuffle = output_dir / "shuffle_label"
    train(model_shuffle, train_loader, val_loader,
          epochs=cfg["epochs"], lr=cfg["lr"],
          temperature=cfg.get("temperature", 0.07), device=device,
          ckpt_dir=ckpt_shuffle, patience=cfg.get("patience", 10),
          warmup_epochs=cfg.get("warmup_epochs", 0))
    model_shuffle = load_best_checkpoint(model_shuffle, ckpt_shuffle, device)
    img_emb, txt_emb = extract_embeddings(model_shuffle, val_loader, device)
    results["shuffle_label"] = recall_at_k(img_emb, txt_emb)
    print(f"  Shuffle-label: {results['shuffle_label']}")

    # ------------------------------------------------------------------ #
    # 2. Zero-Mahalanobis (pure JL) control
    # ------------------------------------------------------------------ #
    print("\n=== Control 2: Zero-Mahalanobis (pure JL) ===")
    set_seed(train_seed_0)
    train_loader, val_loader, _ = _make_loaders(
        img_feats, txt_feats, cfg["batch_size"], seed=train_seed_0
    )
    model_zero = RandomProjectionPipeline(
        vision_dim=cfg["vision_dim"], text_dim=cfg["text_dim"], embed_dim=m,
        jl_eps=cfg["jl_eps"], jl_seed=cfg["jl_seed"],
    )
    model_zero.to(device).eval()
    img_emb, txt_emb = extract_embeddings(model_zero, val_loader, device)
    results["zero_mahalanobis"] = recall_at_k(img_emb, txt_emb)
    print(f"  Pure JL: {results['zero_mahalanobis']}")

    # ------------------------------------------------------------------ #
    # 3. Random-seed variability — 2D grid: JL seed × training seed
    # ------------------------------------------------------------------ #
    print("\n=== Control 3: Random-seed variability (2D: JL seed × training seed) ===")
    jl_seeds      = cfg.get("jl_seeds", [42, 43, 44, 45, 46])
    training_seeds = cfg.get("training_seeds", [cfg.get("seed", 0)])
    seed_results: list[dict] = []

    for train_seed in training_seeds:
        for jl_seed in jl_seeds:
            set_seed(train_seed)
            train_loader, val_loader, _ = _make_loaders(
                img_feats, txt_feats, cfg["batch_size"], seed=train_seed
            )
            model_seed = CLIPJSTPipeline(
                vision_dim=cfg["vision_dim"], text_dim=cfg["text_dim"], embed_dim=m,
                jl_eps=cfg["jl_eps"], jl_seed=jl_seed,
            )
            ckpt_seed = output_dir / f"seed_jl{jl_seed}_tr{train_seed}"
            train(model_seed, train_loader, val_loader,
                  epochs=cfg["epochs"], lr=cfg["lr"],
                  temperature=cfg.get("temperature", 0.07), device=device,
                  ckpt_dir=ckpt_seed, patience=cfg.get("patience", 10),
                  warmup_epochs=cfg.get("warmup_epochs", 0))
            model_seed = load_best_checkpoint(model_seed, ckpt_seed, device)
            img_emb, txt_emb = extract_embeddings(model_seed, val_loader, device)
            metrics = recall_at_k(img_emb, txt_emb)
            metrics["jl_seed"] = jl_seed
            metrics["train_seed"] = train_seed
            seed_results.append(metrics)
            print(f"  jl_seed={jl_seed} train_seed={train_seed}: avg_R={metrics.get('avg_R',0):.4f}")

    if seed_results:
        all_avg_r = [r["avg_R"] for r in seed_results if "avg_R" in r]
        grand_mean = sum(all_avg_r) / len(all_avg_r)
        grand_std  = statistics.stdev(all_avg_r) if len(all_avg_r) > 1 else 0.0

        # Marginal variance over JL seeds (avg over training seeds).
        jl_marginal: dict[int, list[float]] = {}
        for r in seed_results:
            jl_marginal.setdefault(r["jl_seed"], []).append(r["avg_R"])
        jl_means = [sum(v)/len(v) for v in jl_marginal.values()]
        jl_std   = statistics.stdev(jl_means) if len(jl_means) > 1 else 0.0

        # Marginal variance over training seeds (avg over JL seeds).
        tr_marginal: dict[int, list[float]] = {}
        for r in seed_results:
            tr_marginal.setdefault(r["train_seed"], []).append(r["avg_R"])
        tr_means = [sum(v)/len(v) for v in tr_marginal.values()]
        tr_std   = statistics.stdev(tr_means) if len(tr_means) > 1 else 0.0

        results["seed_variability"] = {
            "per_run": seed_results,
            "grand_mean_avg_R": grand_mean,
            "grand_std_avg_R":  grand_std,
            "jl_seed_marginal_std":       jl_std,
            "training_seed_marginal_std": tr_std,
        }
        print(f"\n  Grand mean avg_R={grand_mean:.4f} ± {grand_std:.4f}")
        print(f"  JL-seed marginal std={jl_std:.4f}  |  Training-seed marginal std={tr_std:.4f}")

    save_json(results, output_dir / "controls_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run(cfg)
