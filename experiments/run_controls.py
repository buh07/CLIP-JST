"""
Negative controls and sanity checks (Section 5.5).

1. Shuffle-label control: randomly permute image–text pairings.
   Performance should collapse to chance, confirming genuine learning.

2. Zero-Mahalanobis control: freeze Mahalanobis at identity (pure JL).
   Should be strictly worse than trained Mahalanobis.

3. Random-seed variability: repeat with 5 JL random seeds.
   Low variance across seeds empirically validates obliviousness.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.cache import PairedFeatureDataset
from models.baselines import RandomProjectionPipeline
from models.pipeline import CLIPJSTPipeline
from training.trainer import train, extract_embeddings
from eval.retrieval import recall_at_k
from utils.common import set_seed, save_json, load_best_checkpoint


def _make_loaders(img_feats, txt_feats, batch_size: int):
    n = len(img_feats)
    n_val = max(1, int(n * 0.1))
    ds = TensorDataset(img_feats, txt_feats)
    train_ds, val_ds = random_split(ds, [n - n_val, n_val])
    kw = dict(batch_size=min(batch_size, len(train_ds)), num_workers=0)
    return (
        DataLoader(train_ds, shuffle=True, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
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
    m = cfg["embed_dim"]

    # ------------------------------------------------------------------ #
    # 1. Shuffle-label control
    # ------------------------------------------------------------------ #
    print("\n=== Control 1: Shuffle-label ===")
    set_seed(cfg.get("seed", 0))
    perm = torch.randperm(len(txt_feats))
    shuffled_txt = txt_feats[perm]
    train_loader, val_loader = _make_loaders(img_feats, shuffled_txt, cfg["batch_size"])

    model_shuffle = CLIPJSTPipeline(
        vision_dim=cfg["vision_dim"], text_dim=cfg["text_dim"], embed_dim=m,
        jl_eps=cfg["jl_eps"], jl_seed=cfg["jl_seed"],
    )
    ckpt_shuffle = output_dir / "shuffle_label"
    train(model_shuffle, train_loader, val_loader,
          epochs=cfg["epochs"], lr=cfg["lr"],
          temperature=cfg["temperature"], device=device,
          ckpt_dir=ckpt_shuffle, patience=cfg.get("patience", 5))
    model_shuffle = load_best_checkpoint(model_shuffle, ckpt_shuffle, device)
    img_emb, txt_emb = extract_embeddings(model_shuffle, val_loader, device)
    results["shuffle_label"] = recall_at_k(img_emb, txt_emb)
    print(f"  Shuffle-label: {results['shuffle_label']}")

    # ------------------------------------------------------------------ #
    # 2. Zero-Mahalanobis (pure JL) control
    # ------------------------------------------------------------------ #
    print("\n=== Control 2: Zero-Mahalanobis (pure JL) ===")
    set_seed(cfg.get("seed", 0))
    train_loader, val_loader = _make_loaders(img_feats, txt_feats, cfg["batch_size"])

    model_zero = RandomProjectionPipeline(
        vision_dim=cfg["vision_dim"], text_dim=cfg["text_dim"], embed_dim=m,
        jl_eps=cfg["jl_eps"], jl_seed=cfg["jl_seed"],
    )
    # No training needed (all params frozen); just evaluate.
    model_zero.to(device).eval()
    img_emb, txt_emb = extract_embeddings(model_zero, val_loader, device)
    results["zero_mahalanobis"] = recall_at_k(img_emb, txt_emb)
    print(f"  Pure JL: {results['zero_mahalanobis']}")

    # ------------------------------------------------------------------ #
    # 3. Random-seed variability
    # ------------------------------------------------------------------ #
    print("\n=== Control 3: Random-seed variability ===")
    seed_results = []
    for jl_seed in cfg.get("jl_seeds", [42, 43, 44, 45, 46]):
        set_seed(cfg.get("seed", 0))
        train_loader, val_loader = _make_loaders(img_feats, txt_feats, cfg["batch_size"])

        model_seed = CLIPJSTPipeline(
            vision_dim=cfg["vision_dim"], text_dim=cfg["text_dim"], embed_dim=m,
            jl_eps=cfg["jl_eps"], jl_seed=jl_seed,
        )
        ckpt_seed = output_dir / f"seed_{jl_seed}"
        train(model_seed, train_loader, val_loader,
              epochs=cfg["epochs"], lr=cfg["lr"],
              temperature=cfg["temperature"], device=device,
              ckpt_dir=ckpt_seed, patience=cfg.get("patience", 5))
        model_seed = load_best_checkpoint(model_seed, ckpt_seed, device)
        img_emb, txt_emb = extract_embeddings(model_seed, val_loader, device)
        metrics = recall_at_k(img_emb, txt_emb)
        metrics["jl_seed"] = jl_seed
        seed_results.append(metrics)
        print(f"  seed={jl_seed}: {metrics}")

    # Compute mean ± std over seeds.
    if seed_results:
        key_avg = "avg_R"
        vals = [r[key_avg] for r in seed_results if key_avg in r]
        mean_val = sum(vals) / len(vals)
        std_val  = (sum((v - mean_val) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5
        results["seed_variability"] = {
            "per_seed": seed_results,
            f"mean_{key_avg}": mean_val,
            f"std_{key_avg}": std_val,
        }
        print(f"  avg_R mean={mean_val:.4f} ± std={std_val:.4f}")

    save_json(results, output_dir / "controls_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run(cfg)
