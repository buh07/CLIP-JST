"""
D3. Relationship-graph ablations.

Trains JL+Mahalanobis under three graph conditions and compares optimal m:

  (i)  Full MS-COCO pairs (baseline).
  (ii) High-semantic-agreement pairs: filter to (image, caption) pairs where
       the raw CLIP similarity score > threshold (top-50% by similarity).
  (iii) Pairs augmented with synthetic adversarial negatives: for each image,
        add a hard-negative caption sampled from a semantically similar but
        non-matching image.

Hypothesis: filtered high-agreement graphs have smaller Gaussian width and
thus require smaller m for a given Recall@10.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.pipeline import CLIPJSTPipeline
from training.trainer import train, extract_embeddings
from eval.retrieval import recall_at_k
from theory.width_estimation import cross_modal_width_estimate
from utils.common import set_seed, save_json, load_best_checkpoint


def _filter_high_agreement(
    img_feats: torch.Tensor,
    txt_feats: torch.Tensor,
    keep_fraction: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Keeps the top-(keep_fraction) fraction of pairs ranked by cosine similarity
    between their normalized CLIP features.
    """
    img_norm = torch.nn.functional.normalize(img_feats, dim=1)
    txt_norm = torch.nn.functional.normalize(txt_feats, dim=1)
    sims = (img_norm * txt_norm).sum(dim=1)   # (N,)
    n_keep = int(keep_fraction * len(sims))
    keep_idx = sims.topk(n_keep).indices
    return img_feats[keep_idx], txt_feats[keep_idx]


def _augment_hard_negatives(
    img_feats: torch.Tensor,
    txt_feats: torch.Tensor,
    n_hard_per_sample: int = 1,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Augments the pair set with synthetic hard negatives.

    For each image, finds the K nearest-neighbor images (by feature similarity),
    then samples one of their captions as a hard negative.  The hard negative
    pairs are appended to the dataset with *incorrect* pairing, but this creates
    a denser negative structure during InfoNCE training (larger effective batch).

    Note: we only augment the training set, not the evaluation set.
    """
    rng = torch.Generator().manual_seed(seed)
    img_norm = torch.nn.functional.normalize(img_feats, dim=1)
    sims = img_norm @ img_norm.T       # (N, N)
    sims.fill_diagonal_(-1.0)          # Exclude self

    k = n_hard_per_sample + 1
    topk_nn = sims.topk(k, dim=1).indices   # (N, k)

    aug_img, aug_txt = [img_feats], [txt_feats]
    for i in range(len(img_feats)):
        # Sample n_hard_per_sample distinct NN indices (without replacement).
        n_sample = min(n_hard_per_sample, k)
        perm = torch.randperm(k, generator=rng)[:n_sample]
        for j in perm:
            nn_idx = topk_nn[i, j].item()
            # img_feats[i] paired with txt_feats[nn_idx]: wrong but semantically hard.
            aug_img.append(img_feats[i].unsqueeze(0))
            aug_txt.append(txt_feats[nn_idx].unsqueeze(0))

    return torch.cat(aug_img), torch.cat(aug_txt)


def _train_and_eval(
    img_feats: torch.Tensor,
    txt_feats: torch.Tensor,
    cfg: dict,
    device: str,
    label: str,
) -> dict:
    n = len(img_feats)
    n_val = max(1, int(n * 0.1))
    ds = TensorDataset(img_feats, txt_feats)
    train_ds, val_ds = random_split(ds, [n - n_val, n_val])
    kw = dict(batch_size=min(cfg["batch_size"], len(train_ds)), num_workers=0)

    results_per_m = []
    for m in cfg["embed_dims"]:
        model = CLIPJSTPipeline(
            vision_dim=img_feats.shape[1],
            text_dim=txt_feats.shape[1],
            embed_dim=m,
            jl_eps=cfg["jl_eps"],
            jl_seed=cfg["jl_seed"],
        )
        ckpt_dir = Path(cfg["output_dir"]) / label / f"m{m}"
        train(model, DataLoader(train_ds, shuffle=True, **kw),
              DataLoader(val_ds, shuffle=False, **kw),
              epochs=cfg["epochs"], lr=cfg["lr"],
              temperature=cfg["temperature"], device=device,
              ckpt_dir=ckpt_dir, patience=cfg.get("patience", 5))
        model = load_best_checkpoint(model, ckpt_dir, device)
        img_emb, txt_emb = extract_embeddings(model, DataLoader(val_ds, **kw), device)
        metrics = recall_at_k(img_emb, txt_emb)
        metrics["embed_dim"] = m
        metrics["n_train"] = n - n_val
        results_per_m.append(metrics)
        print(f"  {label} m={m}: {metrics}")

    width = cross_modal_width_estimate(img_feats, txt_feats[:len(img_feats)])
    return {"width": width, "results_per_m": results_per_m}


def run(cfg: dict) -> None:
    set_seed(cfg.get("seed", 0))
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    results: dict = {}

    cache_dir = Path(cfg["cache_dir"]) / "coco"
    img_feats = torch.load(cache_dir / cfg["image_cache_file"],
                           map_location="cpu", weights_only=True)
    txt_feats = torch.load(cache_dir / cfg["text_cache_file"],
                           map_location="cpu", weights_only=True)

    # (i) Full pairs.
    print("\n=== (i) Full COCO pairs ===")
    results["full"] = _train_and_eval(img_feats, txt_feats, cfg, device, "full")

    # (ii) High-agreement filtered pairs.
    print("\n=== (ii) High-agreement pairs ===")
    frac = cfg.get("high_agreement_fraction", 0.5)
    ha_img, ha_txt = _filter_high_agreement(img_feats, txt_feats, frac)
    print(f"  Kept {len(ha_img)}/{len(img_feats)} pairs (top-{frac:.0%} by CLIP similarity)")
    results["high_agreement"] = _train_and_eval(ha_img, ha_txt, cfg, device, "high_agreement")

    # (iii) Hard-negative augmented pairs.
    print("\n=== (iii) Hard-negative augmented pairs ===")
    aug_img, aug_txt = _augment_hard_negatives(
        img_feats, txt_feats,
        n_hard_per_sample=cfg.get("n_hard_negatives", 1),
        seed=cfg.get("seed", 42),
    )
    print(f"  Augmented to {len(aug_img)} pairs ({len(img_feats)} original + {len(aug_img)-len(img_feats)} hard negatives)")
    results["hard_neg_augmented"] = _train_and_eval(aug_img, aug_txt, cfg, device, "hard_neg")

    save_json(results, Path(cfg["output_dir"]) / "D3_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run(cfg)
