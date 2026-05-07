"""
D3. Relationship-graph ablations.

Trains JL+Mahalanobis under four graph conditions and compares optimal m:

  (i)   Full MS-COCO pairs (all 5 captions per image expanded → 530K pairs).
  (ii)  High-semantic-agreement pairs: filter to top-50% by CLIP cosine sim.
  (ii-b) Random 50%: randomly subsample the same count as (ii) — quantity control.
         If (ii) >> (ii-b), pair *quality* drives the improvement.
         If (ii) ≈ (ii-b), it is *quantity* that drives it.
  (iii) Pairs augmented with synthetic adversarial negatives.
        Hard negatives are added ONLY to the training portion; the held-out
        val set is restricted to original true-positive pairs so that recall
        is measured against correct targets.

Width is estimated on L2-normalized features (matching retrieval geometry).
Results aggregated across seeds for statistical significance.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.pipeline import CLIPJSTPipeline
from training.trainer import train
from eval.retrieval import recall_at_k
from theory.width_estimation import cross_modal_width_estimate
from utils.common import set_seed, save_json, load_best_checkpoint


def _top_agreement_indices(
    img_feats: torch.Tensor,
    txt_feats: torch.Tensor,
    keep_fraction: float = 0.5,
) -> torch.Tensor:
    """Returns indices for the top-(keep_fraction) pairs by cosine similarity."""
    img_norm = F.normalize(img_feats, dim=1)
    txt_norm = F.normalize(txt_feats, dim=1)
    sims = (img_norm * txt_norm).sum(dim=1)
    n_keep = int(keep_fraction * len(sims))
    return sims.topk(n_keep).indices


def _augment_hard_negatives(
    img_feats: torch.Tensor,
    txt_feats: torch.Tensor,
    n_hard_per_sample: int = 1,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each image, finds the nearest-neighbor images and appends one of their
    captions as a hard negative (incorrect pairing).

    Output layout: [original_0, ..., original_N-1, hardneg_0, ..., hardneg_N-1]
    The first len(img_feats) entries are always the original positive pairs.
    """
    rng = torch.Generator().manual_seed(seed)
    img_norm = F.normalize(img_feats, dim=1)
    sims = img_norm @ img_norm.T
    sims.fill_diagonal_(-1.0)
    k = n_hard_per_sample + 1
    topk_nn = sims.topk(k, dim=1).indices
    aug_img, aug_txt = [img_feats], [txt_feats]
    for i in range(len(img_feats)):
        n_sample = min(n_hard_per_sample, k)
        perm = torch.randperm(k, generator=rng)[:n_sample]
        for j in perm:
            nn_idx = topk_nn[i, j].item()
            aug_img.append(img_feats[i].unsqueeze(0))
            aug_txt.append(txt_feats[nn_idx].unsqueeze(0))
    return torch.cat(aug_img), torch.cat(aug_txt)


def _train_and_eval(
    img_feats: torch.Tensor,
    txt_feats: torch.Tensor,
    cfg: dict,
    device: str,
    label: str,
    seeds: list[int],
    val_img: torch.Tensor | None = None,
    val_txt: torch.Tensor | None = None,
    width_img_feats: torch.Tensor | None = None,
    width_txt_feats: torch.Tensor | None = None,
) -> dict:
    """
    Train for all embed_dims; aggregate results across seeds.

    val_img / val_txt: optional external validation set of clean true-positive
    pairs.  When provided, ALL of img_feats/txt_feats are used for training and
    the model is evaluated (both early-stopping and final recall) on this clean
    set.  This is required for the hard-negative condition where the augmented
    dataset contains mislabeled pairs that must not leak into evaluation.

    When not provided, a random 10% internal split is used for both early
    stopping and final recall (as in all other conditions).
    """
    n = len(img_feats)

    # Width estimated on matched same-dimensional features to keep estimates
    # comparable when training features are heterogeneous (e.g., 768 vs 512).
    if width_img_feats is None:
        width_img_feats = val_img if val_img is not None else img_feats
    if width_txt_feats is None:
        width_txt_feats = val_txt if val_txt is not None else txt_feats[:n]
    width = cross_modal_width_estimate(width_img_feats, width_txt_feats)

    results_per_m = []
    for m in cfg["embed_dims"]:
        seed_metrics: list[dict] = []
        for seed in seeds:
            set_seed(seed)

            if val_img is not None:
                # External clean val set: train on the full augmented dataset,
                # validate and evaluate exclusively on clean original pairs.
                train_ds = TensorDataset(img_feats, txt_feats)
                val_ds   = TensorDataset(val_img, val_txt)
                n_train  = len(train_ds)
            else:
                # Standard internal random 90/10 split.
                n_val = max(1, int(n * 0.1))
                ds = TensorDataset(img_feats, txt_feats)
                train_ds, val_ds = random_split(
                    ds, [n - n_val, n_val],
                    generator=torch.Generator().manual_seed(seed)
                )
                n_train = n - n_val

            kw_train = dict(batch_size=min(cfg["batch_size"], len(train_ds)), num_workers=0)
            kw_val   = dict(batch_size=cfg["batch_size"], num_workers=0)
            model = CLIPJSTPipeline(
                vision_dim=img_feats.shape[1],
                text_dim=txt_feats.shape[1],
                embed_dim=m,
                jl_eps=cfg["jl_eps"],
                jl_seed=cfg["jl_seed"],
            )
            ckpt_dir = Path(cfg["output_dir"]) / label / f"m{m}" / f"seed{seed}"
            run_lr = cfg.get("lr")
            run_epochs = cfg.get("epochs")
            run_patience = cfg.get("patience", 10)
            # Hard-negative condition is intentionally noisier; a lower LR
            # improves stability and reduces seed collapse at large m.
            if label == "hard_neg":
                run_lr = cfg.get("hard_neg_lr", run_lr)
                run_epochs = cfg.get("hard_neg_epochs", run_epochs)
                run_patience = cfg.get("hard_neg_patience", run_patience)
            train(model, DataLoader(train_ds, shuffle=True, **kw_train),
                  DataLoader(val_ds, shuffle=False, **kw_val),
                  epochs=run_epochs, lr=run_lr,
                  temperature=cfg.get("temperature", 0.07), device=device,
                  ckpt_dir=ckpt_dir, patience=run_patience,
                  warmup_epochs=cfg.get("warmup_epochs", 0))
            model = load_best_checkpoint(model, ckpt_dir, device)

            # Final recall on val_ds (clean pairs if external, random split if internal).
            all_img_emb, all_txt_emb = [], []
            val_loader_eval = DataLoader(val_ds, batch_size=cfg["batch_size"], num_workers=0)
            model.eval()
            with torch.no_grad():
                for iv, it in val_loader_eval:
                    all_img_emb.append(model.encode_image(iv.to(device)).cpu())
                    all_txt_emb.append(model.encode_text(it.to(device)).cpu())
            img_emb = torch.cat(all_img_emb)
            txt_emb = torch.cat(all_txt_emb)
            metrics = recall_at_k(img_emb, txt_emb)
            metrics["embed_dim"] = m
            metrics["n_train"] = n_train
            seed_metrics.append(metrics)
            print(f"  {label} m={m} seed={seed}: {metrics}")

        # Aggregate across seeds.
        agg: dict = {"embed_dim": m, "n_train": n_train}
        for key in ["i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10", "avg_R"]:
            vals = [s[key] for s in seed_metrics if key in s]
            if vals:
                agg[key] = {"mean": sum(vals)/len(vals),
                            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0}
        results_per_m.append(agg)

    return {"width": width, "n_pairs": n, "results_per_m": results_per_m}


def run(cfg: dict) -> None:
    set_seed(cfg.get("seed", 0))
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    seeds  = cfg.get("seeds", [cfg.get("seed", 0)])
    results: dict = {}

    cache_dir = Path(cfg["cache_dir"]) / "coco"
    img_feats = torch.load(cache_dir / cfg["image_cache_file"],
                           map_location="cpu", weights_only=True)
    txt_feats = torch.load(cache_dir / cfg["text_cache_file"],
                           map_location="cpu", weights_only=True)

    # Optional aligned-space features for agreement filtering and width
    # estimation when raw training features have different dimensions.
    agree_img = None
    agree_txt = None
    agree_img_file = cfg.get("agreement_image_cache_file")
    agree_txt_file = cfg.get("agreement_text_cache_file")
    if agree_img_file and agree_txt_file:
        agree_img_path = cache_dir / agree_img_file
        agree_txt_path = cache_dir / agree_txt_file
        if agree_img_path.exists() and agree_txt_path.exists():
            agree_img = torch.load(agree_img_path, map_location="cpu", weights_only=True)
            agree_txt = torch.load(agree_txt_path, map_location="cpu", weights_only=True)
            print(f"Using aligned agreement caches: {agree_img_file}, {agree_txt_file}")
        else:
            print("Agreement cache files not found; falling back to training features.")

    # Keep original (pre-expansion) image features for NN-based hard-neg augmentation.
    img_feats_orig = img_feats  # (N_images, d)
    txt_feats_orig_all = txt_feats  # (N_images * n_cap, d)
    agree_img_orig = agree_img
    agree_txt_orig_all = agree_txt

    # Expand image features to match multi-caption text features.
    # This gives the full relationship graph (all 5 captions per image).
    n_cap = 1
    if len(txt_feats) != len(img_feats):
        n_cap = len(txt_feats) // len(img_feats)
        img_feats = img_feats.repeat_interleave(n_cap, dim=0)
        if agree_img is not None and agree_txt is not None:
            agree_img = agree_img.repeat_interleave(n_cap, dim=0)
    # Now img_feats and txt_feats are both (N*n_cap, d).

    # Choose features used to compute agreement-filtering scores.
    if agree_img is not None and agree_txt is not None:
        score_img, score_txt = agree_img, agree_txt
    else:
        if img_feats.shape[1] != txt_feats.shape[1]:
            raise ValueError(
                "High-agreement filtering requires same-dimensional score features. "
                "Provide agreement_image_cache_file/agreement_text_cache_file in config."
            )
        score_img, score_txt = img_feats, txt_feats

    # (i) Full pairs.
    print("\n=== (i) Full COCO pairs ===")
    results["full"] = _train_and_eval(
        img_feats, txt_feats, cfg, device, "full", seeds,
        width_img_feats=score_img, width_txt_feats=score_txt,
    )

    # (ii) High-agreement filtered pairs.
    print("\n=== (ii) High-agreement pairs ===")
    frac = cfg.get("high_agreement_fraction", 0.5)
    keep_idx = _top_agreement_indices(score_img, score_txt, frac)
    ha_img, ha_txt = img_feats[keep_idx], txt_feats[keep_idx]
    ha_w_img, ha_w_txt = score_img[keep_idx], score_txt[keep_idx]
    print(f"  Kept {len(ha_img)}/{len(img_feats)} pairs (top-{frac:.0%} by CLIP cosine sim)")
    results["high_agreement"] = _train_and_eval(
        ha_img, ha_txt, cfg, device, "high_agreement", seeds,
        width_img_feats=ha_w_img, width_txt_feats=ha_w_txt,
    )

    # (ii-b) Random 50% — quantity-matched baseline.
    print("\n=== (ii-b) Random 50% (quantity-matched baseline) ===")
    rng = torch.Generator().manual_seed(cfg.get("seed", 0))
    rand_idx = torch.randperm(len(img_feats), generator=rng)[:len(ha_img)]
    rand_img, rand_txt = img_feats[rand_idx], txt_feats[rand_idx]
    rand_w_img, rand_w_txt = score_img[rand_idx], score_txt[rand_idx]
    print(f"  Randomly sampled {len(rand_img)} pairs (same count as high-agreement)")
    results["random_50pct"] = _train_and_eval(
        rand_img, rand_txt, cfg, device, "random_50pct", seeds,
        width_img_feats=rand_w_img, width_txt_feats=rand_w_txt,
    )

    # (iii) Hard-negative augmented pairs.
    # NN search runs on unique image features (pre-expansion) to avoid OOM on
    # the 591K-row expanded set.  Hard negatives are added ONLY to the training
    # portion; the held-out val set contains exclusively original true-positive
    # pairs so that recall metrics are not measured against mislabeled targets.
    print("\n=== (iii) Hard-negative augmented pairs ===")
    txt_feats_cap0 = txt_feats_orig_all[::n_cap] if n_cap > 1 else txt_feats_orig_all
    agree_txt_cap0 = None
    if agree_txt_orig_all is not None:
        agree_txt_cap0 = agree_txt_orig_all[::n_cap] if n_cap > 1 else agree_txt_orig_all

    # _augment_hard_negatives outputs [originals | hard-negatives] in that order.
    # The first len(img_feats_orig) entries are the unmodified positive pairs.
    aug_img_orig, aug_txt_orig = _augment_hard_negatives(
        img_feats_orig, txt_feats_cap0,
        n_hard_per_sample=cfg.get("n_hard_negatives", 1),
        seed=cfg.get("seed", 42),
    )
    n_orig_hn = len(img_feats_orig)
    n_aug_total = len(aug_img_orig)

    # Reserve a 10% clean eval/val set from the original positive pairs.
    # The remaining 90% of originals + ALL hard-negatives form the training set.
    n_clean_val = max(1, int(n_orig_hn * 0.1))
    rng_hn = torch.Generator().manual_seed(cfg.get("seed", 0))
    perm_hn = torch.randperm(n_orig_hn, generator=rng_hn)
    clean_val_idx   = perm_hn[:n_clean_val]
    clean_train_idx = perm_hn[n_clean_val:]

    # Clean val: original true-positive pairs only (used for both early
    # stopping and final recall — no contamination from hard negatives).
    clean_val_img = img_feats_orig[clean_val_idx]
    clean_val_txt = txt_feats_cap0[clean_val_idx]
    if agree_img_orig is not None and agree_txt_cap0 is not None:
        clean_val_w_img = agree_img_orig[clean_val_idx]
        clean_val_w_txt = agree_txt_cap0[clean_val_idx]
    else:
        clean_val_w_img = clean_val_img
        clean_val_w_txt = clean_val_txt

    # Training set: 90% of original pairs + all hard negatives.
    # Build by selecting the kept original rows and appending the hard-neg block.
    train_orig_img = img_feats_orig[clean_train_idx]
    train_orig_txt = txt_feats_cap0[clean_train_idx]
    hard_neg_img   = aug_img_orig[n_orig_hn:]   # the appended hard-neg portion
    hard_neg_txt   = aug_txt_orig[n_orig_hn:]
    aug_train_img  = torch.cat([train_orig_img, hard_neg_img])
    aug_train_txt  = torch.cat([train_orig_txt, hard_neg_txt])

    print(f"  Training set: {len(aug_train_img):,} pairs "
          f"({len(train_orig_img):,} original + {len(hard_neg_img):,} hard negatives)")
    print(f"  Clean val/eval set: {len(clean_val_img):,} original true-positive pairs")

    results["hard_neg_augmented"] = _train_and_eval(
        aug_train_img, aug_train_txt, cfg, device, "hard_neg", seeds,
        val_img=clean_val_img, val_txt=clean_val_txt,
        width_img_feats=clean_val_w_img, width_txt_feats=clean_val_w_txt,
    )

    save_json(results, Path(cfg["output_dir"]) / "D3_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run(cfg)
