"""
E7. JLT-in-projection-loop pilot.

Compares four projection strategies on frozen raw CLIP features:
  1) CLIP head (learned dense linear d->m).
  2) Fixed random JL + full Mahalanobis (post-hoc random projection baseline).
  3) Trainable orthogonal JL-style projection head (data-adaptive JL analogue).
  4) Fixed random JL only (no trainable head).

Reports held-out Recall@{1,5,10} and diagnostics:
  - modality gap (||mean(img_emb)-mean(txt_emb)||_2)
  - effective rank / participation ratio of image and text embeddings.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.cache import MultiCaptionDataset, PairedFeatureDataset
from eval.retrieval import recall_at_k
from models.baselines import (
    CLIPProjectionHead,
    OrthogonalProjectionHead,
    RandomProjectionPipeline,
)
from models.pipeline import CLIPJSTPipeline
from training.trainer import train
from utils.common import load_best_checkpoint, save_json, set_seed


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


def _make_splits(ds, seed: int):
    n = len(ds)
    n_test = int(n * 0.10)
    n_val = int(n * 0.10)
    n_train = n - n_val - n_test
    gen = torch.Generator().manual_seed(seed)
    if isinstance(ds, MultiCaptionDataset):
        ds.train(True)
    train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test], generator=gen)
    if isinstance(ds, MultiCaptionDataset):
        val_ds.dataset.train(False)
    return train_ds, val_ds, test_ds


def _make_loaders(cfg: dict, train_ds, val_ds):
    kw = dict(batch_size=cfg["batch_size"], num_workers=4, pin_memory=True)
    return DataLoader(train_ds, shuffle=True, **kw), DataLoader(val_ds, shuffle=False, **kw)


def _effective_rank(x: torch.Tensor, max_rows: int = 5000) -> dict[str, float]:
    if x.shape[0] > max_rows:
        idx = torch.randperm(x.shape[0])[:max_rows]
        x = x[idx]
    x = x.float()
    x = x - x.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(x)
    s = s[s > 1e-12]
    if len(s) == 0:
        return {"effective_rank": 0.0, "participation_ratio": 0.0}
    p = s / s.sum()
    ent = -(p * (p + 1e-12).log()).sum()
    erank = ent.exp().item()
    pr = (s.sum().pow(2) / (s.pow(2).sum() + 1e-12)).item()
    return {"effective_rank": erank, "participation_ratio": pr}


def _diagnostics(img_emb: torch.Tensor, txt_emb: torch.Tensor) -> dict[str, float]:
    gap = (img_emb.mean(dim=0) - txt_emb.mean(dim=0)).norm().item()
    i_rank = _effective_rank(img_emb)
    t_rank = _effective_rank(txt_emb)
    return {
        "modality_gap_l2": gap,
        "img_effective_rank": i_rank["effective_rank"],
        "txt_effective_rank": t_rank["effective_rank"],
        "img_participation_ratio": i_rank["participation_ratio"],
        "txt_participation_ratio": t_rank["participation_ratio"],
    }


@torch.no_grad()
def _eval_on_test(model, ds, test_ds, device: str, batch_size: int) -> tuple[dict, torch.Tensor, torch.Tensor]:
    model.eval()
    test_indices = list(test_ds.indices)

    if isinstance(ds, MultiCaptionDataset):
        img_feats, txt_feats, _ = ds.get_eval_tensors()
        n_cap = ds.n_captions
        test_img = img_feats[test_indices]
        txt_rows = [idx * n_cap + k for idx in test_indices for k in range(n_cap)]
        test_txt = txt_feats[txt_rows]
        test_gt_i2t = {
            local_i: list(range(local_i * n_cap, (local_i + 1) * n_cap))
            for local_i in range(len(test_indices))
        }
        all_img_emb, all_txt_emb = [], []
        for start in range(0, len(test_img), batch_size):
            all_img_emb.append(model.encode_image(test_img[start:start + batch_size].to(device)).cpu())
        for start in range(0, len(test_txt), batch_size):
            all_txt_emb.append(model.encode_text(test_txt[start:start + batch_size].to(device)).cpu())
        img_emb = torch.cat(all_img_emb)
        txt_emb = torch.cat(all_txt_emb)
        metrics = recall_at_k(img_emb, txt_emb, gt_i2t=test_gt_i2t)
        return metrics, img_emb, txt_emb

    img_feats = ds.img[test_indices]
    txt_feats = ds.txt[test_indices]
    all_img_emb, all_txt_emb = [], []
    for start in range(0, len(img_feats), batch_size):
        all_img_emb.append(model.encode_image(img_feats[start:start + batch_size].to(device)).cpu())
        all_txt_emb.append(model.encode_text(txt_feats[start:start + batch_size].to(device)).cpu())
    img_emb = torch.cat(all_img_emb)
    txt_emb = torch.cat(all_txt_emb)
    metrics = recall_at_k(img_emb, txt_emb)
    return metrics, img_emb, txt_emb


def _agg_seed_dict(seed_runs: list[dict]) -> dict:
    if len(seed_runs) == 1:
        return {k: {"mean": v, "std": 0.0} for k, v in seed_runs[0].items()}
    out = {}
    for key in seed_runs[0]:
        vals = [r[key] for r in seed_runs if isinstance(r.get(key), (float, int))]
        if vals:
            out[key] = {
                "mean": float(sum(vals) / len(vals)),
                "std": float(statistics.stdev(vals) if len(vals) > 1 else 0.0),
            }
    return out


def run(cfg: dict) -> None:
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    seeds = cfg.get("seeds", [cfg.get("seed", 0)])
    output_dir = Path(cfg["output_dir"])
    results: dict = {"eval_protocol": "multi_caption_1N_10pct_held_out_train_split"}

    model_factories = {
        "clip_head": lambda m: CLIPProjectionHead(cfg["vision_dim"], cfg["text_dim"], m),
        "random_jl_mahal": lambda m: CLIPJSTPipeline(
            vision_dim=cfg["vision_dim"],
            text_dim=cfg["text_dim"],
            embed_dim=m,
            mahal_rank=None,
            jl_eps=cfg["jl_eps"],
            jl_seed=cfg["jl_seed"],
        ),
        "orth_jl_trainable": lambda m: OrthogonalProjectionHead(
            vision_dim=cfg["vision_dim"],
            text_dim=cfg["text_dim"],
            embed_dim=m,
            orth_reg=cfg.get("orth_reg", 1e-3),
        ),
        "random_jl_only": lambda m: RandomProjectionPipeline(
            vision_dim=cfg["vision_dim"],
            text_dim=cfg["text_dim"],
            embed_dim=m,
            jl_eps=cfg["jl_eps"],
            jl_seed=cfg["jl_seed"],
        ),
    }
    trainable = {"clip_head", "random_jl_mahal", "orth_jl_trainable"}

    for dataset_name in cfg["datasets"]:
        ds = _load_dataset(cfg, dataset_name)
        results[dataset_name] = {}
        for m in cfg["embed_dims"]:
            print(f"\n=== E7 | {dataset_name} | m={m} ===")
            results[dataset_name][f"m{m}"] = {}

            for model_name, make_model in model_factories.items():
                per_seed_metrics: list[dict] = []
                per_seed_diag: list[dict] = []
                n_params = None

                for seed in seeds:
                    set_seed(seed)
                    train_ds, val_ds, test_ds = _make_splits(ds, seed)
                    model = make_model(m)
                    n_params = model.n_trainable_params()
                    ckpt_dir = output_dir / dataset_name / f"m{m}" / model_name / f"seed{seed}"

                    if model_name in trainable:
                        tr_loader, va_loader = _make_loaders(cfg, train_ds, val_ds)
                        train(
                            model, tr_loader, va_loader,
                            epochs=cfg["epochs"], lr=cfg["lr"],
                            temperature=cfg.get("temperature", 0.07),
                            device=device, ckpt_dir=ckpt_dir,
                            patience=cfg.get("patience", 10),
                            warmup_epochs=cfg.get("warmup_epochs", 0),
                        )
                        model = load_best_checkpoint(model, ckpt_dir, device)
                    else:
                        model = model.to(device)

                    metrics, img_emb, txt_emb = _eval_on_test(
                        model, ds, test_ds, device, cfg["batch_size"]
                    )
                    diag = _diagnostics(img_emb, txt_emb)
                    per_seed_metrics.append(metrics)
                    per_seed_diag.append(diag)
                    print(f"  {model_name} seed={seed}: avg_R={metrics['avg_R']:.4f}, gap={diag['modality_gap_l2']:.4f}")

                agg_metrics = _agg_seed_dict(per_seed_metrics)
                agg_diag = _agg_seed_dict(per_seed_diag)
                results[dataset_name][f"m{m}"][model_name] = {
                    "n_params": n_params,
                    "retrieval": agg_metrics,
                    "diagnostics": agg_diag,
                }

    save_json(results, output_dir / "E7_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run(cfg)
