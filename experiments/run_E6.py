"""
E6. Convergence analysis.

Trains CLIP projection head and JL+Full Mahalanobis (both at m=256) for 200
epochs WITHOUT early stopping.  Logs train loss and test recall every eval_every
epochs so we can plot full learning curves.

Primary question: Is the performance gap between JL+Mahal and CLIP head a
convergence failure (solvable with more training) or a capacity ceiling?

If the gap closes by epoch 200, the issue is training budget.
If the gap is stable after epoch ~50, it is a capacity limitation.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.cache import MultiCaptionDataset, PairedFeatureDataset
from eval.retrieval import recall_at_k
from models.baselines import CLIPProjectionHead
from models.pipeline import CLIPJSTPipeline
from training.infonce import infonce_loss
from utils.common import set_seed, save_json, load_best_checkpoint


def _make_splits(cfg: dict, ds):
    n = len(ds)
    n_test  = int(n * 0.10)
    n_val   = int(n * 0.10)
    n_train = n - n_val - n_test
    gen = torch.Generator().manual_seed(0)
    if isinstance(ds, MultiCaptionDataset):
        ds.train(True)
    train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test], generator=gen)
    if isinstance(ds, MultiCaptionDataset):
        val_ds.dataset.train(False)
    return train_ds, val_ds, test_ds


@torch.no_grad()
def _test_recall(model, ds, test_ds, device: str, batch_size: int) -> dict:
    model.eval()
    test_indices = list(test_ds.indices)
    if isinstance(ds, MultiCaptionDataset):
        img_feats, txt_feats, _ = ds.get_eval_tensors()
        n_cap = ds.n_captions
        test_img = img_feats[test_indices]
        txt_rows = [idx * n_cap + k for idx in test_indices for k in range(n_cap)]
        test_txt = txt_feats[txt_rows]
        test_gt  = {li: list(range(li*n_cap, (li+1)*n_cap)) for li in range(len(test_indices))}
        all_i, all_t = [], []
        for s in range(0, len(test_img), batch_size):
            all_i.append(model.encode_image(test_img[s:s+batch_size].to(device)).cpu())
        for s in range(0, len(test_txt), batch_size):
            all_t.append(model.encode_text(test_txt[s:s+batch_size].to(device)).cpu())
        return recall_at_k(torch.cat(all_i), torch.cat(all_t), gt_i2t=test_gt)
    else:
        imgs = ds.img[test_indices]
        txts = ds.txt[test_indices]
        all_i, all_t = [], []
        for s in range(0, len(imgs), batch_size):
            all_i.append(model.encode_image(imgs[s:s+batch_size].to(device)).cpu())
            all_t.append(model.encode_text(txts[s:s+batch_size].to(device)).cpu())
        return recall_at_k(torch.cat(all_i), torch.cat(all_t))


def _train_with_curve(
    model,
    train_loader: DataLoader,
    ds,
    test_ds,
    cfg: dict,
    device: str,
    ckpt_dir: Path,
) -> dict:
    """
    Trains for cfg['epochs'] epochs without early stopping, logging every
    cfg['eval_every'] epochs.  Returns the full learning curve.
    """
    import time
    epochs     = cfg["epochs"]
    lr         = cfg["lr"]
    warmup_ep  = cfg.get("warmup_epochs", 0)
    eval_every = cfg.get("eval_every", 10)
    batch_size = cfg["batch_size"]

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
    )
    warmup_ep = min(warmup_ep, epochs)
    if warmup_ep > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_ep
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs - warmup_ep)
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup_sched, cosine_sched], milestones=[warmup_ep]
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    curve: dict = {"epoch": [], "train_loss": [], "test_R@1_i2t": [],
                   "test_R@5_i2t": [], "test_R@10_i2t": [], "avg_R": []}

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0
        for img_feat, txt_feat in train_loader:
            img_feat, txt_feat = img_feat.to(device), txt_feat.to(device)
            img_emb, txt_emb = model(img_feat, txt_feat)
            if hasattr(model, "logit_scale"):
                temp = model.logit_scale.exp().clamp(max=100.0)
            else:
                temp = cfg.get("temperature", 0.07)
            loss = infonce_loss(img_emb, txt_emb, temp)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)

        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            metrics = _test_recall(model, ds, test_ds, device, batch_size)
            curve["epoch"].append(epoch + 1)
            curve["train_loss"].append(avg_loss)
            curve["test_R@1_i2t"].append(metrics.get("i2t_R@1", 0))
            curve["test_R@5_i2t"].append(metrics.get("i2t_R@5", 0))
            curve["test_R@10_i2t"].append(metrics.get("i2t_R@10", 0))
            curve["avg_R"].append(metrics.get("avg_R", 0))
            print(
                f"  Epoch {epoch+1:3d}/{epochs}  loss={avg_loss:.4f}  "
                f"avg_R={metrics.get('avg_R',0):.4f}  ({time.time()-t0:.1f}s)"
            )
        else:
            print(f"  Epoch {epoch+1:3d}/{epochs}  loss={avg_loss:.4f}  ({time.time()-t0:.1f}s)")

    # Save final checkpoint.
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "final.pt")

    # Compute epoch of 95% final performance.
    final_perf = curve["avg_R"][-1] if curve["avg_R"] else 0.0
    threshold  = 0.95 * final_perf
    epoch_95   = next(
        (e for e, r in zip(curve["epoch"], curve["avg_R"]) if r >= threshold),
        curve["epoch"][-1] if curve["epoch"] else epochs
    )
    curve["epoch_95pct_final"] = epoch_95

    return curve


def run(cfg: dict) -> None:
    device    = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    seeds     = cfg.get("seeds", [0])
    output_dir = Path(cfg["output_dir"])
    results: dict = {"eval_protocol": "10pct_held_out_train_split"}

    # Load dataset.
    cache_dir = Path(cfg["cache_dir"]) / cfg["dataset"]
    n_cap = cfg.get("n_captions", 1)
    if n_cap > 1:
        ds = MultiCaptionDataset(
            cache_dir / cfg["image_cache_file"],
            cache_dir / cfg["text_cache_file"],
            n_captions=n_cap,
            training=True,
        )
    else:
        ds = PairedFeatureDataset(
            cache_dir / cfg["image_cache_file"],
            cache_dir / cfg["text_cache_file"],
        )

    train_ds, val_ds, test_ds = _make_splits(cfg, ds)
    kw = dict(batch_size=cfg["batch_size"], num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **kw)

    m          = cfg["embed_dim"]
    mahal_rank = cfg.get("mahal_rank")
    curves_clip, curves_jl = [], []

    for seed in seeds:
        set_seed(seed)
        print(f"\n=== E6 | CLIP head m={m} | seed={seed} ===")
        clip_model = CLIPProjectionHead(cfg["vision_dim"], cfg["text_dim"], m)
        c = _train_with_curve(clip_model, train_loader, ds, test_ds, cfg, device,
                               output_dir / "clip_head" / f"seed{seed}")
        c["seed"] = seed
        curves_clip.append(c)

        set_seed(seed)
        print(f"\n=== E6 | JL+Mahal m={m} rank={'full' if mahal_rank is None else mahal_rank} | seed={seed} ===")
        jl_model = CLIPJSTPipeline(
            vision_dim=cfg["vision_dim"], text_dim=cfg["text_dim"],
            embed_dim=m, mahal_rank=mahal_rank,
            jl_eps=cfg["jl_eps"], jl_seed=cfg["jl_seed"],
        )
        c = _train_with_curve(jl_model, train_loader, ds, test_ds, cfg, device,
                               output_dir / "jl_mahal" / f"seed{seed}")
        c["seed"] = seed
        curves_jl.append(c)

    results["clip_head"]  = curves_clip
    results["jl_mahal"]   = curves_jl

    # Summary: mean ± std final performance across seeds.
    def _final_agg(curves: list[dict]) -> dict:
        final_vals = [c["avg_R"][-1] for c in curves if c.get("avg_R")]
        ep_95      = [c.get("epoch_95pct_final", 0) for c in curves]
        return {
            "final_avg_R_mean": sum(final_vals)/len(final_vals) if final_vals else 0,
            "final_avg_R_std":  statistics.stdev(final_vals) if len(final_vals) > 1 else 0,
            "epoch_95pct_mean": sum(ep_95)/len(ep_95) if ep_95 else 0,
        }

    results["summary"] = {
        "clip_head": _final_agg(curves_clip),
        "jl_mahal":  _final_agg(curves_jl),
    }
    print(f"\nE6 summary: {results['summary']}")

    save_json(results, output_dir / "E6_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run(cfg)
