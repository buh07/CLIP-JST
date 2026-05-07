"""
E2. Parameter efficiency.

Plots retrieval performance vs. trainable parameter count across:
  - JL + rank-{4,16,64,full} Mahalanobis  (main method variants)
  - CLIP projection head
  - LoRA on CLIP projection head (rank-{4,16,64})
  - Mahalanobis only (no JL)

Hypothesis: JL + rank-16 Mahalanobis Pareto-dominates LoRA on CLIP head
at matched parameter budget.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.cache import PairedFeatureDataset, MultiCaptionDataset
from models.baselines import CLIPProjectionHead, MahalanobisOnlyPipeline
from models.pipeline import CLIPJSTPipeline
from training.trainer import train
from eval.retrieval import recall_at_k
from utils.common import set_seed, save_json, load_best_checkpoint


@torch.no_grad()
def _eval_on_test(model, ds, test_ds, device: str, batch_size: int) -> dict:
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
        return recall_at_k(torch.cat(all_img_emb), torch.cat(all_txt_emb), gt_i2t=test_gt_i2t)

    img_feats = ds.img[test_indices]
    txt_feats = ds.txt[test_indices]
    all_img_emb, all_txt_emb = [], []
    for start in range(0, len(img_feats), batch_size):
        all_img_emb.append(model.encode_image(img_feats[start:start + batch_size].to(device)).cpu())
        all_txt_emb.append(model.encode_text(txt_feats[start:start + batch_size].to(device)).cpu())
    return recall_at_k(torch.cat(all_img_emb), torch.cat(all_txt_emb))


# ---------------------------------------------------------------------------
# LoRA-on-CLIP baseline
# ---------------------------------------------------------------------------

def _load_pretrained_clip_projections(
    vision_dim: int,
    text_dim: int,
    embed_dim: int,
    backbone_name: str = "openai/clip-vit-base-patch32",
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """
    Load CLIP's pretrained projection weights for LoRA initialization.

    CLIP ViT-B/32 has visual_projection (512, 768) and text_projection (512, 512).
    If embed_dim != 512, we truncate/pad so the frozen base weight has shape
    (embed_dim, vision_dim) and (embed_dim, text_dim).  This is an approximation;
    document it in the paper.
    """
    try:
        from transformers import CLIPModel
        clip = CLIPModel.from_pretrained(backbone_name)
        W_v_full = clip.visual_projection.weight.data.clone()   # (clip_dim, vision_dim)
        W_t_full = clip.text_projection.weight.data.clone()     # (clip_dim, text_dim)

        clip_dim = W_v_full.shape[0]
        if clip_dim == embed_dim:
            return W_v_full, W_t_full

        # Truncate or zero-pad along the output dimension.
        if clip_dim > embed_dim:
            return W_v_full[:embed_dim], W_t_full[:embed_dim]
        else:
            pad_v = torch.zeros(embed_dim - clip_dim, vision_dim)
            pad_t = torch.zeros(embed_dim - clip_dim, text_dim)
            return torch.cat([W_v_full, pad_v]), torch.cat([W_t_full, pad_t])
    except Exception as e:
        print(f"  Warning: could not load pretrained CLIP projections ({e}).")
        print("  LoRA base initialized from zeros — model is low-rank training, not adaptation.")
        return None, None


class LoRACLIPHead(nn.Module):
    """
    Rank-r LoRA update on top of a frozen CLIP projection head:
        W_eff = W_0 + B @ A  where A in R^{r×d}, B in R^{m×r}.

    W_0 is initialized from pretrained CLIP weights when available.
    Parameter count: r*(d_v + m) + r*(d_t + m) total.
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        embed_dim: int,
        lora_rank: int,
        backbone_name: str = "openai/clip-vit-base-patch32",
    ):
        super().__init__()
        self.embed_dim = embed_dim

        W_v, W_t = _load_pretrained_clip_projections(
            vision_dim, text_dim, embed_dim, backbone_name
        )
        if W_v is None:
            W_v = torch.zeros(embed_dim, vision_dim)
            W_t = torch.zeros(embed_dim, text_dim)

        self.register_buffer("W_v", W_v)
        self.register_buffer("W_t", W_t)

        # LoRA factors — B initialized to zero so initial f = W_0 x.
        self.A_v = nn.Parameter(torch.randn(lora_rank, vision_dim) * (vision_dim ** -0.5))
        self.B_v = nn.Parameter(torch.zeros(embed_dim, lora_rank))
        self.A_t = nn.Parameter(torch.randn(lora_rank, text_dim) * (text_dim ** -0.5))
        self.B_t = nn.Parameter(torch.zeros(embed_dim, lora_rank))

    def encode_image(self, v: torch.Tensor) -> torch.Tensor:
        W = self.W_v + self.B_v @ self.A_v
        return F.normalize(v @ W.T, dim=-1)

    def encode_text(self, t: torch.Tensor) -> torch.Tensor:
        W = self.W_t + self.B_t @ self.A_t
        return F.normalize(t @ W.T, dim=-1)

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_image(v), self.encode_text(t)

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(cfg: dict) -> None:
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = cfg["dataset"]
    cache_dir = Path(cfg["cache_dir"]) / dataset_name
    n_cap = cfg.get("n_captions", 1)
    m = cfg["embed_dim"]
    backbone = cfg.get("backbone_name", "openai/clip-vit-base-patch32")
    seeds = cfg.get("seeds", [cfg.get("seed", 0)])

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

    all_results: dict[str, list[dict]] = {}

    for seed in seeds:
        set_seed(seed)
        n = len(ds)
        n_test  = int(n * 0.10)
        n_val   = int(n * 0.10)
        n_train = n - n_val - n_test

        if isinstance(ds, MultiCaptionDataset):
            ds.train(True)
        train_ds, val_ds, test_ds = random_split(
            ds, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(seed)
        )
        if isinstance(ds, MultiCaptionDataset):
            val_ds.dataset.train(False)
            test_ds.dataset.train(False)

        kw = dict(batch_size=cfg["batch_size"], num_workers=4, pin_memory=True)
        train_loader = DataLoader(train_ds, shuffle=True, **kw)
        val_loader   = DataLoader(val_ds, shuffle=False, **kw)

        def _run_model(label: str, model: nn.Module) -> dict:
            ckpt_dir = Path(cfg["output_dir"]) / label / f"seed{seed}"
            train(model, train_loader, val_loader,
                  epochs=cfg["epochs"], lr=cfg["lr"],
                  temperature=cfg.get("temperature", 0.07), device=device,
                  ckpt_dir=ckpt_dir, patience=cfg.get("patience", 10),
                  warmup_epochs=cfg.get("warmup_epochs", 0))
            model = load_best_checkpoint(model, ckpt_dir, device)
            metrics = _eval_on_test(model, ds, test_ds, device, cfg["batch_size"])
            metrics["n_params"] = model.n_trainable_params()
            print(f"  {label} seed={seed}: {metrics}")
            return metrics

        # CLIP projection head (reference).
        label = "clip_head"
        all_results.setdefault(label, []).append(
            _run_model(label, CLIPProjectionHead(cfg["vision_dim"], cfg["text_dim"], m))
        )

        # JL + Mahalanobis variants.
        for rank in cfg["mahal_ranks"]:
            rank_tag = "full" if rank is None else str(rank)
            label = f"jl_mahal_r{rank_tag}"
            model = CLIPJSTPipeline(
                vision_dim=cfg["vision_dim"], text_dim=cfg["text_dim"],
                embed_dim=m, mahal_rank=rank,
                jl_eps=cfg["jl_eps"], jl_seed=cfg["jl_seed"],
            )
            all_results.setdefault(label, []).append(_run_model(label, model))

        # LoRA on CLIP head (initialized from pretrained CLIP weights).
        for rank in cfg["lora_ranks"]:
            label = f"lora_r{rank}"
            model = LoRACLIPHead(cfg["vision_dim"], cfg["text_dim"], m,
                                 lora_rank=rank, backbone_name=backbone)
            all_results.setdefault(label, []).append(_run_model(label, model))

        # Mahalanobis only (no JL).
        for rank in cfg["mahal_ranks"]:
            rank_tag = "full" if rank is None else str(rank)
            label = f"mahal_only_r{rank_tag}"
            model = MahalanobisOnlyPipeline(
                vision_dim=cfg["vision_dim"], text_dim=cfg["text_dim"],
                embed_dim=m, mahal_rank=rank,
            )
            all_results.setdefault(label, []).append(_run_model(label, model))

    # Aggregate mean ± std across seeds.
    final_results: dict = {}
    for label, runs in all_results.items():
        agg: dict = {"n_params": runs[0]["n_params"]}
        for key in ["i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10", "avg_R"]:
            vals = [r[key] for r in runs if key in r]
            if vals:
                agg[key] = {"mean": sum(vals) / len(vals),
                            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0}
        final_results[label] = agg

    save_json(final_results, Path(cfg["output_dir"]) / "E2_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run(cfg)
