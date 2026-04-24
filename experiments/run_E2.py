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
from utils.common import set_seed, save_json, load_best_checkpoint, eval_dataset


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
    set_seed(cfg.get("seed", 0))
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = cfg["dataset"]
    cache_dir = Path(cfg["cache_dir"]) / dataset_name
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

    n_val = int(len(ds) * 0.1)
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val])
    if isinstance(ds, MultiCaptionDataset):
        val_ds.dataset.train(False)
    kw = dict(batch_size=cfg["batch_size"], num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader   = DataLoader(val_ds, shuffle=False, **kw)

    results: dict = {}
    m = cfg["embed_dim"]
    backbone = cfg.get("backbone_name", "openai/clip-vit-base-patch32")

    def _run_model(label: str, model: nn.Module) -> dict:
        ckpt_dir = Path(cfg["output_dir"]) / label
        train(model, train_loader, val_loader,
              epochs=cfg["epochs"], lr=cfg["lr"],
              temperature=cfg["temperature"], device=device,
              ckpt_dir=ckpt_dir, patience=cfg.get("patience", 5))
        model = load_best_checkpoint(model, ckpt_dir, device)
        # Eval on full dataset with correct GT.
        metrics = eval_dataset(model, ds, device, cfg["batch_size"])
        metrics["n_params"] = model.n_trainable_params()
        print(f"  {label}: {metrics}")
        return metrics

    # CLIP projection head (reference).
    results["clip_head"] = _run_model(
        "clip_head", CLIPProjectionHead(cfg["vision_dim"], cfg["text_dim"], m)
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
        results[label] = _run_model(label, model)

    # LoRA on CLIP head (initialized from pretrained CLIP weights).
    for rank in cfg["lora_ranks"]:
        label = f"lora_r{rank}"
        model = LoRACLIPHead(cfg["vision_dim"], cfg["text_dim"], m,
                             lora_rank=rank, backbone_name=backbone)
        results[label] = _run_model(label, model)

    # Mahalanobis only (no JL).
    for rank in cfg["mahal_ranks"]:
        rank_tag = "full" if rank is None else str(rank)
        label = f"mahal_only_r{rank_tag}"
        model = MahalanobisOnlyPipeline(
            vision_dim=cfg["vision_dim"], text_dim=cfg["text_dim"],
            embed_dim=m, mahal_rank=rank,
        )
        results[label] = _run_model(label, model)

    save_json(results, Path(cfg["output_dir"]) / "E2_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run(cfg)
