"""
Shared utilities: seeding, device selection, JSON I/O, embedding extraction,
and dataset evaluation.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(cfg: dict) -> str:
    return cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")


def save_json(obj: Any, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"Saved {path}")


def load_json(path: Path | str) -> Any:
    with open(path) as f:
        return json.load(f)


@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (image_embs, text_embs) from a DataLoader (val/train split)."""
    model.eval()
    all_img, all_txt = [], []
    for batch in loader:
        img_feat, txt_feat = batch[0], batch[1]
        all_img.append(model.encode_image(img_feat.to(device)).cpu())
        all_txt.append(model.encode_text(txt_feat.to(device)).cpu())
    return torch.cat(all_img), torch.cat(all_txt)


@torch.no_grad()
def eval_dataset(
    model: nn.Module,
    ds,
    device: str | torch.device,
    batch_size: int = 2048,
) -> dict[str, float]:
    """
    Evaluate a model on the *full* dataset with correct multi-GT ground truth.

    Works for both PairedFeatureDataset (1:1 GT) and MultiCaptionDataset
    (multi-GT, 5 captions per image).  Always uses all data — never a split
    subset — so the GT indices are guaranteed to match.

    This is the function that should be used for all reported metrics.
    """
    from eval.retrieval import recall_at_k

    model.eval()

    if hasattr(ds, "get_eval_tensors"):
        # MultiCaptionDataset: N_images images, N_images * n_cap texts.
        img_feats, txt_feats, gt_i2t = ds.get_eval_tensors()
        gt_t2i = {j: [j // ds.n_captions] for j in range(len(txt_feats))}
    else:
        # PairedFeatureDataset: 1:1 pairing, diagonal GT.
        img_feats = ds.img
        txt_feats = ds.txt
        gt_i2t = None
        gt_t2i = None

    all_img: list[torch.Tensor] = []
    for i in range(0, len(img_feats), batch_size):
        all_img.append(model.encode_image(img_feats[i:i+batch_size].to(device)).cpu())

    all_txt: list[torch.Tensor] = []
    for i in range(0, len(txt_feats), batch_size):
        all_txt.append(model.encode_text(txt_feats[i:i+batch_size].to(device)).cpu())

    return recall_at_k(
        torch.cat(all_img), torch.cat(all_txt),
        gt_i2t=gt_i2t, gt_t2i=gt_t2i,
    )


def load_best_checkpoint(
    model: nn.Module,
    ckpt_dir: Path | str,
    device: str | torch.device,
) -> nn.Module:
    path = Path(ckpt_dir) / "best.pt"
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device)
