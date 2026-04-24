"""
Training loop for CLIP-JST experiments.

Assumes features are pre-extracted and cached (see data/cache.py).
All epochs operate on cached tensors — no backbone forward passes at train time.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.infonce import infonce_loss


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    temperature: float = 0.07,
    device: str | torch.device = "cuda",
    ckpt_dir: Path | None = None,
    loss_fn: Callable = infonce_loss,
    patience: int = 5,
) -> dict[str, Any]:
    """
    Generic training loop for any model that implements encode_image / encode_text.

    Returns dict with keys 'train_losses', 'val_recalls', 'best_epoch', 'best_recall'.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_recall = -1.0
    best_epoch = 0
    no_improve = 0
    train_losses: list[float] = []
    val_recalls: list[float] = []

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0

        for img_feat, txt_feat in train_loader:
            img_feat = img_feat.to(device)
            txt_feat = txt_feat.to(device)
            img_emb, txt_emb = model(img_feat, txt_feat)
            # Pass temperature; loss functions that don't need it accept **kwargs or None.
            loss = loss_fn(img_emb, txt_emb, temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        recall = _val_recall(model, val_loader, k=10, device=device)
        val_recalls.append(recall)

        print(
            f"Epoch {epoch+1:3d}/{epochs}  loss={avg_loss:.4f}  "
            f"R@10={recall:.4f}  ({time.time()-t0:.1f}s)"
        )

        if recall > best_recall:
            best_recall = recall
            best_epoch = epoch
            no_improve = 0
            if ckpt_dir is not None:
                ckpt_dir = Path(ckpt_dir)
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt_dir / "best.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break

    return {
        "train_losses": train_losses,
        "val_recalls": val_recalls,
        "best_epoch": best_epoch,
        "best_recall": best_recall,
    }


@torch.no_grad()
def _val_recall(
    model: nn.Module,
    loader: DataLoader,
    k: int,
    device: str | torch.device,
) -> float:
    model.eval()
    all_img, all_txt = [], []
    for img_feat, txt_feat in loader:
        all_img.append(model.encode_image(img_feat.to(device)).cpu())
        all_txt.append(model.encode_text(txt_feat.to(device)).cpu())

    img_mat = torch.cat(all_img)   # (N, d)
    txt_mat = torch.cat(all_txt)   # (N, d)
    sims = img_mat @ txt_mat.T     # (N, N)
    k_eff = min(k, txt_mat.shape[0])
    topk = sims.topk(k_eff, dim=1).indices
    gt = torch.arange(img_mat.shape[0]).unsqueeze(1)
    return (topk == gt).any(dim=1).float().mean().item()


@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (image_embs, text_embs) as CPU tensors for the full loader."""
    model.eval()
    all_img, all_txt = [], []
    for img_feat, txt_feat in loader:
        all_img.append(model.encode_image(img_feat.to(device)).cpu())
        all_txt.append(model.encode_text(txt_feat.to(device)).cpu())
    return torch.cat(all_img), torch.cat(all_txt)
