from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader

from .losses import infonce_loss


@torch.no_grad()
def val_recall_diagonal(
    model,
    loader: DataLoader,
    *,
    device: str,
    k: int = 10,
    mode: str = "standard",
    mrl_eval_dim: int | None = None,
) -> float:
    model.eval()
    all_i, all_t = [], []
    for img, txt in loader:
        img = img.to(device)
        txt = txt.to(device)
        if mode == "mrl":
            all_i.append(model.encode_image(img, dim=mrl_eval_dim).cpu())
            all_t.append(model.encode_text(txt, dim=mrl_eval_dim).cpu())
        else:
            all_i.append(model.encode_image(img).cpu())
            all_t.append(model.encode_text(txt).cpu())
    i = torch.cat(all_i)
    t = torch.cat(all_t)
    sims = i @ t.T
    k_eff = min(k, t.shape[0])
    topk = sims.topk(k_eff, dim=1).indices
    gt = torch.arange(i.shape[0]).unsqueeze(1)
    return float((topk == gt).any(dim=1).float().mean().item())


def train_bimodal(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    device: str,
    ckpt_dir: Path,
    patience: int = 10,
    warmup_epochs: int = 0,
    grad_clip: float = 1.0,
    mode: str = "standard",
    mrl_dims: list[int] | None = None,
) -> dict:
    model = model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    warmup_epochs = min(warmup_epochs, epochs)
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs - warmup_epochs)
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val = -1.0
    best_epoch = -1
    no_improve = 0
    history: dict[str, list[float]] = {"train_loss": [], "val_r10": []}

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        running = 0.0

        for img, txt in train_loader:
            img = img.to(device)
            txt = txt.to(device)

            scale = model.logit_scale.exp().clamp(max=100.0) if hasattr(model, "logit_scale") else 1.0 / 0.07

            if mode == "mrl":
                dims = mrl_dims if mrl_dims else [model.max_dim]
                loss = 0.0
                for d in dims:
                    zi = model.encode_image(img, dim=d)
                    zt = model.encode_text(txt, dim=d)
                    loss = loss + infonce_loss(zi, zt, scale)
                loss = loss / float(len(dims))
            elif mode == "directclr_proxy":
                zi, zt = model.train_views(img, txt)
                loss = infonce_loss(zi, zt, scale)
            else:
                zi, zt = model(img, txt)
                loss = infonce_loss(zi, zt, scale)

            if hasattr(model, "regularization_loss"):
                reg = model.regularization_loss()
                if reg is not None:
                    loss = loss + reg

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()
            if hasattr(model, "project_parameters"):
                model.project_parameters()
            running += float(loss.item())

        scheduler.step()
        avg_loss = running / max(1, len(train_loader))
        val_r10 = val_recall_diagonal(
            model,
            val_loader,
            device=device,
            k=10,
            mode=("mrl" if mode == "mrl" else "standard"),
            mrl_eval_dim=max(mrl_dims) if mrl_dims else None,
        )
        history["train_loss"].append(avg_loss)
        history["val_r10"].append(val_r10)
        print(f"Epoch {epoch+1:03d}/{epochs} loss={avg_loss:.4f} valR10={val_r10:.4f} ({time.time()-t0:.1f}s)")

        if val_r10 > best_val:
            best_val = val_r10
            best_epoch = epoch + 1
            no_improve = 0
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return {
        "best_val_r10": best_val,
        "best_epoch": best_epoch,
        "history": history,
    }


def _cycle_next(it, loader):
    try:
        return next(it), it
    except StopIteration:
        it = iter(loader)
        return next(it), it


def train_trimodal(
    model,
    image_text_loader: DataLoader,
    audio_text_loader: DataLoader,
    *,
    val_eval_fn: Callable[[], dict],
    epochs: int,
    lr: float,
    device: str,
    ckpt_dir: Path,
    patience: int = 10,
    warmup_epochs: int = 0,
    grad_clip: float = 1.0,
    eval_every: int = 1,
) -> dict:
    model = model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    warmup_epochs = min(warmup_epochs, epochs)
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs - warmup_epochs)
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val = -1.0
    best_epoch = -1
    no_improve = 0
    history: dict[str, list[float]] = {"train_loss": [], "val_score": []}

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        running = 0.0

        it_it = iter(image_text_loader)
        it_at = iter(audio_text_loader)
        steps = max(len(image_text_loader), len(audio_text_loader))

        for _ in range(steps):
            (img, txt_i), it_it = _cycle_next(it_it, image_text_loader)
            (aud, txt_a), it_at = _cycle_next(it_at, audio_text_loader)
            img = img.to(device)
            txt_i = txt_i.to(device)
            aud = aud.to(device)
            txt_a = txt_a.to(device)

            scale = model.logit_scale.exp().clamp(max=100.0) if hasattr(model, "logit_scale") else 1.0 / 0.07

            zi = model.encode_image(img)
            zti = model.encode_text(txt_i)
            za = model.encode_audio(aud)
            zta = model.encode_text(txt_a)

            loss_it = infonce_loss(zi, zti, scale)
            loss_at = infonce_loss(za, zta, scale)
            loss = 0.5 * (loss_it + loss_at)

            if hasattr(model, "regularization_loss"):
                reg = model.regularization_loss()
                if reg is not None:
                    loss = loss + reg

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()
            running += float(loss.item())

        scheduler.step()
        avg_loss = running / max(1, steps)
        history["train_loss"].append(avg_loss)

        if (epoch + 1) % eval_every == 0:
            val_metrics = val_eval_fn()
            val_score = float(val_metrics["combined_avg_R"])
            history["val_score"].append(val_score)
            print(
                f"Epoch {epoch+1:03d}/{epochs} loss={avg_loss:.4f} "
                f"valCombined={val_score:.4f} ({time.time()-t0:.1f}s)"
            )

            if val_score > best_val:
                best_val = val_score
                best_epoch = epoch + 1
                no_improve = 0
                torch.save(model.state_dict(), ckpt_dir / "best.pt")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        else:
            print(f"Epoch {epoch+1:03d}/{epochs} loss={avg_loss:.4f} ({time.time()-t0:.1f}s)")

    return {
        "best_val": best_val,
        "best_epoch": best_epoch,
        "history": history,
    }
