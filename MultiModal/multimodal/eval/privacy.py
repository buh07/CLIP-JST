from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


def reconstruction_metrics(x_hat: torch.Tensor, x_true: torch.Tensor) -> dict[str, float]:
    err = x_hat - x_true
    err_sq = err.pow(2).sum(dim=1)
    norm_sq = x_true.pow(2).sum(dim=1).clamp(min=1e-8)
    rel = err_sq / norm_sq
    return {
        "mean_relative_reconstruction_error": float(rel.mean().item()),
        "mean_abs_reconstruction_error": float(err_sq.mean().sqrt().item()),
        "rmse": float(err.pow(2).mean().sqrt().item()),
    }


def split_coordinate_error(
    x_hat: torch.Tensor,
    x_true: torch.Tensor,
    visible_mask: torch.Tensor | None,
) -> dict[str, float | None]:
    if visible_mask is None:
        return {
            "visible_coord_mse": None,
            "hidden_coord_mse": None,
            "visible_coord_count": None,
            "hidden_coord_count": None,
        }
    mask = visible_mask.bool().cpu()
    if mask.ndim != 1:
        raise ValueError("visible_mask must be a 1-D tensor")
    d = x_true.shape[1]
    if mask.numel() != d:
        raise ValueError("visible_mask length must match feature dimension")
    mse_per_coord = (x_hat - x_true).pow(2).mean(dim=0).cpu()
    vis_count = int(mask.sum().item())
    hid_count = int((~mask).sum().item())
    vis_mse = float(mse_per_coord[mask].mean().item()) if vis_count > 0 else math.nan
    hid_mse = float(mse_per_coord[~mask].mean().item()) if hid_count > 0 else math.nan
    return {
        "visible_coord_mse": vis_mse,
        "hidden_coord_mse": hid_mse,
        "visible_coord_count": vis_count,
        "hidden_coord_count": hid_count,
    }


def pseudoinverse_reconstruction(
    z: torch.Tensor,
    x: torch.Tensor,
    transform: torch.Tensor,
) -> dict[str, float]:
    """
    Reconstruct x from z where z = x @ transform.T.

    transform has shape (k, d), where d is raw feature dimension.
    """
    if transform.ndim != 2:
        raise ValueError("transform must be rank-2")
    if x.shape[0] != z.shape[0]:
        raise ValueError("x and z must share batch axis")
    if transform.shape[1] != x.shape[1]:
        raise ValueError("transform second dim must match raw feature dim")
    if transform.shape[0] != z.shape[1]:
        raise ValueError("transform first dim must match projected dim")

    c = transform.T.float()  # (d, k)
    pinv = torch.linalg.pinv(c)  # (k, d)
    x_hat = z.float() @ pinv
    return reconstruction_metrics(x_hat, x.float())


def mlp_inversion_attack(
    z: torch.Tensor,
    x: torch.Tensor,
    *,
    train_frac: float = 0.8,
    hidden_dim: int = 1024,
    epochs: int = 80,
    lr: float = 1e-3,
    batch_size: int = 1024,
    seed: int = 0,
    device: str = "cuda",
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    """
    Train MLP inverter g: z -> x and evaluate on held-out split.

    Returns (metrics, x_hat_test, x_true_test).
    """
    z = z.float().cpu()
    x = x.float().cpu()
    if len(z) != len(x):
        raise ValueError("z and x must have equal sample count")
    if len(z) < 4:
        raise ValueError("need at least 4 samples for train/test split")

    ds = TensorDataset(z, x)
    n_train = max(1, int(train_frac * len(ds)))
    n_test = len(ds) - n_train
    if n_test <= 0:
        n_train = len(ds) - 1
        n_test = 1
    train_ds, test_ds = random_split(
        ds,
        [n_train, n_test],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = nn.Sequential(
        nn.Linear(z.shape[1], hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, x.shape[1]),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        model.train()
        for zb, xb in train_loader:
            zb = zb.to(device)
            xb = xb.to(device)
            opt.zero_grad()
            pred = model(zb)
            loss = loss_fn(pred, xb)
            loss.backward()
            opt.step()

    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for zb, xb in test_loader:
            zb = zb.to(device)
            pred = model(zb).cpu()
            preds.append(pred)
            trues.append(xb.cpu())
    x_hat = torch.cat(preds)
    x_true = torch.cat(trues)
    return reconstruction_metrics(x_hat, x_true), x_hat, x_true


@torch.no_grad()
def _batch_encode_images(model, x: torch.Tensor, *, device: str, batch_size: int) -> torch.Tensor:
    model = model.to(device).eval()
    outs = []
    for i in range(0, len(x), batch_size):
        outs.append(model.encode_image(x[i:i + batch_size].to(device)).cpu())
    return torch.cat(outs, dim=0)


def iterative_feature_inversion_attack(
    model,
    x: torch.Tensor,
    *,
    train_frac: float = 0.8,
    steps: int = 250,
    lr: float = 0.05,
    prior_weight: float = 1e-3,
    batch_size: int = 256,
    seed: int = 0,
    device: str = "cuda",
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    """
    Strong white-box iterative inversion attack in feature space.

    For each target embedding z = f(x), optimize x_hat to minimize
    ||f(x_hat) - z||^2 + prior_weight * ||(x_hat - mu) / sigma||^2.
    """
    x = x.float().cpu()
    if len(x) < 4:
        raise ValueError("need at least 4 samples for train/test split")

    n_train = max(1, int(train_frac * len(x)))
    n_test = len(x) - n_train
    if n_test <= 0:
        n_train = len(x) - 1
        n_test = 1
    perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    x_train = x[train_idx]
    x_test = x[test_idx]

    mu = x_train.mean(dim=0, keepdim=True)
    sigma = x_train.std(dim=0, keepdim=True).clamp(min=1e-3)
    z_test = _batch_encode_images(model, x_test, device=device, batch_size=batch_size)

    rec_batches = []
    for i in range(0, len(x_test), batch_size):
        xb = x_test[i:i + batch_size]
        zb = z_test[i:i + batch_size]
        cur_bs = xb.shape[0]

        x_var = (mu + sigma * torch.randn(cur_bs, x.shape[1], generator=torch.Generator().manual_seed(seed + i))).to(device)
        x_var = torch.nn.Parameter(x_var)
        opt = torch.optim.Adam([x_var], lr=lr)

        zb = zb.to(device)
        mu_d = mu.to(device)
        sigma_d = sigma.to(device)

        for _ in range(steps):
            opt.zero_grad()
            z_hat = model.encode_image(x_var)
            recon = (z_hat - zb).pow(2).mean()
            prior = ((x_var - mu_d) / sigma_d).pow(2).mean()
            loss = recon + prior_weight * prior
            loss.backward()
            opt.step()

        rec_batches.append(x_var.detach().cpu())

    x_hat = torch.cat(rec_batches, dim=0)
    return reconstruction_metrics(x_hat, x_test), x_hat, x_test
