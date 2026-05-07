from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset, random_split

from ..common import env_snapshot, load_json, mark_done, save_json, set_seed
from ..data.datasets import KarpathyCache
from ..eval.privacy import mlp_inversion_attack, reconstruction_metrics, split_coordinate_error
from ..eval.retrieval import recall_at_k
from ..eval.stats import build_metric_report
from ..models import MahalanobisOnlyHead
from ..training.losses import infonce_loss


class _DPInfoNCEWrapper(nn.Module):
    """
    Opacus-friendly wrapper with single-input forward.
    """

    def __init__(self, vision_dim: int, text_dim: int, shared_raw_dim: int = 768):
        super().__init__()
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.head = MahalanobisOnlyHead(
            vision_dim=vision_dim,
            text_dim=text_dim,
            shared_raw_dim=shared_raw_dim,
        )
        # Scalar temperature parameter is not covered by Opacus grad-samplers.
        self.head.logit_scale.requires_grad_(False)

    @property
    def logit_scale(self):
        return self.head.logit_scale

    def forward(self, pair: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v = pair[:, : self.vision_dim]
        t = pair[:, self.vision_dim : self.vision_dim + self.text_dim]
        zi = self.head.encode_image(v)
        zt = self.head.encode_text(t)
        return zi, zt

    def encode_image(self, v: torch.Tensor) -> torch.Tensor:
        return self.head.encode_image(v)

    def encode_text(self, t: torch.Tensor) -> torch.Tensor:
        return self.head.encode_text(t)

    def n_trainable_params(self) -> int:
        return self.head.n_trainable_params()


def _linear_probe_reconstruction(
    z: torch.Tensor,
    x: torch.Tensor,
    *,
    train_frac: float = 0.8,
    seed: int = 0,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    z = z.float().cpu()
    x = x.float().cpu()
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
    z_train = torch.stack([train_ds[i][0] for i in range(len(train_ds))], dim=0)
    x_train = torch.stack([train_ds[i][1] for i in range(len(train_ds))], dim=0)
    z_test = torch.stack([test_ds[i][0] for i in range(len(test_ds))], dim=0)
    x_test = torch.stack([test_ds[i][1] for i in range(len(test_ds))], dim=0)

    b = torch.linalg.pinv(z_train) @ x_train
    x_hat = z_test @ b
    return reconstruction_metrics(x_hat, x_test), x_hat, x_test


def _unwrap_model(model):
    return getattr(model, "_module", model)


@torch.no_grad()
def _eval_karpathy(model, cache: KarpathyCache, split_name: str, *, device: str, batch_size: int) -> dict[str, float]:
    model.eval()
    core = _unwrap_model(model)
    img, txt, gt_i2t, gt_t2i = cache.eval_tensors(split_name)
    all_i, all_t = [], []
    for i in range(0, len(img), batch_size):
        all_i.append(core.encode_image(img[i:i + batch_size].to(device)).cpu())
    for i in range(0, len(txt), batch_size):
        all_t.append(core.encode_text(txt[i:i + batch_size].to(device)).cpu())
    zi = torch.cat(all_i)
    zt = torch.cat(all_t)
    return recall_at_k(zi, zt, gt_i2t=gt_i2t, gt_t2i=gt_t2i)


def _train_dp(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    target_epsilon: float,
    delta: float,
    max_grad_norm: float,
    device: str,
) -> tuple[dict, object]:
    try:
        from opacus import PrivacyEngine
    except Exception as e:
        raise RuntimeError("Opacus is required for Stage 8. Install with `pip install opacus`.") from e

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4,
    )

    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        target_epsilon=target_epsilon,
        target_delta=delta,
        epochs=epochs,
        max_grad_norm=max_grad_norm,
        grad_sample_mode="hooks",
    )

    best_val = -1.0
    best_state = None
    history = {"train_loss": [], "val_r10": [], "epsilon_spent": []}
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for img, txt in train_loader:
            pair = torch.cat([img, txt], dim=1).to(device)
            scale = model.logit_scale.exp().clamp(max=100.0) if hasattr(model, "logit_scale") else 1.0 / 0.07
            zi, zt = model(pair)
            loss = infonce_loss(zi, zt, scale)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.item())

        eps_spent = float(privacy_engine.accountant.get_epsilon(delta=delta))
        val_r10 = _eval_recall10(model, val_loader, device=device)
        avg_loss = running / max(1, len(train_loader))
        history["train_loss"].append(avg_loss)
        history["val_r10"].append(val_r10)
        history["epsilon_spent"].append(eps_spent)
        print(
            f"DP epoch {epoch+1:03d}/{epochs} loss={avg_loss:.4f} valR10={val_r10:.4f} eps={eps_spent:.3f}"
        )
        if val_r10 > best_val:
            best_val = val_r10
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    dp_meta = {
        "target_epsilon": float(target_epsilon),
        "delta": float(delta),
        "max_grad_norm": float(max_grad_norm),
        "noise_multiplier": float(getattr(optimizer, "noise_multiplier", float("nan"))),
        "sample_rate": float(getattr(train_loader, "sample_rate", float("nan"))),
        "epsilon_spent_final": float(privacy_engine.accountant.get_epsilon(delta=delta)),
        "steps": int(sum(1 for _ in train_loader) * epochs),
    }
    return {"best_val_r10": best_val, "history": history, "dp_meta": dp_meta}, model


@torch.no_grad()
def _eval_recall10(model, loader: DataLoader, *, device: str) -> float:
    model.eval()
    core = _unwrap_model(model)
    all_i, all_t = [], []
    for img, txt in loader:
        img = img.to(device)
        txt = txt.to(device)
        all_i.append(core.encode_image(img).cpu())
        all_t.append(core.encode_text(txt).cpu())
    i = torch.cat(all_i)
    t = torch.cat(all_t)
    sims = i @ t.T
    topk = sims.topk(min(10, t.shape[0]), dim=1).indices
    gt = torch.arange(i.shape[0]).unsqueeze(1)
    return float((topk == gt).any(dim=1).float().mean().item())


@torch.no_grad()
def _encode_images(model, x: torch.Tensor, *, device: str, batch_size: int) -> torch.Tensor:
    core = _unwrap_model(model)
    outs = []
    for i in range(0, len(x), batch_size):
        outs.append(core.encode_image(x[i:i + batch_size].to(device)).cpu())
    return torch.cat(outs)


def run(cfg: dict) -> None:
    start = time.time()
    output_root = Path(cfg["output_root"]).resolve()
    stage_root = output_root / "stage8_e8d"
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    # COCO-only by plan.
    cache_dir = Path(cfg["cache_root"]) / "coco"
    cache = KarpathyCache.from_paths(
        cache_dir / cfg["image_cache_file"],
        cache_dir / cfg["text_cache_file"],
        cache_dir / "metadata.json",
    )

    train_ds = cache.make_train_dataset("train_restval", training=True)
    val_ds = cache.make_train_dataset("val", training=False)
    loader_kw = {
        "batch_size": cfg["batch_size"],
        "num_workers": cfg.get("num_workers", 4),
        "pin_memory": True,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)

    seeds = list(cfg["seeds"])
    epsilons = [float(e) for e in cfg["epsilons"]]
    results = {"stage": "stage8_e8d_dpsgd", "config": cfg, "raw": {}, "stats": {}}

    # Inversion probe set.
    probe_split = cfg.get("privacy_split", "train_restval")
    probe_idx = cache.split_indices(probe_split)
    x_probe = cache.image_feats[probe_idx].float()
    max_probe = int(cfg.get("max_probe_samples", 0))
    if max_probe > 0 and len(x_probe) > max_probe:
        x_probe = x_probe[:max_probe]

    for eps in epsilons:
        eps_key = f"eps{str(eps).replace('.', 'p')}"
        rows = []
        for seed in seeds:
            set_seed(seed)
            run_dir = stage_root / eps_key / f"seed{seed}"
            eval_file = run_dir / "eval.json"
            if eval_file.exists():
                rows.append(load_json(eval_file))
                continue

            model = _DPInfoNCEWrapper(
                vision_dim=cfg["vision_dim"],
                text_dim=cfg["text_dim"],
                shared_raw_dim=cfg.get("shared_raw_dim", 768),
            )
            train_result, model = _train_dp(
                model,
                train_loader,
                val_loader,
                epochs=cfg["epochs"],
                lr=cfg["lr"],
                target_epsilon=eps,
                delta=float(cfg["delta"]),
                max_grad_norm=float(cfg["max_grad_norm"]),
                device=cfg["device"],
            )
            if bool(cfg.get("save_checkpoints", False)):
                run_dir.mkdir(parents=True, exist_ok=True)
                torch.save(_unwrap_model(model).state_dict(), run_dir / "best.pt")

            retrieval = _eval_karpathy(
                model,
                cache,
                "test",
                device=cfg["device"],
                batch_size=cfg["eval_batch_size"],
            )

            z_probe = _encode_images(model, x_probe, device=cfg["device"], batch_size=cfg["eval_batch_size"])
            pseudo, xh_pseudo, xt_pseudo = _linear_probe_reconstruction(
                z_probe,
                x_probe,
                train_frac=float(cfg.get("attack_train_frac", 0.8)),
                seed=seed,
            )
            mlp, xh_mlp, xt_mlp = mlp_inversion_attack(
                z_probe,
                x_probe,
                train_frac=float(cfg.get("attack_train_frac", 0.8)),
                hidden_dim=int(cfg.get("attack_hidden_dim", 1024)),
                epochs=int(cfg.get("attack_epochs", 80)),
                lr=float(cfg.get("attack_lr", 1e-3)),
                batch_size=int(cfg.get("attack_batch_size", 1024)),
                seed=seed,
                device=cfg["device"],
            )
            # DP baseline has no hidden coordinates; all coords are visible.
            vis = torch.ones(x_probe.shape[1])
            pseudo_split = split_coordinate_error(xh_pseudo, xt_pseudo, vis)
            mlp_split = split_coordinate_error(xh_mlp, xt_mlp, vis)

            rec = {
                "seed": seed,
                "epsilon_target": eps,
                **retrieval,
                "train_result": train_result,
                "pseudoinverse": {**pseudo, **pseudo_split},
                "mlp_inverter": {**mlp, **mlp_split},
                "n_params": _unwrap_model(model).n_trainable_params(),
            }
            save_json(rec, eval_file)
            rows.append(rec)
            print(
                f"E8d eps={eps} seed={seed} avg_R={retrieval['avg_R']:.4f} "
                f"mlp_rel={mlp['mean_relative_reconstruction_error']:.4f}"
            )
        results["raw"][eps_key] = rows

    # Aggregate stats.
    per_method = {}
    for eps in epsilons:
        eps_key = f"eps{str(eps).replace('.', 'p')}"
        per_method[eps_key] = []
        for r in results["raw"][eps_key]:
            per_method[eps_key].append(
                {
                    "seed": int(r["seed"]),
                    "avg_R": float(r["avg_R"]),
                    "mlp_rel_error": float(r["mlp_inverter"]["mean_relative_reconstruction_error"]),
                    "pseudo_rel_error": float(r["pseudoinverse"]["mean_relative_reconstruction_error"]),
                    "epsilon_spent": float(r["train_result"]["dp_meta"]["epsilon_spent_final"]),
                }
            )
    results["stats"] = build_metric_report(
        per_method,
        metrics=["avg_R", "mlp_rel_error", "pseudo_rel_error", "epsilon_spent"],
        baseline_method=f"eps{str(epsilons[-1]).replace('.', 'p')}",
    )

    provenance = env_snapshot(
        Path(cfg["project_root"]),
        seeds=seeds,
        extra={
            "stage": "stage8_e8d_dpsgd",
            "elapsed_sec": time.time() - start,
        },
    )
    results_name = str(cfg.get("stage8_results_name", "E8d_dpsgd_results.json"))
    marker_name = str(cfg.get("stage8_marker_name", "stage8_e8d.done.json"))
    provenance_name = str(cfg.get("stage8_provenance_name", "provenance_stage8_e8d.json"))
    save_json(provenance, stage_root / provenance_name)
    save_json(results, stage_root / results_name)
    mark_done(markers / marker_name, {"elapsed_sec": time.time() - start, "results_file": results_name})
    print("Stage 8 (E8d) complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)
