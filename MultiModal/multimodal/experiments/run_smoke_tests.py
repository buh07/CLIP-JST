from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from ..common import save_json, set_seed
from ..eval.stats import holm_bonferroni, mean_std_ci
from ..models import (
    CLIPProjectionHead,
    ConcatJLMahalanobisHead,
    DirectCLRProxyHead,
    LearnedJLSparseHead,
    MahalanobisOnlyHead,
    MaskConcatJLMahalanobisHead,
    MRLProjectionHead,
    OrthogonalPlusMahalanobisHead,
    RandomJLOnlyHead,
    SparseJLL1Head,
    SparseJLProjectedHead,
    TriModalCLIPHead,
)
from ..training import train_bimodal, train_trimodal


def _make_correlated(n: int, latent_dim: int, image_dim: int, text_dim: int, audio_dim: int):
    z = torch.randn(n, latent_dim)
    wi = torch.randn(latent_dim, image_dim)
    wt = torch.randn(latent_dim, text_dim)
    wa = torch.randn(latent_dim, audio_dim)
    img = z @ wi + 0.05 * torch.randn(n, image_dim)
    txt = z @ wt + 0.05 * torch.randn(n, text_dim)
    aud = z @ wa + 0.05 * torch.randn(n, audio_dim)
    return img.float(), txt.float(), aud.float()


@torch.no_grad()
def _diag_recall(model, loader, device: str, kind: str = "it") -> float:
    model.eval()
    left, right = [], []
    for a, b in loader:
        a = a.to(device)
        b = b.to(device)
        if kind == "it":
            left.append(model.encode_image(a).cpu())
            right.append(model.encode_text(b).cpu())
        else:
            left.append(model.encode_audio(a).cpu())
            right.append(model.encode_text(b).cpu())
    l = torch.cat(left)
    r = torch.cat(right)
    sims = l @ r.T
    top1 = sims.argmax(dim=1)
    gt = torch.arange(l.shape[0])
    return float((top1 == gt).float().mean().item())


def run(cfg: dict) -> None:
    set_seed(cfg.get("seed", 0))
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    img, txt, aud = _make_correlated(
        n=cfg.get("n", 512),
        latent_dim=cfg.get("latent_dim", 64),
        image_dim=cfg["image_dim"],
        text_dim=cfg["text_dim"],
        audio_dim=cfg["audio_dim"],
    )
    n_train = int(0.8 * len(img))

    train_it = TensorDataset(img[:n_train], txt[:n_train])
    val_it = TensorDataset(img[n_train:], txt[n_train:])
    train_at = TensorDataset(aud[:n_train], txt[:n_train])
    val_at = TensorDataset(aud[n_train:], txt[n_train:])

    loader_kw = dict(batch_size=cfg.get("batch_size", 128), num_workers=0)
    it_train_loader = DataLoader(train_it, shuffle=True, **loader_kw)
    it_val_loader = DataLoader(val_it, shuffle=False, **loader_kw)
    at_train_loader = DataLoader(train_at, shuffle=True, **loader_kw)
    at_val_loader = DataLoader(val_at, shuffle=False, **loader_kw)

    out: dict = {"smoke": {}, "stats_checks": {}, "sanity": {}}

    # Bimodal families.
    bimodal_models = {
        "mrl": (MRLProjectionHead(cfg["image_dim"], cfg["text_dim"], max_dim=512, nested_dims=[64, 128, 256, 512]), "mrl"),
        "directclr_proxy": (DirectCLRProxyHead(cfg["image_dim"], cfg["text_dim"], full_dim=512, train_subdim=128), "directclr_proxy"),
        "learned_jl_sparse": (LearnedJLSparseHead(cfg["image_dim"], cfg["text_dim"], embed_dim=128), "standard"),
        "sparse_jl_projected": (SparseJLProjectedHead(cfg["image_dim"], cfg["text_dim"], embed_dim=128), "standard"),
        "sparse_jl_l1": (SparseJLL1Head(cfg["image_dim"], cfg["text_dim"], embed_dim=128, lambda_l1=1e-4), "standard"),
        "orth_jl_plus_mahal": (OrthogonalPlusMahalanobisHead(cfg["image_dim"], cfg["text_dim"], embed_dim=128), "standard"),
        "concat_jl_mahal": (ConcatJLMahalanobisHead(cfg["image_dim"], cfg["text_dim"], embed_dim=128, alpha=1.0, beta=1.0), "standard"),
        "mask_concat_jl_mahal": (MaskConcatJLMahalanobisHead(cfg["image_dim"], cfg["text_dim"], embed_dim=128, p=0.25, mask_seed=7), "standard"),
        "mahal_only_rfull": (MahalanobisOnlyHead(cfg["image_dim"], cfg["text_dim"]), "standard"),
    }
    for name, (model, mode) in bimodal_models.items():
        ckpt = Path(cfg["output_root"]) / "smoke" / name
        train_bimodal(
            model,
            it_train_loader,
            it_val_loader,
            epochs=1,
            lr=1e-3,
            device=device,
            ckpt_dir=ckpt,
            patience=1,
            warmup_epochs=0,
            mode=mode,
            mrl_dims=[64, 128, 256, 512] if mode == "mrl" else None,
        )
        out["smoke"][name] = {"ok": (ckpt / "best.pt").exists()}

    # Projected-gradient invariance: off-support weights must stay exactly zero.
    proj_ckpt = Path(cfg["output_root"]) / "smoke" / "sparse_jl_projected" / "best.pt"
    proj_model = SparseJLProjectedHead(cfg["image_dim"], cfg["text_dim"], embed_dim=128).to(device)
    proj_model.load_state_dict(torch.load(proj_ckpt, map_location=device, weights_only=True))
    off_v = (proj_model.weight_v * (1.0 - proj_model.support_v)).abs().max().item()
    off_t = (proj_model.weight_t * (1.0 - proj_model.support_t)).abs().max().item()
    out["sanity"]["projected_support_exact_zero"] = (off_v == 0.0 and off_t == 0.0)
    out["sanity"]["projected_support_offdiag_max"] = max(off_v, off_t)

    # L1 check on isolated regularizer step: nonzero lambda should shrink |W|.
    base_l1 = SparseJLL1Head(cfg["image_dim"], cfg["text_dim"], embed_dim=128, lambda_l1=0.0).to(device)
    test_l1 = SparseJLL1Head(cfg["image_dim"], cfg["text_dim"], embed_dim=128, lambda_l1=1e-2).to(device)
    test_l1.load_state_dict(base_l1.state_dict())
    opt_base = torch.optim.SGD(base_l1.parameters(), lr=1e-2)
    opt_test = torch.optim.SGD(test_l1.parameters(), lr=1e-2)
    opt_base.zero_grad()
    opt_base.step()
    opt_test.zero_grad()
    test_l1.regularization_loss().backward()
    opt_test.step()
    base_abs = 0.5 * (base_l1.weight_v.abs().mean().item() + base_l1.weight_t.abs().mean().item())
    test_abs = 0.5 * (test_l1.weight_v.abs().mean().item() + test_l1.weight_t.abs().mean().item())
    out["sanity"]["l1_regularization_positive"] = bool(test_l1.regularization_loss().item() > 0.0)
    out["sanity"]["l1_weight_shrinkage"] = bool(test_abs <= base_abs)
    out["sanity"]["l1_abs_weight_baseline"] = base_abs
    out["sanity"]["l1_abs_weight_regularized"] = test_abs

    # Orth+Mahal shape/gradient sanity.
    grad_model = OrthogonalPlusMahalanobisHead(cfg["image_dim"], cfg["text_dim"], embed_dim=128).to(device)
    grad_model.train()
    img_b, txt_b = next(iter(it_train_loader))
    img_b = img_b.to(device)
    txt_b = txt_b.to(device)
    zi, zt = grad_model(img_b, txt_b)
    scale = grad_model.logit_scale.exp().clamp(max=100.0)
    sim = (zi * zt).sum(dim=1).mean()
    loss = -scale * sim + grad_model.regularization_loss()
    loss.backward()
    grad_ok = all(p.grad is not None for p in grad_model.parameters() if p.requires_grad)
    out["sanity"]["orth_plus_mahal_shape_ok"] = (zi.shape == (img_b.shape[0], 128) and zt.shape == (txt_b.shape[0], 128))
    out["sanity"]["orth_plus_mahal_gradients_ok"] = bool(grad_ok)

    # Concat shape + text-padding sanity.
    concat_model = ConcatJLMahalanobisHead(cfg["image_dim"], cfg["text_dim"], embed_dim=128, alpha=1.0, beta=1.0).to(device)
    ci, ct = concat_model(img_b, txt_b)
    out["sanity"]["concat_shape_ok"] = (ci.shape == (img_b.shape[0], 896) and ct.shape == (txt_b.shape[0], 896))
    txt_pad = concat_model._pad_shared(txt_b, cfg["text_dim"])
    out["sanity"]["text_padding_tail_zero"] = bool((txt_pad[:, cfg["text_dim"]:] == 0).all().item())

    # Mask determinism + sparsity sanity.
    mask_a = MaskConcatJLMahalanobisHead(cfg["image_dim"], cfg["text_dim"], embed_dim=128, p=0.25, mask_seed=13)
    mask_b = MaskConcatJLMahalanobisHead(cfg["image_dim"], cfg["text_dim"], embed_dim=128, p=0.25, mask_seed=13)
    mask_c = MaskConcatJLMahalanobisHead(cfg["image_dim"], cfg["text_dim"], embed_dim=128, p=0.25, mask_seed=14)
    out["sanity"]["mask_seed_deterministic"] = bool(torch.equal(mask_a.mask_v, mask_b.mask_v) and torch.equal(mask_a.mask_t, mask_b.mask_t))
    out["sanity"]["mask_seed_changes_mask"] = bool((not torch.equal(mask_a.mask_v, mask_c.mask_v)) or (not torch.equal(mask_a.mask_t, mask_c.mask_t)))
    sparsity_tol = 0.1
    mv = float(mask_a.mask_v.mean().item())
    mt = float(mask_a.mask_t.mean().item())
    out["sanity"]["mask_sparsity_close_target"] = bool(abs(mv - 0.25) <= sparsity_tol and abs(mt - 0.25) <= sparsity_tol)
    out["sanity"]["mask_sparsity_image"] = mv
    out["sanity"]["mask_sparsity_text"] = mt

    # Tri-modal runner smoke.
    tri = TriModalCLIPHead(cfg["image_dim"], cfg["audio_dim"], cfg["text_dim"], embed_dim=128)
    tri_ckpt = Path(cfg["output_root"]) / "smoke" / "trimodal"

    def _val_fn():
        return {
            "combined_avg_R": 0.5 * (
                _diag_recall(tri, it_val_loader, device, kind="it")
                + _diag_recall(tri, at_val_loader, device, kind="at")
            )
        }

    train_trimodal(
        tri,
        it_train_loader,
        at_train_loader,
        val_eval_fn=_val_fn,
        epochs=1,
        lr=1e-3,
        device=device,
        ckpt_dir=tri_ckpt,
        patience=1,
        eval_every=1,
    )
    out["smoke"]["trimodal"] = {"ok": (tri_ckpt / "best.pt").exists()}

    # Statistical unit checks.
    ci = mean_std_ci([1, 2, 3, 4, 5])
    out["stats_checks"]["ci_known_mean"] = abs(ci["mean"] - 3.0) < 1e-8
    holm = holm_bonferroni({"a": 0.001, "b": 0.01, "c": 0.04})
    out["stats_checks"]["holm_monotonic"] = holm["a"]["p_holm"] <= holm["b"]["p_holm"] <= holm["c"]["p_holm"]

    # Sanity checks.
    jl = RandomJLOnlyHead(cfg["image_dim"], cfg["text_dim"], embed_dim=128).to(device)
    chance = _diag_recall(jl, it_val_loader, device, kind="it")
    out["sanity"]["random_jl_only_near_chance"] = chance < 0.1
    out["sanity"]["random_jl_only_r1"] = chance

    # Shuffled-label control.
    shuf_perm = torch.randperm(n_train)
    shuf_loader = DataLoader(TensorDataset(img[:n_train], txt[:n_train][shuf_perm]), shuffle=True, **loader_kw)
    shuf_model = CLIPProjectionHead(cfg["image_dim"], cfg["text_dim"], embed_dim=128)
    shuf_ckpt = Path(cfg["output_root"]) / "smoke" / "shuffle_label"
    train_bimodal(
        shuf_model,
        shuf_loader,
        it_val_loader,
        epochs=1,
        lr=1e-3,
        device=device,
        ckpt_dir=shuf_ckpt,
        patience=1,
    )
    shuf_model.load_state_dict(torch.load(shuf_ckpt / "best.pt", map_location=device, weights_only=True))
    shuf_model.to(device)
    shuf_r1 = _diag_recall(shuf_model, it_val_loader, device, kind="it")
    out["sanity"]["shuffle_label_near_chance"] = shuf_r1 < 0.1
    out["sanity"]["shuffle_label_r1"] = shuf_r1

    save_json(out, Path(cfg["output_root"]) / "smoke_tests.json")
    print("Smoke tests complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)
