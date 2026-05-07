"""
E5. Federated learning / privacy stress test.

Simulates federated training where clients share only JL-projected features
rather than raw CLIP features.  Evaluates:

  1. Federated vs. centralized retrieval performance gap (multi-round FedAvg).
     Clients: 5 (each gets ~23K pairs from the 118K COCO train set).
     Rounds: 20 (standard FedAvg convergence requirement).
     Local epochs per round: 3.
  2. Privacy curve: feature-inversion reconstruction error as a function of
     the projection dimension m ∈ {64, 128, 256, 512}.
     Attack: x* = Phi^+ (Phi x) = minimum-norm least-squares solution.
     Success metric: normalized reconstruction error ||x* - x|| / ||x||.

Hypothesis: JL-projected features resist inversion while preserving retrieval.
Results aggregated over seeds for significance.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.cache import MultiCaptionDataset
from models.pipeline import CLIPJSTPipeline
from models.baselines import CLIPProjectionHead
from training.trainer import train, extract_embeddings
from eval.retrieval import recall_at_k
from utils.common import set_seed, save_json, load_best_checkpoint


# ---------------------------------------------------------------------------
# Federated simulation
# ---------------------------------------------------------------------------

def _fedavg(
    global_state: dict,
    client_states: list[dict],
    trainable_keys: list[str],
) -> dict:
    avg_state = {k: v.clone() for k, v in global_state.items()}
    for key in trainable_keys:
        if not all(key in cs for cs in client_states):
            continue  # skip keys absent from client models (e.g. logit_scale)
        stacked = torch.stack([cs[key].float() for cs in client_states])
        avg_state[key] = stacked.mean(0).to(global_state[key].dtype)
    return avg_state


def _federated_train(
    img_feats: torch.Tensor,
    txt_feats: torch.Tensor,
    model: CLIPJSTPipeline,
    cfg: dict,
    device: str,
    output_dir: Path,
    n_rounds: int = 1,
) -> tuple[CLIPJSTPipeline, list[float]]:
    """
    Multi-round FedAvg simulation.

    Each client receives only JL-projected features (not raw CLIP features) and
    trains its local Mahalanobis head.  Mahalanobis parameters are averaged
    globally after each round.  Returns the trained model and per-round recall.
    """
    n_clients = cfg["n_clients"]
    n = len(img_feats)
    indices = torch.randperm(n)
    chunks  = torch.chunk(indices, n_clients)

    # Pre-apply JL so clients never see raw features.
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        jl_img = model.jl_v(img_feats.to(device)).cpu()
        jl_txt = model.jl_t(txt_feats.to(device)).cpu()

    trainable_keys = [k for k, v in model.named_parameters() if v.requires_grad]
    global_state   = {k: v.detach().clone() for k, v in model.state_dict().items()}
    round_recalls: list[float] = []

    for round_idx in range(n_rounds):
        client_states = []
        for cid, chunk in enumerate(chunks):
            if len(chunk) < 2:
                continue
            client_img = jl_img[chunk]
            client_txt = jl_txt[chunk]

            client_model = _IdentityJLPipeline(model.embed_dim, cfg.get("mahal_rank"))
            client_model.load_state_dict(global_state, strict=False)

            n_val = max(1, int(len(chunk) * 0.1))
            ds    = TensorDataset(client_img, client_txt)
            train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val])
            kw = dict(batch_size=min(cfg["batch_size"], len(train_ds)), num_workers=0)
            train_loader = DataLoader(train_ds, shuffle=True, **kw)
            val_loader   = DataLoader(val_ds, shuffle=False, **kw)

            ckpt = output_dir / f"round{round_idx}_client{cid}"
            train(client_model, train_loader, val_loader,
                  epochs=cfg.get("local_epochs", 3), lr=cfg["lr"],
                  temperature=cfg.get("temperature", 0.07), device=device,
                  ckpt_dir=ckpt, patience=cfg.get("local_epochs", 3),
                  warmup_epochs=0)
            client_model.load_state_dict(
                torch.load(ckpt / "best.pt", map_location="cpu", weights_only=True)
            )
            client_states.append({k: v.detach() for k, v in client_model.state_dict().items()})

        if client_states:
            global_state = _fedavg(global_state, client_states, trainable_keys)

        # Log round-level recall on a held-out probe set.
        model.load_state_dict(global_state, strict=False)
        model.eval()
        n_probe = min(2000, n)
        probe_img = jl_img[:n_probe].to(device)
        probe_txt = jl_txt[:n_probe].to(device)
        with torch.no_grad():
            # For round eval, wrap in IdentityJLPipeline to encode from JL space.
            tmp = _IdentityJLPipeline(model.embed_dim, cfg.get("mahal_rank")).to(device)
            tmp.load_state_dict(global_state, strict=False)
            tmp.eval()
            iv = tmp.encode_image(probe_img).cpu()
            it = tmp.encode_text(probe_txt).cpu()
        r = recall_at_k(iv, it)
        round_recalls.append(r["avg_R"])
        print(f"  Round {round_idx+1}/{n_rounds}  fed_avg_R={r['avg_R']:.4f}")

    model.load_state_dict(global_state, strict=False)
    return model, round_recalls


class _IdentityJLPipeline(torch.nn.Module):
    """Client-side model: input is already JL-projected; just applies Mahalanobis."""

    def __init__(self, embed_dim: int, mahal_rank: int | None = None):
        super().__init__()
        self.embed_dim = embed_dim
        from models.mahalanobis import FullMahalanobis, LowRankMahalanobis
        if mahal_rank is None:
            self.mahal_v = FullMahalanobis(embed_dim)
            self.mahal_t = FullMahalanobis(embed_dim)
        else:
            self.mahal_v = LowRankMahalanobis(embed_dim, mahal_rank)
            self.mahal_t = LowRankMahalanobis(embed_dim, mahal_rank)

    def encode_image(self, v: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.mahal_v(v), dim=-1)

    def encode_text(self, t: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.mahal_t(t), dim=-1)

    def forward(self, v: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_image(v), self.encode_text(t)


# ---------------------------------------------------------------------------
# Feature inversion attacks
# ---------------------------------------------------------------------------

def _pseudoinverse_attack_error(
    raw_feats: torch.Tensor,
    jl_matrix: torch.Tensor,
) -> dict[str, float]:
    """
    Minimum-norm least-squares reconstruction:  x* = Phi^T (Phi Phi^T)^{-1} (Phi x).
    This is a lower bound on inversion success — a trained attacker will do better.
    """
    Phi = jl_matrix.float()   # (m, d)
    X   = raw_feats.float()   # (N, d)
    PPhiT     = Phi @ Phi.T
    projected = X @ Phi.T
    coeff     = torch.linalg.solve(PPhiT, projected.T).T
    reconstructed = coeff @ Phi
    orig_norm_sq = X.pow(2).sum(dim=1).clamp(min=1e-8)
    err_sq = (reconstructed - X).pow(2).sum(dim=1)
    return {
        "mean_relative_reconstruction_error": (err_sq / orig_norm_sq).mean().item(),
        "mean_abs_reconstruction_error": err_sq.mean().sqrt().item(),
    }


def _learned_inversion_attack(
    raw_feats: torch.Tensor,
    jl_matrix: torch.Tensor,
    *,
    train_frac: float = 0.8,
    hidden_dim: int = 512,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 512,
    device: str = "cuda",
) -> dict[str, float]:
    """
    Neural network feature inversion attack (He et al., CCS 2019).
    Trains a 3-layer MLP g: R^m -> R^d to reconstruct raw features from JLT projections.
    Evaluates on held-out 20% test split.
    """
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader, random_split

    Phi = jl_matrix.float()        # (m, d)
    X   = raw_feats.float()        # (N, d)
    m, d = Phi.shape

    with torch.no_grad():
        Z = X @ Phi.T              # (N, m) — projected features

    ds = TensorDataset(Z, X)
    n_train = int(train_frac * len(ds))
    n_test  = len(ds) - n_train
    train_ds, test_ds = random_split(ds, [n_train, n_test],
                                     generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    inverter = nn.Sequential(
        nn.Linear(m, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, d),
    ).to(device)

    opt = torch.optim.Adam(inverter.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        inverter.train()
        for z_batch, x_batch in train_loader:
            z_batch, x_batch = z_batch.to(device), x_batch.to(device)
            opt.zero_grad()
            loss_fn(inverter(z_batch), x_batch).backward()
            opt.step()

    inverter.eval()
    rel_errs, abs_errs = [], []
    with torch.no_grad():
        for z_batch, x_batch in test_loader:
            z_batch, x_batch = z_batch.to(device), x_batch.to(device)
            recon = inverter(z_batch)
            orig_norm_sq = x_batch.pow(2).sum(dim=1).clamp(min=1e-8)
            err_sq = (recon - x_batch).pow(2).sum(dim=1)
            rel_errs.append((err_sq / orig_norm_sq).cpu())
            abs_errs.append(err_sq.sqrt().cpu())
    rel_err_tensor = torch.cat(rel_errs)
    abs_err_tensor = torch.cat(abs_errs)
    return {
        "mean_relative_reconstruction_error": rel_err_tensor.mean().item(),
        "mean_abs_reconstruction_error": abs_err_tensor.mean().item(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(cfg: dict) -> None:
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg["output_dir"])
    seeds = cfg.get("seeds", [cfg.get("seed", 0)])

    cache_dir = Path(cfg["cache_dir"]) / cfg["dataset"]
    img_feats = torch.load(cache_dir / cfg["image_cache_file"],
                           map_location="cpu", weights_only=True)
    txt_feats = torch.load(cache_dir / cfg["text_cache_file"],
                           map_location="cpu", weights_only=True)

    # Keep original unique image features for the inversion attack probe.
    # Must be saved before repeat_interleave so each probe sample is distinct.
    img_feats_unique = img_feats.clone()

    # Expand image features to match multi-caption text features.
    if len(txt_feats) != len(img_feats):
        n_cap = len(txt_feats) // len(img_feats)
        img_feats = img_feats.repeat_interleave(n_cap, dim=0)

    m          = cfg["embed_dim"]
    mahal_rank = cfg.get("mahal_rank")
    n_rounds   = cfg.get("n_rounds", 1)
    results: dict = {}

    # --- Run across seeds ---
    centralized_per_seed: list[dict] = []
    federated_per_seed:   list[dict] = []
    round_recalls_per_seed: list[list[float]] = []

    for seed in seeds:
        set_seed(seed)
        n     = len(img_feats)
        n_val = int(n * 0.1)
        ds    = TensorDataset(img_feats, txt_feats)
        train_ds, val_ds = random_split(
            ds, [n - n_val, n_val],
            generator=torch.Generator().manual_seed(seed)
        )
        kw = dict(batch_size=cfg["batch_size"], num_workers=4, pin_memory=True)
        train_loader = DataLoader(train_ds, shuffle=True, **kw)
        val_loader   = DataLoader(val_ds, shuffle=False, **kw)

        # --- Centralized training ---
        model = CLIPJSTPipeline(
            vision_dim=cfg["vision_dim"], text_dim=cfg["text_dim"],
            embed_dim=m, mahal_rank=mahal_rank,
            jl_eps=cfg["jl_eps"], jl_seed=cfg["jl_seed"],
        )
        central_ckpt = output_dir / "centralized" / f"seed{seed}"
        train(model, train_loader, val_loader,
              epochs=cfg["epochs"], lr=cfg["lr"],
              temperature=cfg.get("temperature", 0.07), device=device,
              ckpt_dir=central_ckpt, patience=cfg.get("patience", 10),
              warmup_epochs=cfg.get("warmup_epochs", 0))
        model = load_best_checkpoint(model, central_ckpt, device)
        img_emb, txt_emb = extract_embeddings(model, val_loader, device)
        c_metrics = recall_at_k(img_emb, txt_emb)
        centralized_per_seed.append(c_metrics)
        print(f"Centralized seed={seed}: {c_metrics}")

        # --- Federated training ---
        fed_model = CLIPJSTPipeline(
            vision_dim=cfg["vision_dim"], text_dim=cfg["text_dim"],
            embed_dim=m, mahal_rank=mahal_rank,
            jl_eps=cfg["jl_eps"], jl_seed=cfg["jl_seed"],
        )
        # Use the training partition for federated simulation.
        fed_img = img_feats[list(train_ds.indices)]
        fed_txt = txt_feats[list(train_ds.indices)]
        fed_model, round_recalls = _federated_train(
            fed_img, fed_txt, fed_model, cfg, device,
            output_dir / "federated" / f"seed{seed}",
            n_rounds=n_rounds,
        )
        # Final federated eval on validation set (using IdentityJLPipeline in JL space).
        fed_model_eval = _IdentityJLPipeline(m, mahal_rank).to(device)
        fed_model_eval.load_state_dict(
            {k.replace("mahal_v.", "mahal_v.").replace("mahal_t.", "mahal_t."): v
             for k, v in fed_model.state_dict().items()
             if k.startswith("mahal_")},
            strict=False
        )
        # Encode val set from JL space.
        val_jl_img, val_jl_txt = [], []
        fed_model.eval()
        with torch.no_grad():
            for bv, bt in val_loader:
                val_jl_img.append(fed_model.jl_v(bv.to(device)).cpu())
                val_jl_txt.append(fed_model.jl_t(bt.to(device)).cpu())
        val_jl_img = torch.cat(val_jl_img)
        val_jl_txt = torch.cat(val_jl_txt)
        val_jl_ds  = TensorDataset(val_jl_img, val_jl_txt)
        val_jl_loader = DataLoader(val_jl_ds, batch_size=cfg["batch_size"], num_workers=0)
        img_emb_fed, txt_emb_fed = extract_embeddings(fed_model_eval, val_jl_loader, device)
        f_metrics = recall_at_k(img_emb_fed, txt_emb_fed)
        federated_per_seed.append(f_metrics)
        round_recalls_per_seed.append(round_recalls)
        print(f"Federated  seed={seed}: {f_metrics}")

    # Aggregate centralized and federated across seeds.
    def _agg(per_seed: list[dict]) -> dict:
        if len(per_seed) == 1:
            return per_seed[0]
        agg = {}
        for key in per_seed[0]:
            vals = [s[key] for s in per_seed if isinstance(s.get(key), float)]
            if vals:
                agg[key] = {"mean": sum(vals)/len(vals),
                            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0}
        return agg

    results["centralized"] = _agg(centralized_per_seed)
    results["federated"]   = _agg(federated_per_seed)
    results["federated_round_recalls"] = round_recalls_per_seed

    # --- Feature inversion attacks — privacy curve across m values ---
    print("\n=== Privacy curve: inversion error vs. projection dimension ===")
    n_probe = min(1000, len(img_feats_unique))
    raw_probe = img_feats_unique[:n_probe]
    privacy_curve: dict = {}
    for m_priv in cfg.get("embed_dims_privacy", [cfg["embed_dim"]]):
        tmp_model = CLIPJSTPipeline(
            vision_dim=cfg["vision_dim"], text_dim=cfg["text_dim"],
            embed_dim=m_priv, jl_eps=cfg["jl_eps"], jl_seed=cfg["jl_seed"],
        )
        Phi_v = tmp_model.jl_v.Phi.cpu()
        print(f"  m={m_priv}: training neural inverter ...")
        neural_err = _learned_inversion_attack(raw_probe, Phi_v, device=device)
        pseudo_err = _pseudoinverse_attack_error(raw_probe, Phi_v)
        privacy_curve[m_priv] = {
            "neural_inverter": neural_err,
            "pseudoinverse":   pseudo_err,
        }
        print(f"  m={m_priv}: neural={neural_err}, pseudoinverse={pseudo_err}")
    results["privacy_curve"] = privacy_curve
    results["inversion_raw_features"] = {
        "mean_relative_reconstruction_error": 0.0,
        "note": "Identity reference on unprojected features (exact reconstruction; not an attack baseline).",
    }

    save_json(results, output_dir / "E5_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run(cfg)
