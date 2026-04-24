"""
E5. Federated learning / privacy stress test.

Simulates federated training where clients share only JL-projected features
rather than raw CLIP features.  Evaluates:

  1. Federated vs. centralized retrieval performance gap (FedAvg).
  2. Feature-inversion attack success rate: given JL-projected features,
     how accurately can raw features be reconstructed?
     Attack: x* = Phi^+ (Phi x) = minimum-norm least-squares solution.
     Success metric: normalized reconstruction error ||x* - x|| / ||x||.

Hypothesis: JL-projected features resist inversion while preserving retrieval.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.cache import PairedFeatureDataset
from models.pipeline import CLIPJSTPipeline
from models.jl import kane_nelson_jl
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
    """Averages trainable parameters across clients (FedAvg)."""
    avg_state = {k: v.clone() for k, v in global_state.items()}
    for key in trainable_keys:
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
) -> CLIPJSTPipeline:
    """
    Simulates one round of FedAvg.

    Each client receives the JL-projected features (not raw features) and
    trains its local Mahalanobis head.  Mahalanobis parameters are then
    averaged globally.
    """
    n_clients = cfg["n_clients"]
    n = len(img_feats)
    indices = torch.randperm(n)
    chunks = torch.chunk(indices, n_clients)

    # Pre-apply JL so clients never see raw features.
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        jl_img = model.jl_v(img_feats.to(device)).cpu()
        jl_txt = model.jl_t(txt_feats.to(device)).cpu()

    trainable_keys = [k for k, v in model.named_parameters() if v.requires_grad]
    global_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    client_states = []

    for cid, chunk in enumerate(chunks):
        if len(chunk) < 2:
            continue
        client_img = jl_img[chunk]
        client_txt = jl_txt[chunk]

        # Client model maps from JL space (embed_dim) to embed_dim with Mahalanobis only.
        # We build a thin wrapper that skips JL (features are already projected).
        client_model = _IdentityJLPipeline(model.embed_dim, cfg.get("mahal_rank"))
        client_model.load_state_dict(global_state, strict=False)

        n_val = max(1, int(len(chunk) * 0.1))
        ds = torch.utils.data.TensorDataset(client_img, client_txt)
        train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val])
        kw = dict(batch_size=min(cfg["batch_size"], len(train_ds)), num_workers=0)
        train_loader = DataLoader(train_ds, shuffle=True, **kw)
        val_loader   = DataLoader(val_ds, shuffle=False, **kw)

        ckpt = output_dir / f"client_{cid}"
        train(client_model, train_loader, val_loader,
              epochs=cfg.get("local_epochs", 5), lr=cfg["lr"],
              temperature=cfg["temperature"], device=device,
              ckpt_dir=ckpt, patience=3)
        client_model.load_state_dict(
            torch.load(ckpt / "best.pt", map_location="cpu", weights_only=True)
        )
        client_states.append({k: v.detach() for k, v in client_model.state_dict().items()})

    # Average trainable parameters.
    if client_states:
        new_state = _fedavg(global_state, client_states, trainable_keys)
        model.load_state_dict(new_state, strict=False)

    return model


class _IdentityJLPipeline(torch.nn.Module):
    """
    Thin wrapper used for client-side training: input is already JL-projected,
    so this just applies the Mahalanobis heads.
    """

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
# Feature inversion attack
# ---------------------------------------------------------------------------

def _inversion_attack_error(
    raw_feats: torch.Tensor,
    jl_matrix: torch.Tensor,
) -> dict[str, float]:
    """
    Minimum-norm least-squares reconstruction:  x* = Phi^T (Phi Phi^T)^{-1} (Phi x).

    Phi in R^{m×d}, m < d.  The pseudoinverse Phi^+ = Phi^T (Phi Phi^T)^{-1}.
    Reconstruction error:  ||x* - x||^2 / ||x||^2 averaged over samples.
    """
    Phi = jl_matrix.float()      # (m, d)
    X   = raw_feats.float()      # (N, d)

    # Phi Phi^T  (m×m, small)
    PPhiT = Phi @ Phi.T
    projected = X @ Phi.T        # (N, m)
    # Solve (Phi Phi^T) coeff = projected^T for each sample.
    coeff = torch.linalg.solve(PPhiT, projected.T).T   # (N, m)
    reconstructed = coeff @ Phi  # (N, d)

    orig_norm_sq = X.pow(2).sum(dim=1).clamp(min=1e-8)
    err_sq = (reconstructed - X).pow(2).sum(dim=1)
    rel_err = (err_sq / orig_norm_sq).mean().item()

    return {
        "mean_relative_reconstruction_error": rel_err,
        "mean_abs_reconstruction_error": err_sq.mean().sqrt().item(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(cfg: dict) -> None:
    set_seed(cfg.get("seed", 0))
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg["output_dir"])
    results: dict = {}

    cache_dir = Path(cfg["cache_dir"]) / cfg["dataset"]
    img_feats = torch.load(cache_dir / cfg["image_cache_file"],
                           map_location="cpu", weights_only=True)
    txt_feats = torch.load(cache_dir / cfg["text_cache_file"],
                           map_location="cpu", weights_only=True)
    # Multi-caption cache: collapse to one caption per image (first caption).
    if len(txt_feats) != len(img_feats):
        n_cap = len(txt_feats) // len(img_feats)
        txt_feats = txt_feats[::n_cap]

    m = cfg["embed_dim"]
    mahal_rank = cfg.get("mahal_rank")
    model = CLIPJSTPipeline(
        vision_dim=cfg["vision_dim"],
        text_dim=cfg["text_dim"],
        embed_dim=m,
        mahal_rank=mahal_rank,
        jl_eps=cfg["jl_eps"],
        jl_seed=cfg["jl_seed"],
    )

    # --- Centralized training ---
    n = len(img_feats)
    n_val = int(n * 0.1)
    ds = torch.utils.data.TensorDataset(img_feats, txt_feats)
    train_ds, val_ds = random_split(ds, [n - n_val, n_val])
    kw = dict(batch_size=cfg["batch_size"], num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader   = DataLoader(val_ds, shuffle=False, **kw)

    central_ckpt = output_dir / "centralized"
    history = train(model, train_loader, val_loader,
                    epochs=cfg["epochs"], lr=cfg["lr"],
                    temperature=cfg["temperature"], device=device,
                    ckpt_dir=central_ckpt, patience=cfg.get("patience", 5))
    model = load_best_checkpoint(model, central_ckpt, device)
    img_emb, txt_emb = extract_embeddings(model, val_loader, device)
    results["centralized"] = recall_at_k(img_emb, txt_emb)
    print(f"Centralized: {results['centralized']}")

    # --- Federated training ---
    fed_model = CLIPJSTPipeline(
        vision_dim=cfg["vision_dim"],
        text_dim=cfg["text_dim"],
        embed_dim=m,
        mahal_rank=mahal_rank,
        jl_eps=cfg["jl_eps"],
        jl_seed=cfg["jl_seed"],
    )
    fed_model = _federated_train(
        img_feats, txt_feats, fed_model, cfg, device, output_dir / "federated"
    )
    fed_model.to(device).eval()
    img_emb_fed, txt_emb_fed = extract_embeddings(fed_model, val_loader, device)
    results["federated"] = recall_at_k(img_emb_fed, txt_emb_fed)
    print(f"Federated: {results['federated']}")

    # --- Feature inversion attacks ---
    # On raw CLIP features.
    n_probe = min(1000, len(img_feats))
    raw_probe = img_feats[:n_probe]
    results["inversion_raw_features"] = {"mean_relative_reconstruction_error": 0.0,
                                          "note": "raw features cannot be inverted (identity)"}

    # On JL-projected features (our method).
    Phi_v = model.jl_v.Phi.cpu()   # (m, d_v)
    results["inversion_jl_projected"] = _inversion_attack_error(raw_probe, Phi_v)
    print(f"Inversion (JL): {results['inversion_jl_projected']}")

    save_json(results, output_dir / "E5_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run(cfg)
