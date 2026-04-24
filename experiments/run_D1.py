"""
D1. Expansion/shrinkage decomposition (Gui–Chen–Liu, NeurIPS 2023).

Compares singular-value spectra of:
  - Trained CLIP projection head W_clip (m×d).
  - JL matrix Phi (m×d) — random, data-independent.
  - Composed pipeline L @ Phi (m×d or r×d) — captures full method.

All three matrices map from backbone space R^d, so their right singular vectors
live in the same space and are directly comparable.

Measures subspace alignment between the top-k right singular vectors of
W_clip vs Phi alone, and W_clip vs the composed L @ Phi.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.baselines import CLIPProjectionHead
from models.pipeline import CLIPJSTPipeline
from eval.diagnostics import singular_value_spectrum, expansion_shrinkage_alignment
from utils.common import set_seed, save_json, load_best_checkpoint


def _get_clip_weight(model: CLIPProjectionHead, modality: str) -> torch.Tensor:
    """Returns the (embed_dim, backbone_dim) weight of the CLIP projection head."""
    proj = model.proj_v if modality == "image" else model.proj_t
    return proj.weight.detach().cpu()


def _get_mahal_weight(model: CLIPJSTPipeline, modality: str) -> torch.Tensor:
    """Returns the Mahalanobis factor L as a 2D CPU tensor."""
    mahal = model.mahal_v if modality == "image" else model.mahal_t
    if hasattr(mahal, "L"):
        return mahal.L.detach().cpu()          # LowRankMahalanobis: (r, m)
    return mahal._L().detach().cpu()           # FullMahalanobis: (m, m)


def run(cfg: dict) -> None:
    set_seed(cfg.get("seed", 0))
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    m = cfg["embed_dim"]
    results: dict = {}

    clip_model = CLIPProjectionHead(cfg["vision_dim"], cfg["text_dim"], m)
    clip_ckpt = Path(cfg["clip_head_checkpoint"])
    if not (clip_ckpt / "best.pt").exists():
        raise FileNotFoundError(
            f"CLIP head checkpoint not found at {clip_ckpt}. Run E1 first."
        )
    clip_model = load_best_checkpoint(clip_model, clip_ckpt, device)

    jl_model = CLIPJSTPipeline(
        vision_dim=cfg["vision_dim"],
        text_dim=cfg["text_dim"],
        embed_dim=m,
        mahal_rank=cfg.get("mahal_rank"),
        jl_eps=cfg["jl_eps"],
        jl_seed=cfg["jl_seed"],
    )
    jl_ckpt = Path(cfg["jl_mahal_checkpoint"])
    if not (jl_ckpt / "best.pt").exists():
        raise FileNotFoundError(
            f"JL+Mahalanobis checkpoint not found at {jl_ckpt}. Run E1 first."
        )
    jl_model = load_best_checkpoint(jl_model, jl_ckpt, device)

    for modality in ["image", "text"]:
        W_clip = _get_clip_weight(clip_model, modality)          # (m, d)
        L      = _get_mahal_weight(jl_model, modality)           # (m, m) or (r, m)
        Phi    = (jl_model.jl_v.Phi if modality == "image"
                  else jl_model.jl_t.Phi).cpu()                  # (m, d)

        # Composed pipeline maps R^d → R^m, same as W_clip.
        W_composed = L @ Phi                                      # (m, d) or (r, d)

        sv_clip     = singular_value_spectrum(W_clip)
        sv_phi      = singular_value_spectrum(Phi)
        sv_composed = singular_value_spectrum(W_composed)

        # Subspace alignment — right singular vectors, all in R^d.
        align_jl       = expansion_shrinkage_alignment(W_clip, Phi,        k=32)
        align_composed = expansion_shrinkage_alignment(W_clip, W_composed, k=32)

        results[modality] = {
            "sv_clip":     sv_clip.tolist(),
            "sv_phi":      sv_phi.tolist(),
            "sv_composed": sv_composed.tolist(),
            "subspace_overlap_jl":       align_jl["subspace_overlap"].tolist(),
            "subspace_overlap_composed": align_composed["subspace_overlap"].tolist(),
            "mean_overlap_jl":           float(align_jl["subspace_overlap"].mean()),
            "mean_overlap_composed":     float(align_composed["subspace_overlap"].mean()),
        }
        print(
            f"{modality}: mean subspace overlap — "
            f"JL={results[modality]['mean_overlap_jl']:.3f}  "
            f"Composed={results[modality]['mean_overlap_composed']:.3f}"
        )

    save_json(results, Path(cfg["output_dir"]) / "D1_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run(cfg)
