"""
D2. Class-conditional JL distortion.

For each concept class in NUS-WIDE, measures the empirical JL distortion:
    ratio = ||Phi v - Phi t||^2 / ||v - t||^2
averaged over matched (image, text) pairs within that class.

Tests whether width-adaptive scaling holds per-class: classes with smaller
Gaussian width should require smaller m to achieve a given distortion ratio.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.jl import kane_nelson_jl
from eval.diagnostics import jl_distortion_per_class, gaussian_width_upper_bound
from theory.width_estimation import cross_modal_width_estimate, required_dim
from utils.common import set_seed, save_json


def run(cfg: dict) -> None:
    set_seed(cfg.get("seed", 0))
    results: dict = {}

    cache_dir = Path(cfg["cache_dir"]) / cfg["dataset"]
    img_feats   = torch.load(cache_dir / cfg["image_cache_file"],
                             map_location="cpu", weights_only=True)
    txt_feats   = torch.load(cache_dir / cfg["text_cache_file"],
                             map_location="cpu", weights_only=True)
    labels_file = cache_dir / cfg["labels_file"]
    labels = torch.load(labels_file, map_location="cpu", weights_only=True)  # (N, C) or (N,)

    # If multi-hot labels, convert to integer class labels via dominant class.
    if labels.dim() == 2:
        int_labels = labels.argmax(dim=1)
    else:
        int_labels = labels.long()

    for m in cfg["embed_dims"]:
        for eps in cfg["jl_eps_list"]:
            Phi_v = torch.tensor(
                kane_nelson_jl(img_feats.shape[1], m, eps=eps, seed=cfg["jl_seed"]).toarray(),
                dtype=torch.float32,
            )
            Phi_t = torch.tensor(
                kane_nelson_jl(txt_feats.shape[1], m, eps=eps, seed=cfg["jl_seed"] + 1).toarray(),
                dtype=torch.float32,
            )

            distortion = jl_distortion_per_class(
                img_feats, txt_feats, int_labels, Phi_v, Phi_t
            )

            # Per-class width estimates.
            per_class_width = {}
            for cls in int_labels.unique().tolist():
                mask = int_labels == cls
                if mask.sum() < 10:
                    continue
                w = cross_modal_width_estimate(img_feats[mask], txt_feats[mask],
                                               n_samples=200, subsample=500)
                per_class_width[int(cls)] = w

            label = f"m{m}_eps{eps}"
            results[label] = {
                "distortion_per_class": {str(k): v for k, v in distortion.items()},
                "width_per_class":      {str(k): v for k, v in per_class_width.items()},
                "mean_distortion": float(sum(distortion.values()) / max(1, len(distortion))),
                "embed_dim": m,
                "jl_eps": eps,
            }
            print(f"m={m}, eps={eps}: mean_distortion={results[label]['mean_distortion']:.4f}")

    save_json(results, Path(cfg["output_dir"]) / "D2_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run(cfg)
