"""
E3. Width-complexity scaling.

Constructs MS-COCO subsets with varying Gaussian width by:
  (a) Filtering to concept subsets (using COCO supercategory labels).
  (b) Varying caption-set diversity (1, 2, or 5 captions per image).

For each subset, estimates cross-modal Gaussian width and records the
minimum embedding dimension m needed to achieve a target Recall@10.

Hypothesis: required m scales linearly with estimated width^2, as predicted
by the Bourgain–Dirksen–Nelson width-adaptive JL bound (Claim 1).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.cache import PairedFeatureDataset
from models.pipeline import CLIPJSTPipeline
from training.trainer import train, extract_embeddings
from eval.retrieval import recall_at_k
from theory.width_estimation import cross_modal_width_estimate, required_dim
from utils.common import set_seed, save_json


def _load_coco_supercategory_index(annotations_file: Path) -> dict[str, list[int]]:
    """
    Returns {supercategory: [pair_indices_in_cache]} for concept-subset filtering.
    Requires COCO instances annotations to map image_id → supercategory.
    """
    with open(annotations_file) as f:
        data = json.load(f)

    cat_map = {c["id"]: c["supercategory"] for c in data["categories"]}
    img2super: dict[int, set[str]] = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        img2super.setdefault(img_id, set()).add(cat_map[ann["category_id"]])

    return img2super


def _diversity_subset(
    img_feats: torch.Tensor,
    txt_feats: torch.Tensor,
    n_captions_per_image: int,
    n_cap_total: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Thins the caption set to n_captions_per_image captions per image.
    Assumes txt_feats are in image-major order (5 captions per image).
    """
    n_images = len(img_feats) // n_cap_total
    keep_img, keep_txt = [], []
    for i in range(n_images):
        for c in range(n_captions_per_image):
            keep_img.append(img_feats[i])
            keep_txt.append(txt_feats[i * n_cap_total + c])
    return torch.stack(keep_img), torch.stack(keep_txt)


def _train_and_measure(
    img_feats: torch.Tensor,
    txt_feats: torch.Tensor,
    embed_dim: int,
    cfg: dict,
    device: str,
    label: str,
) -> dict:
    """Train JL+Mahalanobis on given feature subset and return metrics."""
    from torch.utils.data import TensorDataset

    ds = torch.utils.data.TensorDataset(img_feats, txt_feats)
    n_val = max(1, int(len(ds) * 0.1))
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val])
    kw = dict(batch_size=min(cfg["batch_size"], len(train_ds)), num_workers=0)
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader   = DataLoader(val_ds, shuffle=False, **kw)

    model = CLIPJSTPipeline(
        vision_dim=img_feats.shape[1],
        text_dim=txt_feats.shape[1],
        embed_dim=embed_dim,
        jl_eps=cfg["jl_eps"],
        jl_seed=cfg["jl_seed"],
    )
    ckpt_dir = Path(cfg["output_dir"]) / label
    train(model, train_loader, val_loader,
          epochs=cfg["epochs"], lr=cfg["lr"],
          temperature=cfg["temperature"], device=device,
          ckpt_dir=ckpt_dir, patience=cfg.get("patience", 5))
    model.load_state_dict(
        torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=True)
    )
    img_emb, txt_emb = extract_embeddings(model.to(device), val_loader, device)
    metrics = recall_at_k(img_emb, txt_emb)
    metrics["embed_dim"] = embed_dim
    return metrics


def run(cfg: dict) -> None:
    set_seed(cfg.get("seed", 0))
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    results: dict = {}

    cache_dir = Path(cfg["cache_dir"]) / "coco"
    img_feats = torch.load(cache_dir / cfg["image_cache_file"],
                           map_location="cpu", weights_only=True)
    txt_feats = torch.load(cache_dir / cfg["text_cache_file"],
                           map_location="cpu", weights_only=True)

    # ---- (b) Caption-diversity subsets ----
    for n_cap in cfg.get("n_captions_list", [1, 2, 5]):
        if n_cap > cfg.get("n_cap_total", 5):
            continue
        sub_img, sub_txt = _diversity_subset(
            img_feats, txt_feats, n_cap, cfg.get("n_cap_total", 5)
        )
        width = cross_modal_width_estimate(sub_img, sub_txt)
        theory_m = required_dim(width, eps=cfg["jl_eps"])
        print(f"n_cap={n_cap}: width={width:.3f}, theory_m={theory_m}")

        dim_results = []
        for m in cfg["embed_dims"]:
            label = f"ncap{n_cap}_m{m}"
            metrics = _train_and_measure(sub_img, sub_txt, m, cfg, device, label)
            metrics["width"] = width
            metrics["theory_m"] = theory_m
            dim_results.append(metrics)
            print(f"  m={m}: {metrics}")
        results[f"ncap_{n_cap}"] = dim_results

    # ---- (a) Concept subsets (requires COCO instances annotations) ----
    instances_file = Path(cfg.get("instances_annotations", ""))
    if instances_file.exists():
        img2super = _load_coco_supercategory_index(instances_file)
        for supcat in cfg.get("supercategories", ["animal", "vehicle", "person"]):
            # Build index list: image indices in cache that belong to supcat.
            # This requires a separate img_ids_file that lists image_ids in order.
            img_ids_file = cache_dir / "image_ids.json"
            if not img_ids_file.exists():
                print(f"Skipping supcat '{supcat}': no image_ids.json found.")
                continue
            with open(img_ids_file) as f:
                img_ids = json.load(f)  # list of image_ids matching cache rows

            # For multi-caption caches (5 per image), idx in cache = img_idx * 5 + c
            n_cap_total = cfg.get("n_cap_total", 5)
            sub_indices = [
                i for i, img_id in enumerate(img_ids)
                if supcat in img2super.get(img_id, set())
            ]
            if not sub_indices:
                continue

            sub_img = img_feats[[i * n_cap_total for i in sub_indices]]
            sub_txt_rows = []
            for i in sub_indices:
                for c in range(n_cap_total):
                    sub_txt_rows.append(txt_feats[i * n_cap_total + c])
            sub_txt = torch.stack(sub_txt_rows)

            width = cross_modal_width_estimate(sub_img, sub_txt[:len(sub_img)])
            theory_m = required_dim(width, eps=cfg["jl_eps"])
            print(f"supcat={supcat}: N={len(sub_img)}, width={width:.3f}, theory_m={theory_m}")

            dim_results = []
            for m in cfg["embed_dims"]:
                label = f"supcat_{supcat}_m{m}"
                metrics = _train_and_measure(
                    sub_img,
                    sub_txt[:len(sub_img)],
                    m, cfg, device, label,
                )
                metrics["width"] = width
                metrics["theory_m"] = theory_m
                dim_results.append(metrics)
            results[f"supcat_{supcat}"] = dim_results

    save_json(results, Path(cfg["output_dir"]) / "E3_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run(cfg)
