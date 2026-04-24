"""
E4. Out-of-distribution (OOD) retrieval.

Loads models trained on MS-COCO (from E1 checkpoints) and evaluates on
Flickr30K and optionally CC3M without any retraining.

Hypothesis: JL's obliviousness provides better OOD generalization than a
learned CLIP projection head that may overfit to COCO's caption style.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.cache import PairedFeatureDataset, MultiCaptionDataset
from models.baselines import CLIPProjectionHead
from models.pipeline import CLIPJSTPipeline
from utils.common import set_seed, save_json, load_best_checkpoint, eval_dataset


def _load_eval_dataset(cfg: dict, dataset_name: str):
    cache_dir = Path(cfg["cache_dir"]) / dataset_name
    n_cap = cfg.get("n_captions", {}).get(dataset_name, 1)
    if n_cap > 1:
        return MultiCaptionDataset(
            cache_dir / cfg["image_cache_file"],
            cache_dir / cfg["text_cache_file"],
            n_captions=n_cap,
            training=False,
        )
    return PairedFeatureDataset(
        cache_dir / cfg["image_cache_file"],
        cache_dir / cfg["text_cache_file"],
    )


def run(cfg: dict) -> None:
    set_seed(cfg.get("seed", 0))
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    results: dict = {}
    m = cfg["embed_dim"]

    for model_label, model_cfg in cfg["models"].items():
        print(f"\n=== Loading model: {model_label} ===")
        if model_cfg["type"] == "jl_mahal":
            model = CLIPJSTPipeline(
                vision_dim=cfg["vision_dim"],
                text_dim=cfg["text_dim"],
                embed_dim=m,
                mahal_rank=model_cfg.get("mahal_rank"),
                jl_eps=cfg["jl_eps"],
                jl_seed=cfg["jl_seed"],
            )
        elif model_cfg["type"] == "clip_head":
            model = CLIPProjectionHead(cfg["vision_dim"], cfg["text_dim"], m)
        else:
            raise ValueError(f"Unknown model type: {model_cfg['type']}")

        ckpt_path = Path(model_cfg["checkpoint_dir"])
        if not (ckpt_path / "best.pt").exists():
            print(f"  Checkpoint not found at {ckpt_path}, skipping.")
            continue
        model = load_best_checkpoint(model, ckpt_path, device)
        model.eval()
        results[model_label] = {}

        for dataset_name in cfg["eval_datasets"]:
            print(f"  Evaluating on {dataset_name}...")
            ds = _load_eval_dataset(cfg, dataset_name)
            # eval_dataset() always uses the full dataset — correct GT for OOD eval.
            metrics = eval_dataset(model, ds, device, cfg["batch_size"])
            results[model_label][dataset_name] = metrics
            print(f"    {metrics}")

    save_json(results, Path(cfg["output_dir"]) / "E4_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run(cfg)
