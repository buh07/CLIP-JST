"""
D4. Backbone generalization.

Repeats the core E1 experiment with alternative backbone pairs:
  (a) ViT-L/14  (openai/clip-vit-large-patch14,  d_v=1024, d_t=768)
  (b) DINOv2 + BGE text encoder  (d_v=1024, d_t=1024)
  (c) CLAP audio-text encoder    (d_v=512,  d_t=512)

Hypothesis: JL+Mahalanobis advantage is backbone-agnostic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.cache import PairedFeatureDataset
from models.baselines import CLIPProjectionHead
from models.pipeline import CLIPJSTPipeline
from training.trainer import train, extract_embeddings
from eval.retrieval import recall_at_k
from utils.common import set_seed, save_json, load_best_checkpoint


def _build_encoder_fns(backbone_cfg: dict, device: str):
    """
    Returns (image_encoder_fn, text_encoder_fn) callables for non-CLIP backbones.
    Each fn accepts a list of paths / strings and returns a CPU tensor of features.
    """
    btype = backbone_cfg["type"]

    if btype == "dinov2_bge":
        from transformers import AutoModel, AutoProcessor, AutoTokenizer
        from PIL import Image

        dino_name = backbone_cfg.get("dino_model", "facebook/dinov2-large")
        bge_name  = backbone_cfg.get("bge_model",  "BAAI/bge-large-en-v1.5")

        dino_proc  = AutoProcessor.from_pretrained(dino_name)
        dino_model = AutoModel.from_pretrained(dino_name).to(device).eval()
        bge_tok    = AutoTokenizer.from_pretrained(bge_name)
        bge_model  = AutoModel.from_pretrained(bge_name).to(device).eval()

        @torch.no_grad()
        def img_fn(paths: list[str]) -> torch.Tensor:
            images = [Image.open(p).convert("RGB") for p in paths]
            inp = dino_proc(images=images, return_tensors="pt").to(device)
            out = dino_model(**inp).last_hidden_state[:, 0]  # CLS token
            return out.cpu()

        @torch.no_grad()
        def txt_fn(texts: list[str]) -> torch.Tensor:
            inp = bge_tok(texts, return_tensors="pt", padding=True,
                          truncation=True).to(device)
            out = bge_model(**inp).last_hidden_state[:, 0]
            return out.cpu()

        return img_fn, txt_fn

    elif btype == "clap":
        from transformers import ClapModel, ClapProcessor

        clap_name = backbone_cfg.get("clap_model", "laion/clap-htsat-unfused")
        processor = ClapProcessor.from_pretrained(clap_name)
        model     = ClapModel.from_pretrained(clap_name).to(device).eval()

        @torch.no_grad()
        def img_fn(audio_paths: list[str]) -> torch.Tensor:
            import soundfile as sf
            import numpy as np
            audios = [sf.read(p)[0] for p in audio_paths]
            inp = processor(audios=audios, return_tensors="pt",
                            sampling_rate=48000, padding=True).to(device)
            out = model.get_audio_features(**inp)
            return out.cpu()

        @torch.no_grad()
        def txt_fn(texts: list[str]) -> torch.Tensor:
            inp = processor(text=texts, return_tensors="pt",
                            padding=True, truncation=True).to(device)
            out = model.get_text_features(**inp)
            return out.cpu()

        return img_fn, txt_fn

    else:
        raise ValueError(f"Unknown backbone type: {btype}")


def run(cfg: dict) -> None:
    set_seed(cfg.get("seed", 0))
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    results: dict = {}

    for backbone_name, backbone_cfg in cfg["backbones"].items():
        print(f"\n=== Backbone: {backbone_name} ===")
        v_dim = backbone_cfg["vision_dim"]
        t_dim = backbone_cfg["text_dim"]

        cache_dir = Path(cfg["cache_dir"]) / cfg["dataset"] / backbone_name
        # Filenames must match the {tag} pattern written by extract_and_cache_generic.
        tag = backbone_name.replace("/", "_")
        img_cache = cache_dir / f"image_feats_{tag}.pt"
        txt_cache = cache_dir / f"text_feats_{tag}.pt"

        if not (img_cache.exists() and txt_cache.exists()):
            if backbone_cfg["type"] in ("dinov2_bge", "clap"):
                # Load paths from a manifest file.
                import json
                with open(cfg["manifest_file"]) as f:
                    manifest = json.load(f)
                img_fn, txt_fn = _build_encoder_fns(backbone_cfg, device)
                from data.cache import extract_and_cache_generic
                extract_and_cache_generic(
                    manifest["image_paths"], manifest["texts"],
                    cache_dir, backbone_name, img_fn, txt_fn,
                )
            else:
                # Standard CLIP-style extraction.
                import json
                with open(cfg["manifest_file"]) as f:
                    manifest = json.load(f)
                from data.cache import extract_and_cache
                extract_and_cache(
                    manifest["image_paths"], manifest["texts"],
                    cache_dir,
                    backbone_name=backbone_cfg.get("hf_name", backbone_name),
                    device=device,
                )

        ds = PairedFeatureDataset(img_cache, txt_cache)
        n_val = int(len(ds) * 0.1)
        train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val])
        kw = dict(batch_size=cfg["batch_size"], num_workers=4, pin_memory=True)
        train_loader = DataLoader(train_ds, shuffle=True, **kw)
        val_loader   = DataLoader(val_ds, shuffle=False, **kw)
        results[backbone_name] = {}

        for m in cfg["embed_dims"]:
            for model_type in ["clip_head", "jl_mahal"]:
                label = f"{model_type}_m{m}"
                if model_type == "clip_head":
                    model = CLIPProjectionHead(v_dim, t_dim, m)
                else:
                    model = CLIPJSTPipeline(
                        vision_dim=v_dim, text_dim=t_dim, embed_dim=m,
                        jl_eps=cfg["jl_eps"], jl_seed=cfg["jl_seed"],
                    )
                ckpt_dir = Path(cfg["output_dir"]) / backbone_name / label
                train(model, train_loader, val_loader,
                      epochs=cfg["epochs"], lr=cfg["lr"],
                      temperature=cfg["temperature"], device=device,
                      ckpt_dir=ckpt_dir, patience=cfg.get("patience", 5))
                model = load_best_checkpoint(model, ckpt_dir, device)
                img_emb, txt_emb = extract_embeddings(model, val_loader, device)
                metrics = recall_at_k(img_emb, txt_emb)
                metrics["n_params"] = model.n_trainable_params()
                results[backbone_name][label] = metrics
                print(f"  {backbone_name}/{label}: {metrics}")

    save_json(results, Path(cfg["output_dir"]) / "D4_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run(cfg)
