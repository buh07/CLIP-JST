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
import json
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
        import math
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly

        clap_name = backbone_cfg.get("clap_model", "laion/clap-htsat-unfused")
        processor = ClapProcessor.from_pretrained(clap_name)
        model     = ClapModel.from_pretrained(clap_name).to(device).eval()

        def _read_audio_48k(path: str) -> np.ndarray:
            wav, sr = sf.read(path, always_2d=False)
            wav = np.asarray(wav, dtype=np.float32)
            if wav.ndim == 2:
                wav = wav.mean(axis=1)
            if sr != 48000:
                g = math.gcd(int(sr), 48000)
                wav = resample_poly(wav, 48000 // g, int(sr) // g).astype(np.float32, copy=False)
            return wav

        @torch.no_grad()
        def img_fn(audio_paths: list[str]) -> torch.Tensor:
            audios = [_read_audio_48k(p) for p in audio_paths]
            inp = processor(audio=audios, return_tensors="pt",
                            sampling_rate=48000, padding=True).to(device)
            out = model.get_audio_features(**inp)
            if hasattr(out, "audio_embeds"):
                out = out.audio_embeds
            return out.cpu()

        @torch.no_grad()
        def txt_fn(texts: list[str]) -> torch.Tensor:
            inp = processor(text=texts, return_tensors="pt",
                            padding=True, truncation=True).to(device)
            out = model.get_text_features(**inp)
            if hasattr(out, "text_embeds"):
                out = out.text_embeds
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

        manifest_path = Path(backbone_cfg.get("manifest_file", cfg["manifest_file"]))
        if not manifest_path.exists():
            msg = f"manifest missing: {manifest_path}"
            print(f"  Skipping {backbone_name}: {msg}")
            results[backbone_name] = {"status": "skipped", "reason": msg}
            continue

        with open(manifest_path) as f:
            manifest = json.load(f)

        input_key = backbone_cfg.get("input_key")
        if input_key is None:
            input_key = "audio_paths" if backbone_cfg["type"] == "clap" else "image_paths"
        text_key = backbone_cfg.get("text_key", "texts")
        if input_key not in manifest or text_key not in manifest:
            msg = f"manifest keys missing ({input_key}, {text_key}) in {manifest_path}"
            print(f"  Skipping {backbone_name}: {msg}")
            results[backbone_name] = {"status": "skipped", "reason": msg}
            continue

        input_paths = manifest[input_key]
        texts = manifest[text_key]
        if len(input_paths) != len(texts):
            raise ValueError(
                f"Manifest length mismatch for {backbone_name}: "
                f"{input_key}={len(input_paths)} vs {text_key}={len(texts)}"
            )

        cache_dir = Path(cfg["cache_dir"]) / cfg["dataset"] / backbone_name
        # CLIP backbones: extract_and_cache uses hf_name + "_raw" as the tag.
        # Non-CLIP (dinov2_bge, clap): extract_and_cache_generic uses backbone_name.
        if backbone_cfg["type"] == "clip":
            hf_name = backbone_cfg.get("hf_name", backbone_name)
            tag = hf_name.replace("/", "_") + "_raw"
        else:
            tag = backbone_name.replace("/", "_")
        img_cache = cache_dir / f"image_feats_{tag}.pt"
        txt_cache = cache_dir / f"text_feats_{tag}.pt"

        if not (img_cache.exists() and txt_cache.exists()):
            if backbone_cfg["type"] in ("dinov2_bge", "clap"):
                img_fn, txt_fn = _build_encoder_fns(backbone_cfg, device)
                from data.cache import extract_and_cache_generic
                extract_and_cache_generic(
                    input_paths, texts,
                    cache_dir, backbone_name, img_fn, txt_fn,
                )
            else:
                # Standard CLIP-style extraction.
                from data.cache import extract_and_cache
                extract_and_cache(
                    input_paths, texts,
                    cache_dir,
                    backbone_name=backbone_cfg.get("hf_name", backbone_name),
                    device=device,
                )

        ds = PairedFeatureDataset(img_cache, txt_cache)
        n_val = int(len(ds) * 0.1)
        gen = torch.Generator().manual_seed(cfg.get("seed", 0))
        train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val], generator=gen)
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
