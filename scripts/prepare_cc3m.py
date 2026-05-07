#!/usr/bin/env python3
"""
Download Conceptual Captions 3M (CC3M) images and extract CLIP ViT-B/32 features.

CC3M images are fetched by URL from the HuggingFace dataset. Many URLs are dead;
only successfully downloaded images are included. Expect ~1–2M valid images.

Outputs:
  data/cache/cc3m/image_feats_openai_clip-vit-base-patch32_raw.pt  (N, 768)
  data/cache/cc3m/text_feats_openai_clip-vit-base-patch32_raw.pt   (N, 512)
  data/cache/cc3m/captions.json  — list of N captions (aligned with features)

Usage:
  cd "/jumbo/lisp/f004ndc/CLIP JST"
  CUDA_VISIBLE_DEVICES=7 python scripts/prepare_cc3m.py [--max-images N] [--workers W]
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR    = PROJECT_ROOT / "data" / "cache" / "cc3m"
BACKBONE     = "openai/clip-vit-base-patch32"
BATCH_SIZE   = 256
HF_HOME      = os.environ.get("HF_HOME", "/scratch/f004ndc/hf_cache")
TIMEOUT_S    = 12  # overridden by --timeout arg
MAX_RETRIES  = 1


def _fetch_image(url: str, timeout: int = TIMEOUT_S) -> Image.Image | None:
    for _ in range(MAX_RETRIES + 1):
        try:
            r = requests.get(url, timeout=timeout, stream=True)
            if r.status_code != 200:
                return None
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception:
            pass
    return None


def download_cc3m(max_images: int | None, workers: int, timeout: int = 12) -> tuple[list[Image.Image], list[str]]:
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Install 'datasets': pip install datasets")

    print("Loading CC3M metadata from HuggingFace (this may take a minute)...")
    ds = load_dataset("conceptual_captions", split="train", trust_remote_code=True)
    print(f"  Total entries: {len(ds):,}")

    if max_images is not None:
        ds = ds.select(range(min(max_images, len(ds))))
        print(f"  Capped at {len(ds):,} entries")

    urls      = ds["image_url"]
    captions  = ds["caption"]

    images_ok: list[Image.Image] = []
    caps_ok:   list[str]         = []
    n_fail     = 0

    print(f"Fetching images with {workers} workers (timeout={timeout}s) ...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_image, url, timeout): (url, cap)
                   for url, cap in zip(urls, captions)}
        pbar = tqdm(total=len(futures), unit="img")
        for fut in as_completed(futures):
            _, cap = futures[fut]
            img = fut.result()
            if img is not None:
                images_ok.append(img)
                caps_ok.append(cap)
            else:
                n_fail += 1
            pbar.set_postfix(ok=len(images_ok), fail=n_fail)
            pbar.update(1)
        pbar.close()

    print(f"  Downloaded {len(images_ok):,} / {len(futures):,} images "
          f"({n_fail:,} failed/dead URLs)")
    return images_ok, caps_ok


@torch.no_grad()
def encode_images(images: list[Image.Image], model, processor, device) -> torch.Tensor:
    feats = []
    for i in tqdm(range(0, len(images), BATCH_SIZE), desc="  img encode"):
        batch = images[i : i + BATCH_SIZE]
        inp = processor(images=batch, return_tensors="pt").to(device)
        out = model.vision_model(pixel_values=inp["pixel_values"])
        feats.append(out.pooler_output.cpu())
    return torch.cat(feats)


@torch.no_grad()
def encode_texts(captions: list[str], model, processor, device) -> torch.Tensor:
    feats = []
    for i in tqdm(range(0, len(captions), BATCH_SIZE), desc="  txt encode"):
        batch = captions[i : i + BATCH_SIZE]
        inp = processor(text=batch, return_tensors="pt",
                        padding=True, truncation=True).to(device)
        out = model.text_model(input_ids=inp["input_ids"],
                               attention_mask=inp.get("attention_mask"))
        feats.append(out.pooler_output.cpu())
    return torch.cat(feats)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images", type=int, default=None,
                        help="Cap the number of CC3M entries to attempt (default: all ~3.3M)")
    parser.add_argument("--workers", type=int, default=64,
                        help="Parallel HTTP workers for image download (default: 64)")
    parser.add_argument("--timeout", type=int, default=12,
                        help="Per-image HTTP timeout in seconds (default: 12)")
    args = parser.parse_args()

    tag      = BACKBONE.replace("/", "_") + "_raw"
    img_out  = CACHE_DIR / f"image_feats_{tag}.pt"
    txt_out  = CACHE_DIR / f"text_feats_{tag}.pt"
    cap_out  = CACHE_DIR / "captions.json"

    if img_out.exists() and txt_out.exists():
        stored = torch.load(img_out, map_location="cpu", weights_only=True)
        print(f"Cache already exists: {stored.shape[0]:,} images. Delete to regenerate.")
        return

    images, captions = download_cc3m(args.max_images, args.workers, args.timeout)
    if not images:
        sys.exit("No images downloaded — check connectivity or CC3M availability.")

    os.environ["HF_HOME"] = HF_HOME
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading CLIP backbone ({BACKBONE}) on {device}...")
    model     = CLIPModel.from_pretrained(BACKBONE).to(device).eval()
    processor = CLIPProcessor.from_pretrained(BACKBONE)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nEncoding {len(images):,} images...")
    img_feats = encode_images(images, model, processor, device)

    print(f"Encoding {len(captions):,} captions...")
    txt_feats = encode_texts(captions, model, processor, device)

    torch.save(img_feats, img_out)
    torch.save(txt_feats, txt_out)
    with open(cap_out, "w") as f:
        json.dump(captions, f)

    print(f"\nSaved:")
    print(f"  {img_out}  {img_feats.shape}")
    print(f"  {txt_out}  {txt_feats.shape}")
    print(f"  {cap_out}  ({len(captions):,} captions)")


if __name__ == "__main__":
    main()
