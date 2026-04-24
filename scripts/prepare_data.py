#!/usr/bin/env python3
"""
Download COCO train2017 and Flickr30K, then extract CLIP ViT-B/32 features.

Downloads:
  COCO    : images + captions from cocodataset.org  (~19 GB images, ~240 MB annotations)
  Flickr30K: images from nlphuji/flickr30k on HuggingFace + Karpathy splits JSON

Outputs (MultiCaptionDataset-compatible):
  data/cache/coco/image_feats_openai_clip-vit-base-patch32.pt  (N_imgs, 768)
  data/cache/coco/text_feats_openai_clip-vit-base-patch32.pt   (N_imgs*5, 512)
  data/cache/flickr30k/image_feats_openai_clip-vit-base-patch32.pt
  data/cache/flickr30k/text_feats_openai_clip-vit-base-patch32.pt

Usage:
  cd "/jumbo/lisp/f004ndc/CLIP JST"
  CUDA_VISIBLE_DEVICES=6 python scripts/prepare_data.py
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import requests
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT    = PROJECT_ROOT / "data"
CACHE_ROOT   = PROJECT_ROOT / "data" / "cache"
BACKBONE     = "openai/clip-vit-base-patch32"
BATCH_SIZE   = 256
N_CAPTIONS   = 5
HF_HOME      = os.environ.get("HF_HOME", "/scratch/f004ndc/hf_cache")

# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download(url: str, dest: Path, desc: str = "") -> None:
    """Stream-download url to dest, with progress bar. Skips if dest exists."""
    if dest.exists():
        print(f"  Already downloaded: {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {desc or dest.name} ...")
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            bar.update(len(chunk))


def _extract_zip(zip_path: Path, out_dir: Path) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    print(f"  Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)


# ---------------------------------------------------------------------------
# COCO download + parse
# ---------------------------------------------------------------------------

COCO_TRAIN_IMGS_URL = "http://images.cocodataset.org/zips/train2017.zip"
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def prepare_coco() -> tuple[list[str], list[str]]:
    """
    Downloads COCO train2017 images and annotations if not present.
    Returns (image_paths, captions) compatible with extract_and_cache_multi_caption().
    """
    coco_dir = DATA_ROOT / "coco"
    img_dir  = coco_dir / "images" / "train2017"
    ann_file = coco_dir / "annotations" / "captions_train2017.json"

    if not ann_file.exists():
        ann_zip = coco_dir / "annotations_trainval2017.zip"
        _download(COCO_ANNOTATIONS_URL, ann_zip, "COCO annotations")
        _extract_zip(ann_zip, coco_dir)

    if not img_dir.exists() or not any(img_dir.iterdir()):
        img_zip = coco_dir / "train2017.zip"
        _download(COCO_TRAIN_IMGS_URL, img_zip, "COCO train2017 images (~18 GB)")
        _extract_zip(img_zip, coco_dir / "images")

    print("Parsing COCO annotations...")
    with open(ann_file) as f:
        data = json.load(f)

    id2file = {img["id"]: img["file_name"] for img in data["images"]}
    id2caps: dict[int, list[str]] = defaultdict(list)
    for ann in data["annotations"]:
        id2caps[ann["image_id"]].append(ann["caption"])

    image_paths: list[str] = []
    captions: list[str] = []
    for img_id in sorted(id2caps):
        img_path = str(img_dir / id2file[img_id])
        if not Path(img_path).exists():
            continue
        image_paths.append(img_path)
        caps = id2caps[img_id][:N_CAPTIONS]
        while len(caps) < N_CAPTIONS:
            caps.append(caps[0])
        captions.extend(caps)

    print(f"COCO: {len(image_paths)} images, {len(captions)} captions")
    assert len(captions) == len(image_paths) * N_CAPTIONS
    return image_paths, captions


# ---------------------------------------------------------------------------
# Flickr30K — reads from git-cloned nlphuji/flickr30k repo
# ---------------------------------------------------------------------------

def prepare_flickr30k() -> tuple[list[str], list[str]] | None:
    """
    Reads Flickr30K from the git-cloned nlphuji/flickr30k repo layout:
      data/flickr30k/flickr_annotations_30k.csv  — columns: raw, split, filename
      data/flickr30k/flickr30k_images/*.jpg       — extracted from flickr30k-images.zip

    Returns (image_paths, captions) for the train split, or None if images
    are not yet extracted.
    """
    import ast
    import csv

    f30k_dir = DATA_ROOT / "flickr30k"
    img_dir  = f30k_dir / "flickr30k_images"
    csv_file = f30k_dir / "flickr_annotations_30k.csv"

    if not csv_file.exists():
        print(f"  Missing {csv_file}. Clone nlphuji/flickr30k into data/flickr30k first.")
        return None

    existing = list(img_dir.glob("*.jpg"))
    if len(existing) < 1000:
        zip_file = f30k_dir / "flickr30k-images.zip"
        if zip_file.exists():
            print(f"  Extracting {zip_file.name} ({zip_file.stat().st_size // 1_000_000} MB) ...")
            img_dir.mkdir(exist_ok=True)
            with zipfile.ZipFile(zip_file) as zf:
                members = [m for m in zf.namelist() if m.endswith(".jpg")]
                for m in tqdm(members, desc="  extracting"):
                    fname = Path(m).name
                    (img_dir / fname).write_bytes(zf.read(m))
        else:
            print(f"  No images in {img_dir} and no zip file found.")
            print(f"  Run: cd data/flickr30k && unzip flickr30k-images.zip -d flickr30k_images/")
            return None

    image_paths: list[str] = []
    captions: list[str] = []
    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] != "train":
                continue
            img_path = str(img_dir / row["filename"])
            if not Path(img_path).exists():
                continue
            caps = ast.literal_eval(row["raw"])[:N_CAPTIONS]
            while len(caps) < N_CAPTIONS:
                caps.append(caps[0])
            image_paths.append(img_path)
            captions.extend(caps)

    print(f"Flickr30K train: {len(image_paths)} images, {len(captions)} captions")
    assert len(captions) == len(image_paths) * N_CAPTIONS
    return image_paths, captions


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def _encode_images(paths: list[str], model, processor, device, batch_size: int) -> torch.Tensor:
    feats = []
    for i in tqdm(range(0, len(paths), batch_size), desc="  img"):
        batch_paths = paths[i : i + batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        inp = processor(images=imgs, return_tensors="pt").to(device)
        vision_out = model.vision_model(pixel_values=inp["pixel_values"])
        f = model.visual_projection(vision_out.pooler_output).cpu()
        feats.append(f)
    return torch.cat(feats)


@torch.no_grad()
def _encode_texts(texts: list[str], model, processor, device, batch_size: int) -> torch.Tensor:
    feats = []
    for i in tqdm(range(0, len(texts), batch_size), desc="  txt"):
        batch = texts[i : i + batch_size]
        inp = processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(device)
        text_out = model.text_model(
            input_ids=inp["input_ids"],
            attention_mask=inp.get("attention_mask"),
        )
        f = model.text_projection(text_out.pooler_output).cpu()
        feats.append(f)
    return torch.cat(feats)


def extract_and_save(
    image_paths: list[str],
    captions: list[str],
    out_dir: Path,
    model,
    processor,
    device: str,
) -> None:
    tag = BACKBONE.replace("/", "_")
    img_out = out_dir / f"image_feats_{tag}.pt"
    txt_out = out_dir / f"text_feats_{tag}.pt"

    if img_out.exists() and txt_out.exists():
        stored_img = torch.load(img_out, map_location="cpu", weights_only=True)
        stored_txt = torch.load(txt_out, map_location="cpu", weights_only=True)
        N = len(image_paths)
        if stored_img.shape[0] == N and stored_txt.shape[0] == N * N_CAPTIONS:
            print(f"  Cache hit: {img_out.name} and {txt_out.name}, skipping.")
            return

    out_dir.mkdir(parents=True, exist_ok=True)
    N = len(image_paths)
    print(f"  Encoding {N} images ...")
    img_feats = _encode_images(image_paths, model, processor, device, BATCH_SIZE)

    print(f"  Encoding {len(captions)} captions ...")
    txt_feats = _encode_texts(captions, model, processor, device, BATCH_SIZE)

    torch.save(img_feats, img_out)
    torch.save(txt_feats, txt_out)
    print(f"  Saved {img_out} ({img_feats.shape}) and {txt_out} ({txt_feats.shape})")


# ---------------------------------------------------------------------------
# Manifest for D4
# ---------------------------------------------------------------------------

def write_coco_manifest(image_paths: list[str], captions: list[str]) -> None:
    manifest_path = DATA_ROOT / "coco" / "manifest.json"
    if manifest_path.exists():
        return
    # D4 expects {"image_paths": [...], "texts": [...]} with 1:1 pairing.
    # Use first caption per image.
    texts = captions[::N_CAPTIONS]
    assert len(texts) == len(image_paths)
    with open(manifest_path, "w") as f:
        json.dump({"image_paths": image_paths, "texts": texts}, f)
    print(f"  Wrote D4 manifest: {manifest_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    print(f"\nLoading {BACKBONE} ...")
    model     = CLIPModel.from_pretrained(BACKBONE).to(device).eval()
    processor = CLIPProcessor.from_pretrained(BACKBONE)
    print("Model loaded.\n")

    # COCO
    print("=== COCO ===")
    coco_img, coco_cap = prepare_coco()
    extract_and_save(coco_img, coco_cap, CACHE_ROOT / "coco", model, processor, device)
    write_coco_manifest(coco_img, coco_cap)

    # Flickr30K
    print("\n=== Flickr30K ===")
    f30k_result = prepare_flickr30k()
    if f30k_result is not None:
        f30k_img, f30k_cap = f30k_result
        extract_and_save(f30k_img, f30k_cap, CACHE_ROOT / "flickr30k", model, processor, device)
    else:
        print("  Skipping Flickr30K feature extraction (images unavailable).")

    # Sentinel: COCO is ready; GPU 7 can start COCO-dependent experiments.
    sentinel = CACHE_ROOT / ".data_ready"
    sentinel.touch()
    print(f"\nData preparation complete. Sentinel written: {sentinel}")


if __name__ == "__main__":
    main()
