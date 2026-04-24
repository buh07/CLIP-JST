#!/usr/bin/env python3
"""
Prepare NUS-WIDE feature cache from the Lxyhaha/NUS-WIDE HuggingFace repo.

Since raw Flickr images are not redistributed, this script encodes the Flickr
tag strings with CLIP text encoder and uses those as both image and text
features.  D2's distortion-per-class metric is still valid: it tests whether JL
preserves within-class structure of the NUS-WIDE tag feature space stratified by
the 81 concept labels.

Input (after extracting NUS-WIDE.zip and NUS_WID_Tags.zip):
  data/NUS-WIDE/NUS_WID_Tags/All_Tags.txt        — 269648 rows: <flickr_id> <tags...>
  data/NUS-WIDE/NUS_WID_Tags/AllTags81.txt        — 269648×81 binary matrix
  data/NUS-WIDE/NUS_WID_Tags/Train_Tags81.txt     — 161789×81 (train split)
  data/NUS-WIDE/NUS_WID_Tags/Test_Tags81.txt      — 107859×81 (test split)
  data/NUS-WIDE/ConceptsList/Concepts81.txt        — 81 concept names

Output:
  data/cache/nuswide/image_feats_openai_clip-vit-base-patch32.pt  (N_train, 512)
  data/cache/nuswide/text_feats_openai_clip-vit-base-patch32.pt   (N_train, 512)
  data/cache/nuswide/labels.pt                                      (N_train, 81)

Usage:
  cd "/jumbo/lisp/f004ndc/CLIP JST"
  CUDA_VISIBLE_DEVICES=6 python scripts/prepare_nuswide.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NUS_ROOT     = PROJECT_ROOT / "data" / "NUS-WIDE"
CACHE_DIR    = PROJECT_ROOT / "data" / "cache" / "nuswide"
BACKBONE     = "openai/clip-vit-base-patch32"
BATCH_SIZE   = 512


def load_data() -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """
    Returns:
      tag_strings : list[str] length N_train  — one tag string per training sample
      train_labels: (N_train, 81) float32
      test_labels : (N_test, 81)  float32  (saved separately for reference)
    """
    tags_file   = NUS_ROOT / "NUS_WID_Tags" / "All_Tags.txt"
    all_lbl_file = NUS_ROOT / "NUS_WID_Tags" / "AllTags81.txt"
    train_file  = NUS_ROOT / "NUS_WID_Tags" / "Train_Tags81.txt"
    test_file   = NUS_ROOT / "NUS_WID_Tags" / "Test_Tags81.txt"

    for p in [tags_file, all_lbl_file, train_file, test_file]:
        if not p.exists():
            print(f"Missing: {p}")
            print("Run: cd data/NUS-WIDE && unzip NUS-WIDE.zip && unzip NUS_WID_Tags.zip")
            sys.exit(1)

    # Parse tag strings for all 269,648 samples (first token is Flickr ID).
    print("Parsing All_Tags.txt ...")
    all_tags: list[str] = []
    with open(tags_file) as f:
        for line in f:
            parts = line.strip().split()
            # Skip Flickr ID (first token); rest are tags.
            tags = parts[1:] if len(parts) > 1 else ["photo"]
            all_tags.append(" ".join(tags) if tags else "photo")

    # Determine train/test split sizes.
    n_train = sum(1 for _ in open(train_file))
    n_test  = sum(1 for _ in open(test_file))
    n_total = len(all_tags)
    assert n_train + n_test == n_total, (
        f"Split mismatch: {n_train}+{n_test} != {n_total}"
    )

    # All_Tags rows are ordered train then test (standard NUS-WIDE convention).
    train_tags = all_tags[:n_train]

    # Parse AllTags81 labels.
    print("Parsing AllTags81.txt ...")
    all_labels: list[list[int]] = []
    with open(all_lbl_file) as f:
        for line in f:
            row = [int(x) for x in line.strip().split()]
            all_labels.append(row)

    train_labels = torch.tensor(all_labels[:n_train], dtype=torch.float32)
    test_labels  = torch.tensor(all_labels[n_train:], dtype=torch.float32)

    print(f"Train: {n_train} samples, Test: {n_test} samples, Concepts: {train_labels.shape[1]}")
    return train_tags, train_labels, test_labels


@torch.no_grad()
def encode_texts(
    texts: list[str],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str,
) -> torch.Tensor:
    feats = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="encoding tags"):
        batch = texts[i : i + BATCH_SIZE]
        inp = processor(
            text=batch, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        text_out = model.text_model(
            input_ids=inp["input_ids"],
            attention_mask=inp.get("attention_mask"),
        )
        feats.append(model.text_projection(text_out.pooler_output).cpu())
    return torch.cat(feats)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tag        = BACKBONE.replace("/", "_")
    feat_path  = CACHE_DIR / f"image_feats_{tag}.pt"
    txt_path   = CACHE_DIR / f"text_feats_{tag}.pt"
    label_path = CACHE_DIR / "labels.pt"

    if feat_path.exists() and txt_path.exists() and label_path.exists():
        print("Cache already exists, skipping.")
        return

    train_tags, train_labels, _ = load_data()

    print(f"Loading {BACKBONE} ...")
    model     = CLIPModel.from_pretrained(BACKBONE).to(device).eval()
    processor = CLIPProcessor.from_pretrained(BACKBONE)
    print("Model loaded.\n")

    tag_feats = encode_texts(train_tags, model, processor, device)

    # Use the same tag features for both "image" and "text" slots.
    # D2 measures within-class JL distortion; using identical features gives
    # symmetric block-diagonal distances, which is still a valid distortion test.
    torch.save(tag_feats, feat_path)
    torch.save(tag_feats, txt_path)
    torch.save(train_labels, label_path)
    print(f"\nSaved:")
    print(f"  {feat_path}  {tuple(tag_feats.shape)}")
    print(f"  {txt_path}   {tuple(tag_feats.shape)}")
    print(f"  {label_path} {tuple(train_labels.shape)}")


if __name__ == "__main__":
    main()
