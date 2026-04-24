"""
Dataset-specific loaders for COCO, Flickr30K, NUS-WIDE, and MIR-Flickr.

Multi-caption loaders (COCO, Flickr30K) return:
  image_paths : list[str]   length = N_images  (one unique path per image)
  captions    : list[str]   length = N_images * n_captions  (image-major order)

These are passed to extract_and_cache_multi_caption() for feature extraction,
which produces a cache compatible with MultiCaptionDataset.

Multi-label loaders (NUS-WIDE, MIR-Flickr) return:
  image_paths : list[str]
  texts       : list[str]   (one tag-string per image)
  labels      : torch.Tensor (N, C) binary multi-hot

These are passed to extract_and_cache() (1:1 paired).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# MS-COCO Captions
# ---------------------------------------------------------------------------

def load_coco_captions(
    root: Path | str,
    split: str = "train",
    n_captions: int = 5,
) -> tuple[list[str], list[str]]:
    """
    Loads MS-COCO image paths and captions.

    Returns:
        image_paths : length N_images  (one unique path per image)
        captions    : length N_images * n_captions  (image-major order)

    Directory layout:
        root/
          images/train2017/  (or val2017/)
          annotations/captions_train2017.json
    """
    root = Path(root)
    ann_file = root / "annotations" / f"captions_{split}2017.json"
    img_dir  = root / "images" / f"{split}2017"

    with open(ann_file) as f:
        data = json.load(f)

    id2file = {img["id"]: img["file_name"] for img in data["images"]}

    from collections import defaultdict
    id2caps: dict[int, list[str]] = defaultdict(list)
    for ann in data["annotations"]:
        id2caps[ann["image_id"]].append(ann["caption"])

    image_paths: list[str] = []
    captions: list[str] = []
    for img_id, caps in sorted(id2caps.items()):
        image_paths.append(str(img_dir / id2file[img_id]))
        for cap in caps[:n_captions]:
            captions.append(cap)
        # Pad with repeated first caption if fewer than n_captions available.
        for _ in range(max(0, n_captions - len(caps))):
            captions.append(caps[0])

    assert len(captions) == len(image_paths) * n_captions
    return image_paths, captions


# ---------------------------------------------------------------------------
# Flickr30K
# ---------------------------------------------------------------------------

def load_flickr30k(
    root: Path | str,
    split: str = "train",
    n_captions: int = 5,
) -> tuple[list[str], list[str]]:
    """
    Loads Flickr30K image paths and captions using the Karpathy split JSON.

    Returns:
        image_paths : length N_images  (unique)
        captions    : length N_images * n_captions  (image-major order)

    Directory layout:
        root/
          flickr30k_images/
          dataset_flickr30k.json
    """
    root = Path(root)
    img_dir  = root / "flickr30k_images"
    ann_file = root / "dataset_flickr30k.json"

    with open(ann_file) as f:
        data = json.load(f)

    image_paths: list[str] = []
    captions: list[str] = []
    for item in data["images"]:
        if item["split"] != split:
            continue
        img_path = str(img_dir / item["filename"])
        sents = [s["raw"] for s in item["sentences"]]
        image_paths.append(img_path)
        for cap in sents[:n_captions]:
            captions.append(cap)
        for _ in range(max(0, n_captions - len(sents))):
            captions.append(sents[0])

    assert len(captions) == len(image_paths) * n_captions
    return image_paths, captions


# ---------------------------------------------------------------------------
# NUS-WIDE
# ---------------------------------------------------------------------------

def load_nuswide(
    root: Path | str,
    split: str = "train",
    top_k_labels: int = 21,
) -> tuple[list[str], list[str], torch.Tensor]:
    """
    Loads NUS-WIDE image paths, tag-strings, and multi-hot label tensors.

    Returns (image_paths, tag_strings, labels) where labels is (N, top_k_labels).

    Directory layout:
        root/
          images/
          nuswide_tags.json    — {filename: [tag, ...]}
          nuswide_labels.json  — {filename: [0/1 per class]}
          splits/train.txt / test.txt
    """
    root = Path(root)
    split_file  = root / "splits" / f"{split}.txt"
    tags_file   = root / "nuswide_tags.json"
    labels_file = root / "nuswide_labels.json"
    img_dir     = root / "images"

    with open(split_file)  as f: fnames      = [l.strip() for l in f if l.strip()]
    with open(tags_file)   as f: tags_map    = json.load(f)
    with open(labels_file) as f: labels_map  = json.load(f)

    image_paths, tag_strings, label_rows = [], [], []
    for fname in fnames:
        tags = tags_map.get(fname, [])
        lbl  = labels_map.get(fname, [0] * top_k_labels)
        image_paths.append(str(img_dir / fname))
        tag_strings.append(" ".join(tags) if tags else "<no tags>")
        label_rows.append(lbl[:top_k_labels])

    labels = torch.tensor(label_rows, dtype=torch.float32)
    return image_paths, tag_strings, labels


# ---------------------------------------------------------------------------
# MIR-Flickr-25K
# ---------------------------------------------------------------------------

def load_mirflickr(
    root: Path | str,
    split: str = "train",
    top_k_labels: int = 38,
) -> tuple[list[str], list[str], torch.Tensor]:
    """
    Loads MIR-Flickr-25K image paths, tag-strings, and multi-hot labels.

    Returns (image_paths, tag_strings, labels) where labels is (N, top_k_labels).

    Directory layout:
        root/
          images/
          annotations/  (one .txt per concept)
          tags/          (one .txt per image)
          splits/train.txt / test.txt
    """
    root = Path(root)
    ann_dir  = root / "annotations"
    tags_dir = root / "tags"
    img_dir  = root / "images"

    split_file = root / "splits" / f"{split}.txt"
    with open(split_file) as f:
        fnames = [l.strip() for l in f if l.strip()]

    concept_files = sorted(ann_dir.glob("*.txt"))[:top_k_labels]
    concept_sets: list[set[str]] = []
    for cf in concept_files:
        with open(cf) as f:
            concept_sets.append({l.strip() for l in f if l.strip()})

    image_paths, tag_strings, label_rows = [], [], []
    for fname in fnames:
        img_path = str(img_dir / fname)
        tag_file = tags_dir / fname.replace(".jpg", ".txt")
        tags = []
        if tag_file.exists():
            with open(tag_file) as f:
                tags = [l.strip() for l in f if l.strip()]
        lbl = [1.0 if fname in cs else 0.0 for cs in concept_sets]
        image_paths.append(img_path)
        tag_strings.append(" ".join(tags) if tags else "<no tags>")
        label_rows.append(lbl)

    labels = torch.tensor(label_rows, dtype=torch.float32)
    return image_paths, tag_strings, labels


# ---------------------------------------------------------------------------
# Multi-label dataset wrapper
# ---------------------------------------------------------------------------

class MultiLabelFeatureDataset(Dataset):
    """
    Wraps pre-cached image/text features and multi-hot labels for NUS-WIDE
    and MIR-Flickr evaluation (mAP metric).
    """

    def __init__(
        self,
        image_cache: Path | str,
        text_cache: Path | str,
        labels: torch.Tensor,
    ):
        self.img    = torch.load(image_cache, map_location="cpu", weights_only=True)
        self.txt    = torch.load(text_cache,  map_location="cpu", weights_only=True)
        self.labels = labels
        assert len(self.img) == len(self.txt) == len(self.labels)

    def __len__(self) -> int:
        return len(self.img)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.img[idx], self.txt[idx], self.labels[idx]
