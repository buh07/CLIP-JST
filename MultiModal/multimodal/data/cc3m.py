from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..common import load_json, save_json
from .datasets import PairedFeatureDataset


@dataclass
class CC3MCache:
    image_feats: torch.Tensor
    text_feats: torch.Tensor
    split_to_indices: dict[str, list[int]]

    @classmethod
    def from_paths(cls, image_cache: Path | str, text_cache: Path | str, metadata_path: Path | str) -> "CC3MCache":
        img = torch.load(image_cache, map_location="cpu", weights_only=True)
        txt = torch.load(text_cache, map_location="cpu", weights_only=True)
        if len(img) != len(txt):
            raise ValueError("CC3M feature length mismatch")
        meta = load_json(metadata_path)
        split_to_indices = {k: [int(i) for i in v] for k, v in meta["split_to_indices"].items()}
        return cls(image_feats=img, text_feats=txt, split_to_indices=split_to_indices)

    def split_indices(self, split_names: str | list[str]) -> list[int]:
        if isinstance(split_names, str):
            split_names = [split_names]
        out: list[int] = []
        for s in split_names:
            out.extend(self.split_to_indices[s])
        return out

    def make_train_dataset(self, split_names: str | list[str]) -> PairedFeatureDataset:
        idx = self.split_indices(split_names)
        return PairedFeatureDataset(self.image_feats[idx], self.text_feats[idx])

    def eval_tensors(self, split_names: str | list[str]) -> tuple[torch.Tensor, torch.Tensor, dict[int, list[int]], dict[int, list[int]]]:
        idx = self.split_indices(split_names)
        img = self.image_feats[idx]
        txt = self.text_feats[idx]
        gt_i2t = {i: [i] for i in range(len(idx))}
        gt_t2i = {i: [i] for i in range(len(idx))}
        return img, txt, gt_i2t, gt_t2i


def build_cc3m_adapter(
    *,
    cc3m_cache_root: Path,
    out_dir: Path,
    image_cache_file: str = "image_feats_openai_clip-vit-base-patch32_raw.pt",
    text_cache_file: str = "text_feats_openai_clip-vit-base-patch32_raw.pt",
    split_seed: int = 2026,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> dict[str, Path]:
    """
    Build deterministic train/val/test split metadata over an existing CC3M feature cache.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    img_in = cc3m_cache_root / image_cache_file
    txt_in = cc3m_cache_root / text_cache_file
    meta_out = out_dir / "metadata.json"

    if meta_out.exists():
        meta = load_json(meta_out)
        expected = {
            "split_seed": int(split_seed),
            "train_frac": float(train_frac),
            "val_frac": float(val_frac),
            "image_cache": str(img_in.resolve()),
            "text_cache": str(txt_in.resolve()),
        }
        if all(meta.get(k) == v for k, v in expected.items()):
            return {"image": img_in, "text": txt_in, "meta": meta_out}

    img = torch.load(img_in, map_location="cpu", weights_only=True)
    txt = torch.load(txt_in, map_location="cpu", weights_only=True)
    if len(img) != len(txt):
        raise RuntimeError("CC3M cache mismatch: image/text row count differ")

    n = len(img)
    rng = np.random.default_rng(int(split_seed))
    perm = rng.permutation(n)

    n_train = int(round(n * float(train_frac)))
    n_val = int(round(n * float(val_frac)))
    n_train = min(max(n_train, 1), n - 2)
    n_val = min(max(n_val, 1), n - n_train - 1)
    n_test = n - n_train - n_val

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    meta = {
        "dataset": "cc3m",
        "protocol": "cc3m_adapter_split",
        "split_seed": int(split_seed),
        "train_frac": float(train_frac),
        "val_frac": float(val_frac),
        "num_pairs": int(n),
        "image_dim": int(img.shape[1]),
        "text_dim": int(txt.shape[1]),
        "image_cache": str(img_in.resolve()),
        "text_cache": str(txt_in.resolve()),
        "split_to_indices": {
            "train": [int(x) for x in train_idx.tolist()],
            "val": [int(x) for x in val_idx.tolist()],
            "test": [int(x) for x in test_idx.tolist()],
        },
        "counts": {"train": int(n_train), "val": int(n_val), "test": int(n_test)},
    }
    save_json(meta, meta_out)
    return {"image": img_in, "text": txt_in, "meta": meta_out}
