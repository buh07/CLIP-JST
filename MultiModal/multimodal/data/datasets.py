from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ..common import load_json


class ImageCaptionTrainDataset(Dataset):
    def __init__(self, img_feats: torch.Tensor, txt_feats: torch.Tensor, image_indices: list[int], n_captions: int):
        self.img_feats = img_feats
        self.txt_feats = txt_feats
        self.indices = image_indices
        self.n_captions = n_captions
        self.training = True

    def train(self, mode: bool = True) -> "ImageCaptionTrainDataset":
        self.training = mode
        return self

    def eval(self) -> "ImageCaptionTrainDataset":
        return self.train(False)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        if self.training:
            cap_offset = int(torch.randint(self.n_captions, (1,)).item())
        else:
            cap_offset = 0
        txt_idx = idx * self.n_captions + cap_offset
        return self.img_feats[idx], self.txt_feats[txt_idx]


@dataclass
class KarpathyCache:
    image_feats: torch.Tensor
    text_feats: torch.Tensor
    n_captions: int
    split_to_indices: dict[str, list[int]]

    @classmethod
    def from_paths(cls, image_cache: Path | str, text_cache: Path | str, metadata_path: Path | str) -> "KarpathyCache":
        img = torch.load(image_cache, map_location="cpu", weights_only=True)
        txt = torch.load(text_cache, map_location="cpu", weights_only=True)
        meta = load_json(metadata_path)
        n_cap = int(meta["n_captions"])
        if txt.shape[0] != img.shape[0] * n_cap:
            raise ValueError("Cache shape mismatch for multi-caption format")
        return cls(image_feats=img, text_feats=txt, n_captions=n_cap, split_to_indices=meta["split_to_indices"])

    def split_indices(self, split_names: str | list[str]) -> list[int]:
        if isinstance(split_names, str):
            split_names = [split_names]
        out: list[int] = []
        for s in split_names:
            out.extend(self.split_to_indices[s])
        return out

    def make_train_dataset(self, split_names: str | list[str], training: bool = True) -> ImageCaptionTrainDataset:
        ds = ImageCaptionTrainDataset(
            img_feats=self.image_feats,
            txt_feats=self.text_feats,
            image_indices=self.split_indices(split_names),
            n_captions=self.n_captions,
        )
        ds.train(training)
        return ds

    def eval_tensors(self, split_names: str | list[str]) -> tuple[torch.Tensor, torch.Tensor, dict[int, list[int]], dict[int, list[int]]]:
        indices = self.split_indices(split_names)
        img = self.image_feats[indices]
        txt_rows = [idx * self.n_captions + k for idx in indices for k in range(self.n_captions)]
        txt = self.text_feats[txt_rows]
        gt_i2t = {
            i: list(range(i * self.n_captions, (i + 1) * self.n_captions))
            for i in range(len(indices))
        }
        gt_t2i = {j: [j // self.n_captions] for j in range(len(txt_rows))}
        return img, txt, gt_i2t, gt_t2i


class PairedFeatureDataset(Dataset):
    def __init__(self, left: torch.Tensor, right: torch.Tensor):
        if len(left) != len(right):
            raise ValueError("paired tensors must have equal first dimension")
        self.left = left
        self.right = right

    def __len__(self) -> int:
        return len(self.left)

    def __getitem__(self, idx: int):
        return self.left[idx], self.right[idx]


class TripleFeatureDataset(Dataset):
    def __init__(self, first: torch.Tensor, second: torch.Tensor, third: torch.Tensor):
        if len(first) != len(second) or len(first) != len(third):
            raise ValueError("triple tensors must have equal first dimension")
        self.first = first
        self.second = second
        self.third = third

    def __len__(self) -> int:
        return len(self.first)

    def __getitem__(self, idx: int):
        return self.first[idx], self.second[idx], self.third[idx]


@dataclass
class AudioCapsCache:
    audio_feats: torch.Tensor
    text_feats: torch.Tensor
    split_to_indices: dict[str, list[int]]

    @classmethod
    def from_paths(cls, audio_cache: Path | str, text_cache: Path | str, metadata_path: Path | str) -> "AudioCapsCache":
        audio = torch.load(audio_cache, map_location="cpu", weights_only=True)
        text = torch.load(text_cache, map_location="cpu", weights_only=True)
        meta = load_json(metadata_path)
        if len(audio) != len(text):
            raise ValueError("Audio/text feature length mismatch")
        return cls(audio_feats=audio, text_feats=text, split_to_indices=meta["split_to_indices"])

    def split_indices(self, split_names: str | list[str]) -> list[int]:
        if isinstance(split_names, str):
            split_names = [split_names]
        out: list[int] = []
        for s in split_names:
            out.extend(self.split_to_indices[s])
        return out

    def make_dataset(self, split_names: str | list[str]) -> PairedFeatureDataset:
        idx = self.split_indices(split_names)
        return PairedFeatureDataset(self.audio_feats[idx], self.text_feats[idx])

    def eval_tensors(self, split_names: str | list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        idx = self.split_indices(split_names)
        return self.audio_feats[idx], self.text_feats[idx]


@dataclass
class AudioCapsAVCache:
    image_feats: torch.Tensor
    audio_feats: torch.Tensor
    text_feats: torch.Tensor
    split_to_indices: dict[str, list[int]]
    youtube_ids: list[str]
    thumbnail_ok: list[bool]

    @classmethod
    def from_paths(
        cls,
        image_cache: Path | str,
        audio_cache: Path | str,
        text_cache: Path | str,
        metadata_path: Path | str,
    ) -> "AudioCapsAVCache":
        img = torch.load(image_cache, map_location="cpu", weights_only=True)
        audio = torch.load(audio_cache, map_location="cpu", weights_only=True)
        text = torch.load(text_cache, map_location="cpu", weights_only=True)
        meta = load_json(metadata_path)
        if len(img) != len(audio) or len(img) != len(text):
            raise ValueError("AudioCapsAV cache feature length mismatch")
        youtube_ids = [str(x) for x in meta.get("youtube_ids", [])]
        thumb_ok = [bool(x) for x in meta.get("thumbnail_ok", [True] * len(img))]
        if youtube_ids and len(youtube_ids) != len(img):
            raise ValueError("youtube_ids length mismatch in AudioCapsAV metadata")
        if thumb_ok and len(thumb_ok) != len(img):
            raise ValueError("thumbnail_ok length mismatch in AudioCapsAV metadata")
        return cls(
            image_feats=img,
            audio_feats=audio,
            text_feats=text,
            split_to_indices=meta["split_to_indices"],
            youtube_ids=youtube_ids,
            thumbnail_ok=thumb_ok,
        )

    def split_indices(self, split_names: str | list[str]) -> list[int]:
        if isinstance(split_names, str):
            split_names = [split_names]
        out: list[int] = []
        for s in split_names:
            out.extend(self.split_to_indices[s])
        return out

    def make_dataset(self, split_names: str | list[str]) -> TripleFeatureDataset:
        idx = self.split_indices(split_names)
        return TripleFeatureDataset(self.image_feats[idx], self.audio_feats[idx], self.text_feats[idx])

    def eval_tensors(self, split_names: str | list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = self.split_indices(split_names)
        return self.image_feats[idx], self.audio_feats[idx], self.text_feats[idx]
