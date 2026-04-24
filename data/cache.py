"""
Feature extraction and caching for frozen backbone features.

Run once per dataset; all training and evaluation operate on cached tensors.

Cache format contract
---------------------
Single-caption datasets (PairedFeatureDataset):
  image_feats_{tag}.pt : (N, d_v)
  text_feats_{tag}.pt  : (N, d_t)   — paired 1:1 with images

Multi-caption datasets (MultiCaptionDataset), e.g. COCO/Flickr30K:
  image_feats_{tag}.pt : (N_images, d_v)           — ONE row per unique image
  text_feats_{tag}.pt  : (N_images * n_cap, d_t)   — captions in image-major order
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


def extract_and_cache(
    image_paths: list[str],
    captions: list[str],
    cache_dir: Path | str,
    backbone_name: str = "openai/clip-vit-base-patch32",
    device: str = "cuda",
    batch_size: int = 256,
) -> tuple[Path, Path]:
    """
    Single-caption extraction: len(image_paths) == len(captions).
    Returns (image_cache_path, text_cache_path). Skips if cache exists.
    """
    assert len(image_paths) == len(captions), (
        f"image_paths ({len(image_paths)}) and captions ({len(captions)}) must have equal length. "
        "For multi-caption datasets use extract_and_cache_multi_caption()."
    )
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        raise ImportError("pip install transformers Pillow")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = backbone_name.replace("/", "_")
    img_path = cache_dir / f"image_feats_{tag}.pt"
    txt_path = cache_dir / f"text_feats_{tag}.pt"

    if img_path.exists() and txt_path.exists():
        print(f"Cache hit: {img_path}")
        return img_path, txt_path

    from PIL import Image
    model     = CLIPModel.from_pretrained(backbone_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(backbone_name)

    img_feats: list[torch.Tensor] = []
    txt_feats: list[torch.Tensor] = []

    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            images = [Image.open(p).convert("RGB") for p in image_paths[i:i+batch_size]]
            inputs = processor(images=images, return_tensors="pt").to(device)
            img_feats.append(model.get_image_features(**inputs).cpu())
            if (i // batch_size) % 10 == 0:
                print(f"  images {i}/{len(image_paths)}")

        for i in range(0, len(captions), batch_size):
            inputs = processor(
                text=captions[i:i+batch_size],
                return_tensors="pt", padding=True, truncation=True,
            ).to(device)
            txt_feats.append(model.get_text_features(**inputs).cpu())
            if (i // batch_size) % 10 == 0:
                print(f"  texts {i}/{len(captions)}")

    torch.save(torch.cat(img_feats), img_path)
    torch.save(torch.cat(txt_feats), txt_path)
    print(f"Saved {img_path}, {txt_path}")
    return img_path, txt_path


def extract_and_cache_multi_caption(
    image_paths: list[str],
    captions: list[str],
    n_captions: int,
    cache_dir: Path | str,
    backbone_name: str = "openai/clip-vit-base-patch32",
    device: str = "cuda",
    batch_size: int = 256,
) -> tuple[Path, Path]:
    """
    Multi-caption extraction for COCO/Flickr30K.

    image_paths : N unique image paths (one per image).
    captions    : N * n_captions captions in image-major order.

    Saves:
      image_feats_{tag}.pt : (N, d_v)
      text_feats_{tag}.pt  : (N * n_captions, d_t)
    """
    N = len(image_paths)
    assert len(captions) == N * n_captions, (
        f"Expected {N * n_captions} captions for {N} images × {n_captions} captions/image, "
        f"got {len(captions)}."
    )
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        raise ImportError("pip install transformers Pillow")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = backbone_name.replace("/", "_")
    img_path = cache_dir / f"image_feats_{tag}.pt"
    txt_path = cache_dir / f"text_feats_{tag}.pt"

    if img_path.exists() and txt_path.exists():
        stored_img = torch.load(img_path, map_location="cpu", weights_only=True)
        stored_txt = torch.load(txt_path, map_location="cpu", weights_only=True)
        if stored_img.shape[0] == N and stored_txt.shape[0] == N * n_captions:
            print(f"Cache hit: {img_path}")
            return img_path, txt_path

    from PIL import Image
    model     = CLIPModel.from_pretrained(backbone_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(backbone_name)

    img_feats: list[torch.Tensor] = []
    txt_feats: list[torch.Tensor] = []

    with torch.no_grad():
        for i in range(0, N, batch_size):
            images = [Image.open(p).convert("RGB") for p in image_paths[i:i+batch_size]]
            inputs = processor(images=images, return_tensors="pt").to(device)
            img_feats.append(model.get_image_features(**inputs).cpu())
            if (i // batch_size) % 10 == 0:
                print(f"  images {i}/{N}")

        for i in range(0, len(captions), batch_size):
            inputs = processor(
                text=captions[i:i+batch_size],
                return_tensors="pt", padding=True, truncation=True,
            ).to(device)
            txt_feats.append(model.get_text_features(**inputs).cpu())
            if (i // batch_size) % 10 == 0:
                print(f"  captions {i}/{len(captions)}")

    torch.save(torch.cat(img_feats), img_path)
    torch.save(torch.cat(txt_feats), txt_path)
    print(f"Saved {img_path} ({N}×d), {txt_path} ({N*n_captions}×d)")
    return img_path, txt_path


def extract_and_cache_generic(
    image_paths: list[str],
    texts: list[str],
    cache_dir: Path | str,
    backbone_name: str,
    image_encoder_fn,
    text_encoder_fn,
    batch_size: int = 128,
) -> tuple[Path, Path]:
    """Generic caching for non-CLIP backbones (DINOv2+BGE, CLAP, etc.)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = backbone_name.replace("/", "_")
    img_path = cache_dir / f"image_feats_{tag}.pt"
    txt_path = cache_dir / f"text_feats_{tag}.pt"

    if img_path.exists() and txt_path.exists():
        print(f"Cache hit: {img_path}")
        return img_path, txt_path

    img_feats = [
        image_encoder_fn(image_paths[i:i+batch_size])
        for i in range(0, len(image_paths), batch_size)
    ]
    txt_feats = [
        text_encoder_fn(texts[i:i+batch_size])
        for i in range(0, len(texts), batch_size)
    ]
    torch.save(torch.cat(img_feats), img_path)
    torch.save(torch.cat(txt_feats), txt_path)
    return img_path, txt_path


# ---------------------------------------------------------------------------
# Dataset wrappers
# ---------------------------------------------------------------------------

class PairedFeatureDataset(Dataset):
    """
    Wraps pre-cached (image_feat, text_feat) tensors, paired 1:1.
    """

    def __init__(self, image_cache: Path | str, text_cache: Path | str):
        self.img = torch.load(image_cache, map_location="cpu", weights_only=True)
        self.txt = torch.load(text_cache,  map_location="cpu", weights_only=True)
        assert len(self.img) == len(self.txt), (
            f"Cache length mismatch: {len(self.img)} images vs {len(self.txt)} texts."
        )

    def __len__(self) -> int:
        return len(self.img)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.img[idx], self.txt[idx]


class MultiCaptionDataset(Dataset):
    """
    Dataset for COCO/Flickr30K where each image has n_captions captions.

    Cache format:
      image_cache : (N_images, d_v)                 — ONE row per image
      text_cache  : (N_images * n_captions, d_t)    — captions in image-major order

    Training mode  (training=True):  randomly samples one caption per image each call.
    Eval mode      (training=False): always returns caption 0 (deterministic, for loaders).
    For proper multi-GT evaluation use get_eval_tensors() instead of a DataLoader.
    """

    def __init__(
        self,
        image_cache: Path | str,
        text_cache: Path | str,
        n_captions: int = 5,
        training: bool = True,
    ):
        self.img = torch.load(image_cache, map_location="cpu", weights_only=True)
        self.txt = torch.load(text_cache,  map_location="cpu", weights_only=True)
        self.n_captions = n_captions
        self.training = training
        assert len(self.txt) == len(self.img) * n_captions, (
            f"Text cache has {len(self.txt)} rows; expected {len(self.img)} × {n_captions} = "
            f"{len(self.img) * n_captions}. "
            "Use extract_and_cache_multi_caption() to build the cache."
        )

    # Follow nn.Module convention so callers can do ds.train() / ds.eval().
    def train(self, mode: bool = True) -> "MultiCaptionDataset":
        self.training = mode
        return self

    def eval(self) -> "MultiCaptionDataset":
        return self.train(False)

    def __len__(self) -> int:
        return len(self.img)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        cap_offset = (
            int(torch.randint(self.n_captions, (1,)).item())
            if self.training
            else 0
        )
        return self.img[idx], self.txt[idx * self.n_captions + cap_offset]

    def get_eval_tensors(self) -> tuple[torch.Tensor, torch.Tensor, dict[int, list[int]]]:
        """
        Returns (image_feats, text_feats, gt_i2t) for multi-GT evaluation.
        gt_i2t[i] = [i*n_cap, i*n_cap+1, ..., (i+1)*n_cap - 1]

        Use this instead of a DataLoader for final metric reporting.
        """
        gt_i2t = {
            i: list(range(i * self.n_captions, (i + 1) * self.n_captions))
            for i in range(len(self.img))
        }
        return self.img, self.txt, gt_i2t
