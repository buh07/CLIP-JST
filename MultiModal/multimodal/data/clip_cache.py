from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..common import save_json


def _load_clip(backbone_name: str, device: str):
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(backbone_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(backbone_name)
    return model, processor


@torch.no_grad()
def _encode_images(image_paths: list[str], model, processor, device: str, batch_size: int) -> torch.Tensor:
    from PIL import Image

    feats: list[torch.Tensor] = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inp = processor(images=images, return_tensors="pt").to(device)
        vision_out = model.vision_model(pixel_values=inp["pixel_values"])
        feats.append(vision_out.pooler_output.cpu())
        if (i // batch_size) % 10 == 0:
            print(f"  images {i}/{len(image_paths)}")
    return torch.cat(feats)


@torch.no_grad()
def _encode_texts(texts: list[str], model, processor, device: str, batch_size: int) -> torch.Tensor:
    feats: list[torch.Tensor] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inp = processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(device)
        out = model.text_model(input_ids=inp["input_ids"], attention_mask=inp.get("attention_mask"))
        feats.append(out.pooler_output.cpu())
        if (i // batch_size) % 10 == 0:
            print(f"  captions {i}/{len(texts)}")
    return torch.cat(feats)


def extract_karpathy_clip_cache(
    *,
    manifest_path: Path,
    out_dir: Path,
    backbone_name: str,
    batch_size: int,
    device: str,
    existing_image_cache: Path | None = None,
    existing_image_ids_json: Path | None = None,
) -> dict[str, Path]:
    """
    Extract CLIP raw backbone features for one Karpathy manifest.

    Saves:
      image_feats_<tag>_raw.pt   (N_img, 768)
      text_feats_<tag>_raw.pt    (N_img*n_cap, 512)
      metadata.json              split indices + IDs
    """
    import json

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    dataset = manifest["dataset"]
    n_captions = int(manifest["n_captions"])
    entries = manifest["images"]

    image_paths = [e["image_path"] for e in entries]
    image_ids = [e["image_id"] for e in entries]
    split_labels = [e["split"] for e in entries]
    all_texts = [cap for e in entries for cap in e["captions"]]

    tag = backbone_name.replace("/", "_")
    img_out = out_dir / f"image_feats_{tag}_raw.pt"
    txt_out = out_dir / f"text_feats_{tag}_raw.pt"
    meta_out = out_dir / "metadata.json"

    if img_out.exists() and txt_out.exists() and meta_out.exists():
        img = torch.load(img_out, map_location="cpu", weights_only=True)
        txt = torch.load(txt_out, map_location="cpu", weights_only=True)
        if img.shape[0] == len(image_paths) and txt.shape[0] == len(all_texts):
            print(f"Cache hit for {dataset}: {img_out}")
            return {"image": img_out, "text": txt_out, "meta": meta_out}

    print(f"Encoding {dataset}: {len(image_paths)} images, {len(all_texts)} captions")
    model, processor = _load_clip(backbone_name, device)

    img_feats: torch.Tensor
    if (
        dataset == "coco"
        and existing_image_cache is not None
        and existing_image_ids_json is not None
        and existing_image_cache.exists()
        and existing_image_ids_json.exists()
    ):
        print(f"Reusing existing COCO image cache: {existing_image_cache}")
        existing_img = torch.load(existing_image_cache, map_location="cpu", weights_only=True)
        import json

        with open(existing_image_ids_json, encoding="utf-8") as f:
            existing_ids = json.load(f)
        id_to_row = {int(cid): i for i, cid in enumerate(existing_ids)}

        # Pull features for IDs already present and only encode the missing set.
        missing_indices = [i for i, cid in enumerate(image_ids) if int(cid) not in id_to_row]
        print(f"  Found {len(image_ids) - len(missing_indices)} reused images, {len(missing_indices)} missing images.")

        missing_feats_map: dict[int, torch.Tensor] = {}
        if missing_indices:
            missing_paths = [image_paths[i] for i in missing_indices]
            missing_ids = [int(image_ids[i]) for i in missing_indices]
            missing_feats = _encode_images(missing_paths, model, processor, device, batch_size)
            for cid, feat in zip(missing_ids, missing_feats):
                missing_feats_map[cid] = feat

        rows: list[torch.Tensor] = []
        for cid in image_ids:
            cid_int = int(cid)
            if cid_int in id_to_row:
                rows.append(existing_img[id_to_row[cid_int]])
            else:
                rows.append(missing_feats_map[cid_int])
        img_feats = torch.stack(rows, dim=0)
    else:
        img_feats = _encode_images(image_paths, model, processor, device, batch_size)

    txt_feats = _encode_texts(all_texts, model, processor, device, batch_size)

    torch.save(img_feats, img_out)
    torch.save(txt_feats, txt_out)

    split_to_indices: dict[str, list[int]] = {}
    for i, sp in enumerate(split_labels):
        split_to_indices.setdefault(sp, []).append(i)
    if dataset == "coco":
        tr = split_to_indices.get("train", [])
        rv = split_to_indices.get("restval", [])
        split_to_indices["train_restval"] = tr + rv

    meta: dict[str, Any] = {
        "dataset": dataset,
        "protocol": "karpathy",
        "backbone": backbone_name,
        "n_captions": n_captions,
        "n_images": len(image_paths),
        "n_text": len(all_texts),
        "vision_dim": int(img_feats.shape[1]),
        "text_dim": int(txt_feats.shape[1]),
        "split_to_indices": split_to_indices,
        "image_ids": image_ids,
    }
    save_json(meta, meta_out)
    print(f"Saved {dataset} cache to {out_dir}")
    return {"image": img_out, "text": txt_out, "meta": meta_out}
