from __future__ import annotations

import argparse
import io
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

from ..common import env_snapshot, load_json, mark_done, save_json
from ..data.audiocaps import _decode_audio_with_ffmpeg, _resample_audio


def _coerce_text(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _decode_image_blob(image_blob: Any) -> Image.Image:
    if isinstance(image_blob, Image.Image):
        return image_blob.convert("RGB")
    if isinstance(image_blob, dict):
        b = image_blob.get("bytes")
        p = image_blob.get("path")
        if b is not None:
            return Image.open(io.BytesIO(b)).convert("RGB")
        if p:
            return Image.open(str(p)).convert("RGB")
    raise RuntimeError(f"Unsupported image payload type: {type(image_blob)}")


def _to_audio_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if hasattr(x, "audio_embeds") and x.audio_embeds is not None:
        return x.audio_embeds
    if hasattr(x, "pooler_output") and x.pooler_output is not None:
        return x.pooler_output
    if isinstance(x, (tuple, list)) and len(x) > 0 and isinstance(x[0], torch.Tensor):
        return x[0]
    raise RuntimeError(f"Unsupported CLAP audio output type: {type(x)}")


def _load_split(dataset_name: str, split: str, hf_cache_dir: str | None, token: str | None):
    from datasets import Audio, Image as HFImage, load_dataset

    kwargs: dict[str, Any] = {"split": split}
    if hf_cache_dir:
        kwargs["cache_dir"] = hf_cache_dir
    if token:
        kwargs["token"] = token

    ds = load_dataset(dataset_name, **kwargs)
    ds = ds.cast_column("audio", Audio(sampling_rate=None, decode=False))
    ds = ds.cast_column("image", HFImage(decode=False))
    return ds


def run(cfg: dict[str, Any]) -> None:
    start = time.time()
    stage_name = "stage69_speechcoco_cache_encode_shard"

    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(
        cfg.get(
            "manifest_path",
            output_root / "stage69_speechcoco_cache_manifest" / "stage69_speechcoco_manifest.json",
        )
    ).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = load_json(manifest_path)

    shard_index = int(cfg["shard_index"])
    shard_count = int(cfg["shard_count"])
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"Invalid shard_index={shard_index} shard_count={shard_count}")

    shard_root = Path(cfg.get("shard_root", output_root / "stage69_speechcoco_cache_shards")).resolve()
    shard_dir = shard_root / f"shard{shard_index}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = str(manifest["dataset"])
    hf_cache_dir = cfg.get("speechcoco_hf_cache_dir")
    hf_token = cfg.get("hf_token")
    if not hf_token:
        import os

        hf_token = os.environ.get("HF_TOKEN")

    device = str(cfg["device"])
    clap_model_name = str(cfg["clap_model"])
    clip_backbone_name = str(cfg["clip_backbone"])
    target_sr = int(cfg.get("speechcoco_target_sr", 48_000))
    audio_bs = int(cfg.get("speechcoco_audio_batch_size", 32))
    image_bs = int(cfg.get("speechcoco_image_batch_size", 32))
    text_bs = int(cfg.get("speechcoco_text_batch_size", 128))
    flush_bs = max(1, min(audio_bs, image_bs, text_bs))

    # Load models once per shard worker.
    from transformers import AutoProcessor, CLIPModel, CLIPProcessor, ClapModel

    clap_model = ClapModel.from_pretrained(clap_model_name).to(device).eval()
    clap_processor = AutoProcessor.from_pretrained(clap_model_name)
    clip_model = CLIPModel.from_pretrained(clip_backbone_name).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_backbone_name)

    train_ds = _load_split(dataset_name, "train", hf_cache_dir=hf_cache_dir, token=hf_token)
    val_ds = _load_split(dataset_name, "validation", hf_cache_dir=hf_cache_dir, token=hf_token)

    rows_all: list[dict[str, Any]] = manifest["rows"]
    rows = [r for r in rows_all if int(r["row_id"]) % shard_count == shard_index]
    rows.sort(key=lambda r: int(r["row_id"]))

    img_chunks: list[torch.Tensor] = []
    aud_chunks: list[torch.Tensor] = []
    txt_chunks: list[torch.Tensor] = []
    success_row_ids: list[int] = []

    decode_failures = 0
    empty_audio = 0
    dropped_empty_text = 0

    batch_images: list[Image.Image] = []
    batch_audio: list[np.ndarray] = []
    batch_text: list[str] = []
    batch_row_ids: list[int] = []

    @torch.no_grad()
    def flush_batch() -> None:
        if not batch_audio:
            return
        a_in = clap_processor(
            audio=batch_audio,
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True,
        ).to(device)
        a_out = clap_model.get_audio_features(**a_in)
        a_feat = _to_audio_tensor(a_out).cpu()

        i_in = clip_processor(images=batch_images, return_tensors="pt", padding=True).to(device)
        i_out = clip_model.vision_model(pixel_values=i_in["pixel_values"])
        i_feat = i_out.pooler_output.cpu()

        t_in = clip_processor(text=batch_text, return_tensors="pt", padding=True, truncation=True).to(device)
        t_out = clip_model.text_model(
            input_ids=t_in["input_ids"],
            attention_mask=t_in.get("attention_mask"),
        )
        t_feat = t_out.pooler_output.cpu()

        if not (len(a_feat) == len(i_feat) == len(t_feat) == len(batch_row_ids)):
            raise RuntimeError("Feature count mismatch inside shard flush")

        img_chunks.append(i_feat)
        aud_chunks.append(a_feat)
        txt_chunks.append(t_feat)
        success_row_ids.extend(int(x) for x in batch_row_ids)

        batch_images.clear()
        batch_audio.clear()
        batch_text.clear()
        batch_row_ids.clear()

    for r in rows:
        src_split = str(r["source_split"])
        src_idx = int(r["source_index"])
        row_id = int(r["row_id"])

        if src_split == "train":
            ex = train_ds[src_idx]
        elif src_split == "validation":
            ex = val_ds[src_idx]
        else:
            raise RuntimeError(f"Unexpected source_split={src_split}")

        txt = _coerce_text(ex.get("text"))
        if not txt:
            dropped_empty_text += 1
            continue

        try:
            img = _decode_image_blob(ex["image"])
            wav, sr = _decode_audio_with_ffmpeg(ex["audio"], target_sr)
            wav = _resample_audio(wav, src_sr=int(sr), tgt_sr=target_sr)
            if wav.size == 0:
                empty_audio += 1
                continue
        except Exception:
            decode_failures += 1
            continue

        batch_images.append(img)
        batch_audio.append(wav)
        batch_text.append(txt)
        batch_row_ids.append(row_id)
        if len(batch_audio) >= flush_bs:
            flush_batch()

    flush_batch()

    if success_row_ids:
        image_feats = torch.cat(img_chunks, dim=0)
        audio_feats = torch.cat(aud_chunks, dim=0)
        text_feats = torch.cat(txt_chunks, dim=0)
    else:
        image_feats = torch.empty((0, 768), dtype=torch.float32)
        audio_feats = torch.empty((0, 512), dtype=torch.float32)
        text_feats = torch.empty((0, 512), dtype=torch.float32)

    if not (len(image_feats) == len(audio_feats) == len(text_feats) == len(success_row_ids)):
        raise RuntimeError("Final shard feature length mismatch")

    torch.save(image_feats, shard_dir / "image_feats.pt")
    torch.save(audio_feats, shard_dir / "audio_feats.pt")
    torch.save(text_feats, shard_dir / "text_feats.pt")
    save_json(success_row_ids, shard_dir / "row_ids_success.json")

    shard_meta = {
        "stage": stage_name,
        "manifest_path": str(manifest_path),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "rows_assigned": int(len(rows)),
        "rows_success": int(len(success_row_ids)),
        "decode_failures": int(decode_failures),
        "empty_audio": int(empty_audio),
        "dropped_empty_text": int(dropped_empty_text),
        "elapsed_sec": float(time.time() - start),
    }
    save_json(shard_meta, shard_dir / "shard_meta.json")

    provenance = env_snapshot(
        Path(cfg["project_root"]),
        seeds=[],
        extra={
            "stage": stage_name,
            "manifest_path": str(manifest_path),
            "shard_index": shard_index,
            "shard_count": shard_count,
            "rows_assigned": int(len(rows)),
            "rows_success": int(len(success_row_ids)),
            "elapsed_sec": float(time.time() - start),
        },
    )
    save_json(provenance, shard_dir / "provenance_stage69_cache_encode_shard.json")

    mark_done(
        markers / f"stage69_cache_encode_shard{shard_index}.done.json",
        {
            "shard_index": shard_index,
            "shard_count": shard_count,
            "rows_assigned": int(len(rows)),
            "rows_success": int(len(success_row_ids)),
            "elapsed_sec": float(time.time() - start),
        },
    )
    print(
        f"{stage_name} complete shard={shard_index}/{shard_count} "
        f"assigned={len(rows)} success={len(success_row_ids)}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

