from __future__ import annotations

import io
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from ..common import load_json, save_json
from .audiocaps import _decode_audio_with_ffmpeg, _resample_audio


def _extract_first_text(rec: dict[str, Any]) -> str:
    for key in ("audio_visual_captions", "GPT_AV_captions", "visual_captions", "audio_captions"):
        v = rec.get(key)
        if isinstance(v, list):
            for x in v:
                if isinstance(x, str) and x.strip():
                    return x.strip()
        elif isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _decode_video_preview_frame(video_bytes: bytes) -> Image.Image:
    """
    Decode a representative frame from an mp4 blob via ffmpeg.
    Uses thumbnail filter to avoid always picking the very first frame.
    """
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-vframes",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, input=video_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode == 0 and proc.stdout:
        return Image.open(io.BytesIO(proc.stdout)).convert("RGB")

    # Fallback for hosts where ffmpeg CLI lacks frame options.
    try:
        import av

        with av.open(io.BytesIO(video_bytes), mode="r", format="mp4") as container:
            for frame in container.decode(video=0):
                return frame.to_image().convert("RGB")
    except Exception:
        pass

    err = proc.stderr.decode("utf-8", errors="replace")
    raise RuntimeError(f"video frame decode failed (ffmpeg_code={proc.returncode}): {err[-500:]}")


def extract_avcaps_av_cache(
    *,
    out_dir: Path,
    dataset_name: str,
    clap_model_name: str,
    clip_backbone_name: str,
    target_sampling_rate: int,
    device: str,
    audio_batch_size: int,
    image_batch_size: int,
    text_batch_size: int,
    max_examples_per_split: int | None = None,
) -> dict[str, Path]:
    """
    Build video-frame/audio/text raw feature cache for AVCaps.

    - video modality: one representative frame per video -> CLIP image encoder
    - audio modality: full clip audio -> CLAP audio encoder
    - text modality: first available AV caption -> CLIP text encoder
    """
    from huggingface_hub import hf_hub_download
    from transformers import AutoProcessor, CLIPModel, CLIPProcessor, ClapModel

    out_dir.mkdir(parents=True, exist_ok=True)
    image_out = out_dir / "image_feats_clip_raw.pt"
    audio_out = out_dir / "audio_feats_clap_raw.pt"
    text_out = out_dir / "text_feats_clip_raw.pt"
    meta_out = out_dir / "metadata.json"

    expected = {
        "dataset": dataset_name,
        "audio_encoder": clap_model_name,
        "image_text_encoder": clip_backbone_name,
        "target_sampling_rate": int(target_sampling_rate),
        "max_examples_per_split": int(max_examples_per_split) if max_examples_per_split is not None else None,
    }
    if image_out.exists() and audio_out.exists() and text_out.exists() and meta_out.exists():
        meta = load_json(meta_out)
        if all(meta.get(k) == v for k, v in expected.items()):
            return {"image": image_out, "audio": audio_out, "text": text_out, "meta": meta_out}

    clap_model = ClapModel.from_pretrained(clap_model_name).to(device).eval()
    clap_processor = AutoProcessor.from_pretrained(clap_model_name)

    clip_model = CLIPModel.from_pretrained(clip_backbone_name).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_backbone_name)

    all_img_feats: list[torch.Tensor] = []
    all_aud_feats: list[torch.Tensor] = []
    all_txt_feats: list[torch.Tensor] = []

    split_to_indices: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    video_ids: list[str] = []
    next_idx = 0

    batch_images: list[Image.Image] = []
    batch_audio: list[np.ndarray] = []
    batch_text: list[str] = []

    def _to_audio_tensor(x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        if hasattr(x, "audio_embeds") and x.audio_embeds is not None:
            return x.audio_embeds
        if hasattr(x, "pooler_output") and x.pooler_output is not None:
            return x.pooler_output
        if isinstance(x, (tuple, list)) and len(x) > 0 and isinstance(x[0], torch.Tensor):
            return x[0]
        raise RuntimeError(f"Unsupported CLAP output type: {type(x)}")

    def flush(split_name: str) -> None:
        nonlocal next_idx
        if not batch_audio:
            return
        with torch.no_grad():
            a_in = clap_processor(
                audio=batch_audio,
                sampling_rate=target_sampling_rate,
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

        all_aud_feats.append(a_feat)
        all_img_feats.append(i_feat)
        all_txt_feats.append(t_feat)

        idxs = list(range(next_idx, next_idx + len(batch_audio)))
        split_to_indices[split_name].extend(idxs)
        next_idx += len(batch_audio)

        batch_images.clear()
        batch_audio.clear()
        batch_text.clear()

    split_files = [
        ("train", "train_captions.json", "train_videos.zip"),
        ("validation", "val_captions.json", "val_videos.zip"),
        ("test", "test_captions.json", "test_videos.zip"),
    ]

    decode_failures = 0
    missing_videos = 0

    for split_name, captions_name, videos_name in split_files:
        cap_path = hf_hub_download(repo_id=dataset_name, repo_type="dataset", filename=captions_name)
        vid_zip_path = hf_hub_download(repo_id=dataset_name, repo_type="dataset", filename=videos_name)

        with open(cap_path, encoding="utf-8") as f:
            cap_json = json.load(f)
        if not isinstance(cap_json, dict):
            raise RuntimeError(f"Unexpected captions format in {captions_name}")

        ids = sorted(cap_json.keys())
        if max_examples_per_split is not None:
            ids = ids[: max(0, int(max_examples_per_split))]

        with zipfile.ZipFile(vid_zip_path, "r") as zf:
            name_set = set(zf.namelist())
            prefix = "train_videos" if split_name == "train" else ("val_videos" if split_name == "validation" else "test_videos")

            for i, vid in enumerate(ids, start=1):
                rec = cap_json.get(vid, {})
                txt = _extract_first_text(rec)
                if not txt:
                    continue

                member_name = f"{prefix}/{vid}.mp4"
                if member_name not in name_set:
                    missing_videos += 1
                    continue

                try:
                    video_bytes = zf.read(member_name)
                    img = _decode_video_preview_frame(video_bytes)
                    wav, sr = _decode_audio_with_ffmpeg({"bytes": video_bytes}, target_sampling_rate)
                    wav = _resample_audio(wav, src_sr=sr, tgt_sr=target_sampling_rate)
                    if wav.size == 0:
                        raise RuntimeError("empty waveform")
                except Exception:
                    decode_failures += 1
                    continue

                batch_images.append(img)
                batch_audio.append(wav)
                batch_text.append(txt)
                video_ids.append(str(vid))

                if len(batch_audio) >= min(audio_batch_size, image_batch_size, text_batch_size):
                    flush(split_name)

                if i % 100 == 0:
                    print(f"AVCaps {split_name}: {i}/{len(ids)} processed")

        flush(split_name)
        print(f"AVCaps {split_name}: kept {len(split_to_indices[split_name])}")

    if not all_img_feats or not all_aud_feats or not all_txt_feats:
        raise RuntimeError("No AVCaps features were extracted")

    img_feats = torch.cat(all_img_feats)
    aud_feats = torch.cat(all_aud_feats)
    txt_feats = torch.cat(all_txt_feats)

    if len(img_feats) != len(aud_feats) or len(img_feats) != len(txt_feats):
        raise RuntimeError("AVCaps cache length mismatch")
    if len(video_ids) != len(img_feats):
        raise RuntimeError("AVCaps metadata length mismatch")

    torch.save(img_feats, image_out)
    torch.save(aud_feats, audio_out)
    torch.save(txt_feats, text_out)

    meta = {
        **expected,
        "split_to_indices": split_to_indices,
        "num_pairs": int(len(img_feats)),
        "image_dim": int(img_feats.shape[1]),
        "audio_dim": int(aud_feats.shape[1]),
        "text_dim": int(txt_feats.shape[1]),
        "video_ids": video_ids,
        "decode_failures": int(decode_failures),
        "missing_videos": int(missing_videos),
    }
    save_json(meta, meta_out)
    print(f"Saved AVCaps cache to {out_dir}")
    return {"image": image_out, "audio": audio_out, "text": text_out, "meta": meta_out}
