from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from ..common import load_json, save_json
from .audiocaps import _decode_audio_with_ffmpeg, _resample_audio


def _coerce_text(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    return s


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


def _perm_indices(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    idx = np.arange(int(n), dtype=np.int64)
    rng.shuffle(idx)
    return idx


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


def extract_speechcoco_av_cache(
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
    train_max_examples: int,
    phase_b_val_examples: int,
    eval_test_examples: int,
    sample_seed: int,
    eval_seed: int,
    phase_b_split_seed: int,
    strict_disjoint: bool,
    hf_cache_dir: str | None = None,
    force_rebuild: bool = False,
) -> dict[str, Path]:
    """
    Build image-audio-text cache for SpeechCoco with deterministic splits.

    Split protocol:
    - phase_b_train / phase_b_val sampled from HF train split
    - eval_test sampled from HF validation split
    - If strict_disjoint=True, image_ids in eval_test are excluded from phase_b_train/phase_b_val.
    """
    from datasets import Audio, Image as HFImage, load_dataset
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
        "train_max_examples": int(train_max_examples),
        "phase_b_val_examples": int(phase_b_val_examples),
        "eval_test_examples": int(eval_test_examples),
        "sample_seed": int(sample_seed),
        "eval_seed": int(eval_seed),
        "phase_b_split_seed": int(phase_b_split_seed),
        "strict_disjoint": bool(strict_disjoint),
    }

    if (not force_rebuild) and image_out.exists() and audio_out.exists() and text_out.exists() and meta_out.exists():
        try:
            meta = load_json(meta_out)
            if all(meta.get(k) == v for k, v in expected.items()):
                return {"image": image_out, "audio": audio_out, "text": text_out, "meta": meta_out}
        except Exception:
            pass

    clap_model = ClapModel.from_pretrained(clap_model_name).to(device).eval()
    clap_processor = AutoProcessor.from_pretrained(clap_model_name)

    clip_model = CLIPModel.from_pretrained(clip_backbone_name).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_backbone_name)

    train_ds = load_dataset(dataset_name, split="train", cache_dir=hf_cache_dir)
    val_ds = load_dataset(dataset_name, split="validation", cache_dir=hf_cache_dir)

    train_ds = train_ds.cast_column("audio", Audio(sampling_rate=None, decode=False))
    val_ds = val_ds.cast_column("audio", Audio(sampling_rate=None, decode=False))
    train_ds = train_ds.cast_column("image", HFImage(decode=False))
    val_ds = val_ds.cast_column("image", HFImage(decode=False))

    n_train = len(train_ds)
    n_val = len(val_ds)

    if train_max_examples <= phase_b_val_examples:
        raise ValueError("train_max_examples must be > phase_b_val_examples")

    target_train_pool = int(train_max_examples)
    pool_slack = max(2048, int(0.10 * target_train_pool))
    target_train_candidates = target_train_pool + pool_slack
    target_eval = int(eval_test_examples)

    all_img_feats: list[torch.Tensor] = []
    all_aud_feats: list[torch.Tensor] = []
    all_txt_feats: list[torch.Tensor] = []

    split_to_indices: dict[str, list[int]] = {
        "phase_b_train": [],
        "phase_b_val": [],
        "eval_test": [],
    }

    image_ids_all: list[int] = []
    row_sources: list[str] = []
    row_source_indices: list[int] = []

    decode_failures = 0
    empty_audio = 0
    dropped_empty_text = 0
    skipped_disjoint_train = 0

    batch_images: list[Image.Image] = []
    batch_audio: list[np.ndarray] = []
    batch_text: list[str] = []
    batch_target: list[str] = []
    batch_image_ids: list[int] = []
    batch_src: list[str] = []
    batch_src_idx: list[int] = []

    next_idx = 0

    def flush_batch() -> None:
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

        if not (len(a_feat) == len(i_feat) == len(t_feat) == len(batch_target)):
            raise RuntimeError("SpeechCoco batch feature count mismatch")

        all_aud_feats.append(a_feat)
        all_img_feats.append(i_feat)
        all_txt_feats.append(t_feat)

        for j in range(len(batch_target)):
            split_to_indices[batch_target[j]].append(next_idx + j)
            image_ids_all.append(int(batch_image_ids[j]))
            row_sources.append(str(batch_src[j]))
            row_source_indices.append(int(batch_src_idx[j]))

        next_idx += len(batch_target)

        batch_images.clear()
        batch_audio.clear()
        batch_text.clear()
        batch_target.clear()
        batch_image_ids.clear()
        batch_src.clear()
        batch_src_idx.clear()

    # -------------------------
    # 1) Build eval_test first
    # -------------------------
    eval_perm = _perm_indices(n_val, eval_seed)
    eval_image_ids: set[int] = set()

    for ridx in eval_perm.tolist():
        if len(split_to_indices["eval_test"]) >= target_eval:
            break
        ex = val_ds[int(ridx)]
        txt = _coerce_text(ex.get("text"))
        if not txt:
            dropped_empty_text += 1
            continue
        try:
            img = _decode_image_blob(ex["image"])
            wav, sr = _decode_audio_with_ffmpeg(ex["audio"], target_sampling_rate)
            wav = _resample_audio(wav, src_sr=int(sr), tgt_sr=int(target_sampling_rate))
            if wav.size == 0:
                empty_audio += 1
                continue
            image_id = int(ex.get("image_id"))
        except Exception:
            decode_failures += 1
            continue

        batch_images.append(img)
        batch_audio.append(wav)
        batch_text.append(txt)
        batch_target.append("eval_test")
        batch_image_ids.append(image_id)
        batch_src.append("validation")
        batch_src_idx.append(int(ridx))
        eval_image_ids.add(image_id)

        if len(batch_audio) >= min(audio_batch_size, image_batch_size, text_batch_size):
            flush_batch()

    flush_batch()

    if len(split_to_indices["eval_test"]) < target_eval:
        raise RuntimeError(
            f"SpeechCoco eval_test underfilled: got {len(split_to_indices['eval_test'])}, need {target_eval}"
        )

    # ----------------------------------------
    # 2) Build phase_b pool from train split
    # ----------------------------------------
    train_perm = _perm_indices(n_train, sample_seed)

    phase_b_pool_indices: list[int] = []

    for ridx in train_perm.tolist():
        if len(phase_b_pool_indices) >= target_train_candidates:
            break
        ex = train_ds[int(ridx)]
        txt = _coerce_text(ex.get("text"))
        if not txt:
            dropped_empty_text += 1
            continue

        image_id = int(ex.get("image_id"))
        if strict_disjoint and image_id in eval_image_ids:
            skipped_disjoint_train += 1
            continue

        # lightweight validity checks before expensive feature encode
        aud = ex.get("audio")
        img = ex.get("image")
        if not isinstance(aud, dict) or not isinstance(img, dict):
            continue

        phase_b_pool_indices.append(int(ridx))

    if len(phase_b_pool_indices) < target_train_pool:
        raise RuntimeError(
            f"SpeechCoco phase_b pool underfilled: got {len(phase_b_pool_indices)}, need {target_train_pool}"
        )

    # deterministic train/val split over pool with reserve candidates for decode failures
    rng_split = np.random.default_rng(int(phase_b_split_seed))
    perm_local = np.arange(len(phase_b_pool_indices), dtype=np.int64)
    rng_split.shuffle(perm_local)

    n_val_target = int(phase_b_val_examples)
    n_train_target = int(target_train_pool - n_val_target)
    if n_train_target <= 0:
        raise RuntimeError("Invalid split: phase_b_train target must be positive")

    core = perm_local[:target_train_pool]
    reserve = perm_local[target_train_pool:]

    train_core_local = set(core[:n_train_target].tolist())
    val_core_local = set(core[n_train_target:].tolist())

    train_ok = 0
    val_ok = 0
    reserve_used = 0

    def _try_append(local_idx: int, split_name: str) -> bool:
        nonlocal decode_failures, empty_audio, dropped_empty_text
        ridx = int(phase_b_pool_indices[int(local_idx)])
        ex = train_ds[ridx]
        txt = _coerce_text(ex.get("text"))
        if not txt:
            dropped_empty_text += 1
            return False
        try:
            img = _decode_image_blob(ex["image"])
            wav, sr = _decode_audio_with_ffmpeg(ex["audio"], target_sampling_rate)
            wav = _resample_audio(wav, src_sr=int(sr), tgt_sr=int(target_sampling_rate))
            if wav.size == 0:
                empty_audio += 1
                return False
            image_id = int(ex.get("image_id"))
        except Exception:
            decode_failures += 1
            return False

        batch_images.append(img)
        batch_audio.append(wav)
        batch_text.append(txt)
        batch_target.append(split_name)
        batch_image_ids.append(image_id)
        batch_src.append("train")
        batch_src_idx.append(ridx)
        if len(batch_audio) >= min(audio_batch_size, image_batch_size, text_batch_size):
            flush_batch()
        return True

    # Process core assignments first.
    for local_idx in core.tolist():
        if local_idx in train_core_local:
            if train_ok >= n_train_target:
                continue
            if _try_append(local_idx, "phase_b_train"):
                train_ok += 1
        elif local_idx in val_core_local:
            if val_ok >= n_val_target:
                continue
            if _try_append(local_idx, "phase_b_val"):
                val_ok += 1

    # Use reserve candidates to backfill decode failures deterministically.
    for local_idx in reserve.tolist():
        if train_ok >= n_train_target and val_ok >= n_val_target:
            break
        split_name = "phase_b_train" if train_ok < n_train_target else "phase_b_val"
        reserve_used += 1
        if _try_append(local_idx, split_name):
            if split_name == "phase_b_train":
                train_ok += 1
            else:
                val_ok += 1

    flush_batch()

    if val_ok < n_val_target:
        raise RuntimeError(
            f"SpeechCoco phase_b_val underfilled after decode+reserve: got {val_ok}, need {n_val_target}"
        )
    if train_ok < n_train_target:
        raise RuntimeError(
            f"SpeechCoco phase_b_train underfilled after decode+reserve: got {train_ok}, need {n_train_target}"
        )

    img_feats = torch.cat(all_img_feats, dim=0)
    aud_feats = torch.cat(all_aud_feats, dim=0)
    txt_feats = torch.cat(all_txt_feats, dim=0)

    if len(img_feats) != len(aud_feats) or len(img_feats) != len(txt_feats):
        raise RuntimeError("SpeechCoco cache feature length mismatch")
    if len(image_ids_all) != len(img_feats):
        raise RuntimeError("SpeechCoco image_ids length mismatch")

    # Strict disjoint diagnostics from finalized splits
    train_split_ids = {image_ids_all[i] for i in split_to_indices["phase_b_train"]}
    val_split_ids = {image_ids_all[i] for i in split_to_indices["phase_b_val"]}
    eval_split_ids = {image_ids_all[i] for i in split_to_indices["eval_test"]}

    overlap_train_eval = sorted(train_split_ids.intersection(eval_split_ids))
    overlap_val_eval = sorted(val_split_ids.intersection(eval_split_ids))

    if strict_disjoint and (overlap_train_eval or overlap_val_eval):
        raise RuntimeError(
            "SpeechCoco strict_disjoint violated in finalized splits: "
            f"train_eval_overlap={len(overlap_train_eval)} val_eval_overlap={len(overlap_val_eval)}"
        )

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
        "image_ids": [int(x) for x in image_ids_all],
        "row_sources": row_sources,
        "row_source_indices": [int(x) for x in row_source_indices],
        "split_counts": {k: int(len(v)) for k, v in split_to_indices.items()},
        "overlap_train_eval_count": int(len(overlap_train_eval)),
        "overlap_val_eval_count": int(len(overlap_val_eval)),
        "decode_failures": int(decode_failures),
        "empty_audio": int(empty_audio),
        "dropped_empty_text": int(dropped_empty_text),
        "skipped_disjoint_train": int(skipped_disjoint_train),
        "phase_b_pool_target": int(target_train_pool),
        "phase_b_pool_candidates": int(len(phase_b_pool_indices)),
        "phase_b_pool_slack": int(pool_slack),
        "phase_b_reserve_used": int(reserve_used),
        "hf_split_sizes": {"train": int(n_train), "validation": int(n_val)},
    }
    save_json(meta, meta_out)
    return {"image": image_out, "audio": audio_out, "text": text_out, "meta": meta_out}
