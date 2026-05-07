from __future__ import annotations

import json
import math
import os
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..common import load_json, save_json
from .audiocaps import _decode_audio_with_ffmpeg, _resample_audio


def _extract_caption(ex: dict[str, Any]) -> str:
    for key in ["caption", "text", "description", "prompt", "title"]:
        v = ex.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    conv = ex.get("conversations")
    if isinstance(conv, list) and conv:
        for item in conv:
            if isinstance(item, dict):
                v = item.get("value")
                if isinstance(v, str) and v.strip():
                    return v.strip()
    return ""


def _extract_source(ex: dict[str, Any]) -> str:
    for key in ["source", "dataset", "subset", "category"]:
        v = ex.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return "unknown"


def _extract_audio_blob(ex: dict[str, Any]) -> dict[str, Any] | None:
    a = ex.get("audio")
    if isinstance(a, dict):
        if "bytes" in a or "path" in a:
            return a
        if "array" in a and "sampling_rate" in a:
            arr = np.asarray(a["array"], dtype=np.float32)
            sr = int(a["sampling_rate"])
            return {"array": arr, "sampling_rate": sr}
    return None


def _is_torchcodec_error(e: Exception) -> bool:
    msg = f"{type(e).__name__}: {e}".lower()
    return "torchcodec" in msg or "libtorchcodec" in msg


def extract_wavcaps_audio_text_cache(
    *,
    out_dir: Path,
    dataset_name: str,
    clap_model_name: str,
    clip_backbone_name: str,
    target_sampling_rate: int,
    max_examples: int,
    sampling_policy: str,
    device: str,
    audio_batch_size: int,
    text_batch_size: int,
    split_name: str = "train",
    stream: bool = True,
) -> dict[str, Path]:
    """
    Extract audio-text raw features from WavCaps-like HF datasets.

    The extractor is resilient to schema variation and emits a status JSON even on failure.
    """
    from transformers import AutoProcessor, CLIPModel, CLIPProcessor, ClapModel

    out_dir.mkdir(parents=True, exist_ok=True)
    audio_out = out_dir / "audio_feats_clap_raw.pt"
    text_out = out_dir / "text_feats_clip_raw.pt"
    meta_out = out_dir / "metadata.json"

    if audio_out.exists() and text_out.exists() and meta_out.exists():
        meta = load_json(meta_out)
        expected = {
            "dataset": dataset_name,
            "audio_encoder": clap_model_name,
            "text_encoder": clip_backbone_name,
            "target_sampling_rate": int(target_sampling_rate),
            "max_examples": int(max_examples),
            "sampling_policy": str(sampling_policy),
        }
        if all(meta.get(k) == v for k, v in expected.items()):
            return {"audio": audio_out, "text": text_out, "meta": meta_out}

    status = {
        "dataset": dataset_name,
        "audio_encoder": clap_model_name,
        "text_encoder": clip_backbone_name,
        "target_sampling_rate": int(target_sampling_rate),
        "max_examples": int(max_examples),
        "sampling_policy": str(sampling_policy),
    }

    clap_model = ClapModel.from_pretrained(clap_model_name).to(device).eval()
    clap_processor = AutoProcessor.from_pretrained(clap_model_name)
    clip_model = CLIPModel.from_pretrained(clip_backbone_name).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_backbone_name)

    all_audio_feats: list[torch.Tensor] = []
    all_text_feats: list[torch.Tensor] = []
    source_counts: dict[str, int] = {}
    sample_sources: list[str] = []
    sample_ids: list[str] = []

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

    def flush() -> None:
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

            t_in = clip_processor(
                text=batch_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)
            t_out = clip_model.text_model(
                input_ids=t_in["input_ids"],
                attention_mask=t_in.get("attention_mask"),
            )
            t_feat = t_out.pooler_output.cpu()

        all_audio_feats.append(a_feat)
        all_text_feats.append(t_feat)
        batch_audio.clear()
        batch_text.clear()

    accepted = 0
    n_seen = 0
    seen_sources: set[str] = set()
    decode_failures = 0

    def maybe_accept(*, blob: dict[str, Any] | None, ex: dict[str, Any]) -> None:
        nonlocal accepted, n_seen, decode_failures
        n_seen += 1
        if blob is None:
            return
        caption = _extract_caption(ex)
        if not caption:
            return

        source = _extract_source(ex)
        seen_sources.add(source)
        n_src = max(1, len(seen_sources))
        per_source_cap = int(math.ceil(max_examples / n_src)) if sampling_policy == "stratified" else max_examples
        src_count = source_counts.get(source, 0)
        if src_count >= per_source_cap:
            return

        try:
            if "array" in blob:
                wav = np.asarray(blob["array"], dtype=np.float32)
                sr = int(blob["sampling_rate"])
            else:
                wav, sr = _decode_audio_with_ffmpeg(blob, target_sampling_rate)
        except Exception:
            decode_failures += 1
            return

        wav48 = _resample_audio(wav, src_sr=sr, tgt_sr=target_sampling_rate)
        if wav48.size == 0:
            decode_failures += 1
            return

        batch_audio.append(wav48)
        batch_text.append(caption)
        sid = str(ex.get("id") or ex.get("filename") or ex.get("audio_id") or f"{source}:{n_seen}")
        sample_ids.append(sid)
        sample_sources.append(source)
        source_counts[source] = src_count + 1
        accepted += 1

        if len(batch_audio) >= min(audio_batch_size, text_batch_size):
            flush()

    def extract_via_datasets() -> None:
        from datasets import Audio, load_dataset

        ds = load_dataset(dataset_name, split=split_name, streaming=bool(stream))
        try:
            ds = ds.cast_column("audio", Audio(sampling_rate=None, decode=False))
        except Exception:
            pass

        for ex in ds:
            blob = _extract_audio_blob(ex)
            maybe_accept(blob=blob, ex=ex)
            if accepted >= max_examples:
                break

    def extract_via_tar_shards() -> None:
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi()
        files = api.list_repo_files(dataset_name, repo_type="dataset")
        tar_files = sorted([f for f in files if f.endswith(".tar")])
        if not tar_files:
            raise RuntimeError(f"No .tar shards found in dataset repo {dataset_name}")

        for tar_name in tar_files:
            tar_path = hf_hub_download(repo_id=dataset_name, repo_type="dataset", filename=tar_name)
            pending: dict[str, dict[str, Any]] = {}
            with tarfile.open(tar_path, "r") as tf:
                for member in tf:
                    if not member.isfile():
                        continue
                    base = os.path.basename(member.name)
                    stem, ext = os.path.splitext(base)
                    ext = ext.lower()
                    if ext not in {".json", ".flac", ".wav", ".mp3", ".ogg", ".m4a"}:
                        continue

                    fobj = tf.extractfile(member)
                    if fobj is None:
                        continue
                    raw = fobj.read()

                    rec = pending.setdefault(stem, {})
                    if ext == ".json":
                        try:
                            rec["meta"] = json.loads(raw.decode("utf-8"))
                        except Exception:
                            pending.pop(stem, None)
                            continue
                    else:
                        rec["audio_bytes"] = raw

                    if "meta" in rec and "audio_bytes" in rec:
                        meta = rec["meta"]
                        blob = {"bytes": rec["audio_bytes"]}
                        maybe_accept(blob=blob, ex=meta)
                        pending.pop(stem, None)
                        if accepted >= max_examples:
                            break
                if accepted >= max_examples:
                    break

    try:
        try:
            extract_via_datasets()
        except Exception as e:
            if _is_torchcodec_error(e):
                print("WavCaps datasets/torchcodec path unavailable; falling back to tar-shard extraction.")
                extract_via_tar_shards()
            else:
                raise

        flush()

        if not all_audio_feats or not all_text_feats:
            raise RuntimeError("No usable wavcaps examples decoded")

        audio_feats = torch.cat(all_audio_feats)
        text_feats = torch.cat(all_text_feats)
        if len(audio_feats) != len(text_feats):
            raise RuntimeError("WavCaps feature length mismatch")
        if len(sample_sources) != len(audio_feats) or len(sample_ids) != len(audio_feats):
            raise RuntimeError("WavCaps metadata/feature length mismatch")

        torch.save(audio_feats, audio_out)
        torch.save(text_feats, text_out)

        status.update(
            {
                "status": "ok",
                "num_pairs": int(len(audio_feats)),
                "audio_dim": int(audio_feats.shape[1]),
                "text_dim": int(text_feats.shape[1]),
                "source_counts": {k: int(v) for k, v in sorted(source_counts.items())},
                "sample_ids": sample_ids,
                "sample_sources": sample_sources,
                "n_seen_examples": int(n_seen),
                "n_decode_failures": int(decode_failures),
                "split_to_indices": {
                    "train": list(range(int(len(audio_feats)))),
                    "validation": [],
                    "test": [],
                },
            }
        )
        save_json(status, meta_out)
        return {"audio": audio_out, "text": text_out, "meta": meta_out}
    except Exception as e:
        status.update({"status": "failed", "error_type": type(e).__name__, "error": str(e)})
        save_json(status, meta_out)
        raise


def _list_wavcaps_tar_files(dataset_name: str) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(dataset_name, repo_type="dataset")
    tar_files = sorted([f for f in files if f.endswith(".tar")])
    if not tar_files:
        raise RuntimeError(f"No .tar shards found in dataset repo {dataset_name}")
    return tar_files


def build_wavcaps_audio_text_cache_tar_shard(
    *,
    shard_out_dir: Path,
    dataset_name: str,
    clap_model_name: str,
    clip_backbone_name: str,
    target_sampling_rate: int,
    max_examples: int,
    sampling_policy: str,
    device: str,
    audio_batch_size: int,
    text_batch_size: int,
    shard_index: int,
    shard_count: int,
) -> dict[str, Path]:
    """
    Build one deterministic shard of WavCaps cache directly from dataset tar files.
    This avoids torchcodec decode paths and enables multi-GPU parallel cache building.
    """
    from huggingface_hub import hf_hub_download
    from transformers import AutoProcessor, CLIPModel, CLIPProcessor, ClapModel

    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"Invalid shard_index={shard_index} for shard_count={shard_count}")

    shard_out_dir.mkdir(parents=True, exist_ok=True)
    audio_out = shard_out_dir / "audio_feats_clap_raw.pt"
    text_out = shard_out_dir / "text_feats_clip_raw.pt"
    meta_out = shard_out_dir / "metadata.json"

    expected = {
        "dataset": dataset_name,
        "audio_encoder": clap_model_name,
        "text_encoder": clip_backbone_name,
        "target_sampling_rate": int(target_sampling_rate),
        "max_examples": int(max_examples),
        "sampling_policy": str(sampling_policy),
        "shard_index": int(shard_index),
        "shard_count": int(shard_count),
    }
    if audio_out.exists() and text_out.exists() and meta_out.exists():
        meta = load_json(meta_out)
        if meta.get("status") == "ok" and all(meta.get(k) == v for k, v in expected.items()):
            return {"audio": audio_out, "text": text_out, "meta": meta_out}

    status = dict(expected)

    tar_files = _list_wavcaps_tar_files(dataset_name)
    assigned_tar_files = [name for i, name in enumerate(tar_files) if i % shard_count == shard_index]
    if not assigned_tar_files:
        raise RuntimeError(f"No tar files assigned to shard {shard_index}/{shard_count}")

    clap_model = ClapModel.from_pretrained(clap_model_name).to(device).eval()
    clap_processor = AutoProcessor.from_pretrained(clap_model_name)
    clip_model = CLIPModel.from_pretrained(clip_backbone_name).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_backbone_name)

    all_audio_feats: list[torch.Tensor] = []
    all_text_feats: list[torch.Tensor] = []
    sample_ids: list[str] = []
    sample_sources: list[str] = []
    source_counts: dict[str, int] = {}
    seen_sources: set[str] = set()

    batch_audio: list[np.ndarray] = []
    batch_text: list[str] = []
    batch_ids: list[str] = []
    batch_sources: list[str] = []

    accepted = 0
    n_seen = 0
    decode_failures = 0

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

    def flush() -> None:
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

            t_in = clip_processor(
                text=batch_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)
            t_out = clip_model.text_model(
                input_ids=t_in["input_ids"],
                attention_mask=t_in.get("attention_mask"),
            )
            t_feat = t_out.pooler_output.cpu()

        all_audio_feats.append(a_feat)
        all_text_feats.append(t_feat)
        sample_ids.extend(batch_ids)
        sample_sources.extend(batch_sources)
        batch_audio.clear()
        batch_text.clear()
        batch_ids.clear()
        batch_sources.clear()

    def maybe_accept(*, meta: dict[str, Any], audio_bytes: bytes) -> None:
        nonlocal accepted, n_seen, decode_failures
        n_seen += 1
        caption = _extract_caption(meta)
        if not caption:
            return

        source = _extract_source(meta)
        seen_sources.add(source)
        n_src = max(1, len(seen_sources))
        per_source_cap = int(math.ceil(max_examples / n_src)) if sampling_policy == "stratified" else max_examples
        src_count = source_counts.get(source, 0)
        if src_count >= per_source_cap:
            return

        sample_id = str(meta.get("id") or meta.get("filename") or f"{source}:{n_seen}")

        try:
            wav, sr = _decode_audio_with_ffmpeg({"bytes": audio_bytes}, target_sampling_rate)
        except Exception:
            decode_failures += 1
            return
        wav48 = _resample_audio(wav, src_sr=sr, tgt_sr=target_sampling_rate)
        if wav48.size == 0:
            decode_failures += 1
            return

        batch_audio.append(wav48)
        batch_text.append(caption)
        batch_ids.append(sample_id)
        batch_sources.append(source)
        source_counts[source] = src_count + 1
        accepted += 1
        if len(batch_audio) >= min(audio_batch_size, text_batch_size):
            flush()

    try:
        for tar_name in assigned_tar_files:
            tar_path = hf_hub_download(repo_id=dataset_name, repo_type="dataset", filename=tar_name)
            pending: dict[str, dict[str, Any]] = {}
            with tarfile.open(tar_path, "r") as tf:
                for member in tf:
                    if not member.isfile():
                        continue
                    base = os.path.basename(member.name)
                    stem, ext = os.path.splitext(base)
                    ext = ext.lower()
                    if ext not in {".json", ".flac", ".wav", ".mp3", ".ogg", ".m4a"}:
                        continue

                    fobj = tf.extractfile(member)
                    if fobj is None:
                        continue
                    raw = fobj.read()
                    rec = pending.setdefault(stem, {})
                    if ext == ".json":
                        try:
                            rec["meta"] = json.loads(raw.decode("utf-8"))
                        except Exception:
                            pending.pop(stem, None)
                            continue
                    else:
                        rec["audio_bytes"] = raw

                    if "meta" in rec and "audio_bytes" in rec:
                        maybe_accept(meta=rec["meta"], audio_bytes=rec["audio_bytes"])
                        pending.pop(stem, None)
                        if accepted >= max_examples:
                            break
                if accepted >= max_examples:
                    break
        flush()

        if not all_audio_feats or not all_text_feats:
            raise RuntimeError("No usable wavcaps examples decoded in shard")

        audio_feats = torch.cat(all_audio_feats)
        text_feats = torch.cat(all_text_feats)
        if len(audio_feats) != len(text_feats):
            raise RuntimeError("WavCaps shard feature length mismatch")
        if len(sample_ids) != len(audio_feats):
            raise RuntimeError("WavCaps shard metadata/feature length mismatch")

        torch.save(audio_feats, audio_out)
        torch.save(text_feats, text_out)

        status.update(
            {
                "status": "ok",
                "assigned_tar_files": assigned_tar_files,
                "num_pairs": int(len(audio_feats)),
                "audio_dim": int(audio_feats.shape[1]),
                "text_dim": int(text_feats.shape[1]),
                "source_counts": {k: int(v) for k, v in sorted(source_counts.items())},
                "n_seen_examples": int(n_seen),
                "n_decode_failures": int(decode_failures),
                "sample_ids": sample_ids,
                "sample_sources": sample_sources,
                "split_to_indices": {
                    "train": list(range(int(len(audio_feats)))),
                    "validation": [],
                    "test": [],
                },
            }
        )
        save_json(status, meta_out)
        return {"audio": audio_out, "text": text_out, "meta": meta_out}
    except Exception as e:
        status.update({"status": "failed", "error_type": type(e).__name__, "error": str(e)})
        save_json(status, meta_out)
        raise


def merge_wavcaps_audio_text_cache_shards(
    *,
    shard_dirs: list[Path],
    out_dir: Path,
    dataset_name: str,
    clap_model_name: str,
    clip_backbone_name: str,
    target_sampling_rate: int,
    max_examples: int,
    sampling_policy: str,
    merge_seed: int = 2026,
    include_sources: list[str] | None = None,
    compat_fields: dict | None = None,
) -> dict[str, Path]:
    """
    Deterministically merge shard caches into a single cache with fixed sampling policy.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_out = out_dir / "audio_feats_clap_raw.pt"
    text_out = out_dir / "text_feats_clip_raw.pt"
    meta_out = out_dir / "metadata.json"

    expected = {
        "dataset": dataset_name,
        "audio_encoder": clap_model_name,
        "text_encoder": clip_backbone_name,
        "target_sampling_rate": int(target_sampling_rate),
        "max_examples": int(max_examples),
        "sampling_policy": str(sampling_policy),
        "merge_seed": int(merge_seed),
        "include_sources": sorted(include_sources) if include_sources else None,
    }
    if audio_out.exists() and text_out.exists() and meta_out.exists():
        meta = load_json(meta_out)
        if meta.get("status") == "ok" and all(meta.get(k) == v for k, v in expected.items()):
            return {"audio": audio_out, "text": text_out, "meta": meta_out}

    status = dict(expected)
    rows: list[dict[str, Any]] = []
    shard_blocks: list[dict[str, Any]] = []
    total_decode_failures = 0
    total_seen = 0
    shard_meta_paths: list[str] = []

    for shard_dir in shard_dirs:
        shard_dir = Path(shard_dir).resolve()
        meta_p = shard_dir / "metadata.json"
        a_p = shard_dir / "audio_feats_clap_raw.pt"
        t_p = shard_dir / "text_feats_clip_raw.pt"
        if not meta_p.exists() or not a_p.exists() or not t_p.exists():
            raise FileNotFoundError(f"Missing shard artifacts under {shard_dir}")

        meta = load_json(meta_p)
        if meta.get("status") != "ok":
            raise RuntimeError(f"Shard not successful: {meta_p}")
        shard_meta_paths.append(str(meta_p))
        total_decode_failures += int(meta.get("n_decode_failures", 0))
        total_seen += int(meta.get("n_seen_examples", 0))

        audio = torch.load(a_p, map_location="cpu", weights_only=True)
        text = torch.load(t_p, map_location="cpu", weights_only=True)
        if len(audio) != len(text):
            raise RuntimeError(f"Feature length mismatch in shard {shard_dir}")

        sample_ids = meta.get("sample_ids") or [str(i) for i in range(len(audio))]
        sample_sources = meta.get("sample_sources") or ["unknown"] * len(audio)
        if len(sample_ids) != len(audio) or len(sample_sources) != len(audio):
            raise RuntimeError(f"Sample metadata length mismatch in shard {shard_dir}")

        shard_idx = len(shard_blocks)
        shard_blocks.append(
            {
                "audio": audio,
                "text": text,
                "sample_ids": sample_ids,
                "sample_sources": sample_sources,
            }
        )
        for i in range(len(audio)):
            rows.append(
                {
                    "source": str(sample_sources[i]),
                    "sample_id": str(sample_ids[i]),
                    "shard_idx": int(shard_idx),
                    "row_idx": int(i),
                }
            )

    if not rows:
        raise RuntimeError("No rows loaded from shard caches")

    if include_sources:
        include_set = set(include_sources)
        rows = [r for r in rows if r["source"] in include_set]
        if not rows:
            raise RuntimeError(f"No rows remaining after include_sources filtering: {include_sources}")

    rng = np.random.default_rng(int(merge_seed))
    target_n = min(int(max_examples), len(rows))

    if str(sampling_policy) == "stratified":
        source_to_idx: dict[str, list[int]] = {}
        for i, r in enumerate(rows):
            source_to_idx.setdefault(r["source"], []).append(i)
        selected: list[int] = []
        n_src = max(1, len(source_to_idx))
        cap = int(math.ceil(target_n / n_src))
        leftovers: list[int] = []
        for source in sorted(source_to_idx):
            idxs = source_to_idx[source]
            idxs = sorted(idxs, key=lambda k: rows[k]["sample_id"])
            order = rng.permutation(len(idxs))
            picked_local = [idxs[j] for j in order[: min(cap, len(idxs))]]
            picked_set = set(picked_local)
            selected.extend(picked_local)
            leftovers.extend([k for k in idxs if k not in picked_set])
        if len(selected) < target_n and leftovers:
            order = rng.permutation(len(leftovers))
            need = target_n - len(selected)
            selected.extend([leftovers[j] for j in order[:need]])
        if len(selected) > target_n:
            order = rng.permutation(len(selected))
            selected = [selected[j] for j in order[:target_n]]
    else:
        order = rng.permutation(len(rows))
        selected = [int(i) for i in order[:target_n]]

    selected_rows = [rows[i] for i in selected]
    selected_rows.sort(key=lambda r: (r["source"], r["sample_id"]))

    audio_feats = torch.stack(
        [shard_blocks[r["shard_idx"]]["audio"][r["row_idx"]] for r in selected_rows],
        dim=0,
    )
    text_feats = torch.stack(
        [shard_blocks[r["shard_idx"]]["text"][r["row_idx"]] for r in selected_rows],
        dim=0,
    )

    if len(audio_feats) != len(text_feats):
        raise RuntimeError("Merged feature length mismatch")

    torch.save(audio_feats, audio_out)
    torch.save(text_feats, text_out)

    source_counts: dict[str, int] = {}
    for r in selected_rows:
        source_counts[r["source"]] = source_counts.get(r["source"], 0) + 1

    status.update(
        {
            "status": "ok",
            "num_pairs": int(len(audio_feats)),
            "audio_dim": int(audio_feats.shape[1]),
            "text_dim": int(text_feats.shape[1]),
            "source_counts": {k: int(v) for k, v in sorted(source_counts.items())},
            "n_seen_examples": int(total_seen),
            "n_decode_failures": int(total_decode_failures),
            "sample_ids": [r["sample_id"] for r in selected_rows],
            "sample_sources": [r["source"] for r in selected_rows],
            "merged_from_shards": shard_meta_paths,
            "split_to_indices": {
                "train": list(range(int(len(audio_feats)))),
                "validation": [],
                "test": [],
            },
        }
    )
    if compat_fields:
        status.update(compat_fields)
    save_json(status, meta_out)
    return {"audio": audio_out, "text": text_out, "meta": meta_out}
