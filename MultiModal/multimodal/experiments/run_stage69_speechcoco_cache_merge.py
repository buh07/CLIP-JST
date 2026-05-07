from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from ..common import env_snapshot, load_json, mark_done, save_json


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _select_row_ids(
    *,
    manifest_rows_by_id: dict[int, dict[str, Any]],
    available_row_ids: set[int],
    role_core: str,
    role_reserve: str,
    target_count: int,
) -> list[int]:
    core = sorted(
        rid
        for rid, row in manifest_rows_by_id.items()
        if row.get("role") == role_core and rid in available_row_ids
    )
    reserve = sorted(
        rid
        for rid, row in manifest_rows_by_id.items()
        if row.get("role") == role_reserve and rid in available_row_ids
    )
    selected = core[:target_count]
    if len(selected) < target_count:
        need = target_count - len(selected)
        selected.extend(reserve[:need])
    if len(selected) < target_count:
        raise RuntimeError(
            f"Underfilled split for roles ({role_core},{role_reserve}): "
            f"need={target_count} got={len(selected)}"
        )
    return selected


def run(cfg: dict[str, Any]) -> None:
    start = time.time()
    stage_name = "stage69_speechcoco_cache_merge"

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

    shard_root = Path(cfg.get("shard_root", output_root / "stage69_speechcoco_cache_shards")).resolve()
    shard_count = int(cfg["shard_count"])
    if shard_count <= 0:
        raise ValueError("shard_count must be > 0")

    cache_root = Path(cfg.get("speechcoco_cache_root", output_root / "caches" / "speechcoco_av")).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    image_out = cache_root / "image_feats_clip_raw.pt"
    audio_out = cache_root / "audio_feats_clap_raw.pt"
    text_out = cache_root / "text_feats_clip_raw.pt"
    meta_out = cache_root / "metadata.json"

    rows = manifest["rows"]
    rows_by_id: dict[int, dict[str, Any]] = {int(r["row_id"]): r for r in rows}
    targets = manifest["targets"]

    # Map each successful row_id to (shard_dir, local_index).
    row_to_loc: dict[int, tuple[Path, int]] = {}
    shard_stats: list[dict[str, Any]] = []
    total_decode_failures = 0
    total_empty_audio = 0
    total_dropped_empty_text = 0

    for i in range(shard_count):
        sdir = shard_root / f"shard{i}"
        row_ids_path = sdir / "row_ids_success.json"
        meta_path = sdir / "shard_meta.json"
        img_path = sdir / "image_feats.pt"
        aud_path = sdir / "audio_feats.pt"
        txt_path = sdir / "text_feats.pt"
        if not (row_ids_path.exists() and meta_path.exists() and img_path.exists() and aud_path.exists() and txt_path.exists()):
            raise FileNotFoundError(f"Incomplete shard outputs at {sdir}")

        row_ids = [int(x) for x in load_json(row_ids_path)]
        s_meta = load_json(meta_path)
        shard_stats.append(s_meta)
        total_decode_failures += int(s_meta.get("decode_failures", 0))
        total_empty_audio += int(s_meta.get("empty_audio", 0))
        total_dropped_empty_text += int(s_meta.get("dropped_empty_text", 0))

        img = torch.load(img_path, map_location="cpu", weights_only=True)
        aud = torch.load(aud_path, map_location="cpu", weights_only=True)
        txt = torch.load(txt_path, map_location="cpu", weights_only=True)
        if not (len(img) == len(aud) == len(txt) == len(row_ids)):
            raise RuntimeError(f"Shard tensor length mismatch at {sdir}")

        for local_idx, rid in enumerate(row_ids):
            if rid in row_to_loc:
                raise RuntimeError(f"Duplicate row_id across shards: {rid}")
            row_to_loc[rid] = (sdir, local_idx)

    available_row_ids = set(row_to_loc.keys())

    selected_eval = _select_row_ids(
        manifest_rows_by_id=rows_by_id,
        available_row_ids=available_row_ids,
        role_core="eval_test_core",
        role_reserve="eval_test_reserve",
        target_count=int(targets["eval_test"]),
    )
    selected_train = _select_row_ids(
        manifest_rows_by_id=rows_by_id,
        available_row_ids=available_row_ids,
        role_core="phase_b_train_core",
        role_reserve="phase_b_train_reserve",
        target_count=int(targets["phase_b_train"]),
    )
    selected_val = _select_row_ids(
        manifest_rows_by_id=rows_by_id,
        available_row_ids=available_row_ids,
        role_core="phase_b_val_core",
        role_reserve="phase_b_val_reserve",
        target_count=int(targets["phase_b_val"]),
    )

    selected_by_split = {
        "phase_b_train": selected_train,
        "phase_b_val": selected_val,
        "eval_test": selected_eval,
    }
    final_order = selected_train + selected_val + selected_eval

    # Cache loaded shard tensors to avoid repeated disk reads.
    shard_tensor_cache: dict[Path, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    image_rows: list[torch.Tensor] = []
    audio_rows: list[torch.Tensor] = []
    text_rows: list[torch.Tensor] = []
    image_ids: list[int] = []
    row_sources: list[str] = []
    row_source_indices: list[int] = []

    for rid in final_order:
        sdir, local_idx = row_to_loc[rid]
        if sdir not in shard_tensor_cache:
            shard_tensor_cache[sdir] = (
                torch.load(sdir / "image_feats.pt", map_location="cpu", weights_only=True),
                torch.load(sdir / "audio_feats.pt", map_location="cpu", weights_only=True),
                torch.load(sdir / "text_feats.pt", map_location="cpu", weights_only=True),
            )
        img_t, aud_t, txt_t = shard_tensor_cache[sdir]
        image_rows.append(img_t[local_idx])
        audio_rows.append(aud_t[local_idx])
        text_rows.append(txt_t[local_idx])

        row = rows_by_id[rid]
        image_ids.append(int(row["image_id"]))
        row_sources.append(str(row["source_split"]))
        row_source_indices.append(int(row["source_index"]))

    image_feats = torch.stack(image_rows, dim=0)
    audio_feats = torch.stack(audio_rows, dim=0)
    text_feats = torch.stack(text_rows, dim=0)
    if not (len(image_feats) == len(audio_feats) == len(text_feats)):
        raise RuntimeError("Merged tensor length mismatch")

    split_to_indices: dict[str, list[int]] = {
        "phase_b_train": list(range(0, len(selected_train))),
        "phase_b_val": list(range(len(selected_train), len(selected_train) + len(selected_val))),
        "eval_test": list(range(len(selected_train) + len(selected_val), len(final_order))),
    }

    train_ids = {image_ids[i] for i in split_to_indices["phase_b_train"]}
    val_ids = {image_ids[i] for i in split_to_indices["phase_b_val"]}
    eval_ids = {image_ids[i] for i in split_to_indices["eval_test"]}
    overlap_train_eval = sorted(train_ids.intersection(eval_ids))
    overlap_val_eval = sorted(val_ids.intersection(eval_ids))
    strict_disjoint = bool(manifest.get("strict_disjoint", True))
    if strict_disjoint and (overlap_train_eval or overlap_val_eval):
        raise RuntimeError(
            "Strict disjoint violated after merge: "
            f"train_eval={len(overlap_train_eval)} val_eval={len(overlap_val_eval)}"
        )

    torch.save(image_feats, image_out)
    torch.save(audio_feats, audio_out)
    torch.save(text_feats, text_out)

    meta = {
        "dataset": manifest["dataset"],
        "audio_encoder": str(cfg["clap_model"]),
        "image_text_encoder": str(cfg["clip_backbone"]),
        "target_sampling_rate": int(cfg.get("speechcoco_target_sr", 48_000)),
        "train_max_examples": int(manifest["speechcoco_train_max_examples"]),
        "phase_b_val_examples": int(manifest["speechcoco_phase_b_val_examples"]),
        "eval_test_examples": int(manifest["speechcoco_eval_test_examples"]),
        "sample_seed": int(manifest["speechcoco_sample_seed"]),
        "eval_seed": int(manifest["speechcoco_eval_seed"]),
        "phase_b_split_seed": int(manifest["phase_b_split_seed"]),
        "strict_disjoint": strict_disjoint,
        "split_to_indices": split_to_indices,
        "num_pairs": int(len(image_feats)),
        "image_dim": int(image_feats.shape[1]),
        "audio_dim": int(audio_feats.shape[1]),
        "text_dim": int(text_feats.shape[1]),
        "image_ids": [int(x) for x in image_ids],
        "row_sources": row_sources,
        "row_source_indices": [int(x) for x in row_source_indices],
        "split_counts": {k: int(len(v)) for k, v in split_to_indices.items()},
        "overlap_train_eval_count": int(len(overlap_train_eval)),
        "overlap_val_eval_count": int(len(overlap_val_eval)),
        "decode_failures": int(total_decode_failures),
        "empty_audio": int(total_empty_audio),
        "dropped_empty_text": int(total_dropped_empty_text),
        "manifest_path": str(manifest_path),
        "manifest_rows_sha256": str(manifest["rows_sha256"]),
        "manifest_row_count": int(len(rows)),
        "selected_row_count": int(len(final_order)),
        "shard_count": shard_count,
        "phase_b_reserve_used_train": int(max(0, len(selected_train) - len([r for r in rows if r["role"] == "phase_b_train_core"]))),
        "phase_b_reserve_used_val": int(max(0, len(selected_val) - len([r for r in rows if r["role"] == "phase_b_val_core"]))),
        "eval_reserve_used": int(max(0, len(selected_eval) - len([r for r in rows if r["role"] == "eval_test_core"]))),
    }
    save_json(meta, meta_out)

    merge_report = {
        "stage": stage_name,
        "manifest_path": str(manifest_path),
        "shard_root": str(shard_root),
        "shard_count": shard_count,
        "selected_counts": {
            "phase_b_train": int(len(selected_train)),
            "phase_b_val": int(len(selected_val)),
            "eval_test": int(len(selected_eval)),
            "total": int(len(final_order)),
        },
        "decode_failures_total": int(total_decode_failures),
        "empty_audio_total": int(total_empty_audio),
        "dropped_empty_text_total": int(total_dropped_empty_text),
        "cache_files": {
            "image": str(image_out),
            "audio": str(audio_out),
            "text": str(text_out),
            "meta": str(meta_out),
            "image_sha256": _sha256_file(image_out),
            "audio_sha256": _sha256_file(audio_out),
            "text_sha256": _sha256_file(text_out),
            "meta_sha256": _sha256_file(meta_out),
        },
        "elapsed_sec": float(time.time() - start),
        "shards": shard_stats,
    }
    save_json(merge_report, output_root / stage_name / f"{stage_name}_report.json")

    provenance = env_snapshot(
        Path(cfg["project_root"]),
        seeds=[],
        extra={
            "stage": stage_name,
            "elapsed_sec": float(time.time() - start),
            "manifest_path": str(manifest_path),
            "shard_root": str(shard_root),
            "shard_count": shard_count,
        },
    )
    save_json(provenance, output_root / stage_name / f"provenance_{stage_name}.json")

    mark_done(
        markers / "stage69_cache_ready.done.json",
        {
            "cache_root": str(cache_root),
            "meta_path": str(meta_out),
            "elapsed_sec": float(time.time() - start),
        },
    )
    # Plain sentinel for legacy waiters.
    (markers / "stage69_cache_ready.done").touch()
    print(f"{stage_name} complete: cache_root={cache_root}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)

