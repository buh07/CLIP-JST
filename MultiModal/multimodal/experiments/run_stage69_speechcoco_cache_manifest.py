from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from ..common import env_snapshot, mark_done, save_json


def _coerce_text(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _topk_random_candidates(
    ds,
    *,
    k: int,
    seed: int,
    build_candidate: Callable[[int, dict[str, Any]], dict[str, Any] | None],
) -> list[dict[str, Any]]:
    """
    Deterministically select k random valid candidates while scanning sequentially.

    We avoid permutation-driven random indexing (slow on parquet-backed datasets)
    and instead perform a single sequential pass with random scores.
    """
    if k <= 0:
        return []
    rng = np.random.default_rng(int(seed))
    heap: list[tuple[int, int, dict[str, Any]]] = []
    n = len(ds)
    for idx in range(n):
        ex = ds[idx]
        cand = build_candidate(idx, ex)
        if cand is None:
            continue
        score = int(rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64))
        item = (-score, -idx, cand)
        if len(heap) < k:
            heapq.heappush(heap, item)
        else:
            root = heap[0]
            if item[0] > root[0] or (item[0] == root[0] and item[1] > root[1]):
                heapq.heapreplace(heap, item)
    # Return deterministic order by increasing random score, then index.
    heap.sort(key=lambda t: (-t[0], -t[1]))
    return [t[2] for t in heap]


def _safe_image_id(ex: dict[str, Any]) -> int | None:
    v = ex.get("image_id")
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _is_lightweight_candidate(ex: dict[str, Any]) -> bool:
    # Manifest stage intentionally validates only lightweight fields.
    # Heavy media validity is checked during shard encoding, with reserves for backfill.
    if not _coerce_text(ex.get("text")):
        return False
    if _safe_image_id(ex) is None:
        return False
    return True


def _load_split(dataset_name: str, split: str, hf_cache_dir: str | None, token: str | None):
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": split}
    if hf_cache_dir:
        kwargs["cache_dir"] = hf_cache_dir
    if token:
        kwargs["token"] = token

    ds = load_dataset(dataset_name, **kwargs)
    keep_cols = [c for c in ["image_id", "text"] if c in ds.column_names]
    if keep_cols:
        ds = ds.select_columns(keep_cols)
    return ds


def _rows_hash(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run(cfg: dict[str, Any]) -> None:
    start = time.time()
    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stage_root = output_root / "stage69_speechcoco_cache_manifest"
    stage_root.mkdir(parents=True, exist_ok=True)
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    dataset_name = str(cfg.get("speechcoco_dataset", "mteb/SpeechCoco"))
    hf_cache_dir = cfg.get("speechcoco_hf_cache_dir")
    hf_token = cfg.get("hf_token") or None
    if not hf_token:
        # Prefer env token to avoid writing secrets to config files.
        import os

        hf_token = os.environ.get("HF_TOKEN")

    train_max = int(cfg.get("speechcoco_train_max_examples", 120_000))
    val_max = int(cfg.get("speechcoco_phase_b_val_examples", 20_000))
    eval_max = int(cfg.get("speechcoco_eval_test_examples", 5_000))
    sample_seed = int(cfg.get("speechcoco_sample_seed", 69001))
    eval_seed = int(cfg.get("speechcoco_eval_seed", 69002))
    split_seed = int(cfg.get("phase_b_split_seed", 69003))
    strict_disjoint = bool(cfg.get("strict_disjoint", True))

    if train_max <= val_max:
        raise ValueError("speechcoco_train_max_examples must be > speechcoco_phase_b_val_examples")

    eval_reserve = int(cfg.get("speechcoco_eval_reserve_examples", max(512, int(0.20 * eval_max))))
    phaseb_slack = int(cfg.get("speechcoco_phase_b_slack_examples", max(4096, int(0.15 * train_max))))

    train_ds = _load_split(dataset_name, "train", hf_cache_dir=hf_cache_dir, token=hf_token)
    val_ds = _load_split(dataset_name, "validation", hf_cache_dir=hf_cache_dir, token=hf_token)

    n_train = len(train_ds)
    n_val = len(val_ds)

    rows: list[dict[str, Any]] = []
    dropped = {
        "eval_invalid": 0,
        "train_invalid": 0,
        "train_disjoint_skipped": 0,
    }

    # 1) Eval candidates from validation split (single sequential pass).
    eval_need = eval_max + eval_reserve

    def _build_eval_candidate(ridx: int, ex: dict[str, Any]) -> dict[str, Any] | None:
        if not _is_lightweight_candidate(ex):
            dropped["eval_invalid"] += 1
            return None
        iid = _safe_image_id(ex)
        assert iid is not None
        return {
            "source_split": "validation",
            "source_index": int(ridx),
            "image_id": int(iid),
        }

    eval_cands = _topk_random_candidates(
        val_ds,
        k=eval_need,
        seed=eval_seed,
        build_candidate=_build_eval_candidate,
    )
    eval_image_ids: set[int] = {int(x["image_id"]) for x in eval_cands}

    if len(eval_cands) < eval_need:
        raise RuntimeError(f"Underfilled eval candidates: got {len(eval_cands)}, need {eval_need}")

    eval_core = eval_cands[:eval_max]
    eval_res = eval_cands[eval_max:eval_need]

    # 2) Phase-B pool candidates from train split (single sequential pass).
    phaseb_need = train_max + phaseb_slack

    def _build_train_candidate(ridx: int, ex: dict[str, Any]) -> dict[str, Any] | None:
        if not _is_lightweight_candidate(ex):
            dropped["train_invalid"] += 1
            return None
        iid = _safe_image_id(ex)
        assert iid is not None
        if strict_disjoint and int(iid) in eval_image_ids:
            dropped["train_disjoint_skipped"] += 1
            return None
        return {
            "source_split": "train",
            "source_index": int(ridx),
            "image_id": int(iid),
        }

    phaseb_cands = _topk_random_candidates(
        train_ds,
        k=phaseb_need,
        seed=sample_seed,
        build_candidate=_build_train_candidate,
    )

    if len(phaseb_cands) < train_max:
        raise RuntimeError(
            f"Underfilled phase-B candidates: got {len(phaseb_cands)}, need at least {train_max}"
        )

    core = phaseb_cands[:train_max]
    reserve = phaseb_cands[train_max:]

    n_train_target = train_max - val_max
    n_val_target = val_max
    if n_train_target <= 0:
        raise RuntimeError("Computed phase_b_train target <= 0")

    rng_split = np.random.default_rng(split_seed)
    perm = np.arange(len(core), dtype=np.int64)
    rng_split.shuffle(perm)
    tr_core_set = set(perm[:n_train_target].tolist())

    row_id = 0
    split_to_row_ids: dict[str, list[int]] = {
        "eval_test_core": [],
        "eval_test_reserve": [],
        "phase_b_train_core": [],
        "phase_b_val_core": [],
        "phase_b_train_reserve": [],
        "phase_b_val_reserve": [],
    }

    def _append(entry: dict[str, Any], target_split: str, role: str) -> None:
        nonlocal row_id
        rows.append(
            {
                "row_id": int(row_id),
                "source_split": str(entry["source_split"]),
                "source_index": int(entry["source_index"]),
                "image_id": int(entry["image_id"]),
                "target_split": str(target_split),
                "role": str(role),
            }
        )
        split_to_row_ids[role].append(int(row_id))
        row_id += 1

    for e in eval_core:
        _append(e, "eval_test", "eval_test_core")
    for e in eval_res:
        _append(e, "eval_test", "eval_test_reserve")

    for local_i, e in enumerate(core):
        if local_i in tr_core_set:
            _append(e, "phase_b_train", "phase_b_train_core")
        else:
            _append(e, "phase_b_val", "phase_b_val_core")

    # Deterministically assign reserve rows half/half by current deficit parity.
    tr_res = 0
    va_res = 0
    for e in reserve:
        assign_train = tr_res <= va_res
        if assign_train:
            _append(e, "phase_b_train", "phase_b_train_reserve")
            tr_res += 1
        else:
            _append(e, "phase_b_val", "phase_b_val_reserve")
            va_res += 1

    manifest = {
        "stage": "stage69_speechcoco_cache_manifest",
        "dataset": dataset_name,
        "manifest_version": 1,
        "strict_disjoint": strict_disjoint,
        "speechcoco_train_max_examples": train_max,
        "speechcoco_phase_b_val_examples": val_max,
        "speechcoco_eval_test_examples": eval_max,
        "speechcoco_eval_reserve_examples": eval_reserve,
        "speechcoco_phase_b_slack_examples": phaseb_slack,
        "speechcoco_sample_seed": sample_seed,
        "speechcoco_eval_seed": eval_seed,
        "phase_b_split_seed": split_seed,
        "targets": {
            "eval_test": eval_max,
            "phase_b_train": n_train_target,
            "phase_b_val": n_val_target,
        },
        "source_sizes": {
            "train": n_train,
            "validation": n_val,
        },
        "dropped": dropped,
        "split_to_row_ids": split_to_row_ids,
        "rows": rows,
        "rows_sha256": _rows_hash(rows),
    }

    out_json = stage_root / "stage69_speechcoco_manifest.json"
    save_json(manifest, out_json)

    provenance = env_snapshot(
        Path(cfg["project_root"]),
        seeds=[],
        extra={
            "stage": "stage69_speechcoco_cache_manifest",
            "elapsed_sec": float(time.time() - start),
            "dataset": dataset_name,
            "rows": int(len(rows)),
        },
    )
    save_json(provenance, stage_root / "provenance_stage69_cache_manifest.json")
    mark_done(
        markers / "stage69_cache_manifest.done.json",
        {
            "elapsed_sec": float(time.time() - start),
            "manifest_path": str(out_json),
            "rows": int(len(rows)),
        },
    )
    print(f"stage69_speechcoco_cache_manifest complete: rows={len(rows)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)
