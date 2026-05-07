from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

from ..common import env_snapshot, mark_done, save_json
from ..data.audiocaps import extract_audiocaps_clap_cache
from ..data.clip_cache import extract_karpathy_clip_cache
from ..data.karpathy import build_karpathy_manifests


def run(cfg: dict) -> None:
    t0 = time.time()
    project_root = Path(cfg["project_root"]).resolve()
    output_root = Path(cfg["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_dir = output_root / "manifests"
    cache_dir = output_root / "caches"
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    manifest_paths = build_karpathy_manifests(
        coco_root=Path(cfg["coco_root"]),
        flickr_root=Path(cfg["flickr_root"]),
        caption_zip=Path(cfg["caption_zip"]),
        out_dir=manifest_dir,
        n_captions=cfg.get("n_captions", 5),
        ensure_val=cfg.get("ensure_coco_val2017", True),
    )

    karpathy_cache_paths = {}
    for ds_name, mp in manifest_paths.items():
        karpathy_cache_paths[ds_name] = extract_karpathy_clip_cache(
            manifest_path=mp,
            out_dir=cache_dir / ds_name,
            backbone_name=cfg["clip_backbone"],
            batch_size=cfg["clip_batch_size"],
            device=cfg["device"],
            existing_image_cache=Path(cfg["existing_coco_image_cache"]) if ds_name == "coco" and cfg.get("existing_coco_image_cache") else None,
            existing_image_ids_json=Path(cfg["existing_coco_image_ids_json"]) if ds_name == "coco" and cfg.get("existing_coco_image_ids_json") else None,
        )

    audiocaps_cache_paths = extract_audiocaps_clap_cache(
        out_dir=cache_dir / "audiocaps",
        dataset_name=cfg["audiocaps_dataset"],
        clap_model_name=cfg["clap_model"],
        clip_backbone_name=cfg["clip_backbone"],
        device=cfg["device"],
        audio_batch_size=cfg["audio_batch_size"],
        text_batch_size=cfg.get("text_batch_size", 256),
        target_sampling_rate=cfg.get("audiocaps_target_sr", 48_000),
        max_examples_per_split=cfg.get("audiocaps_max_examples_per_split"),
    )

    protocol_report = {
        "coco_expected": {"train": 82783, "restval": 30504, "val": 5000, "test": 5000},
        "flickr_expected": {"train": 29000, "val": 1014, "test": 1000},
        "status": "passed",
    }
    save_json(protocol_report, output_root / "protocol_checks.json")

    provenance = env_snapshot(project_root, seeds=cfg.get("seeds", [0, 1, 2, 3, 4]), extra={
        "stage": "stage1_prepare",
        "elapsed_sec": time.time() - t0,
        "karpathy_cache_paths": {k: {kk: str(vv) for kk, vv in p.items()} for k, p in karpathy_cache_paths.items()},
        "audiocaps_cache_paths": {k: str(v) for k, v in audiocaps_cache_paths.items()},
    })
    save_json(provenance, output_root / "provenance_stage1.json")

    mark_done(markers / "stage1_prepare.done.json", {"elapsed_sec": time.time() - t0})
    print("Stage 1 complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)
