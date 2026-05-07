#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from MultiModal.multimodal.data import build_wavcaps_audio_text_cache_tar_shard


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-out-dir", required=True)
    ap.add_argument("--dataset", default="humanify/AS-WavCaps")
    ap.add_argument("--clap-model", default="laion/clap-htsat-unfused")
    ap.add_argument("--clip-backbone", default="openai/clip-vit-base-patch32")
    ap.add_argument("--target-sr", type=int, default=48_000)
    ap.add_argument("--max-examples", type=int, required=True)
    ap.add_argument("--sampling-policy", default="stratified")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--audio-batch-size", type=int, default=64)
    ap.add_argument("--text-batch-size", type=int, default=256)
    ap.add_argument("--shard-index", type=int, required=True)
    ap.add_argument("--shard-count", type=int, required=True)
    ap.add_argument("--done-marker", required=True)
    ap.add_argument("--fail-marker", required=True)
    args = ap.parse_args()

    done_marker = Path(args.done_marker)
    fail_marker = Path(args.fail_marker)
    done_marker.parent.mkdir(parents=True, exist_ok=True)
    fail_marker.parent.mkdir(parents=True, exist_ok=True)

    try:
        res = build_wavcaps_audio_text_cache_tar_shard(
            shard_out_dir=Path(args.shard_out_dir).resolve(),
            dataset_name=str(args.dataset),
            clap_model_name=str(args.clap_model),
            clip_backbone_name=str(args.clip_backbone),
            target_sampling_rate=int(args.target_sr),
            max_examples=int(args.max_examples),
            sampling_policy=str(args.sampling_policy),
            device=str(args.device),
            audio_batch_size=int(args.audio_batch_size),
            text_batch_size=int(args.text_batch_size),
            shard_index=int(args.shard_index),
            shard_count=int(args.shard_count),
        )
        done_marker.write_text(
            json.dumps(
                {
                    "status": "ok",
                    "shard_index": int(args.shard_index),
                    "shard_count": int(args.shard_count),
                    "artifacts": {k: str(v) for k, v in res.items()},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wavcaps shard build ok: {res}")
    except Exception as e:
        fail_marker.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "shard_index": int(args.shard_index),
                    "shard_count": int(args.shard_count),
                    "error_type": type(e).__name__,
                    "error": str(e),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()

