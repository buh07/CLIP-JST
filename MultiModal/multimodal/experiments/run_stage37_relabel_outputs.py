from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

from ..common import load_json, mark_done, save_json


def _relabel(ret: dict, left_to_right_prefix: str, right_to_left_prefix: str) -> dict:
    out = dict(ret)
    for k, v in ret.items():
        if k.startswith("i2t_R@"):
            kk = k.replace("i2t_", left_to_right_prefix + "_")
            out[kk] = v
        elif k.startswith("t2i_R@"):
            kk = k.replace("t2i_", right_to_left_prefix + "_")
            out[kk] = v
    return out


def run(cfg: dict) -> None:
    output_root = Path(cfg["output_root"]).resolve()
    stage_root = output_root / "stage37_imagebind_comparison"
    markers = output_root / "markers"
    markers.mkdir(parents=True, exist_ok=True)

    p = stage_root / "stage37_imagebind_comparison_results.json"
    obj = load_json(p)

    obj["image_audio_relabeled"] = _relabel(obj["image_audio"], "i2a", "a2i")
    obj["audio_text_relabeled"] = _relabel(obj["audio_text"], "a2t", "t2a")
    obj["image_text_relabeled"] = _relabel(obj["image_text"], "i2t", "t2i")

    thumb_ok_frac = float(obj.get("thumbnail_ok_fraction", 1.0))
    obj["reporting_caveats"] = {
        "naming_note": "Raw recall_at_k keys are generic (i2t/t2i). Use *_relabeled blocks for modality-correct direction labels.",
        "thumbnail_placeholder_fraction": float(1.0 - thumb_ok_frac),
        "thumbnail_placeholder_note": "Unavailable thumbnails are replaced with deterministic gray 224x224 placeholders.",
        "model_scale_note": "ImageBind-Huge is substantially larger than CLIP ViT-B/32 and trained with direct audio-visual supervision.",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    save_json(obj, p)

    md = [
        "# Stage37 Relabel Addendum",
        "",
        "- Added modality-correct aliases under `image_audio_relabeled`, `audio_text_relabeled`, and `image_text_relabeled`.",
        "- Core values are unchanged; this is a reporting-label cleanup only.",
        f"- Placeholder thumbnail fraction: `{1.0 - thumb_ok_frac:.6f}`",
        "",
        "## Direction Labels",
        "- image-audio: `i2a_R@k`, `a2i_R@k`",
        "- audio-text: `a2t_R@k`, `t2a_R@k`",
        "- image-text: `i2t_R@k`, `t2i_R@k`",
    ]
    (stage_root / "stage37_relabel_addendum.md").write_text("\n".join(md), encoding="utf-8")
    mark_done(markers / "stage37_relabel_outputs.done.json", {"updated_file": str(p)})
    print("Stage37 relabel complete")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)
