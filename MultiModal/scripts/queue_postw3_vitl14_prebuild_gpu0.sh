#!/usr/bin/env bash
set -euo pipefail

ROOT="/jumbo/lisp/f004ndc/CLIP JST"
MM_ROOT="$ROOT/MultiModal"
LOG_DIR="$MM_ROOT/logs/next_run_suite/post_w3_sequence"
MARK_DIR="$MM_ROOT/results/next_run_suite/post_w3_sequence/markers"
CACHE_BASE="$MM_ROOT/results/next_run_suite/caches_vitl14"
COCO_DIR="$CACHE_BASE/coco"
mkdir -p "$LOG_DIR" "$MARK_DIR" "$COCO_DIR"

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Prepare COCO cache layout expected by Stage44/Stage29 runner.
ln -sf "/jumbo/lisp/f004ndc/CLIP JST/data/cache/coco/vit_l14/image_feats_openai_clip-vit-large-patch14_raw.pt" \
  "$COCO_DIR/image_feats_openai_clip-vit-large-patch14_raw.pt"
ln -sf "/jumbo/lisp/f004ndc/CLIP JST/data/cache/coco/vit_l14/text_feats_openai_clip-vit-large-patch14_raw.pt" \
  "$COCO_DIR/text_feats_openai_clip-vit-large-patch14_raw.pt"

# Build metadata that is index-consistent with the available ViT-L tensors.
# The ViT-L cache has 118,287 rows (not full 123,287), so we remap Karpathy
# splits by image_id onto this reduced row set.
python - <<'PY'
import json
from pathlib import Path

import torch

full_meta_path = Path("/jumbo/lisp/f004ndc/CLIP JST/MultiModal/results/full_suite/caches/coco/metadata.json")
vit_ids_path = Path("/jumbo/lisp/f004ndc/CLIP JST/data/cache/coco/image_ids.json")
img_path = Path("/jumbo/lisp/f004ndc/CLIP JST/data/cache/coco/vit_l14/image_feats_openai_clip-vit-large-patch14_raw.pt")
txt_path = Path("/jumbo/lisp/f004ndc/CLIP JST/data/cache/coco/vit_l14/text_feats_openai_clip-vit-large-patch14_raw.pt")
dst_path = Path("/jumbo/lisp/f004ndc/CLIP JST/MultiModal/results/next_run_suite/caches_vitl14/coco/metadata.json")

full_meta = json.loads(full_meta_path.read_text())
full_ids = [int(x) for x in full_meta["image_ids"]]

full_idx_to_split = {}
for split_name, idxs in full_meta["split_to_indices"].items():
    if split_name == "train_restval":
        continue
    for idx in idxs:
        full_idx_to_split[int(idx)] = split_name
id_to_split = {full_ids[i]: full_idx_to_split[i] for i in range(len(full_ids)) if i in full_idx_to_split}

vit_ids = [int(x) for x in json.loads(vit_ids_path.read_text())]
img = torch.load(img_path, map_location="cpu", weights_only=True)
txt = torch.load(txt_path, map_location="cpu", weights_only=True)
if len(vit_ids) != img.shape[0] or img.shape[0] != txt.shape[0]:
    raise RuntimeError(
        f"ViT-L cache size mismatch: ids={len(vit_ids)} img={img.shape[0]} txt={txt.shape[0]}"
    )

split_to_indices = {"train": [], "restval": [], "val": [], "test": []}
for row_idx, image_id in enumerate(vit_ids):
    split_name = id_to_split.get(image_id)
    if split_name is None:
        raise RuntimeError(f"image_id {image_id} missing in full Karpathy metadata")
    split_to_indices[split_name].append(row_idx)

split_to_indices["train_restval"] = split_to_indices["train"] + split_to_indices["restval"]

meta = {
    "dataset": "coco",
    "protocol": "karpathy",
    "backbone": "openai/clip-vit-large-patch14",
    "n_captions": 1,
    "n_images": int(img.shape[0]),
    "n_text": int(txt.shape[0]),
    "vision_dim": int(img.shape[1]),
    "text_dim": int(txt.shape[1]),
    "split_to_indices": split_to_indices,
    "image_ids": vit_ids,
}
dst_path.write_text(json.dumps(meta, indent=2))
print("wrote", dst_path)
print({k: len(v) for k, v in split_to_indices.items()})
PY

bash "$MM_ROOT/scripts/bootstrap_env.sh"
source "$MM_ROOT/.venv/bin/activate"
export PYTHONPATH="$MM_ROOT:${PYTHONPATH:-}"
cd "$ROOT"

python -m MultiModal.multimodal.experiments.run_stage44_zero_shot_baseline_control \
  --config MultiModal/configs/stage44_vitl14_prebuild_gpu0.yaml 2>&1 | tee "$LOG_DIR/stage44_vitl14_prebuild_gpu0.log"

touch "$MARK_DIR/gpu0.vitl14_prebuild.done"
echo "[$(date '+%F %T')] queue_postw3_vitl14_prebuild_gpu0 complete"
