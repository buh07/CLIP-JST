from __future__ import annotations

import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..common import save_json

COCO_VAL2017_URL = "http://images.cocodataset.org/zips/val2017.zip"


def _read_json_from_zip(zip_path: Path, member: str) -> dict[str, Any]:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as f:
            return json.load(f)


def _read_karpathy_json(caption_zip: Path, member: str, fallback_dir: Path | None = None) -> dict[str, Any]:
    if caption_zip.exists():
        return _read_json_from_zip(caption_zip, member)
    if fallback_dir is not None:
        fp = fallback_dir / member
        if fp.exists():
            with open(fp, encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(f"Could not find {member} in {caption_zip} or fallback directory.")


def _download_file(url: str, dest: Path) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def ensure_coco_val2017(coco_root: Path) -> Path:
    """Ensure COCO val2017 images are available (needed for Karpathy val/test IDs)."""
    val_dir = coco_root / "images" / "val2017"
    if val_dir.exists():
        n = len(list(val_dir.glob("*.jpg")))
        if n >= 5000:
            print(f"COCO val2017 present: {n} images")
            return val_dir

    zip_path = coco_root / "val2017.zip"
    if not zip_path.exists():
        _download_file(COCO_VAL2017_URL, zip_path)

    val_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path} -> {val_dir.parent}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(val_dir.parent)

    n = len(list(val_dir.glob("*.jpg")))
    if n < 5000:
        raise RuntimeError(f"Expected >=5000 val2017 images, found {n}")
    return val_dir


def _load_coco_id_map(coco_root: Path) -> dict[int, Path]:
    """Map COCO image_id -> local 2017 image path using train2017 + val2017 caption JSONs."""
    ann = coco_root / "annotations"
    train_json = ann / "captions_train2017.json"
    val_json = ann / "captions_val2017.json"
    if not train_json.exists() or not val_json.exists():
        raise FileNotFoundError("COCO annotations missing captions_train2017.json/captions_val2017.json")

    with open(train_json, encoding="utf-8") as f:
        train_data = json.load(f)
    with open(val_json, encoding="utf-8") as f:
        val_data = json.load(f)

    id_to_path: dict[int, Path] = {}
    train_dir = coco_root / "images" / "train2017"
    val_dir = coco_root / "images" / "val2017"
    for img in train_data["images"]:
        id_to_path[int(img["id"])] = train_dir / img["file_name"]
    for img in val_data["images"]:
        id_to_path[int(img["id"])] = val_dir / img["file_name"]
    return id_to_path


def _assert_split_integrity(entries: list[dict[str, Any]], expected_counts: dict[str, int], id_key: str) -> None:
    counts = Counter(e["split"] for e in entries)
    for split, expected in expected_counts.items():
        got = counts.get(split, 0)
        if got != expected:
            raise AssertionError(f"Split count mismatch for {split}: expected {expected}, got {got}")

    split_to_ids: dict[str, set[Any]] = defaultdict(set)
    for e in entries:
        split_to_ids[e["split"]].add(e[id_key])
    splits = list(split_to_ids.keys())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            a, b = splits[i], splits[j]
            overlap = split_to_ids[a] & split_to_ids[b]
            if overlap:
                raise AssertionError(f"Leakage between {a} and {b}: {len(overlap)} overlapping IDs")


def build_karpathy_manifests(
    *,
    coco_root: Path,
    flickr_root: Path,
    caption_zip: Path,
    out_dir: Path,
    n_captions: int = 5,
    ensure_val: bool = True,
) -> dict[str, Path]:
    """
    Build Karpathy-standard manifests for COCO and Flickr30K.

    COCO expected counts:
      train=82,783 restval=30,504 val=5,000 test=5,000
    Flickr30K expected counts:
      train=29,000 val=1,014 test=1,000
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if ensure_val:
        ensure_coco_val2017(coco_root)

    coco_k = _read_karpathy_json(caption_zip, "dataset_coco.json", fallback_dir=flickr_root)
    flickr_k = _read_karpathy_json(caption_zip, "dataset_flickr30k.json", fallback_dir=flickr_root)

    # ---- COCO ----
    id_to_path = _load_coco_id_map(coco_root)
    coco_entries: list[dict[str, Any]] = []
    unresolved = 0
    missing_paths = 0
    for item in coco_k["images"]:
        cocoid = int(item["cocoid"])
        img_path = id_to_path.get(cocoid)
        if img_path is None:
            unresolved += 1
            continue
        if not img_path.exists():
            missing_paths += 1
            continue
        sents = [s["raw"].strip() for s in item["sentences"]]
        if len(sents) < n_captions:
            sents = sents + [sents[0]] * (n_captions - len(sents))
        coco_entries.append(
            {
                "image_id": cocoid,
                "split": item["split"],
                "image_path": str(img_path),
                "captions": sents[:n_captions],
            }
        )

    if unresolved or missing_paths:
        raise RuntimeError(
            f"COCO mapping issues: unresolved_ids={unresolved}, missing_paths={missing_paths}"
        )

    _assert_split_integrity(
        coco_entries,
        expected_counts={"train": 82783, "restval": 30504, "val": 5000, "test": 5000},
        id_key="image_id",
    )

    coco_manifest = {
        "dataset": "coco",
        "protocol": "karpathy",
        "n_captions": n_captions,
        "num_images": len(coco_entries),
        "counts": dict(Counter(e["split"] for e in coco_entries)),
        "images": coco_entries,
    }
    coco_manifest_path = out_dir / "karpathy_coco_manifest.json"
    save_json(coco_manifest, coco_manifest_path)

    # ---- Flickr30K ----
    flickr_img_dir = flickr_root / "flickr30k_images"
    flickr_entries: list[dict[str, Any]] = []
    missing_flickr = 0
    for item in flickr_k["images"]:
        fname = item["filename"]
        img_path = flickr_img_dir / fname
        if not img_path.exists():
            missing_flickr += 1
            continue
        sents = [s["raw"].strip() for s in item["sentences"]]
        if len(sents) < n_captions:
            sents = sents + [sents[0]] * (n_captions - len(sents))
        flickr_entries.append(
            {
                "image_id": fname,
                "split": item["split"],
                "image_path": str(img_path),
                "captions": sents[:n_captions],
            }
        )

    if missing_flickr:
        raise RuntimeError(f"Missing Flickr30K image files: {missing_flickr}")

    _assert_split_integrity(
        flickr_entries,
        expected_counts={"train": 29000, "val": 1014, "test": 1000},
        id_key="image_id",
    )

    flickr_manifest = {
        "dataset": "flickr30k",
        "protocol": "karpathy",
        "n_captions": n_captions,
        "num_images": len(flickr_entries),
        "counts": dict(Counter(e["split"] for e in flickr_entries)),
        "images": flickr_entries,
    }
    flickr_manifest_path = out_dir / "karpathy_flickr30k_manifest.json"
    save_json(flickr_manifest, flickr_manifest_path)

    return {
        "coco": coco_manifest_path,
        "flickr30k": flickr_manifest_path,
    }
