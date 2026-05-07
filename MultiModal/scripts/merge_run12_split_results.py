#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path('/jumbo/lisp/f004ndc/CLIP JST/MultiModal/results')
COCO_PATH = ROOT / 'run12_gpu6' / 'stage2_e7' / 'E7_karpathy_full_results.json'
FLICKR_PATH = ROOT / 'run12_gpu4' / 'stage2_e7' / 'E7_karpathy_full_results.json'
OUT_DIR = ROOT / 'run12_split_combined' / 'stage2_e7'
OUT_PATH = OUT_DIR / 'E7_karpathy_full_results.json'


def _load(path: Path) -> dict:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def main() -> None:
    if not COCO_PATH.exists() or not FLICKR_PATH.exists():
        missing = [str(p) for p in [COCO_PATH, FLICKR_PATH] if not p.exists()]
        raise FileNotFoundError(f'Missing split result files: {missing}')

    coco = _load(COCO_PATH)
    flickr = _load(FLICKR_PATH)

    merged = {
        'stage': 'stage2_e7_karpathy',
        'datasets': {},
        'config': {
            'merged_from': [str(COCO_PATH), str(FLICKR_PATH)],
        },
    }
    merged['datasets'].update(coco.get('datasets', {}))
    merged['datasets'].update(flickr.get('datasets', {}))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2)

    print(f'Wrote merged results: {OUT_PATH}')


if __name__ == '__main__':
    main()
