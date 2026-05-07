# REDO — Unreliable and Incomplete Experiments

This document lists every experiment whose results cannot be used as-is, with exact file paths, error locations, and scientific consequences. Read alongside `UNIFIED_RESULTS.md` (reliable results) and `CLIP-JST.md` (research proposal).

---

## Summary table

| Experiment | Status | Priority | Issue |
|---|---|---|---|
| D4 — Backbone generalization | **COMPLETE** 2026-04-30 — ViT-L/14 and DINOv2+BGE done; results in `results/D4/D4_results.json` | ~~HIGH~~ **DONE** | CLAP still open — needs audio data (see below) |
| E3 — Width-complexity scaling | **COMPLETE** 2026-04-29 — full rerun results in `results/rerun_fix_20260429/E3/` | ~~HIGH~~ **DONE** | Core theoretical claim may be based on stale cache features |
| D3 — Relationship-graph ablations rerun | **COMPLETE** 2026-04-30 — results in `results/rerun_fix_20260429/D3/` | ~~HIGH~~ **DONE** | Rerun confirmed original results on fixed cache |
| E4 — OOD retrieval (CC3M) | **COMPLETE** 2026-04-30 — ~125K CC3M images; results in `results/E4_cc3m/E4_results.json` | ~~MEDIUM~~ **DONE** | Half of the planned OOD evaluation is missing |
| E5 — rerun on fixed features | **COMPLETE** 2026-04-30 — results in `results/rerun_fix_20260429/E5/E5_results.json` | ~~MEDIUM~~ **DONE** | Confirmed statistically identical to original E5 results |
| rerun_fix_20260429/D3 | Smoke test results only (10K pairs, 1 seed, 2 epochs) | **INFO** | Not a bug — but original results/D3/ are correct; ignore this dir |
| rerun_fix_20260429/E5 | Smoke test results only (pathological centralized avg_R=0.005) | **INFO** | Not a bug — but original results/E5/ are correct; ignore this dir |
| E5 — no-op state dict replace | Cosmetic code bug, no effect on results | **LOW** | Should be fixed before next code refactor |
| D4 — CLAP backbone | **OPEN** — no audio data in COCO manifest | **MEDIUM** | Requires TTS synthesis of captions or an audio-captioned dataset |

---

## D4 — Backbone generalization: NO RESULTS JSON

### What was supposed to happen

`experiments/run_D4.py` with `configs/D4.yaml` was supposed to:
1. Extract features using three backbones: ViT-L/14 (`openai/clip-vit-large-patch14`), DINOv2+BGE (`facebook/dinov2-large` + `BAAI/bge-large-en-v1.5`), and CLAP (`laion/clap-htsat-unfused`)
2. Train JL+Mahalanobis and CLIP-head baselines on each backbone at m∈{128,256}
3. Save a summary `results/D4/D4_results.json`

### What actually exists

```
results/D4/
  vit_l14/          ← checkpoint files from a partial ViT-L/14 run
  (no D4_results.json)
```

`results/D4/D4_results.json` **does not exist**. The other two backbones (DINOv2+BGE, CLAP) have no output at all.

### Where the run failed

`experiments/run_D4.py` — the `save_json(results, ...)` call at the end of `run()` was never reached. The script either crashed during the DINOv2+BGE or CLAP feature extraction/training phases, or was cancelled before completing. The ViT-L/14 backbone likely ran and produced checkpoints but did not trigger the final JSON save (which only happens after all three backbones complete).

### Why it matters

D4 is the only evidence for claim: *"JL+Mahalanobis advantage is backbone-agnostic."* Without results from DINOv2+BGE and CLAP, no backbone-agnosticism claim can be made. CLAP in particular is a qualitatively different modality (audio+text), which would be the strongest test of generality.

### How to rerun

```bash
python experiments/run_D4.py --config configs/_rerun_fix_D4.yaml
```

`configs/_rerun_fix_D4.yaml` targets `results/rerun_fix_20260429/D4/`. Verify that the DINOv2 (`facebook/dinov2-large`) and BGE (`BAAI/bge-large-en-v1.5`) checkpoints are accessible; CLAP requires `laion/clap-htsat-unfused`. The rerun uses `embed_dims: [128, 256]`, `epochs: 30`, `patience: 5`, `seed: 0` (single seed to limit compute; can extend to 3 seeds after verifying the runs complete).

---

## E3 — Width-complexity scaling: RERUN COMPLETE ✓

**Update 2026-04-29**: The full rerun finished at 20:26:56. `results/rerun_fix_20260429/E3/E3_results.json` contains all 8 subset types (ncap_1, ncap_2, ncap_5, supcat_animal, supcat_vehicle, supcat_person, supcat_food, supcat_furniture) × 5 embed_dims, computed on current (fixed) cache files. Use this file instead of `results/E3/E3_results.json`. The original E3 results should be considered stale.

### What was supposed to happen

`experiments/run_E3.py` with `configs/E3.yaml` tests the width-adaptive JL bound (Claim 1): the embedding dimension needed to achieve a target recall should scale linearly with estimated Gaussian width. It creates subsets of COCO by varying caption count per image (1, 2, 5) and by filtering to supercategories (animal, vehicle, person, food, furniture), then reports `theory_m` (the width-predicted required dimension) alongside empirical recall at each m.

### What exists

`results/E3/E3_results.json` — appears complete (8 subset types × 5 embed_dims), **but was likely generated before the "Fix CLIP feature extraction for transformers 5.x" bug fix**.

### Evidence of staleness

1. A smoke-test rerun (`configs/_smoke_fix_E3.yaml`) was executed **today** (2026-04-29 20:26:56, per `results/rerun_fix_20260429/markers/E3.done.json`), confirming the feature extraction bug needed correction.
2. The full rerun config (`configs/_rerun_fix_E3.yaml`) has **not been executed** — its output directory `results/rerun_fix_20260429/E3/` contains only the smoke test output (1 supercategory `animal`, m=64, 1 epoch), not a full run.
3. `data/cache.py` has uncommitted working-tree modifications (git status: ` M data/cache.py`), indicating the cache loading code is still being refined.

### Where the bug was

The commit `a48b984 Fix CLIP feature extraction for transformers 5.x` changed how frozen backbone features are extracted. If the cache files `data/cache/coco/image_feats_openai_clip-vit-base-patch32_raw.pt` and `data/cache/coco/text_feats_openai_clip-vit-base-patch32_raw.pt` were regenerated after that fix but **before E3 ran**, the stored E3 results use correct features. If E3 ran with the old (buggy) extraction, both the features and the width estimates computed from them are wrong.

The width estimate `cross_modal_width_estimate()` is called directly on the cached features. If features had incorrect pooling or normalization before the fix, the estimated widths (and therefore `theory_m` values) in `results/E3/E3_results.json` are unreliable.

### Why it matters

E3 is the **primary experiment for Claim 1** (width-adaptive distortion preservation). The scaling plot of "required dimension vs. Gaussian width" is a central theoretical contribution. If the width estimates and retrieval metrics were computed on buggy features, this plot and the claim it supports are invalid.

### How to rerun

```bash
python experiments/run_E3.py --config configs/_rerun_fix_E3.yaml
```

This runs on the full dataset (all 5 supercategories + 3 caption counts) with the current (fixed) cache files, writing to `results/rerun_fix_20260429/E3/`. Note: the smoke test already wrote a partial result to that directory; the full run will overwrite it. Budget: ~30 epochs × 8 subsets × 5 dims = significant GPU time; see the config for exact settings (`epochs: 30`, `warmup_epochs: 5`, `batch_size: 2048`, `seed: 0`).

---

## E4 — OOD retrieval: CC3M NOT RUN

### What was supposed to happen

`experiments/run_E4.py` evaluates models trained on COCO (from E1) on **two** OOD datasets: Flickr30K and Conceptual Captions 3M (CC3M).

### What actually exists

`results/E4/E4_results.json` contains only `flickr30k` sub-keys for each model. CC3M was never evaluated.

```python
# Each model entry looks like:
{"jl_mahal_m256_rfull": {"flickr30k": {...}}}
# Expected to also have:
{"jl_mahal_m256_rfull": {"flickr30k": {...}, "cc3m": {...}}}
```

### Root cause

CC3M requires downloading ~3M image-caption pairs and extracting CLIP features — a multi-hour preprocessing step. The cache files for CC3M are absent from `data/cache/`. Without the cache, `run_E4.py` silently skips or fails on the CC3M dataset. The Flickr30K evaluation completed because its cache exists (`data/cache/flickr30k/image_feats_openai_clip-vit-base-patch32_raw.pt`).

### Why it matters

The hypothesis is that JL's fixed-random structure generalizes better OOD. With only one OOD dataset, the evidence is weak. Additionally, the existing Flickr30K OOD results *contradict* the hypothesis (CLIP head drops 61% OOD vs. JL+Mahal dropping 68–72%), so a second OOD dataset is needed to confirm whether this is a genuine effect or Flickr30K-specific. CC3M is also a larger and more diverse dataset, making it a stronger generalization test.

### How to complete

1. Prepare CC3M cache: run `scripts/prepare_data.py` with CC3M configuration to extract features into `data/cache/cc3m/`
2. Re-run E4 with the updated cache: `python experiments/run_E4.py --config configs/E4.yaml`

The existing Flickr30K results in `results/E4/E4_results.json` are valid and can be included in the final paper as a partial result.

---

## Smoke test artefacts in results/rerun_fix_20260429/

### What happened

The smoke test configs (`_smoke_fix_D3.yaml`, `_smoke_fix_E5.yaml`) were designed to quickly verify bug fixes on a small dataset (`data/cache_smoke/`). However, they both write to the **same output directories** as the planned full reruns (`_rerun_fix_D3.yaml`, `_rerun_fix_E5.yaml`):

```
configs/_smoke_fix_D3.yaml:   output_dir: results/rerun_fix_20260429/D3   ← smoke
configs/_rerun_fix_D3.yaml:   output_dir: results/rerun_fix_20260429/D3   ← full rerun
```

The smoke tests ran first and wrote invalid results to these directories.

### results/rerun_fix_20260429/D3/D3_results.json — DO NOT USE

- `n_pairs: 10,000` (smoke cache, not full COCO 591K)
- Single seed, m=512 only, 2 epochs
- **Correct data is in `results/D3/D3_results.json`** (n_pairs=591,435, 3 seeds, 4 dims). Use that file.

### results/rerun_fix_20260429/E5/E5_results.json — DO NOT USE

- `centralized avg_R: 0.005` (pathological — 1 epoch on 10K smoke samples)
- `federated avg_R: 0.420` (inconsistently high relative to centralized; smoke artefact)
- **Correct data is in `results/E5/E5_results.json`** (3 seeds, 60 epochs, full COCO). Use that file.

### What to do if running the full reruns

If `_rerun_fix_D3.yaml` or `_rerun_fix_E5.yaml` are run in the future, they will overwrite the smoke results in their output directories. This is the correct behavior — no special handling needed. The original `results/D3/` and `results/E5/` are unaffected.

---

## E5 — Cosmetic code bug: no-op state dict replacement

### Location

`experiments/run_E5.py`, lines 344–347:

```python
fed_model_eval.load_state_dict(
    {k.replace("mahal_v.", "mahal_v.").replace("mahal_t.", "mahal_t."): v
     for k, v in fed_model.state_dict().items()
     if k.startswith("mahal_")},
    strict=False
)
```

### The bug

Both `.replace()` calls replace a string with itself — they are no-ops. The key dictionary comprehension does nothing to the key names. This is dead code from an incomplete refactoring, likely left over from a rename of `mahal_v` → something else (or vice versa) that was then reverted.

### Does it affect the E5 results?

**No.** The state dict keys in `fed_model` already match the expected keys in `_IdentityJLPipeline` (both use `mahal_v.*` and `mahal_t.*`). The `strict=False` argument causes `load_state_dict` to load matching keys and silently ignore missing/extra keys, so the Mahalanobis weights are correctly transferred. The E5 results in `results/E5/E5_results.json` are valid.

### Why document it

If `CLIPJSTPipeline` or `_IdentityJLPipeline` attribute names change in a future refactor (e.g., `mahal_v` → `mahal_img`), these no-op replaces will not perform the intended rename. The `strict=False` will then silently load a randomly-initialized Mahalanobis instead of the trained federated weights, producing wrong federated evaluation results without any error or warning. The fix is either to remove the dead replaces or replace them with the actual intended rename.

### Fix (one line)

```python
# Either remove the no-op replaces:
fed_model_eval.load_state_dict(
    {k: v for k, v in fed_model.state_dict().items() if k.startswith("mahal_")},
    strict=False
)
```
