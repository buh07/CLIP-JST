# CLIP-JST / MultiModal: Full Narrative (Evidence-Linked)

## 1) What this paper is actually about

This project is about a specific scientific question:

**When can fixed random JL projections support cross-modal retrieval and modular multimodal extension, and when must projection parameters be learned (or unconstrained)?**

The experiments show a clear answer:

1. **Frozen post-hoc random JL is not a replacement for a learned CLIP projection head** in bimodal retrieval.
2. **If projection parameters are trainable, performance recovers to CLIP-head levels** (often statistically indistinguishable at practical dimensions).
3. **For modular transitivity (image↔audio via text), JL sharing is not universally required**; its effect depends strongly on embedding dimension and training phase.
4. **In exact-retrieval benchmarks, modular JL variants are not the top performers** against simpler direct audio-to-CLIP baselines in this codebase.
5. **Privacy/federated gains are real but conditional**: claims depend on whether comparisons are strict equal-budget or high-bandwidth hybrid regimes.
6. **Semantic top-k retrieval is substantially above chance even when exact-pair recall is low**, so the system’s failure mode is fine-grained instance matching, not absence of semantic alignment.

---

## 2) Core claims and exactly which experiments support them

## Claim A — Fixed random JL (post-hoc) loses too much semantic structure

### Supporting experiments

- `E1` (`results/E1/E1_results.json`)
- `E6` (`results/E6/E6_results.json`)
- `E2` (`results/E2/E2_results.json`)
- `D1` (`results/D1/D1_results.json`)
- `D4` (`results/D4/D4_results.json`)

### Why these support the claim

1. **E1** shows a persistent gap at matched embedding size. At `m=256`, CLIP head is far above `jl_mahal_rfull` on both COCO and Flickr30K.
2. **E6** shows the gap is not a convergence artifact: both models converge by similar epochs, but the final gap remains.
3. **E2** shows the gap is not parameter-efficiency in JL’s favor: low-parameter baselines with pretrained initialization (e.g., LoRA variants) strongly outperform frozen-JL variants.
4. **D1** shows mechanism: JL has near-flat singular spectrum and low subspace overlap with CLIP head; Mahalanobis improves overlap only partially, not enough to close retrieval gap.
5. **D4** shows backbone robustness of the failure mode: the same relative JL deficit appears across ViT-B/32, ViT-L/14, and DINOv2+BGE.

**Inference**: the failure is structural to fixed random post-hoc projection in this setup, not just undertraining or one unlucky backbone.

---

## Claim B — Training the projection removes most of the deficit

### Supporting experiments

- `E7` (`results/E7/E7_results.json`)
- `E7-Run12` Karpathy 5-seed extension (`MultiModal/results/run12_gpu{4,6}/stage2_e7/...`)
- `Stage18` spectral regularization sweep (`MultiModal/results/modular_transitivity_suite/aggregate/stage18_spectral_results.json`)

### Why these support the claim

1. **E7**: `orth_jl_trainable` (same parameter count class as CLIP head) matches/slightly exceeds CLIP-head means, while `random_jl_only` is near chance and `random_jl_mahal` remains much lower.
2. **Run12 (5 seeds, Karpathy protocol)** replicates this with stronger statistics and Holm-corrected paired tests:
   - trainable orthogonal methods are near CLIP-head;
   - frozen random JL remains significantly lower at most settings.
3. **Stage18** clarifies regularization behavior:
   - moderate spectral alignment pressure (`reg=0.1`) stays near CLIP-head behavior;
   - overly strong pressure (`reg=10`) collapses low-dim performance.

**Inference**: “JL-style structure” is not inherently bad; **freezing** a random projection is the central issue in high-performing regimes.

---

## Claim C — Modular transitivity exists, but the mechanism is subtler than “shared JL is necessary”

### Supporting experiments

- `Stage19` pseudo-modality gate (`.../stage19_pseudomodality_results.json`)
- `Stage20/21` modular vs joint tri-modal (`.../stage21_modular_transitivity_aggregate.json`)
- `Stage25/26` 4-way JL factorial mechanism lock (`MultiModal/results/modular_transitivity_jl_ablation/aggregate/stage26_jlablation_aggregate.json`)

### Why these support the claim

1. **Stage19** shows modular training is viable in controlled pseudo-modality settings (ratios near/above joint baseline at key `m`), justifying real modular tri-modal runs.
2. **Stage21** demonstrates nontrivial zero-shot image↔audio retrieval above chance for modular shared-JL, while preserving strong COCO image-text.
3. **Stage25/26** is the decisive mechanistic test (all four methods, same run, same seeds, same dims):
   - `shared`, `separate`, `hybrid_it`, `hybrid_at` at `m={64,128,256,512}`.
   - Result: **no universal dominance** of all-shared JL.

### Stage25/26 specific interpretation

- At `m=128`, sharing in at least one phase helps vs all-separate (strong Holm-significant pairwise deltas).
- At `m=256`, all-separate beats shared and hybrid_it (Holm-significant).
- At `m=512`, methods largely converge (mostly non-significant pairwise differences).
- Factorial effects show dimension-dependent behavior:
  - positive sharing main effects at low dims,
  - negative Phase-A effect at `m=256`,
  - diminishing effects at `m=512`.

**Inference**: transitivity does not require globally shared JL. A better mechanistic statement is:

**Phase-wise text anchoring enables modular transfer; JL sharing acts as a dimension-dependent inductive bias, not a universal requirement.**

### Performance caveat from Domain-Gap Closure Suite

- `Stage30` (`stage30_modular_vs_nonmodular_results.json`) shows that at `m=512`, non-modular baselines outperform modular-shared JL on exact retrieval:
  - `audio_linear_probe`: `combined=0.2798`, `av_ia=0.0462`
  - `audio_text_lora_proxy`: `combined=0.2783`, `av_ia=0.0389`
  - `modular_shared_jl`: `combined=0.2613`, `av_ia=0.0328`
- `Stage31` (`stage31_wavcaps_scaling_results.json`) shows WavCaps Phase-B scaling did **not** improve modular JL in this setup:
  - best modular at `m=512` (`modular_separate_jl`): `combined=0.2210`, `av_ia=0.0231`
  - this is below Stage32 AudioCaps-ordering results at `m=512`.

**Inference**: the modular transitivity mechanism is present, but in this benchmark suite it does not win the primary exact-pair retrieval metric versus simpler baselines.

### Semantic follow-up (Stage34/35) resolves the “does it learn semantics?” question

- `Stage34/35` (`MultiModal/results/semantic_followup/aggregate/stage35_semantic_topk_aggregate/stage35_semantic_topk_aggregate.json`) evaluates category-level top-k semantic retrieval on **existing Stage30/31 checkpoints** (no retraining), with 20 coarse classes and `k={1,5,10}`.
- Chance: `P@1=0.05`, `Hit@10≈0.401`.

Key results:

1. **Stage30 (AudioCaps bridge), m=512**:
   - `audio_linear_probe`: `avg_cat_P@1=0.2395`, `avg_cat_P@5=0.2328`, `avg_cat_Hit@10=0.5141`
   - `modular_shared_jl`: `avg_cat_P@1=0.2340`, `avg_cat_P@5=0.2255`, `avg_cat_Hit@10=0.5158`
2. **Stage30 (m=256)**:
   - `audio_linear_probe` beats `modular_shared_jl` on `avg_cat_P@1` by `+0.0193` (Holm-significant, `p=5.32e-4`).
3. **Stage31 (WavCaps bridge), m=512**:
   - `modular_separate_jl` beats `modular_shared_jl` on `avg_cat_P@1` by `+0.0125` (Holm-significant, `p=0.0079`),
   - but absolute semantic performance remains below Stage30, consistent with weaker bridge quality.

**Inference**: semantic routing is real and stable (well above chance), but exact-pair weakness is not a “no-signal” failure. The bottleneck is high-precision cross-instance matching under bridge/data constraints.

---

## Claim D — Privacy/federated story is valid but must be split into two regimes

### Supporting experiments

- `E5` (federated + inversion curve)
- `E8` (concat/mask-concat high-bandwidth regime)
- `Stage13/14/15` strengthened federated + stronger attacks
- `Stage16/17` strict budget-matched frontier (corrected methodology)

### Why these support the claim

1. **E5** shows a real utility-privacy tradeoff for projected features and strong federated retention vs centralized under that specific protocol.
2. **E8** shows strong retrieval/privacy tradeoffs for concat/mask-concat, but in a larger transmitted representation regime (not strict 256-d equal-budget).
3. **Stage13/14/15** adds stronger attacks and broader federated partitions, reducing over-optimistic conclusions.
4. **Stage16/17 strict rerun** resolves earlier methodological issues (shard safety, randomized probe sampling, strict budget matching, corrected baselines) and becomes the reliable equal-budget reference.

**Inference**: paper should explicitly separate:

- **High-bandwidth hybrid privacy regime** (E8-style), and
- **Strict budget-matched regime** (Stage16/17),

because conclusions differ materially across them.

## Claim E — Result integrity required shard-safe aggregation and stage-aware merging

### Supporting evidence

- Stage result writer/merging logic in:
  - `run_stage29_cc3m_phaseA_modular.py` (shared by Stage29/30/31/32 wrappers)
  - `run_stage33_domain_gap_aggregate.py`

### Why this matters

1. Parallel shard runs could previously leave partial `*_results.json` snapshots if one shard overwrote another.
2. Stage33 previously merged identical method names across different stages into one namespace, creating cross-stage collisions.
3. After fixes (file-lock + merge-on-write for stage summaries, stage-prefixed method keys in Stage33), the final aggregate is reproducible and internally consistent.

---

## 3) What all experiments together say (end-to-end)

This is the integrated storyline from E1→E7→Run12→Stage25/26→Stage29–35:

1. **Negative control established**: Frozen random JL + Mahalanobis underperforms learned CLIP-head projection (E1/E6/E2/D1/D4).
2. **Positive control established**: If projection is trainable, CLIP-level retrieval is recoverable (E7 + Run12).
3. **Mechanism refined** for modular multimodal:
   - sharing can help at low dimensions,
   - can hurt at intermediate dimension (`m=256`),
   - matters less at larger dimension (`m=512`),
   - one shared phase may be sufficient in some regimes.
4. **Privacy/federated conclusions are conditional on budget/fairness protocol**, and strict methodology changes earlier ranking claims.
5. **Domain-gap closure extensions** show that scaling Phase-B data (WavCaps in Stage31) did not improve the modular methods here; strongest exact-retrieval numbers come from simpler non-modular baselines (Stage30).
6. **Semantic follow-up (Stage34/35)** shows category-level retrieval is substantially above chance, so the failure is not semantic collapse but insufficient instance-level discrimination.

So the project matured from a binary thesis (“shared JL works/fails”) into a **regime map** over:

- trainable vs frozen projection,
- dimension (`m`),
- phase-sharing pattern,
- and communication/privacy budget.

But it also narrows the strongest defensible claim: this is primarily a **mechanistic mapping and negative/conditional result paper**, not a “new SOTA modular multimodal method” paper.

---

## 4) How this changes the paper narrative

## Old narrative (too strong)

- “Shared random JL is the key mechanism for modular transitivity.”
- “Shared-JL modular methods dominate simpler alternatives on retrieval.”
- “Hybrid privacy methods dominate DP-SGD-like baselines.”

## New narrative (supported)

1. **Main claim**: fixed post-hoc random JL is not sufficient; trainable projections are the main performance lever.
2. **Mechanistic claim**: modular transitivity is enabled by phase-wise alignment through text; JL sharing is a tunable bias whose value changes with `m`.
3. **Benchmark claim**: in this stack, direct audio-to-CLIP baselines outperform modular JL variants on exact retrieval metrics.
4. **Semantic claim**: category-level routing remains strong even where exact-pair retrieval is weak.
5. **Systems claim**: privacy-utility tradeoffs exist, but claims must be scoped by bandwidth/fairness regime.
6. **Practical recommendation**:
   - low `m`: keep at least one phase shared,
   - mid `m` (notably 256): test separate/hybrid variants explicitly,
   - high `m`: expect convergence across sharing variants on exact retrieval, while semantic metrics may remain robust.

This narrative is more nuanced and scientifically stronger on mechanism/protocol rigor, but weaker as a pure methods-performance paper.

---

## 5) Why these findings matter

1. **For multimodal learning**: shows modular addition can work without full joint retraining, but quality depends on bridge/domain quality.
2. **For representation geometry**: quantifies where projection sharing helps, hurts, or washes out (dimension- and phase-dependent).
3. **For federated systems**: provides a conditional utility/privacy/bandwidth map under stricter protocols.
4. **For evaluation design**: shows why exact-pair and category-level metrics should be reported together for modular transitivity claims.
5. **For methodology**: demonstrates that sharded aggregation correctness and fair budget protocols can materially change conclusions.

---

## 6) Where this can be applied (and where not)

## Strong-fit applications

1. **Federated cross-silo cross-modal retrieval** (primary): organizations keep raw data local and exchange model updates or projected embeddings.
2. **Enterprise self-hosted multimodal search** (secondary): controllable embedding dimension and privacy posture, with explicit performance tradeoffs.
3. **Incremental modality onboarding** (secondary): adding modality-specific components without full retraining of all previous modalities.

## Caution / out-of-scope without more work

1. **Formal-privacy-critical deployments**: current privacy evidence is empirical attacker-based, not formal `(epsilon, delta)` guarantees for core method.
2. **Severe OOD shifts**: E4 indicates substantial degradation under distribution shift; deployment claims should be scoped accordingly.
3. **Safety/regulatory high-stakes domains**: require stronger attacker suite + formal guarantees + domain-specific validation.

---

## 7) Current best one-paragraph abstract-level framing

We study when Johnson–Lindenstrauss-style projections can support cross-modal retrieval and modular multimodal transfer. Across controlled experiments, fixed post-hoc random JL projections consistently underperform learned projection heads, while trainable JL-like projections recover near-CLIP performance. In modular tri-modal transfer (image, text, audio), zero-shot image–audio retrieval emerges without direct image–audio supervision, and a 4-way phase-sharing ablation shows that JL sharing is a dimension-dependent inductive bias rather than a universal requirement. Exact-pair retrieval is strongest for simpler direct audio-to-CLIP baselines in our stack, but category-level top-k evaluation remains substantially above chance across modular and non-modular methods, indicating preserved semantic routing despite weak instance-level matching. In federated/privacy evaluations, conclusions depend strongly on budget matching, attacker protocol, and shard-safe aggregation. Overall, the contribution is a mechanistic and methodological regime map for projection design, modular transfer, semantic evaluation, and privacy-utility tradeoffs.

---

## 8) Evidence index (quick lookup)

- Core failure/success mechanism: `E1`, `E2`, `E6`, `E7`, `Run12`, `D1`, `D4`
- Theory-linked diagnostics: `D1`, `D2`, `E3`, `D3`, `Stage18`
- OOD robustness limits: `E4`
- Federated/privacy (historical + strict): `E5`, `E8`, `Stage13/14/15`, `Stage16/17`
- Modular transitivity narrative: `Stage19`, `Stage20/21`, `Stage25/26`, `Stage29/30/31/32`, `Stage33`, `Stage34/35`
