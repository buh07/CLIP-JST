# Augmenting JL+Mahalanobis with Original-Input Concatenation: Design, Experiments, and NeurIPS Implications

**Context.** The CLIP-JST project's central hypothesis (frozen sparse JL + small Mahalanobis ≈ CLIP projection head) was empirically refuted: at $m=256$ on COCO, JL+Mahal reaches only ~40% of the CLIP head's retrieval (avg_R 0.0668 vs 0.1661), with the gap consistent across three backbones. E7+D1 located the mechanism: JL flattens the feature singular spectrum (2.64–2.73 vs CLIP head's 5.29→1.74 decay), discarding the data-dependent expansion/shrinkage capacity that contrastive learning needs. The N6 narrative — *the same property that makes JL lose at retrieval makes JL good at inversion-resistance* — is the paper's strongest available framing, but its retrieval-side argument needs a positive proposal, not just a characterized failure.

This document analyzes two such proposals: (1) **full concatenation** of the JL output with the raw input feature, and (2) **mask concatenation**, where the second component is a random coordinate-wise mask of the input rather than the full input. Both can be added to the existing pipeline as new model variants without changing the training loop.

---

## 1. Proposal A: Full concatenation $z(x) = [\alpha R x \,;\, \beta x]$

### 1.1 Design

Replace the embedding $\Phi(x) = R x \in \mathbb{R}^m$ with the augmented embedding
$$
z(x) = [\alpha R x \,;\, \beta x] \in \mathbb{R}^{m+d},
$$
where $R$ is the existing frozen sparse JL matrix and $\alpha, \beta \geq 0$ are scalar weights. A learnable Mahalanobis head $M^{1/2} \in \mathbb{R}^{(m+d) \times (m+d)}$ is then trained on top with the existing InfoNCE loss.

The endpoints of the $\alpha/\beta$ ratio recover known baselines:

- $\beta = 0$: pure JL+Mahalanobis (the refuted baseline, E1's `jl_mahal_rfull`).
- $\alpha = 0$: Mahalanobis on raw features only (E2's `mahal_only_rfull`, which already achieves avg_R = 0.5385 on COCO at $m=256$ — beating the CLIP head).
- $\alpha, \beta > 0$: a tunable interpolation between the two.

### 1.2 Theoretical grounding

The construction is a special case of the Indyk–Vakilian–Yuan (NeurIPS 2019) vertical concatenation trick. Because the JL component alone satisfies the Bourgain–Dirksen–Nelson distortion bound, the augmented matrix inherits a JL-style guarantee: for any $(x, y)$,
$$
\|z(x) - z(y)\|^2 \in (\alpha^2 + \beta^2) \|x - y\|^2 \cdot \big[1 - \varepsilon \tfrac{\alpha^2}{\alpha^2 + \beta^2},\; 1 + \varepsilon \tfrac{\alpha^2}{\alpha^2 + \beta^2}\big]
$$
with probability $\geq 1 - \delta$ over $R$. The effective distortion is $\varepsilon \cdot \alpha^2 / (\alpha^2 + \beta^2)$, which is at most $\varepsilon$ (recovering ordinary JL when $\beta = 0$) and strictly smaller when $\beta > 0$. **Distance preservation can only improve** relative to pure JL.

The representational story is even simpler: the hypothesis class for $z(x)$ strictly contains the hypothesis class for $Rx$. Any expansion/shrinkage along original-feature axes — exactly what Gui–Chen–Liu (NeurIPS 2023) identify as the load-bearing component of CLIP-style heads — is now reachable by the Mahalanobis on top of $z(x)$, but unreachable by the Mahalanobis on top of $Rx$ alone. **This addresses the precise failure mode that E7+D1 diagnosed.**

### 1.3 What the design buys and what it costs

| Property | Pure JL+Mahal | Full concat | Mahal-only |
|---|---|---|---|
| Distance-preservation guarantee | $\varepsilon$ | $\leq \varepsilon$ | exact |
| Expansion/shrinkage capacity | none on raw axes | full on raw axes | full |
| Inversion-resistance (E5-style) | strong (32% recon error at $m=256$) | weak ($x$ is in the clear) | none |
| Trainable params | $m^2$ | $(m+d)^2$ | $d^2 + m \cdot d$ |
| Federated transmission | only $Rx$ | $Rx$ + $x$ (defeats privacy) | $x$ (no privacy) |

The headline observation: **full concatenation does not preserve E5's privacy story**, because the unprojected $\beta x$ component is transmitted in clear. So Proposal A is a retrieval-side win and a privacy-side loss.

---

## 2. Proposal B: Mask concatenation $z(x) = [\alpha R x \,;\, \beta M x]$

### 2.1 Design

Replace the second component with a random coordinate-wise mask: $M \in \{0, 1\}^{d \times d}$ is a fixed diagonal matrix with iid Bernoulli($p$) diagonal entries, drawn once and frozen alongside $R$. The augmented embedding becomes
$$
z(x) = [\alpha R x \,;\, \beta M x] \in \mathbb{R}^{m+d}
$$
with effective dimensionality $m + pd$ (since $(1-p)d$ entries are deterministically zero). The endpoints recover Proposal A at $p = 1$ and pure JL+Mahalanobis at $p = 0$.

### 2.2 What masking actually preserves (and what it doesn't)

A common intuition — that masking preserves "local distances" while JL preserves "global distances" — is **not quite right** in the standard $\ell_2$ sense. For $M$ with diagonal Bernoulli($p$) entries:
$$
\mathbb{E}\|M(x-y)\|^2 = p \|x - y\|^2, \quad \mathrm{Var}\,\|(M/\sqrt{p})(x-y)\|^2 = \tfrac{1-p}{p} \sum_i (x_i - y_i)^4.
$$
Variance is dominated by the fourth moment of coordinate differences, which is *large* when $x - y$ is sparse. So masking is actually **worse** than JL at preserving distances of sparse-difference vectors — exactly the regime ("local" differences in coordinate space) where the intuition predicts it would help. JL's coordinate-mixing is what gives it uniform $(1 \pm \varepsilon)$ concentration regardless of difference sparsity; masking lacks this.

What masking *does* preserve is **coordinate identity**. The surviving entries of $Mx$ are literal values of $x$, not random combinations. This matters because:

1. The downstream Mahalanobis head can route directly to specific input coordinates rather than learning to undo random mixing. Axis-aligned expansion/shrinkage is preserved verbatim.
2. **Per-coordinate privacy becomes analyzable**: an honest-but-curious adversary observing $Mx$ learns the values of a uniformly random $p$-fraction of $x$'s coordinates and *exactly nothing* about the other $(1-p)$-fraction (information-theoretically). This is a coordinate-level differential-privacy-like guarantee, not a property [Rx ; x] has.

So the framing should shift: masking doesn't preserve "local distances," it preserves *axis-aligned discriminative directions* (a useful subset of expansion/shrinkage capacity) while exposing only a random fraction of coordinates.

### 2.3 The privacy-utility trade-off, with two knobs

This is the conceptually important part. Pure JL+Mahalanobis offers a 1-D privacy-utility curve parameterized by $m$. Mask concatenation offers a **2-D frontier** parameterized by $(m, p)$:

- $m$ controls the JL component: more dimensions → better retrieval + worse inversion-resistance on the $Rx$ channel.
- $p$ controls the mask component: higher sparsity (lower $p$) → fewer leaked raw coordinates + less retrieval capacity from axis-aligned directions.

The 2-D knob is more informative experimentally and more compelling rhetorically — privacy frontiers, not privacy points.

### 2.4 Comparison to Proposal A

| Property | Full concat ($p=1$) | Mask concat ($p \in (0,1)$) | Pure JL ($p=0$) |
|---|---|---|---|
| Retrieval ceiling | highest | interpolates | lowest (refuted) |
| Distance guarantee | $\leq \varepsilon$ (Indyk-style) | $\leq \varepsilon$ (Indyk-style; mask is contractive) | $\varepsilon$ |
| Privacy on raw features | none | per-coordinate ($1-p$ hidden) | full (only $Rx$ visible) |
| Privacy story | binary | tunable | binary |
| Federated transmission | reveals $x$ | reveals random $p \cdot d$ entries of $x$ | reveals only $Rx$ |

**Mask concatenation strictly dominates full concatenation in the privacy dimension** while losing some retrieval. For a paper organized around the privacy-utility trade-off (the N6 narrative), this is the more interesting variant.

---

## 3. Predictions before running the experiments

Before any experiment is run, the predictions worth pre-registering:

1. **Full concatenation closes most of the JL+Mahal-vs-CLIP-head gap.** At $m=256$ on COCO, expect avg_R for `[Rx ; x] + Mahal` to be in the range 0.45–0.55 — likely *above* the CLIP head (0.166) and at or near `mahal_only_rfull` (0.539). Confidence: high. The hypothesis class strictly contains `mahal_only_rfull`; if optimization works at all, retrieval should match or slightly exceed.

2. **Mask concatenation interpolates monotonically in $p$.** At $p=0$, avg_R = 0.067 (pure JL+Mahal). At $p=1$, avg_R ≈ Proposal A. In between, retrieval should increase smoothly with $p$. Whether the curve is concave (most gain at small $p$) or roughly linear is the empirically interesting question.

3. **Inversion resistance on the $Rx$ channel is unchanged from E5** (~32% reconstruction error at $m=256$), regardless of $p$. The mask channel reveals exactly $p \cdot d$ raw coordinates and reveals zero information about the other $(1-p) \cdot d$ — this is a structural property, not an empirical one.

4. **At small $p$ (say $p = 0.1$–$0.25$), retrieval should already be well above pure JL+Mahal** while exposing only 10–25% of raw coordinates. If true, this is the publishable result: the privacy-utility curve has a favorable knee.

5. **Full concatenation does not improve a fully-learned CLIP head.** Concatenating $[Rx ; x]$ at the input of a learned linear projection $W$ gives $W' x$ for some $W'$ — same hypothesis class. So Proposal A is a no-op on `clip_head` and `orth_jl_trainable`. The improvement is specific to the Mahalanobis-on-frozen-projection setting.

---

## 4. Experimental plan

### 4.1 Stage 1 — Proposal A baseline (1–3 days of compute)

**E8a: Full concatenation sweep.** Train a `concat_jl_mahal` model with input $[\alpha R x \,;\, \beta x]$ and full Mahalanobis on the $(m+d)$-dim concatenated vector.

- Backbone: CLIP ViT-B/32 (matches E1).
- Datasets: COCO and Flickr30K.
- Sweep $m \in \{64, 128, 256, 512\}$.
- Sweep $\alpha/\beta \in \{(1, 0), (1, 0.1), (1, 0.5), (1, 1), (1, 2), (0, 1)\}$ — endpoints recover existing baselines for free.
- 3 seeds.
- Compare against E1's `clip_head`, `jl_mahal_rfull`, and E2's `mahal_only_rfull`.

**Decision criterion:** if at $m=256, \alpha=\beta=1$, avg_R is within 5% of `mahal_only_rfull` (≥ 0.51), Proposal A works as predicted and the mechanism is confirmed. If avg_R lands well below `mahal_only_rfull` (say < 0.30), the optimization is leaving expressivity on the table and something else is wrong — investigate before continuing.

**Cost:** 6 configurations × 4 dimensions × 3 seeds × 2 datasets = 144 runs, each ~30 sec/epoch × 50 epochs ≈ 25 min. Total ~60 hours of A100 time. Fits in 3 days on a single GPU.

### 4.2 Stage 2 — Proposal B sweep (3–5 days of compute)

**E8b: Mask concatenation sweep.** Same setup as E8a, but the second component is $\beta M x$ where $M$ is a frozen Bernoulli($p$) diagonal mask.

- Sweep $p \in \{0.05, 0.1, 0.25, 0.5, 0.75\}$ (skip $p \in \{0, 1\}$ — covered by E1 and E8a).
- $m \in \{128, 256\}$ (the most informative dimensions).
- $\alpha = \beta = 1$ (fix the ratio at the value chosen from E8a).
- 3 seeds per $(p, m)$.
- 3 mask seeds per condition (to check mask-draw variance, paralleling controls).

**Cost:** 5 values of $p$ × 2 dimensions × 3 model seeds × 3 mask seeds × 2 datasets = 180 runs. ~75 hours on A100.

### 4.3 Stage 3 — Privacy curve (2–3 days of compute)

**E8c: Inversion-attack reruns** on the new variants, paralleling E5's protocol.

- For each of: pure JL+Mahal ($p=0$), mask concat at $p \in \{0.1, 0.5\}$, full concat ($p=1$):
  - Train a 3-layer MLP attacker to reconstruct $x$ from the visible component(s).
  - Report relative reconstruction error, separately for the JL channel and the mask channel.
  - For mask concat, also report per-coordinate reconstruction error to verify that hidden coordinates remain hidden.
- Add the pseudoinverse baseline as in E5.

**Cost:** 4 configurations × 3 seeds × 2 attackers = 24 attacker training runs, ~2 hours each ≈ 50 hours.

### 4.4 Stage 4 — DP-SGD comparison (1 week of compute, the highest-leverage experiment)

**E8d: DP-SGD on `mahal_only_rfull`** at $\varepsilon \in \{1, 4, 8\}$ with the same MLP attacker. This is the experiment that decides whether the project's privacy story is publishable: if mask concat lands on the Pareto frontier of (retrieval, inversion error) at any privacy level, N6 is in. If DP dominates everywhere, the privacy framing collapses regardless of how good Proposals A/B are.

**Cost:** 3 privacy budgets × 3 seeds = 9 DP-SGD runs (each slower than vanilla because of per-sample gradient clipping; budget ~6 hours each) + attacker reruns. ~80 hours.

### 4.5 Total and prioritization

Stages 1 + 2 + 3 + 4 ≈ 10 days of single-A100 time, parallelizable across multiple GPUs to ~3 days. Strict ordering matters:

1. **Run Stage 1 first.** If full concat doesn't recover `mahal_only_rfull`-level retrieval, the whole storyline doesn't work and you stop here.
2. **Run Stage 4 in parallel with Stage 2.** Stage 4 is the experiment most likely to kill the N6 framing entirely; learning that early is valuable.
3. **Run Stage 3 last**, conditional on Stages 1 and 4 producing favorable results.

---

## 5. NeurIPS implications

### 5.1 If the predictions hold

The paper becomes the principled-trade-off story that N6 anticipated, but with the positive retrieval result that the original project lacked. The headline becomes something like:

> *Hybrid JL projections offer a tunable privacy-utility frontier for cross-modal retrieval. A frozen sparse JL alone discards data-dependent expansion/shrinkage directions essential for contrastive learning, losing 60% of CLIP-projection-head retrieval on COCO. Concatenating the JL output with a random coordinate-wise mask of the input recovers most of the lost retrieval (within X% of the CLIP head at $m=256$, $p=0.5$) while preserving inversion-resistance for $(1-p)$ of the input coordinates by construction. The same mechanism — JL's spectral flattening — explains both the failure of pure JL+Mahal at retrieval and the privacy guarantee of the mask channel: lost discriminative information is, equivalently, lost reconstructive information. We characterize the 2-D Pareto frontier in $(m, p)$, demonstrate it dominates DP-SGD on raw features in the [moderate privacy / high utility] regime, and validate the federated training story across three vision-text backbones.*

The structural strengths of this framing for NeurIPS:

- **No refuted claim left standing.** The original parity-with-CLIP-head hypothesis is reframed as a mechanistic finding (E7+D1) that *motivates* the hybrid design.
- **Unified mechanism.** Spectral flattening explains both the retrieval cost and the privacy benefit. This is the kind of argument NeurIPS reviewers respond to: not "we built a thing that works" but "we identified a property that has consequences."
- **Tunable trade-off, not a point.** A 2-D Pareto frontier is more publishable than a single design point.
- **Positive practical result.** Federated cross-modal retrieval with formal-DP-comparable privacy and competitive retrieval is a real contribution to a real problem.
- **Multiple supporting experiments.** E1, E2, E5, E6, E7, D1, D3 all become evidence for the unified story instead of damage control. D3's relationship-graph quality finding becomes a clean side-result on top.

Estimated NeurIPS acceptance odds in this scenario: **~22–32%**, contingent on the DP-SGD comparison being favorable. Above baseline if the mask channel offers a regime DP-SGD doesn't.

### 5.2 If the predictions partially hold (likely scenario)

Most realistic outcome: full concat works well (Proposal A recovers `mahal_only_rfull`-level retrieval), but the mask interpolation has a less clean shape than predicted, and the DP-SGD comparison is mixed — mask concat dominates in some regime but not uniformly. In this case the paper still works, but the framing has to be more careful: "we characterize a hybrid family that admits a tunable privacy-utility trade-off; under [specific conditions] it offers an alternative to formal DP that may be preferable for [specific use cases]."

This is still publishable at NeurIPS but probably closer to **~18–25%** odds. The narrower claim is more defensible but less rhetorically strong. TMLR remains the safe alternative at any level.

### 5.3 If the predictions fail

If full concatenation doesn't help (avg_R stays in the 0.10–0.20 range despite the strictly larger hypothesis class), something deeper than expansion/shrinkage is the bottleneck. This would actually be the most scientifically interesting outcome — it would suggest that contrastive learning's failure mode under JL is not just spectral but optimization-dynamical. But it would kill the N6 paper. In this case the right move is TMLR or a workshop, with the negative result and mechanism characterization (E7+D1) as the core contribution.

NeurIPS odds in this scenario: **~5–10%**. The paper becomes a pure mechanism-and-negative-results story, which NeurIPS reviewers usually undervalue.

### 5.4 What this path is not

Worth being explicit about three things this path will *not* deliver:

1. **It will not vindicate the original proposal's central hypothesis.** Pure JL + small Mahalanobis remains decisively worse than learned projection heads. The pivot is to the hybrid family and the privacy story; the original parity claim is dead.
2. **It will not provide a formal generalization bound** of the form Claim 2 in the proposal. The Indyk-style distortion bound for the augmented embedding is straightforward; the InfoNCE generalization bound on top of it remains hard, and is not necessary for an empirically-driven NeurIPS paper. Save it for a follow-up.
3. **It will not demonstrate width-adaptive scaling.** E3 already showed the width estimator is too coarse; nothing in Proposals A or B addresses this. The width theory thread should be quietly dropped or relegated to a "limitations" section.

### 5.5 Recommended decision tree

After Stages 1 and 4 complete (≈ 2 weeks):

- **Both favorable** (full concat closes the gap, mask concat lands on Pareto frontier): commit to N6. Run Stages 2–3 to fill in the curve. Write for NeurIPS. Estimated acceptance ~22–32%.
- **Stage 1 favorable, Stage 4 unfavorable** (DP dominates everywhere): the privacy story is dead, but the retrieval story still works. Pivot to a paper about hybrid JL/raw embeddings as a parameter-efficient projection family; submit to TMLR or an efficient-ML venue. Estimated acceptance ~50%.
- **Stage 1 unfavorable**: the deeper-than-expected failure mode story. Workshop or TMLR with E7+D1 as the core mechanism. Estimated NeurIPS odds ~5–10%, TMLR ~50–60%.

The first ~2 weeks of experiments resolve most of the uncertainty. That's a remarkably efficient way to know which paper you're writing.

---

## 6. Summary

Proposal A (full concatenation) is the obvious move for retrieval and has clean theory: the hypothesis class strictly contains `mahal_only_rfull`, so retrieval should match or exceed it. But it destroys E5's privacy story by transmitting $x$ in clear.

Proposal B (mask concatenation) is the more interesting design for a privacy-framed paper. It interpolates between Proposal A and pure JL+Mahalanobis with a single knob $p$, generalizing rather than replacing both, and offers a per-coordinate privacy guarantee that full concatenation lacks. The "local distance preservation" intuition is technically backwards in the $\ell_2$ sense, but the underlying point — that the mask preserves axis-aligned discriminative directions while protecting non-revealed coordinates — is sound.

Together, the two proposals turn the project from "characterized failure with a privacy spinoff" into "characterized failure that *motivated* a hybrid design with a tunable Pareto frontier." That reframe is what makes N6 a real NeurIPS contender, conditional on the DP-SGD comparison going favorably. Two weeks of experiments resolve most of the remaining uncertainty about which paper you're writing, and which venue.
