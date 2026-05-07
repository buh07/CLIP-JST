# Revised Narrative (Theory-Driven): Why This Work Matters

## 0) Purpose of this document
This file is a **logic map** for the paper, not just a result dump.

For each core claim, it specifies:
- the theory/hypothesis statement,
- the identification strategy,
- the exact experiments that test it,
- what alternative explanations are ruled out,
- and why the result matters in the bigger picture.

---

## 0.1) Latest Experimental Results (2026-05-06)

All NeurIPS reviewer-fix experiments are complete. Summary of new findings:

**W3 / Stage67 — Causal geometry intervention**
- Centroid-gap regularization in Phase B causally increases av_ia at m=256 (Δ=+0.0032, raw p=0.0019) and m=512 (Δ=+0.0039, raw p=0.0051). Both Holm-significant.
- av_at decreases slightly (Δ≈−0.0045) but Holm non-significant (p=0.086).
- Language update: "centroid gap **is causally associated with** (not merely predicts) transfer efficiency."

**W5 — Sharing reversal mechanism: centroid gap ruled out**
- W5 analysis across all 4 dims: at m=256 (reversal dim), shared vs separate ia_gap = 0.585 vs 0.582 (Δ=−0.003, smallest divergence across all dims).
- The alignment hypothesis (shared JL increases modality gap → causes reversal) is refuted.
- At m=128, separate has *smaller* gap but *lower* av_ia — sign reversal. Mechanism remains open.

**W12 — Shuffled caption control at m=64 and m=256**
- Chance behavior (av_at≈0.001, av_ia≈0.0005) confirmed at m=64 and m=256, consistent with m=512 (W11).
- The causal Phase-B bridge evidence holds dimension-independently.

**W13 — 5th WavCaps condition (mixed100k)**
- mixed100k at m=512: av_at=0.1100±0.0011, av_ia=0.0267±0.0018, wav_holdout_at=0.0865±0.0029.
- Confirms non-monotone WavCaps–av_ia scaling: mixed46k (40K) > mixed100k (95K) ≈ mixed200k (142K) on av_ia.
- av_at IS monotone with mixed scale. Quality (annotation richness) limits av_ia, not training size.

**W14 — Phase A source ablation: COCO-subsampled vs CC3M vs COCO-full**
- At matched scale (100K pairs), COCO-sub outperforms CC3M by +24% (m=512) to +53% (m=256) on av_ia.
- COCO-sub also outperforms COCO-full (5.6× more pairs) at every dimension — more COCO training data does not help beyond 100K pairs.
- Phase A data quality (human-written COCO captions) dominates Phase A data volume.

**Stage69 — Third modality triple: SpeechCoco spoken captions (pre-registered 2026-05-06)**
- 60 cells (3 methods × 4 dims × 5 seeds), all complete. Audio: spoken COCO captions via CLAP HTSAT-unfused; same encoder and backbone as AudioCaps experiments.
- Bottleneck theory holds: av_ia ≈ α·sqrt(av_it·av_at) is accurate at every (dim, method) cell.
- α (pooled m≥128, all methods) = **0.0198 ± 0.0033** — 13× lower than AudioCaps (α≈0.27), 34× lower than AVCaps (α≈0.67).
- Proximate cause: av_at for speech audio is weak (≈0.11 at m=512 vs ≈0.23 for AudioCaps). CLAP HTSAT-unfused was not trained on speech; its audio→text alignment for spoken language is structurally weaker than for environmental sounds.
- Alpha plateaus after m=64 (0.037 → 0.020 at m=128, then flat through m=512) with tight std (0.002 at m=512). More embedding capacity cannot compensate for the encoder ceiling.
- Method convergence: at m=512, all three methods give α within 0.0025 of each other — when the encoder is the bottleneck, method choice is irrelevant.
- Three-triple summary: AudioCaps α=0.27, AVCaps α=0.67, SpeechCoco α=0.020 — two orders of magnitude, one geometric mean structure.

---

## 1) Paper scope and central question

### Central question
Can we add a new modality (audio) to a pretrained image-text model **without** direct image-audio supervision and **without** full joint retraining, while still getting useful zero-shot image-audio transfer?

### Scope boundaries
- This is a **modular transfer** paper under supervision constraints.
- It is **not** a claim to outperform direct-supervision systems (ImageBind/AudioCLIP) on absolute retrieval.
- "Theory" is used as shorthand for an empirical regularity; we do not claim a modality-agnostic physical rule.

---

## 2) Executive story in one paragraph
The paper establishes that text-anchored modular transfer is real, measurable, and partially predictable. Transfer emerges only when Phase-B semantic supervision is meaningful — identity and shuffled-caption controls collapse to chance across all tested embedding dimensions (m=64, 256, 512), confirming the causal role of the text bridge. Transfer quality is dominated by data quality in both training phases: high-quality human-written supervision (COCO > CC3M in Phase A; AudioCaps > WavCaps in Phase B) consistently outperforms matched-scale but lower-quality alternatives, and additional scale shows diminishing or negative returns in several conditions. Geometry is a genuine causal pathway, not just a correlational one: directly reducing the image-audio centroid gap in Phase B significantly increases image-audio transfer (Δ≈+0.003, p<0.01 at both tested dimensions). Across many controlled conditions, image-audio transfer is well approximated by a multiplicative bottleneck relation `av_ia ≈ α·sqrt(av_it·av_at)`, with strong held-out predictive performance. Cell-mean reanalysis (Stage68) confirms the fit is not a seed-replication artifact — r² improves on cell means (0.83→0.84 AudioCaps; 0.87→0.90 AVCaps) and mixed-effects models give tight α CIs. The geometric form is the best cross-regime generalizer: on held-out WavCaps prediction after AudioCaps training, geometric achieves r²=0.85 while all other tested forms collapse or fail catastrophically. The same pattern appears on a second triple (AVCaps, α≈0.67) and a third triple (SpeechCoco spoken captions, α≈0.020), which together span two orders of magnitude of α and define both the strength and limits of the claim: the geometric mean structure is consistent across tested triples, but α is regime-dependent and serves as a diagnostic of encoder-domain fit. The SpeechCoco result establishes an explicit boundary condition — using CLAP HTSAT with speech audio collapses the transitive bridge to near chance because av_at is structurally weak, regardless of embedding dimension or training volume. Geometry is a controllable post-hoc lever (Stage67), not a pre-flight predictor (Stage68-D: pre-Phase-B gap explains only 4.6% of final α variance).

---

## 3) Theory/Hypothesis map with concrete reasoning

## T1. Emergence theory (existence): text-anchored transfer emerges without direct image-audio pairs

### Statement
Given Phase A (image-text) and Phase B (audio-text), both text-anchored, zero-shot image-audio retrieval should be above chance **even without direct image-audio training pairs**.

### Identification logic
To show true emergence, we must rule out two confounds:
1. accidental encoder pre-alignment,
2. non-semantic optimization artifacts.

### Tests and results
- **Stage44** (full Phase-B training, COCO-matched):
  - m=512 modular_shared_jl: `av_at=0.2256`, `av_ia=0.0339`
  - m=512 audio_linear_probe: `av_at=0.2328`, `av_ia=0.0352`
- **Stage47 identity ablation** (no Phase-B learning): chance-like across all dims
  - m64 `av_ia=0.00039`, m128 `0.00066`, m256 `0.00134`, m512 `0.00149`
- **W11 shuffled-caption control** (m=512, 5 seeds): `av_at=0.0012`, `av_ia=0.0013` — chance level.
- **W12 extension** (m=64 and m=256, 2 seeds each): chance behavior confirmed at all additional dimensions.
  - m=64: `av_at=0.0012`, `av_ia=0.0005`; m=256: `av_at=0.0010`, `av_ia=0.0005`
  - Chance behavior is dimension-independent.

### Why this is strong evidence
- Stage44 demonstrates nontrivial transfer.
- Stage47 removes learned Phase-B alignment; performance collapses.
- W11+W12 keep the optimization process intact but destroy semantic supervision; collapse is consistent at m=64, 256, and 512.
- Critically, W12 rules out the dimension-capacity confound: even at m=64, where the embedding space is severely constrained, the shuffled control still reaches chance. The emergence result is not a large-m artifact.

Together these reject both confounds and support the existence claim across the full range of tested embedding dimensions.

### Why this matters
This is the minimum condition for modular modality onboarding in real systems: if this fails, the whole strategy is unusable. The dimension-independence established by W12 means the causal claim holds at deployable small-m settings, not just at the largest tested configuration.

---

## T2. Bottleneck theory (dominant factor): Phase-B supervision quality is the primary limiter

### Statement
Transfer quality is constrained more by Phase-B supervision quality than by naive data scale alone.

### Identification logic
Need to separate:
- scale effect,
- domain-shift effect,
- supervision-quality effect.

### Tests and results
- **Stage38 quality mechanism**: AudioCaps margin `0.567` vs WavCaps(real) `0.249`.
- **Stage31/45/56**: WavCaps scale improves `av_at` but not uniformly `av_ia` at large m.
- **W3 Clotho intermediate**: supports mixed decomposition (domain shift + residual quality deficit).
- **W13 — 5th WavCaps condition (mixed100k, 94,500 training pairs)**:
  - av_at=0.1100±0.0011, av_ia=0.0267±0.0018, wav_holdout_at=0.0865±0.0029 (m=512, 3 seeds)
  - Full 5-condition picture at m=512: mixed46k av_ia=0.0285 > mixed200k=0.0279 > mixed100k=0.0267 > clean_source_46k=0.0248 > clean_source=0.0239, despite n_train spanning 40K–142K.
  - av_at and wav_holdout_at both increase monotonically with mixed scale; av_ia does not.
  - The key distinction: **wav_holdout_at (in-distribution WavCaps retrieval) increases with scale** (0.072 → 0.087 → 0.090), while **av_ia (out-of-distribution AudioCaps-eval image-audio) does not**. More WavCaps data makes the model better within the WavCaps domain but does not transfer to cross-modal image-audio performance. This is a domain specialization effect, not simply "quality beats scale."

- **W14 — Phase A source ablation (COCO-subsampled 20K images = 100K pairs)**:
  - Three-condition comparison: CC3M 100K vs COCO-sub 100K vs COCO-full 566K (all same Phase B: AudioCaps).
  - Results across dims (av_ia):

  | m | CC3M 100K | COCO-sub 100K | COCO-full 566K |
  |---:|---|---|---|
  | 64 | 0.0043 | **0.0117** (2.72×) | 0.0068 |
  | 128 | **0.0195** | 0.0140 (0.72×) | 0.0116 |
  | 256 | 0.0190 | **0.0291** (1.53×) | 0.0208 |
  | 512 | 0.0328 | **0.0407** (1.24×) | 0.0339 |

  - COCO-sub beats COCO-full at every dimension: more COCO training data beyond 100K pairs actively degrades downstream transfer, likely due to redundancy and overfitting to COCO's specific image distribution.
  - COCO-sub vs CC3M: COCO-sub wins at m=64, 256, 512, but **CC3M wins at m=128** (0.0195 vs 0.0140). At m=128, the diversity of 100K distinct web-scraped CC3M images outweighs COCO-sub's deeper but narrower coverage (20K images × 5 captions). At m=256+, COCO's caption quality advantage reasserts itself as the larger space benefits from fine-grained descriptions over noisy alt-text.

### What this rules out
It rules out the simple claim "more noisy web pairs always improve transfer." More WavCaps data reliably improves in-distribution audio retrieval but does not improve — and can slightly hurt — out-of-distribution image-audio transfer (W13). It rules out the claim that COCO's Phase-A advantage over CC3M is purely quantitative — COCO-sub 100K outperforms CC3M 100K in three of four dimension settings (W14). And it rules out the claim that more Phase-A training data is always beneficial — COCO-sub 100K beats COCO-full 566K at every dimension (W14).

### Why this matters
The quality signal propagates through both training phases and in related but distinct ways. In Phase B, the binding constraint is distribution match between training audio and evaluation domain: scale helps within-distribution but does not transfer cross-modality. In Phase A, the binding constraint is caption precision and image diversity: high-quality human captions matter at large m, but image variety matters at small m. An actionable design rule follows: (1) for Phase B, prioritize domain-matched or high-quality supervised audio-text pairs over scale; (2) for Phase A at m≥256, prefer clean image-text supervision over larger noisy web-scraped datasets at matched scale; (3) do not increase Phase-A training volume beyond a sufficiency point.

---

## T3. Geometry theory (mechanistic pathway): geometry predicts transmission efficiency

### Statement
Centroid-level geometry is a predictive mechanism for transfer efficiency α, and intervening on geometry can move transfer.

### Identification logic
A purely correlational claim is weak; require both:
1. predictive association,
2. intervention response.

### Tests and results
- **Stage60 (joint extension)**:
  - theory fit on joint rows: `r=0.949`, `alpha=0.302`, `n=40`
  - gap→alpha: `r=-0.865`
- **Stage65 calibration**:
  - linear models generalize more reliably than isotonic for `alpha_local`
- **Stage66/67 intervention** (centroid regularization, Phase-B loss = InfoNCE + 0.1 × centroid_alignment_penalty):
  - m=256 (4 seeds): baseline av_ia=0.0201 → gap-reg av_ia=0.0233, **Δ=+0.0032**, raw p=0.0019, Holm p=0.0115 — **significant**
  - m=512 (4 seeds): baseline av_ia=0.0340 → gap-reg av_ia=0.0379, **Δ=+0.0039**, raw p=0.0051, Holm p=0.0256 — **significant**
  - av_at decreases slightly (Δ≈−0.0045 at m=256) but Holm p=0.086 (non-significant)
  - Combined combined_avg_R is unchanged (Δ≈0, non-significant) — the gain is specifically in the cross-modal leg

### Scope of the geometry claim (W5 and Stage68-D caveats)
The geometry claim applies to overall transfer efficiency: reducing the image-audio centroid gap causally improves av_ia (Stage67). It does **not** fully explain every between-method ordering, and the gap is a post-Phase-B observable, not a pre-Phase-B predictor. Specifically:

- The m=256 sharing reversal (separate_jl av_ia=0.0263 > shared_jl av_ia=0.0204) is **not** explained by centroid-gap divergence. W5 analysis across 5+ seeds per dim finds:
  - m=256: shared ia_gap=0.585±0.004 vs separate ia_gap=0.582±0.003, Δ=−0.003 — essentially zero, the smallest divergence across all four dimensions
  - m=128: separate has a *smaller* gap (0.535 vs 0.569) yet *lower* av_ia (0.007 vs 0.012). The sign inverts — a smaller gap at m=128 is associated with *worse* performance, directly contradicting the alignment hypothesis.

- The gap therefore explains aggregate efficiency (how well a given method transmits across all conditions) but not the specific ordering of shared vs separate JL at a particular dimension. The reversal mechanism remains an open research question. The most parsimonious remaining hypothesis is a capacity interaction: at m=256, forcing shared projection coordinates for both image and audio creates a conflicting subspace allocation that hurts audio specifically at that dimension. But this is untested.

- **The W5 finding does not weaken the geometry claim in T3.** Stage67 shows that when you directly reduce the gap by regularization, av_ia improves causally. W5 shows that the choice between JL variants does not meaningfully change the gap at m=256, so gap is not the *mechanism* for the reversal — they are separate phenomena.

- **Stage68-D (Phase-A geometry feasibility)**: Pre-Phase-B image-audio centroid gap does not predict post-Phase-B α: r=0.2149, r²=0.046, n=20. The gap at initialization explains only 4.6% of variance in the final transmission efficiency. Phase-B training itself shapes the gap, so geometry is only a useful observable and control tool **post-hoc** — not a pre-flight signal.

### Why this matters
Without intervention evidence (Stage67), geometry is just a plotting story. With Stage67, geometry becomes a controllable engineering lever for improving cross-modal transfer. The W5 scope clarification means the lever is real and effective, but it is not a complete mechanistic explanation of every method-ordering effect. Stage68-D further scopes the claim: the lever is post-hoc (you can monitor and reduce the gap during or after Phase B), not predictive of what gap you will get before running Phase B.

---

## T4. Bottleneck relation theory (predictive regularity):
`av_ia ≈ α·sqrt(av_it·av_at)`

### Statement
Across controlled conditions, transitive retrieval scales approximately with the geometric mean of the two bridge legs, modulated by method/regime efficiency α.

### Identification logic
Need:
1. in-sample fit,
2. held-out prediction,
3. robustness to seed replication,
4. cross-regime functional form stability.

### Tests and results
- **Stage36 (primary suite)**:
  - `alpha=0.2694`, `r=0.9207`, `r^2=0.8477`, `n=300`
- **Stage43 (OOS test)**:
  - fit on AudioCaps-trained rows (`n_train=300`)
  - predict held-out WavCaps rows (`n_test=80`)
  - `alpha_train=0.2820`, `r=0.9531`, `MAE=0.00220`
- **Stage64 (AVCaps form comparison)**:
  - geometric mean strong (`CV R^2=0.855`) and better than arithmetic/hard-min,
  - product/free-power fit AVCaps better (`0.876` / `0.913`).
- **Stage68-A (cell-mean refit)**:
  - AudioCaps: seed-level r²=0.8317 → cell-mean r²=0.8413; mixed-effects α=0.2701, 95% CI=[0.2550, 0.2852]
  - AVCaps: seed-level r²=0.8712 → cell-mean r²=0.9030; mixed-effects α=0.6795, 95% CI=[0.6477, 0.7113]
  - Fit quality improves on cell means — theory is not a seed-replication artifact.
- **Stage68-B (functional form, cross-regime held-out)**:
  - Geometric mean held-out r²=0.8530 (AudioCaps→WavCaps transfer); hard-min 0.2030; product 0.2101; free-power −1.1726
  - Geometric mean does NOT win within-suite CV on every suite (hard-min best on AudioCaps CV; free-power best on AVCaps CV) — but is the only form that maintains strong cross-regime held-out performance.
  - Hard-min collapses because the identity of the minimum leg (av_it vs av_at) swaps between AudioCaps and WavCaps Phase-B regimes. Free-power catastrophically fails because fitted exponents (a≈0.82, b≈−0.02) don't transfer when the balance of legs reverses.
- **Stage68-C (α-locked W14 prediction)**:
  - α=0.2820 from Stage43 AudioCaps training predicts W14 CC3M domain-gap conditions (n=80): r=0.9435, r²=0.8201, MAE=0.00350
  - Strong α-transfer across Phase-A variations within the same Phase-B regime. Moderate for COCO-sub (r=0.928, r²=0.65, n=12). W13 WavCaps Phase-B fails as expected (wrong regime α).

### What this means mathematically
The geometric relation is the best **parsimonious cross-regime generalizer**: it wins on held-out WavCaps prediction after AudioCaps training (r²=0.853), while all other forms collapse or fail catastrophically. Within a single suite, hard-min or free-power forms may fit better. The functional-form CI on free-power (b∈[−0.069, +0.038]) is consistent with b=0, supporting the (0.5, 0.5) geometric exponents as the correct null model.

α is regime-specific: within a Phase-B protocol, α transfers across Phase-A variations (CC3M, COCO-sub) with near-perfect rank correlation. Across Phase-B protocols (AudioCaps vs WavCaps), α shifts and must be refitted.

### Why this matters
This converts a qualitative bridge intuition into a quantitative planning tool. Stage68 confirms the theory is not a data-collection artifact (cell-mean) and that the geometric form is the correct cross-regime default even when within-suite alternatives appear to perform better — exactly the property needed for practical deployment, where you cannot tune the form to each new triple.

---

## T5. Cross-triple theory (portability with regime dependence)

### Statement
The bottleneck structure appears across multiple modality triples, but α is regime-dependent and reflects encoder-domain fit quality. Three distinct α regimes have now been identified and characterized.

### Tests and results

- **Stage58 AVCaps** (image + text + video-associated audio):
  - global: `n=80`, `alpha=0.6731`, `r=0.9595`, `r^2=0.8712`
  - per-method r: linear `0.820`, LoRA `0.878`, separate `0.963`, shared `0.990`
  - α=0.67 — higher than AudioCaps because video-associated audio is semantically and acoustically well-aligned with CLAP's training distribution.

- **Stage69 SpeechCoco** (image + text + spoken COCO captions, pre-registered 2026-05-06):
  - **60 cells** (3 methods × 4 dims × 5 seeds), all complete.
  - α (m≥128 pooled, n=45): **0.0198 ± 0.0033** — 13× lower than AudioCaps, 34× lower than AVCaps.
  - At m=512: av_it≈0.62, av_at≈0.11, av_ia≈0.005; geometric prediction exact.
  - Alpha drops from 0.037 (m=64) to 0.020 (m=128), then plateaus through m=512. Tight std (0.002 at m=512) confirms this is a stable regime, not noise.
  - Method convergence at m=512: all three methods give α within 0.0025 (audio_linear_probe=0.0215, modular_shared_jl=0.0221, modular_separate_jl=0.0196).

### Three-triple summary

| Triple | Audio type | n | α (high-dim) | Relative |
|---|---|---:|---:|---:|
| AudioCaps | Environmental sounds | 300 | 0.270 | 1.0× |
| AVCaps | Video-associated audio | 80 | 0.673 | 2.5× |
| SpeechCoco | Spoken captions (speech) | 60 | 0.020 | 0.07× |

All three use the same CLAP HTSAT-unfused encoder and CLIP ViT-B/32 backbone. The variable is audio type and the CLAP encoder's audio→text alignment quality for that type.

### What drives regime variation

The α difference is traced to av_at, not av_it. In SpeechCoco, av_it≈0.63 at m=512 — comparable to AudioCaps. But av_at≈0.11 for SpeechCoco vs ≈0.23 for AudioCaps. CLAP HTSAT-unfused was trained primarily on environmental sounds and music; it was not designed for speech recognition or spoken-language alignment. Speech waveforms carry semantic content that is fully aligned with the written text, but CLAP's representations don't exploit this alignment. The result is a 2× weaker audio-text bridge, which propagates through the geometric mean into a 13× lower α.

This interpretation is supported by the method convergence at all dims for SpeechCoco: when av_at is the binding constraint, no projection architecture can improve the cross-modal path. In AudioCaps, method-level variation (including the m=256 sharing reversal) reflects genuine architectural effects; in SpeechCoco, the encoder bottleneck swamps all architectural signal.

### α as an encoder-domain fit diagnostic

Stage69 converts α into a practical diagnostic: the ratio of observed av_ia to the geometric mean prediction (`α = av_ia / sqrt(av_it·av_at)`) measures how efficiently a given audio encoder and training protocol bridges its audio domain through text to image. Higher α generally indicates better encoder-domain fit; very low α values (as seen in SpeechCoco, α≈0.020) indicate likely mismatch between encoder pretraining and target audio type.

This makes the Bottleneck theory actionable for encoder selection: before investing in Phase-B training, a small-scale α pilot can quantify encoder-domain fit and predict whether the transitive bridge will be practically useful.

### The deployment boundary condition

SpeechCoco establishes that the text-bridge strategy has a hard lower bound on usefulness governed by the audio encoder's av_at. With av_at≈0.11 at m=512, av_ia≈0.005 — barely above chance for a 5,000-pair eval set (R@1≈0.001–0.002). This is not practically useful for retrieval. The boundary condition is: the strategy produces useful image-audio transfer only when the audio encoder achieves adequate audio-text alignment for the specific audio domain. CLAP HTSAT meets this condition for environmental sounds (av_at≈0.23, av_ia≈0.034 at m=512) but fails for speech.

To operate in the AudioCaps-regime α range with speech audio, a speech-specialized contrastive encoder (e.g., trained on spoken caption–written caption pairs) would be required. Stage69 defines where the boundary is; it does not require moving it for the current paper's claims.

### Why this is not overclaiming

We claim structure portability, not a shared constant across all settings. α shifts across triples — from 0.020 to 0.673 — which is expected under different audio encoder alignments and modality coupling strengths. The theory makes no prediction about the absolute magnitude of α, only its functional relationship to av_it and av_at. Stage69 is fully consistent with the theory; it reveals the range of regime variation the theory must accommodate.

### Why this matters

Three confirmed triples across two orders of magnitude of α move the result from "candidate principle" to a more robustly supported cross-modal regularity. The SpeechCoco result also provides the first explicit boundary condition and failure mode for the text-bridge strategy, making the theory more useful as a practical engineering tool: you can predict in advance whether a proposed encoder-domain combination will yield useful transfer by estimating the expected α from a small pilot experiment.

---

## 4) Relationship to prior work (and what is genuinely new)

### Relative to ImageBind / AudioCLIP
- Prior work: direct cross-modal supervision, stronger absolute image-audio retrieval expected.
- Our setting: no direct image-audio supervision by design.
- Contribution is not SOTA replacement; it is constrained-regime transfer characterization.

### Relative to language-pivot multimodal alignment
- Prior work establishes that language can anchor modalities.
- Our new value is the **controlled decomposition with causal verification**:
  - what governs transfer quality in both training phases (Phase-A and Phase-B quality dominate scale),
  - when scale helps vs hurts (in-distribution vs out-of-distribution; phase-specific scale limits in both phases),
  - how geometry enters and that it is **causally**, not merely correlationally, connected to transfer (Stage67),
  - an OOS-tested bottleneck relation that predicts held-out conditions,
  - and a mechanistic null for the sharing reversal (gap is not the driver — W5), scoping what the geometry claim can and cannot explain.

### Relative to scaling-theory discourse
- We should frame this as an empirical bottleneck regularity over controlled design conditions, not broad compute scaling.

---

## 5) Big-picture significance: do these findings matter?

### 5.1 Scientific significance

- Turns "text bridge" from intuition into testable, causally verified quantitative structure. Every major claim has independent experimental support: emergence (W11/W12), quality bottlenecks (Stage38/56/W13/W14), geometric mechanism (Stage67), theory robustness (Stage68), cross-triple portability (Stage58, Stage69).
- Separates emergence, quality bottlenecks, and geometric mechanism using independent controls — each can be switched on and off experimentally, and each collapses in the expected direction when turned off.
- Establishes that quality constraints hold across both training phases but manifest differently: in Phase B, the binding constraint is distribution match between training audio and evaluation domain (scale helps within-distribution but not cross-modal); in Phase A, it is caption precision and image diversity at m≥256, with image diversity mattering more at m≤128.
- Identifies open failure modes with concrete next steps: the m=256 sharing reversal mechanism is open (gap ruled out by W5; capacity-interaction hypothesis is the surviving candidate and directly testable); the SpeechCoco boundary condition motivates a speech-specialized encoder follow-up.
- Establishes a cross-regime functional form: the geometric mean is uniquely stable across Phase-B regimes (Stage68-B). This is a structural finding about the algebra of transitivity bottlenecks — the geometric form wins not because it fits any single suite best, but because it is the only form that does not collapse when predicting into a new regime.
- Provides three independent replications (AudioCaps α=0.27, AVCaps α=0.67, SpeechCoco α=0.020) spanning two orders of magnitude of α. The theory's form is consistent across tested triples; the constant is regime-specific. This is precisely the expected behavior of a genuine empirical regularity.
- Introduces α as an interpretable measure of encoder-domain fit. Stage69 isolates the audio type as the sole variable (same encoder, same backbone, same protocol, same COCO Phase A) and observes a 13× α difference — turning α into a quantity with a direct causal interpretation.

### 5.2 Engineering significance: concise deployment guidance (appendix has full details)

A concise, experiment-backed deployment guide for practitioners using modular multimodal extension (with the full decision workflow moved to appendix material):

**Step 1 — Verify emergence prerequisites.**
Run a shuffled-caption control before investing in Phase B infrastructure. If av_ia collapses to chance with shuffled labels, the setup is correct. If av_ia is above chance with shuffled labels, there is data contamination. Cost: one short ablation run. The control holds from m=64 to m=512 (W11/W12) — embedding dimension is not an escape hatch.

**Step 2 — Run an α pilot before committing to full Phase B.**
Take your candidate audio encoder, train Phase B for a few epochs on 10–20% of your planned dataset, measure av_at and av_it, and compute α = av_ia / √(av_it × av_at). Compare against prior successful runs in similar domains: very low α (SpeechCoco-like values) suggests encoder-domain mismatch, while AudioCaps/AVCaps-like values suggest a more favorable regime.

**Step 3 — Choose Phase B data strategically.**
Prioritize domain-matched, high-quality supervised audio-text pairs over scale. AudioCaps-quality annotation at 40K pairs outperforms 142K noisy WavCaps pairs on cross-modal (image-audio) transfer (Stage38, W13). More in-domain scale reliably improves within-distribution audio retrieval but does NOT improve — and can slightly hurt — out-of-distribution image-audio transfer. The binding constraint is annotation quality and distribution match, not volume.

**Step 4 — Choose Phase A data strategically.**
At m≥256, prefer clean image-text supervision over larger noisy web-scraped datasets at matched scale (W14: COCO-subsampled 100K beats CC3M 100K by +24–53% on av_ia at m≥256). Do not increase Phase A training volume beyond ~100K pairs — COCO-sub 100K beats COCO-full 566K on av_ia at every dimension. At m≤128, image diversity per pair matters more than caption depth — CC3M's wider variety can match or exceed COCO-sub's quality advantage at small embedding sizes.

**Step 5 — Monitor and regularize geometry post-Phase-B.**
If av_ia is lower than the theory predicts given your av_it and av_at, check the image-audio centroid gap. Centroid gap regularization (Stage67) adds ≈15% to av_ia at m=256 and m=512 with a small non-significant cost to av_at. Geometry is a post-training lever — do not try to predict α from pre-training geometry (Stage68-D: r²=0.046 only, near zero predictive power before Phase B runs).

**Step 6 — Use the Bottleneck theory for expected transfer budgeting.**
Once α is measured from a pilot, predict av_ia for any (av_it, av_at) combination: av_ia ≈ α × √(av_it × av_at). The theory transfers across Phase-A variations within the same Phase-B regime (Stage68-C: r=0.9435 on held-out CC3M conditions using AudioCaps-fitted α). For cross-regime prediction (different audio encoder or audio domain), α must be refitted. Do not apply AudioCaps α to WavCaps or SpeechCoco conditions.

### 5.3 Practical applications in full

The following use cases flow directly from specific experimental findings. For each, the relevant paper results are cited precisely.

---

#### Application 1: Incremental modality onboarding in deployed image-text systems

**Scenario**: A company has a deployed image-text retrieval system (e.g., Google Photos, Pinterest visual search, e-commerce product search) and wants to add audio search — let users query by audio clip or ambient sound — without rebuilding the system from scratch.

**What the paper enables**: A two-phase protocol that adds the audio modality by training only small projection heads (~300K parameters each) on audio-text pairs. The original image-text model is never modified. No paired image-audio examples are needed at training time.

**Why this is practically valuable**: Full joint multimodal retraining (ImageBind-style) requires: (a) massive paired datasets across all modalities, (b) a full training run on the combined corpus, (c) re-deployment of the entire model — large infrastructure costs with substantial regression risk. The modular protocol reduces audio onboarding to: collect ~40K high-quality audio-caption pairs, train two small projection heads, deploy. Phase B training converges in hours on a single GPU.

**Key results that apply**:
- *Existence* (Stage44): av_ia=0.034 at m=512 with zero image-audio pairs, on a 59K-image eval pool. Transfer is real and above chance.
- *Quality over scale* (Stage38, W13): 40K AudioCaps pairs outperform 142K noisy WavCaps pairs on av_ia. Don't scrape — annotate carefully.
- *Bottleneck theory* (Stage36/43): predict expected av_ia before committing to full training. Know performance tier before building infrastructure.
- *Encoder diagnostic* (Stage69): if considering non-standard audio types (music, speech, specialized sounds), run an α pilot first. CLAP fails for speech (α=0.020). Know before building the pipeline.

---

#### Application 2: Regulated and federated multimodal systems

**Scenario**: Healthcare, finance, legal — domains where raw data cannot leave each institution (HIPAA, GDPR, institutional policy). A consortium of hospitals wants multimodal retrieval (radiology images + clinical notes + patient audio) without centralizing any data.

**What the paper enables**: The modular protocol decouples training from data centralization. Phase A uses a public image-text corpus. Phase B is trained locally at each institution on its own audio-text pairs; only trained weights are shared, not raw data. Full federated training experiments under differential privacy budgets and adversarial attacks (Stage13–17) are left as future work; the current paper validates the modularity claim in centralized settings.

**Key results that apply**:
- *Quality over scale* (W13): in a federated setting where local data is limited, annotation quality matters more than pooled scale. Institutions should invest in careful local annotation rather than aggregating more data across privacy boundaries.
- *Bottleneck theory as early feasibility guidance*: if federated Phase B constraints strongly limit av_at, the theory suggests av_ia will remain near chance regardless of additional infrastructure effort. Treat this as an early warning signal rather than a fixed rule.
- *Modular independence* (Stage44 + Section 3): Phase B training runs independently with only audio-text data; no image data or image head updates required during Phase B. This structural property is the federated deployment enabler.

---

#### Application 3: Low-resource new modality addition

**Scenario**: A research team has a specialist audio type (wildlife recordings, industrial machine sounds, ultrasound, seismic data) with very few or no image-audio pairs, but some audio-text pairs (biologist annotations of birdsong recordings, engineer logs for machine faults).

**What the paper enables**: Phase B requires only audio-text pairs. Even at AudioCaps scale (~40K pairs), transfer is real. No image-audio pairs needed at training time.

**Key results that apply**:
- *Quality over scale* (W13): invest annotation budget in quality, not quantity. 40K precise human descriptions beat 142K noisy ones for cross-modal transfer. A team with a limited annotation budget should focus on high-quality descriptions rather than web scraping.
- *Phase A data choice* (W14): for Phase A, COCO or a COCO-quality subset at 100K pairs is a reasonable starting point even for specialist image domains. Do not use the full COCO training set — 100K curated pairs outperforms it.
- *Encoder pilot* (Stage69): before spending annotation budget on Phase B pairs, run an α pilot. For wildlife audio, CLAP likely works (similar to AudioCaps distribution); for speech-based audio, it does not. Know before annotating.

---

#### Application 4: Encoder selection via the α diagnostic

**Scenario**: An ML team is evaluating multiple audio encoders (CLAP-HTSAT, CLAP-large, wav2vec-2, music-specialist, whisper-based contrastive). Running full-scale Phase B training with each candidate is prohibitively expensive.

**What Stage69 specifically enables**: The α pilot protocol. For each candidate: train Phase B for a few epochs on 10–20% of planned data, compute α = av_ia / √(av_it × av_at), and compare with previously observed regimes. Extremely low α relative to prior successful runs signals a likely mismatch before full-scale investment.

**Why α is more informative than av_at alone**: An encoder could achieve decent av_at by mapping audio to a text representation that is locally correct but globally misaligned with the image embedding space. α captures whether the alignment is the *right kind* for transitive bridging. Stage69 proves this explicitly: CLAP achieves av_at=0.11 on SpeechCoco (non-zero, apparently functional), but α=0.020 because the alignment does not compose with image-text transitivity.

**Practical guidance from current data**:
- Higher α is generally associated with better encoder-domain fit and more useful transitive transfer.
- Very low α values (e.g., SpeechCoco with CLAP) indicate that changing encoder or domain alignment strategy is likely more valuable than increasing training volume.
- Intermediate regimes should be treated as pilot-and-validate settings rather than hard pass/fail cases.

**Broader use as benchmark**: α can standardize audio encoder evaluation for multimodal compatibility, analogous to how BLiMP evaluates language models on structural properties. Report (av_at, α) from a standard Phase B training run to characterize encoder multimodal-bridge quality. This is a more informative characterization than audio-text recall alone.

---

#### Application 5: Multimodal product search and media indexing

**Scenario**: E-commerce or media platforms want users to search image/video inventory by audio query — environmental soundscapes, ambient recordings, musical themes. A large image-text index already exists; no audio-image pairs do.

**What the paper enables**: Adding audio retrieval to an existing image-text index without re-indexing images or retraining the image encoder. Only a new audio projection head is trained and deployed. Audio queries at inference time map through the text bridge into the existing embedding space.

**Key results that apply**:
- *Theory as pre-deployment predictor*: if you know av_it (from existing system evaluation) and can pilot av_at (from a small Phase B run), you can predict av_ia before full deployment.
- *Quality interpretation*: AudioCaps-level results (e.g., av_ia≈0.034 at m=512) are detectable but still modest, so deployment targets should be matched to product needs.
- *Encoder selection*: use CLAP HTSAT for environmental/ambient sounds. Do not use it for spoken-language queries (Stage69: α=0.020 → av_ia≈0.005, near chance).

---

#### Application 6: Budget planning for multimodal retrieval projects

**Scenario**: A team has a fixed annotation and compute budget and must decide: how many audio-text pairs to annotate, what embedding dimension to deploy at (storage/latency cost vs quality), whether to fine-tune Phase A.

**What the paper enables**: A cost-quality curve parameterized by four levers — Phase B annotation quality, Phase A data quality, embedding dimension, and encoder fit (α). The theory provides the planning formula: av_ia ≈ α × √(av_it × av_at).

**Concrete planning example**: Targeting m=256 with AudioCaps-quality Phase B:
- Pilot shows av_it ≈ 0.040 (from Phase A eval), av_at ≈ 0.192 (from Phase B eval)
- Predicted av_ia ≈ 0.27 × √(0.040 × 0.192) = 0.27 × 0.088 = **0.024**
- Decision: is avg_R=0.024 on the target eval pool sufficient for deployment? If not, upgrade Phase B quality (not volume) or increase embedding dimension (if budget permits storage cost).

This pre-commitment evaluation requires no image-audio pairs and costs only one pilot Phase B run — potentially saving weeks of full training before discovering quality is insufficient.

---

#### Application 7: Sparse or delayed cross-pair supervision

**Scenario**: A team wants audio-image search but can only collect image-audio pairs slowly (crowdsourced annotation at 1K pairs/week). They need a working system now and want to improve incrementally.

**What the paper enables**: Phase B requires only audio-text pairs, which are much cheaper to collect (existing audio + transcription + annotation). A working system can be deployed immediately. As image-audio pairs accumulate over time, they can be used for evaluation and optional fine-tuning — but the text-bridge system is already useful without them.

**Key results that apply**:
- *Theory as ongoing diagnostic*: as image-audio pairs accumulate, periodically measure av_ia and compare to the theory prediction α × √(av_it × av_at). If observed av_ia << predicted, the geometry may have drifted — apply centroid regularization (Stage67). If it matches the prediction, the theory provides a lightweight health-check metric that does not require any new image-audio pairs.

---

#### Honest scope limitations (what this paper does NOT enable)

- **Beating direct-supervision systems on absolute retrieval numbers**: ImageBind and AudioCLIP use paired image-audio training and will always win on absolute metrics. This framework is for settings where such pairs are unavailable, too expensive, or privacy-restricted.
- **Speech audio retrieval with CLAP** (Stage69 proves this): av_ia≈0.005 at m=512, near-chance. A speech-specialized contrastive encoder is required.
- **Settings without a text modality as bridge**: the entire mechanism requires text as the anchor. Video-audio retrieval without text is outside scope.
- **Highly specialized image domains far from COCO's distribution**: the COCO Phase A generalizes to AudioCaps-domain images (generic scenes) but likely does not generalize to microscopy, satellite imagery, or other specialist domains without domain-adapted Phase A.
- **When av_at remains structurally weak at the target embedding dimension**: the Bottleneck theory suggests av_ia will remain near chance despite additional training effort. SpeechCoco (av_at=0.11 → av_ia=0.005) is the clearest current example.

### 5.4 Why the Bottleneck theory is the practical core of the paper

The theory does three things, in increasing order of practical importance:

**Summarizes**: av_ia ≈ α × √(av_it × av_at) compresses the entire cross-modal retrieval quality of a system into two observables (av_it, av_at) plus one constant (α). Instead of reporting a 6-number recall vector per modality pair, you get a single interpretable structure that connects both training phases to the final product.

**Predicts**: given av_it and av_at from a pilot, predict av_ia before full training. No prior work provided a planning formula for zero-shot cross-modal transfer quality. Stage43 demonstrates this is not just in-sample: r=0.9531 on held-out WavCaps conditions fitted purely on AudioCaps data, with MAE=0.00220. The theory is accurate enough to be a real planning tool.

**Diagnoses**: α = observed av_ia / √(av_it × av_at) is an encoder-domain fit diagnostic. Persistently low α relative to historical baselines suggests mismatch in encoder, domain, or supervision quality. This turns a black-box system property into a debuggable, interpretable number.

The Stage68-B finding is what makes the theory practically useful rather than just descriptively accurate: the geometric mean form is robust to regime changes in a way that all other forms are not. Hard-min collapses when the minimum-leg identity swaps (AudioCaps → WavCaps regime). Free-power fails catastrophically because fitted exponents reflect the specific balance of legs in one regime and break when that balance reverses. The geometric mean is the only form a practitioner can use *without knowing in advance which regime they will be predicting into* — exactly the real-world scenario. This is what makes it a deployable planning tool, not just a post-hoc statistical description.

---

## 6) What is settled vs open

### Settled (high confidence)
- Emergence under text anchoring is real and requires genuine semantic Phase-B supervision.
- The causal role of Phase-B text supervision is dimension-independent (W11 at m=512; W12 at m=64 and m=256).
- Phase-B data quality is a dominant bottleneck; more in-domain scale does not substitute (Stage38/56/W13).
- Phase-A data quality matters too: at m≥256, clean human-annotated pairs outperform matched-scale web-scraped data; more data beyond ~100K pairs is counterproductive (W14).
- WavCaps scale improves in-distribution (WavCaps holdout) retrieval but does not improve — and can slightly hurt — out-of-distribution image-audio transfer. The binding constraint is distribution match, not volume (W13).
- Multiplicative bottleneck relation has strong in-scope predictive validity including held-out cross-condition test and is confirmed as not a seed-replication artifact (Stage68-A: cell-mean r² improves; mixed-effects α CI tight).
- Geometric mean form is the best cross-regime generalizer: held-out r²=0.853 (AudioCaps→WavCaps); all other tested forms collapse or fail catastrophically on the same cross-regime test (Stage68-B).
- Within a Phase-B protocol, α transfers across Phase-A variations (Stage68-C: W14 CC3M r=0.9435, r²=0.82); α shifts across Phase-B protocols and must be refitted.
- Geometry (centroid gap) is a **causally verified** pathway: direct gap regularization significantly increases av_ia at m=256 (p=0.0019) and m=512 (p=0.0051), with a small non-significant tradeoff against av_at (W3/Stage67).
- Geometry is a **post-training** diagnostic and control tool, not a pre-flight predictor: pre-Phase-B gap explains only r²=0.046 of post-Phase-B α variance (Stage68-D).
- The bottleneck structure is confirmed on three distinct modality triples (AudioCaps, AVCaps, SpeechCoco), spanning two orders of magnitude in α (0.020 to 0.673) under the same geometric mean form. α is a measure of encoder-domain fit: the 13× gap between AudioCaps (α=0.27) and SpeechCoco (α=0.020) traces to CLAP HTSAT's weaker audio→text alignment for speech vs environmental sounds. (Stage69)
- The text-bridge strategy has an explicit lower-bound failure mode: when the audio encoder's av_at is structurally weak (due to encoder-domain mismatch), av_ia collapses to near-chance regardless of embedding dimension, training volume, or method choice. SpeechCoco establishes this boundary: av_at≈0.11, av_ia≈0.005 at m=512, α≈0.020. (Stage69)

### Open (important)
- Exact mechanism of m=256 sharing reversal. Centroid gap has been ruled out as the driver (W5). The most plausible remaining hypothesis is a capacity-related subspace-conflict, but this is untested.
- Full universality of the bottleneck relation across more triples, backbones, and domains. Three triples confirmed; functional form untested with other backbone encoders.
- Functional-form uniqueness beyond current suites (geometric mean is best parsimonious default, but product/free-power fit better on AVCaps).
- Whether the Phase-A quality finding at m=128 (CC3M diversity > COCO-sub depth) generalises, and where the crossover point lies (W14).
- Whether a speech-specialized contrastive encoder (e.g., trained on spoken caption–written caption pairs) would recover AudioCaps-like α values for SpeechCoco audio. Stage69 identifies the failure mode but does not test the remedy. The boundary condition (encoder-domain fit) is established, and current evidence suggests the weak-av_at regime is where failure is most likely.

---

## 7) Detailed paper outline (section-by-section with figure assignments)

This section is the full structural plan for the NeurIPS paper. For each section: narrative function, key claim(s), experiments cited, figure/table assignments, and the key numbers that must appear.

---

### Section 1: Introduction — Problem, constraints, and contributions

**Narrative function**: Establish the constrained-regime setting and motivate why it is common and important. Distinguish the paper from direct-supervision multimodal work (ImageBind, AudioCLIP, ALIGN) without positioning as a competition.

**Key claims**:
1. Adding a new modality to a deployed system without direct cross-modal pairs is the common case, not a niche. (Motivation: privacy, annotation cost, federated settings, incremental deployment.)
2. Text can serve as a shared anchor for zero-shot cross-modal transfer.
3. This paper establishes: when it works, what governs quality, how to quantitatively predict it, and when it fails — all with causal verification.

**What NOT to claim in Section 1**: Do not claim SOTA retrieval. Do not claim universality of the bottleneck constant. Position the paper as a *characterization under constraints*, not a *beat the baseline* paper.

**Experiments to cite**: None in detail. Stage44 av_ia=0.034 can appear as a teaser number.

**Figure assignment**: 
- **Figure 1 (system architecture)**: Two-panel diagram. Left panel: Phase A — CLIP ViT-B/32 (frozen), image projection head, text projection head, InfoNCE training on COCO pairs. Right panel: Phase B — CLAP HTSAT-unfused (frozen), audio projection head, same text head (shared or separate, shown as two options), InfoNCE training on AudioCaps pairs. Bottom: inference diagram showing zero-shot image-audio query path (image → image proj → text space; audio → audio proj → text space; cosine similarity). Arrow labels showing "text as bridge."
- This figure should make the modular mechanism obvious at a glance. All three modality paths converge at text-space, with no direct image-audio edge.

**Key numbers for intro**: av_ia=0.034 at m=512 (AudioCaps-quality Phase B, zero image-audio pairs); chance ≈ 0.001.

---

### Section 2: Protocol, architecture, and evaluation regimes

**Narrative function**: Define the setup precisely enough that all results are interpretable. Readers need to understand what "modular" means architecturally, why three evaluation regimes exist, and what the three methods are.

**Key claims**:
1. The modular projection architecture is minimal: only ~300K parameters per modality head; full encoders frozen.
2. Three evaluation regimes correspond to different distribution assumptions; results are consistent across all three.
3. Three methods (modular_shared_jl, modular_separate_jl, audio_linear_probe) provide a within-paper comparison of projection head designs.

**Experiments to cite**: Stage44 for architecture description; Stage12/Stage20 for protocol validation; Stage32 for full/overlap/Clotho consistency.

**Tables**:
- **Table 1 (Protocol summary)**: Columns: Phase / Data source / Pairs / Modality / Training / Parameters. Two rows: Phase A (COCO train_restval, 566K pairs, image+text, InfoNCE, ~300K params each × 2 heads), Phase B (AudioCaps train, ~40K pairs, audio+text, InfoNCE, ~300K audio head).
- **Table 2 (Evaluation regimes)**: Regime name / Image pool size / Audio pool source / Overlap. Full (59K COCO images, AudioCaps eval ~1K), 1K overlap (1K images, 1K AudioCaps), Clotho.

**Figure assignment**: Could embed a method-comparison schematic here (shared text head vs separate text head vs linear probe), or move to supplement.

---

### Section 3: Emergence and causal controls

**Narrative function**: This is the foundational existence proof. Two questions must be answered: (1) Does transfer happen at all? (2) Is it causally due to semantic Phase-B supervision, not to an artifact? This section must answer both with crisp controls.

**Key claims**:
1. Zero-shot image-audio transfer is real: av_ia=0.034 at m=512 with zero image-audio training pairs (Stage44).
2. Transfer requires semantic Phase-B supervision: identity control collapses to chance (Stage47).
3. The text bridge is the causal mechanism: shuffled-caption control collapses at all tested dimensions (W11 m=512, W12 m=64, m=256), ruling out embedding-dimension as an escape hatch.
4. Emergence is dimension-independent: the causal claim holds from m=64 to m=512.

**Experiments (primary)**:
- Stage44: full Phase-B training, COCO-matched Phase A. av_ia=0.034 (m=512, modular_shared_jl).
- Stage47: identity ablation (Phase B frozen at initialization, no gradient updates). av_ia collapses to near zero at all m.
  - m=64: av_ia=0.00039; m=128: 0.00066; m=256: 0.00134; m=512: 0.00149.
- W11: shuffled-caption control, m=512, 5 seeds. av_at=0.0012, av_ia=0.0013 — chance.
- W12: shuffled-caption control, m=64 (2 seeds) and m=256 (2 seeds).
  - m=64: av_at=0.0012, av_ia=0.0005. m=256: av_at=0.0010, av_ia=0.0005 — chance.

**Figure assignment**:
- **Figure 2 (Emergence and causal controls)**: Grouped bar chart. X-axis: four conditions (Stage44 full training, Stage47 identity, W11 shuffled m=512, W12 shuffled m=64/256). Y-axis: av_ia. Color groups within each bar: m=64, m=128, m=256, m=512. The Stage44 bars should be clearly above zero; all control bars should be at floor level. Horizontal dashed line at chance (≈0.001).
  - Key design choice: show all four conditions in a single panel to make the visual argument crisp. Do not split into multiple panels.
  - Caption should highlight: "Transfer is present (Stage44) and collapses to chance when semantic supervision is removed (Stage47) or corrupted (W11, W12), consistently across all embedding dimensions tested."

**Ruling out alternative explanations**:
- Accidental encoder pre-alignment: Stage47 rules this out — if encoders were already aligned, the identity control would be above chance.
- Non-semantic optimization artifact: W11/W12 rule this out — optimization proceeds normally with shuffled labels, but av_ia stays at chance. The emergence is semantics-dependent.
- Large-m artifact: W12 rules this out — m=64 shuffled control is still at chance.

---

### Section 4: Quality bottlenecks — what governs transfer in both phases

**Narrative function**: Having shown transfer exists, explain what determines how well it works. This is the largest empirical section. The structure is: Phase B quality, Phase A quality, geometry pathway, geometry scope. Each subsection builds the argument.

**Key claims**:
1. **Phase-B quality bottleneck**: High-quality human-annotated audio-text pairs outperform matched-scale web-scraped pairs on cross-modal (image-audio) transfer; more web-scraped scale improves within-distribution audio retrieval but does NOT improve image-audio transfer.
2. **Phase-A quality bottleneck**: At m≥256, clean human-annotated image-text (COCO) outperforms matched-scale noisy web-scraped (CC3M); more Phase-A volume beyond ~100K pairs is counterproductive.
3. **Geometry is a causally verified mechanism**: Reducing the image-audio centroid gap by regularization causally increases av_ia at m=256 and m=512.
4. **Geometry scope**: The gap explains aggregate transfer efficiency but does not explain the m=256 sharing reversal (W5 rules out gap as the reversal driver); geometry is a post-Phase-B tool, not a pre-Phase-B predictor (Stage68-D).

**Subsection 4.1: Phase-B quality**

*Experiments*:
- Stage38: AudioCaps margin=0.567 vs WavCaps(real) margin=0.249. First evidence that quality differs.
- Stage31/45/56: WavCaps scale progression (40K→200K pairs). av_at increases monotonically with scale; av_ia is non-monotone at large m.
- W13 (5th condition, mixed100k, 94.5K pairs): av_at=0.110, av_ia=0.0267, wav_holdout_at=0.0865.
  - Full 5-condition picture: mixed46k av_ia=0.0285 > mixed200k=0.0279 > mixed100k=0.0267 > clean_source_46k=0.0248 > clean_source=0.0239 — spanning 40K to 142K training pairs with no monotone improvement.
  - av_at: monotonically increases with scale (more scale → better in-distribution audio retrieval).
  - wav_holdout_at: monotonically increases with scale (within-WavCaps domain improves).
  - av_ia: non-monotone — quality (AudioCaps-style annotation) limits cross-modal bridging, not volume.
  - Key distinction: wav_holdout_at (in-distribution) scales; av_ia (out-of-distribution, cross-modal) does not.

*What this rules out*: The claim "more training data always improves transfer" is false for cross-modal (image-audio) quality, even though it is true for in-distribution (WavCaps holdout) quality. The binding constraint is distribution match and annotation quality, not volume.

**Subsection 4.2: Phase-A quality**

*Experiments*:
- Stage29: CC3M 100K pairs, modular_shared_jl, dims=64/128/256/512.
- Stage44: COCO-full 566K pairs (same Phase B: AudioCaps).
- W14: COCO-subsampled 20K images = 100K pairs, same Phase B.
  - Three-condition comparison at m=512: CC3M=0.0328, COCO-sub=0.0407, COCO-full=0.0339.
  - COCO-sub beats CC3M at m=64/256/512; CC3M beats COCO-sub at m=128 (diversity > depth at small m).
  - COCO-sub beats COCO-full at EVERY dimension — more COCO data beyond 100K pairs actively degrades performance.
  - Interpretation: at m≥256, caption precision matters more than web diversity. At m=128, image diversity per pair matters more (CC3M has 100K distinct images; COCO-sub has only 20K).

*What this rules out*: (a) That COCO's advantage over CC3M is purely quantitative — COCO-sub at 100K pairs beats CC3M 100K in 3/4 dims. (b) That more Phase-A data is always helpful — COCO-sub 100K beats COCO-full 566K at every dim.

**Subsection 4.3: Geometry as a causal mechanism**

*Experiments*:
- Stage39/60: correlational evidence — centroid gap inversely correlates with α (r=−0.865 at n=40).
- Stage66/67 intervention: Phase-B loss = InfoNCE + 0.1 × centroid_alignment_penalty.
  - m=256 (4 seeds): baseline av_ia=0.0201 → gap-reg av_ia=0.0233, Δ=+0.0032, Holm p=0.0115 — **significant**.
  - m=512 (4 seeds): baseline av_ia=0.0340 → gap-reg av_ia=0.0379, Δ=+0.0039, Holm p=0.0256 — **significant**.
  - av_at: Δ≈−0.0045, Holm p=0.086 — non-significant.
  - combined_avg_R: unchanged (Δ≈0) — gain is specifically in the cross-modal leg.

*W5 scope clarification*: At m=256 (the reversal dimension, where separate_jl beats shared_jl), gap does NOT differ: shared ia_gap=0.585±0.004 vs separate ia_gap=0.582±0.003 — essentially zero difference, the smallest divergence across all tested dims. The alignment hypothesis (shared JL → larger gap → lower av_ia at m=256) is falsified. The reversal mechanism remains open.

*Stage68-D feasibility*: Pre-Phase-B image-audio centroid gap → post-Phase-B α: r=0.215, r²=0.046, n=20. Gap at initialization explains only 4.6% of variance in final α. Phase B training itself reshapes geometry. Geometry is a post-hoc lever, not a pre-flight signal.

**Figure assignment**:
- **Figure 3 (Quality bottlenecks — three panels)**:
  - **Panel A (Phase-B quality)**: Line plot with x=training scale (log scale: 40K, 46K, 94.5K, 142K, reference AudioCaps 40K), y=av_ia at m=512. Two lines: AudioCaps-quality sets and WavCaps-scale sets. Shows the non-monotone WavCaps vs consistently strong AudioCaps.
    - Include secondary y-axis or separate small plot showing wav_holdout_at (monotone with scale) to visually contrast in-distribution vs cross-modal scaling behavior.
  - **Panel B (Phase-A quality)**: Grouped bars at each embedding dim (64, 128, 256, 512). Three bars per dim: CC3M 100K, COCO-sub 100K, COCO-full 566K. Shows the pattern: COCO-sub wins at 3/4 dims; CC3M wins at m=128; COCO-full never wins.
  - **Panel C (Geometry intervention)**: Before/after bar chart at m=256 and m=512. Bars: baseline av_ia, gap-regularized av_ia, with error bars (std across 4 seeds). Significance markers (✓ p<0.05) on the gain bars. Small inset showing av_at slight decrease (non-significant) to be transparent about tradeoffs.

---

### Section 5: Bottleneck relation and cross-triple portability

**Narrative function**: This is the theoretical and predictive core of the paper. The theory synthesizes all previous results into a single quantitative structure. The three-triple portability establishes that the structure is not specific to AudioCaps.

**Key claims**:
1. Image-audio transfer is well-approximated by `av_ia ≈ α × √(av_it × av_at)` with strong in-sample and held-out predictive performance.
2. The geometric mean form is the correct cross-regime default: it is the only tested form that does not collapse on held-out cross-regime prediction (Stage68-B).
3. The theory is not a seed-replication artifact: cell-mean refit improves r², and mixed-effects α CIs are tight (Stage68-A).
4. Within a Phase-B protocol, α transfers across Phase-A variations; across Phase-B protocols, α must be refitted (Stage68-C).
5. The structure holds on three distinct modality triples spanning two orders of magnitude in α, confirming cross-triple portability (Stage58, Stage69).
6. α is a measure of encoder-domain fit: the same encoder (CLAP HTSAT) gives α=0.27 (AudioCaps) vs 0.020 (SpeechCoco) — a 13× difference traceable to the encoder's av_at for that audio type.
7. The SpeechCoco result establishes an explicit boundary condition: when av_at is structurally weak (encoder-domain mismatch), av_ia collapses to near-chance regardless of embedding dimension or method.

**Subsection 5.1: Theory fit and OOS prediction**

*Experiments*:
- Stage36 primary: α=0.2694, r=0.9207, r²=0.8477, n=300 (AudioCaps, all methods × all dims × 5 seeds).
- Stage43 OOS: train on n=300 AudioCaps rows, predict n=80 held-out WavCaps rows: r=0.9531, MAE=0.00220.

**Subsection 5.2: Theory robustness (Stage68)**

*Experiments*:
- Stage68-A (cell-mean): AudioCaps seed r²=0.83 → cell-mean r²=0.84; mixed-effects α=0.2701, 95% CI=[0.2550, 0.2852]. AVCaps seed r²=0.87 → cell-mean r²=0.90; mixed-effects α=0.6795, CI=[0.6477, 0.7113]. Theory is not a seed-replication artifact.
- Stage68-B (functional form): Tested: geometric mean, hard-min, product, free-power.
  - Within-suite CV: hard-min best on AudioCaps CV; free-power best on AVCaps CV — geometric mean does NOT win within-suite.
  - Cross-regime held-out (AudioCaps train → WavCaps predict): geometric r²=0.853; hard-min 0.203; product 0.210; free-power −1.173. Geometric mean is the ONLY form that does not collapse.
  - Why hard-min fails: the identity of the minimum leg (av_it vs av_at) swaps between AudioCaps and WavCaps regimes. Trained on AudioCaps (av_at is the min), hard-min breaks when predicting WavCaps (where av_at is larger).
  - Why free-power fails: exponents fitted on AudioCaps (a≈0.82, b≈−0.02) break catastrophically when the balance reverses. Free-power CI on b: [−0.069, +0.038] — consistent with b=0, supporting (0.5, 0.5) as the null.
- Stage68-C (α-locked prediction): α=0.2820 (from AudioCaps) predicts W14 CC3M conditions: r=0.9435, r²=0.8201, n=80. COCO-sub weaker (r=0.928, r²=0.65, n=12). W13 WavCaps Phase B fails as expected (wrong regime α). α transfers across Phase-A variations, not Phase-B protocol changes.
- Stage68-D (geometry feasibility): pre-Phase-B gap → post-Phase-B α: r=0.215, r²=0.046, n=20. Gap at initialization does not predict final α.

**Subsection 5.3: Cross-triple portability**

*Experiments*:
- Stage58 AVCaps: α=0.6731, r=0.9595, r²=0.8712, n=80. α=0.67 because video-associated audio is acoustically and semantically well-aligned with CLAP's training distribution.
- Stage69 SpeechCoco: 60 cells (3 methods × 4 dims × 5 seeds). α (m≥128 pooled) = 0.0198±0.0033. Bottleneck theory holds: geometric prediction accurate at every cell. Alpha plateaus after m=64; method convergence at all dims (within-dim α std across methods = 0.0013 at m=512).

*Three-triple table*:

| Triple | Audio type | n cells | α (high-dim) | CLAP av_at (m=512) |
|---|---|---:|---:|---:|
| AudioCaps | Environmental sounds | 300 | 0.270 | ≈0.233 |
| AVCaps | Video-associated audio | 80 | 0.673 | ≈0.35 (est.) |
| SpeechCoco | Spoken captions (speech) | 60 | 0.020 | ≈0.110 |

Same encoder (CLAP HTSAT-unfused), same backbone (CLIP ViT-B/32), same Phase A (COCO), same protocol. Variable: audio type. α reflects CLAP's audio→text alignment quality for each audio type.

*Alpha as encoder-domain fit diagnostic*: The ratio α = av_ia / √(av_it × av_at) measures how efficiently the audio encoder's learned representation bridges through text to image. Larger α indicates better fit; very low α (as in SpeechCoco with CLAP) indicates likely mismatch and motivates encoder/domain re-evaluation.

*Deployment boundary condition from SpeechCoco*: av_at≈0.11 (CLAP on speech) → av_ia≈0.005 at m=512. Near-chance on a 5K-pair eval set. The strategy fails when the encoder's audio→text alignment is structurally weak for the target audio domain. This is an explicit, empirically grounded failure condition — not a hypothetical.

**Figure assignment**:
- **Figure 4 (Bottleneck theory scatter)**: Scatter plot of sqrt(av_it × av_at) (x-axis) vs av_ia (y-axis). Primary data: AudioCaps n=300 points (colored by method). Overlay: held-out WavCaps n=80 points (different marker shape). Theory line: av_ia = 0.2694 × x. Annotation: r²=0.848 in-sample, r²=0.853 OOS. This single figure communicates the core quantitative claim.
  - Design notes: Use small alpha for point transparency to show density. Mark the held-out points clearly (dashed outline or triangle marker). Include AVCaps as a third color cluster (n=80) to show the three-regime structure — they will scatter around a steeper line (α=0.67) and the SpeechCoco cluster (α=0.020) will hug the x-axis.
  - Alternative: separate panel per regime with a common theory line at each α level. Decision: show all three in one plot with per-cluster fit lines labeled.

- **Figure 5 (Cross-triple comparison + functional form robustness — two panels)**:
  - **Panel A**: Three-triple α comparison. Bar chart with three bars: AudioCaps α=0.27, AVCaps α=0.67, SpeechCoco α=0.020. Error bars = 95% CI from mixed-effects model (or std across seeds). Annotate CLAP av_at below each bar. Use shaded regime bands (illustrative, not strict cutoffs) to indicate low-fit vs high-fit operating regions.
  - **Panel B**: Functional form robustness (Stage68-B). Bar chart of cross-regime held-out r² for four functional forms: geometric mean (r²=0.853), product (0.210), hard-min (0.203), free-power (−1.173, shown as a collapsed/negative bar). Annotate: "Geometric mean is the only cross-regime stable form." This figure makes the case for the geometric form without requiring readers to understand the math.

---

### Section 6: Positioning, limitations, and implications

**Narrative function**: Close the paper with intellectual honesty about scope, a fair comparison framing, and actionable engineering guidance. This section converts the scientific findings into a practical decision framework.

**Key claims**:
1. Fair comparison: direct-supervision systems (ImageBind, AudioCLIP) will have higher absolute retrieval; our contribution is not SOTA, it is a constrained-regime characterization and planning tool.
2. The concise engineering guidance in §5.2 is experiment-backed, and the full decision workflow can live in the appendix.
3. The α pilot protocol can eliminate infeasible encoder-domain combinations before full training investment.
4. Known limitations: speech audio with CLAP fails (α=0.020); settings without text bridge are out of scope; specialist image domains may require domain-adapted Phase A.

**Experiments to cite**: Stage69 (boundary condition), Stage67 (geometry lever), Stage68-C (α transfer), W13/W14 (quality/scale tradeoff).

**Figure assignment**:
- **Figure 6 (Decision flowchart — optional)**: A simple flowchart of the full appendix decision workflow. Start → run shuffled-caption control → if fails, stop (contamination detected); if passes → run α pilot → compare against historical regimes → choose Phase B data strategy → budget Phase A → monitor geometry → use theory for planning.
  - *Decision on inclusion*: If the paper is already at page limit, move to supplement. The flowchart is the most practically valuable figure for readers trying to apply the framework.

---

### Appendix structure

**Appendix A: Full per-dimension tables**
- Table A1: All Stage44 metrics by method × dim × seed (mean/std).
- Table A2: All Stage69 SpeechCoco metrics by method × dim × seed.
- Table A3: W13 WavCaps 5-condition comparison at m=512 (all metrics).
- Table A4: W14 Phase-A source comparison at all dims.

**Appendix B: Stage68 robustness details**
- Figure B1: Stage68-A cell-mean scatter (seed vs cell-mean fit comparison for AudioCaps and AVCaps).
- Figure B2: Stage68-B functional form cross-validation matrix (within-suite CV r² and cross-regime held-out r² for all four forms and all suites).
- Figure B3: Stage68-C α-locked prediction scatter (AudioCaps α → predict W14 CC3M conditions).
- Table B1: Mixed-effects model α estimates and 95% CIs for all three triples.

**Appendix C: SpeechCoco Stage69 detail**
- Figure C1: SpeechCoco α by embedding dim (m=64, 128, 256, 512), line plot showing plateau after m=64. Three lines: one per method. Error bars = std across 5 seeds. Annotate: "Method convergence at all dims; encoder bottleneck dominates architectural variation."
- Figure C2: Three-triple scatter plot (full version from Figure 4, with per-cluster labels and individual seed points visible).

**Appendix D: Geometry details**
- Figure D1: Image-audio centroid gap by embedding dim (m=64 through m=512), by method and triple. Shows gap decreases with m for all methods/triples.
- Figure D2: W5 gap analysis — shared vs separate JL ia_gap at each dim. Annotate m=256 (near-zero difference) and m=128 (sign reversal). This is the mechanistic null for the sharing reversal.
- Figure D3: Stage68-D feasibility scatter — pre-Phase-B gap vs post-Phase-B α. Show r²=0.046 annotated. Flat line to show no pre-flight predictive power.

**Appendix E: Protocol and training details**
- Full hyperparameters for Phase A and Phase B training.
- Evaluation pool construction details (full pool, 1K overlap, Clotho).
- SpeechCoco eval: strict disjoint construction, 3,594 image_id-level removals from Phase B train.
- *Note: Federated training (Stage13–17) is future work and is not described in the current paper.*

---

## 7A) Experimental results map

This table maps every significant experiment in the paper to its claim, paper section, and key numbers. It is the complete reference for "what goes where."

| Stage/Experiment | What it tests | Core claim supported | Paper section | Key numbers |
|---|---|---|---|---|
| **Stage44** | Full Phase-B training on AudioCaps, shared JL | Transfer exists without image-audio pairs | §3 | av_ia=0.034 (m=512); av_at=0.226 |
| **Stage47** | Identity ablation (Phase B not trained) | Transfer requires Phase B learning, not encoder pre-alignment | §3 | av_ia≈0.001 at all dims |
| **W11** | Shuffled-caption control, m=512, 5 seeds | Transfer requires semantic supervision | §3 | av_at=0.001, av_ia=0.001 (chance) |
| **W12** | Shuffled-caption control, m=64 and m=256, 2 seeds each | Causal claim is dimension-independent | §3 | av_ia=0.0005 at both dims (chance) |
| **Stage38** | AudioCaps margin vs WavCaps margin comparison | Phase-B quality drives transfer | §4.1 | AudioCaps margin=0.567 vs WavCaps=0.249 |
| **Stage31/45/56** | WavCaps at 3 scales (100K, 140K, 200K) | Scale without quality doesn't help av_ia | §4.1 | av_at monotone, av_ia non-monotone |
| **W13 (mixed100k)** | 5th WavCaps condition (94.5K pairs) | WavCaps scale improves within-distribution, not cross-modal | §4.1 | av_at=0.110, av_ia=0.0267; 5-condition picture complete |
| **Stage29** | CC3M 100K pairs as Phase A | Phase-A data quality effect (CC3M baseline) | §4.2 | av_ia=0.033 (m=512) |
| **W14 (COCO-sub)** | COCO-subsampled 100K pairs as Phase A | Quality dominates; more COCO data hurts | §4.2 | COCO-sub=0.041 vs CC3M=0.033 vs COCO-full=0.034 (m=512) |
| **Stage39/60** | Centroid gap correlation with α | Geometry predicts transfer efficiency | §4.3 | gap→α: r=−0.865 (n=40) |
| **Stage65** | Calibration: linear vs isotonic α prediction | Linear calibration generalizes better | §4.3 (or appendix) | Linear > isotonic for α_local generalization |
| **Stage66/67 (W3)** | Centroid regularization intervention | Geometry is a causal lever, not just correlational | §4.3 | Δav_ia=+0.0032 (m=256, p=0.0019 Holm-sig); +0.0039 (m=512, p=0.0256) |
| **W5 (gap analysis)** | Gap comparison: shared vs separate JL by dim | Centroid gap does not explain m=256 sharing reversal | §4.3 | m=256 Δgap=−0.003 (effectively zero); m=128 sign reversal |
| **Stage68-D** | Pre-Phase-B gap → post-Phase-B α | Geometry is post-hoc, not pre-flight | §4.3 | r²=0.046, n=20 — near zero |
| **Stage36** | Primary Bottleneck theory fit (AudioCaps, n=300) | Theory fit in-sample | §5.1 | α=0.2694, r²=0.848 |
| **Stage43** | OOS prediction: AudioCaps→WavCaps held-out | Theory predicts held-out conditions | §5.1 | r=0.953, MAE=0.0022, n_test=80 |
| **Stage68-A** | Cell-mean refit; mixed-effects α CIs | Theory not a seed-replication artifact | §5.2 | AudioCaps cell-mean r²=0.841; α 95% CI=[0.255, 0.285] |
| **Stage68-B** | Functional form comparison (4 forms, cross-regime) | Geometric mean is the only cross-regime stable form | §5.2 | Geometric r²=0.853; all others collapse (hard-min 0.20; free-power −1.17) |
| **Stage68-C** | α-locked prediction: AudioCaps α → W14 conditions | α transfers across Phase-A variations within same Phase-B regime | §5.2 | r=0.944, r²=0.820, n=80 |
| **Stage58 AVCaps** | Second modality triple (video-associated audio) | Theory holds on second triple; α regime-dependent | §5.3 | α=0.673, r²=0.871, n=80 |
| **Stage69 SpeechCoco** | Third modality triple (spoken captions, CLAP on speech) | Theory holds on third triple; encoder-domain mismatch failure mode | §5.3 | α=0.020±0.003 (m≥128); av_at=0.110; method convergence Δα<0.003 at m=512 |
| ~~Stage13–17~~ | *(Federated training under DP — excluded from paper; treated as future work only)* | N/A | — | — |

---

### Summary: figure count and placement

| Figure | Title | Section | Panels |
|---|---|---|---|
| Figure 1 | System architecture: Phase A + Phase B + inference | §2 | 2 panels (training, inference) |
| Figure 2 | Emergence and causal controls: Stage44 vs Stage47 vs W11/W12 | §3 | 1 panel (grouped bars by condition × dim) |
| Figure 3 | Quality bottlenecks: Phase B (non-monotone), Phase A (3-condition), Geometry intervention | §4 | 3 panels |
| Figure 4 | Bottleneck theory scatter (n=300 AudioCaps + WavCaps overlay + SpeechCoco cluster) | §5 | 1 panel (3 clusters) |
| Figure 5 | Three-triple α comparison + functional form robustness | §5 | 2 panels |
| Figure 6 | Decision flowchart (appendix workflow) | §6 | 1 panel (optional; move to appendix if space-constrained) |

**Appendix figures** (Figures A–D): Stage68 robustness panels (cell-mean, functional form, α-locked), SpeechCoco α plateau by dim, W5 gap analysis (shared vs separate), Stage68-D feasibility scatter. These support the main text claims without cluttering the main paper.

**Target**: 6 main figures, 4–6 appendix figures. All main figures should be readable at half-page width. Figures 4 and 5 are the most important — invest most design effort there.

---

## 8) Final narrative sentence
This paper should be read as a **theory-guided systems result**: a constrained-regime, causally verified framework for modular multimodal extension — establishing that text-anchored transfer emerges dimension-independently, that quality dominates scale in both training phases, that geometry is a controllable post-hoc lever (not a pre-flight predictor), and that the bottleneck relation `av_ia ≈ α·sqrt(av_it·av_at)` robustly predicts held-out transfer quality across conditions and triples (confirmed not a seed artifact; geometric form is the uniquely cross-regime stable default). Three modality triples — AudioCaps (α=0.27), AVCaps (α=0.67), and SpeechCoco (α=0.020, pre-registered) — confirm the structure across two orders of magnitude of α, with α functioning as a diagnostic of encoder-domain fit and the SpeechCoco result defining the explicit boundary condition where the strategy fails: when the audio encoder is mismatched to the audio domain, the transitive bridge collapses to near chance regardless of dimension, volume, or method.
