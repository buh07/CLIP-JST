# Width-Adaptive Cross-Modal Johnson–Lindenstrauss Projections for Relationship-Preserving Multi-Modal Representation Learning

## A Research Proposal

---

## 1. Abstract

This proposal investigates whether a frozen, oblivious Johnson–Lindenstrauss (JL) transform, composed with a lightweight learned per-modality Mahalanobis head, can match or approach the downstream retrieval performance of fully-learned projection heads in CLIP-style dual-encoder architectures — while providing formal, width-adaptive distortion guarantees on a user-specified cross-modal relationship graph. The central hypothesis is that the expensive, data-dependent part of CLIP's projection head is doing work that can be decomposed into (i) an oblivious distance-preserving projection (handled by JL) and (ii) a small data-dependent metric adjustment (handled by the Mahalanobis head), and that the decomposition is nearly lossless when the relationship graph has bounded Gaussian width. If confirmed, this would provide the first formal, distortion-bounded training mechanism for relationship-preserving multi-modal representations, with immediate implications for on-device deployment, federated multi-modal learning, and privacy-preserving retrieval.

---

## 2. Motivation and Research Question

### 2.1 The gap in the literature

Modern multi-modal foundation models (CLIP, ALIGN, ImageBind) all share a common architectural pattern: per-modality backbones produce high-dimensional features, which are then passed through learned linear projection heads to a shared cosine-similarity space and trained with a contrastive InfoNCE loss against specified image–caption pairs. The projection head is data-dependent, opaque, and consumes substantial training compute and memory.

Meanwhile, a parallel theoretical literature — the Klartag–Mendelson (2005) width-based JL bounds, the Bourgain–Dirksen–Nelson (2015) sparse-JL guarantees, the Narayanan–Nelson (2019) optimal terminal embeddings, and recently the Palias–Kabán (2024) result on dimension-free generalization of Mahalanobis metric learning in a JL-projected space — shows that *oblivious* random projections can preserve the geometry of structured sets at a dimension that scales with the set's complexity, and that learning a metric *on top* of such a projection retains dimension-free statistical guarantees.

These two lines of work have never been connected in a single system. No published method uses a frozen, oblivious JL as the sole dimension-reduction step in a CLIP-style pipeline while preserving a user-specified cross-modal relationship graph. Cross-modal hashing work (CMSSH, SePH, DCMH) optimizes relationship-preservation objectives but relies on learned nonlinear projections to Hamming space and does not connect to JL distortion theory. Efficient-attention work (Linformer, Performer) uses JL inside a single modality's attention but not for cross-modal alignment.

### 2.2 The research question

**Does a frozen sparse Johnson–Lindenstrauss transform, composed with a small learned per-modality Mahalanobis head, approach the retrieval performance of a fully-learned CLIP-style projection head at equivalent embedding dimension — and does the gap close predictably as a function of the specified relationship graph's Gaussian width?**

Three sub-questions follow:

1. **Empirical.** At embedding dimensions $m \in \{64, 128, 256, 512\}$, how does JL + Mahalanobis compare to a trained dense $d \times m$ projection head on MS-COCO, Flickr30K, NUS-WIDE, and MIR-Flickr retrieval benchmarks?

2. **Theoretical.** Can we obtain a formal generalization bound of the form *"for a cross-modal relationship graph of Gaussian width $w$ and $n$ pairs, the InfoNCE-trained Mahalanobis-after-JL pipeline achieves retrieval error within $O(w/\sqrt{nm})$ of the learned-projection optimum with probability $1-\delta$"*?

3. **Diagnostic.** When the gap is non-trivial, which directions in the learned projection head does the oblivious JL fail to capture? Specifically, does it fail on the "expansion/shrinkage" directions identified by Gui–Chen–Liu (NeurIPS 2023) as essential to contrastive learning?

---

## 3. Theoretical Framework

### 3.1 Notation and setup

Let $\mathcal{V} \subset \mathbb{R}^{d_v}$ be the image-feature space produced by a frozen vision backbone (e.g., CLIP-ViT-B/32, $d_v = 768$) and $\mathcal{T} \subset \mathbb{R}^{d_t}$ be the text-feature space produced by a frozen text backbone ($d_t = 512$). A cross-modal relationship graph $R \subseteq \mathcal{V} \times \mathcal{T}$ specifies the $n = |R|$ image–text pairs that should be close in the shared embedding space. For MS-COCO, $R$ is the set of (image, caption) pairs; for NUS-WIDE, $R$ is defined by shared tag sets.

Let $\Phi_v : \mathbb{R}^{d_v} \to \mathbb{R}^m$ and $\Phi_t : \mathbb{R}^{d_t} \to \mathbb{R}^m$ be per-modality sparse JL matrices drawn from the Kane–Nelson (2014) distribution with $s = \Theta(\varepsilon^{-1})$ non-zeros per column and $m = \Theta(\varepsilon^{-2} \log n)$ rows. Let $M_v, M_t \in \mathbb{R}^{m \times m}$ be symmetric positive semi-definite Mahalanobis matrices (or their rank-$r$ factorizations for parameter efficiency), trained on the contrastive objective.

The full pipeline is:

$$
f_v(v) = M_v^{1/2} \Phi_v(v), \quad f_t(t) = M_t^{1/2} \Phi_t(t)
$$

with InfoNCE loss

$$
\mathcal{L} = -\frac{1}{n}\sum_{(v,t) \in R} \log \frac{\exp(\langle f_v(v), f_t(t) \rangle / \tau)}{\sum_{t' \in B_t} \exp(\langle f_v(v), f_t(t') \rangle / \tau)}
$$

where $\tau$ is the temperature and $B_t$ is an in-batch negative set.

### 3.2 Theoretical claims to establish

**Claim 1 (Width-adaptive distortion preservation).** *Let $R$ be a cross-modal pair set with effective Gaussian width $w(R) = \mathbb{E}_g \sup_{(v,t) \in R - R} \langle g, (v,-t) \rangle$. For $m \geq C \varepsilon^{-2} (w(R)^2 + \log(1/\delta))$, a Kane–Nelson sparse JL satisfies*

$$
(1 - \varepsilon) \| v - t \|^2 \leq \| \Phi_v(v) - \Phi_t(t) \|^2 \leq (1 + \varepsilon) \| v - t \|^2
$$

*simultaneously for all $(v, t) \in R$ with probability at least $1 - \delta$.*

This follows from Bourgain–Dirksen–Nelson (STOC 2015) applied to the Minkowski difference set $R - R = \{(v, -t) : (v, t) \in R\} \subset \mathbb{R}^{d_v + d_t}$. The novelty is instantiating it for *cross-modal* pair sets where $v$ and $t$ live in different native spaces; the trick is to pad with zeros and apply a block-diagonal JL.

**Claim 2 (Dimension-free Mahalanobis generalization).** *Let $\hat{M}_v, \hat{M}_t$ be the empirical-risk minimizers of the InfoNCE loss over $n$ training pairs in the JL-projected space. Then with probability $1-\delta$ over the sample:*

$$
\mathcal{L}_{\text{pop}}(\hat{M}_v, \hat{M}_t) - \mathcal{L}_{\text{pop}}^* \leq C \cdot \frac{\sqrt{\text{sd}(R)}}{\sqrt{n}} + O(\varepsilon) + \tilde{O}\left(\sqrt{\frac{\log(1/\delta)}{n}}\right)
$$

*where $\text{sd}(R)$ is the statistical dimension of the relationship-graph support and is independent of the ambient backbone dimensions $d_v, d_t$.*

This extends Palias–Kabán (IJCNN 2024)'s single-modality compressive-Mahalanobis generalization bound to the cross-modal contrastive setting. The main technical obstacle is that InfoNCE is not a standard metric-learning loss; we need to adapt the Mendelson-style chaining argument used by Palias–Kabán to the soft-nearest-neighbor objective.

**Claim 3 (Compositionality with frozen backbones).** *When the backbones $\psi_v, \psi_t$ are frozen and the downstream task is retrieval on the same relationship-graph distribution, the composed pipeline $M^{1/2} \Phi \psi$ achieves retrieval error that is optimal over all linear projections $\mathbb{R}^d \to \mathbb{R}^m$ up to a factor of $(1 + \varepsilon)$ in the Bayes regret.*

This is the trickiest claim and is the one most likely to need relaxation — it formalizes the intuition that "JL does the distance-preservation work; the Mahalanobis head does the metric-specialization work; together they match a fully-learned projection head."

### 3.3 Theoretical obstacles

1. **InfoNCE is not a standard metric-learning loss.** Existing compressive-metric-learning bounds (Palias–Kabán, LMNN-style) assume triplet or contrastive *pair* losses with fixed margin. InfoNCE couples all negatives in a batch through the softmax, breaking independence assumptions in chaining arguments. A recent workaround is the spectral-contrastive analysis of HaoChen et al. (NeurIPS 2021), which rewrites InfoNCE as a spectral decomposition of an augmentation/pairing graph. We will compose the spectral-contrastive framework with the Bourgain–Dirksen–Nelson width bound.

2. **Cross-modal Gaussian width is not standard.** Width is typically defined for a set in a single Euclidean space. For a cross-modal pair set, we need a block-diagonal construction: embed $(v, t) \in \mathbb{R}^{d_v} \times \mathbb{R}^{d_t}$ into $\mathbb{R}^{d_v + d_t}$ by concatenation, and analyze the width of the concatenated set. This is mathematically clean but needs care when the two modalities have different scales.

3. **The Gui–Chen–Liu "expansion/shrinkage" phenomenon.** They show that trained projection heads expand some singular directions and shrink others in a data-dependent way. An oblivious JL does neither. Our Claim 3 therefore cannot hold in full generality — it will require the assumption that the Mahalanobis head can recover the expansion/shrinkage pattern. Empirically this means $M_v, M_t$ must have sufficient spectral freedom (i.e., not be restricted to identity + low-rank).

---

## 4. Proposed Method

### 4.1 Architecture

```
   image v                                    text t
     |                                          |
     ▼                                          ▼
  [frozen CLIP-ViT-B/32]                  [frozen CLIP text enc]
     | ψ_v(v) ∈ R^768                         | ψ_t(t) ∈ R^512
     ▼                                          ▼
  [frozen sparse JL Φ_v]                  [frozen sparse JL Φ_t]
     | Φ_v ψ_v(v) ∈ R^m                      | Φ_t ψ_t(t) ∈ R^m
     ▼                                          ▼
  [trainable Mahalanobis M_v^{1/2}]      [trainable Mahalanobis M_t^{1/2}]
     | f_v(v) ∈ R^m                           | f_t(t) ∈ R^m
     ▼                                          ▼
            InfoNCE contrastive loss on R
```

The frozen JL matrices $\Phi_v, \Phi_t$ are drawn once from the Kane–Nelson sparse distribution with sparsity $s = \lceil \varepsilon^{-1} \rceil$ non-zeros per column (for $\varepsilon = 0.1$, $s = 10$) and never updated. The Mahalanobis factors $M_v^{1/2}, M_t^{1/2}$ are $m \times m$ matrices (or their rank-$r$ LoRA-style factorizations for $r \ll m$) and are the only trained parameters.

Total trainable parameter count:
- Full Mahalanobis: $2 \cdot m(m+1)/2 = m(m+1)$ (e.g., $\sim 66{,}000$ for $m=256$)
- Rank-$r$ LoRA: $2 \cdot 2mr$ (e.g., $\sim 16{,}000$ for $m=256, r=16$)

Compare against CLIP's projection heads at $m=512$: $2 \cdot 768 \cdot 512 + 2 \cdot 512 \cdot 512 \approx 1.3$M parameters. This is a **20–100× reduction in trainable parameters.**

### 4.2 Key design choices and rationale

- **Why sparse JL rather than Gaussian?** Kane–Nelson sparse JL achieves the optimal $m = \Theta(\varepsilon^{-2} \log n)$ with only $O(\varepsilon m)$ non-zeros per column, giving fast matrix-vector products and low memory footprint — essential for the "cheap deployment" practical motivation. It also enjoys the same width-adaptive guarantees as Gaussian JL (Bourgain–Dirksen–Nelson 2015).
- **Why frozen backbones?** (i) Isolates the contribution of JL + Mahalanobis from general fine-tuning gains; (ii) matches the practical deployment setting where one wants to take a pretrained CLIP and compress its representations; (iii) keeps experiments feasible on a single GPU.
- **Why Mahalanobis rather than a free MLP?** Mahalanobis preserves the linear-algebraic structure needed for the theoretical claims. An MLP would confound the oblivious-vs-learned decomposition. We will nevertheless run an MLP head as a diagnostic baseline.
- **Why InfoNCE?** Matches CLIP/ALIGN/ImageBind so the baseline comparison is apples-to-apples. We will also report spectral-contrastive-loss variants for theoretical transparency.

### 4.3 Training procedure

1. **Data preparation.** Extract frozen CLIP features once; cache to disk (COCO: ~6 GB).
2. **JL instantiation.** Sample $\Phi_v, \Phi_t$ from Kane–Nelson distribution with fixed random seed (enables reproducibility and theoretical analysis under the specific draw).
3. **Mahalanobis parameterization.** Initialize $M_v^{1/2}, M_t^{1/2}$ as identity (so initial pipeline is pure JL). Option A: full $m \times m$. Option B: rank-$r$ factorization $L^\top L$ with $L \in \mathbb{R}^{r \times m}$.
4. **Training.** AdamW, learning rate 1e-3, batch size 4096 (large batch critical for InfoNCE), temperature 0.07, 50 epochs, early stopping on validation recall@10.
5. **Hyperparameter sweep.** $m \in \{64, 128, 256, 512\}$, $r \in \{4, 16, 64, \text{full}\}$, $\varepsilon$-target $\in \{0.05, 0.1, 0.2\}$.

---

## 5. Experimental Plan

### 5.1 Datasets

| Dataset | Modalities | Train pairs | Test queries | Primary metric |
|---|---|---|---|---|
| MS-COCO Captions | image, text (5 captions/image) | 113 K images × 5 = 565 K pairs | 5 K images | Recall@{1,5,10} (i→t, t→i) |
| Flickr30K | image, text (5 captions/image) | 29 K images × 5 = 145 K pairs | 1 K images | Recall@{1,5,10} |
| NUS-WIDE | image, tag set | 190 K image-tag pairs | 10 K | mAP (cross-modal retrieval by shared tags) |
| MIR-Flickr-25K | image, tag set | 20 K image-tag pairs | 2 K | mAP |
| Conceptual Captions 3M (optional scale test) | image, text | 3 M | 10 K | Recall@{1,5,10} |

### 5.2 Baselines

1. **CLIP projection head (upper-bound reference).** Original $d \times m$ trained linear projection head, trained on the same data, same loss, same schedule.
2. **Random projection only.** JL with $M_v = M_t = I$ (no Mahalanobis). Isolates the contribution of the learned metric.
3. **Mahalanobis only (no JL).** Full $d_v \times d_v$ and $d_t \times d_t$ Mahalanobis on raw CLIP features. Isolates the contribution of dimension reduction.
4. **PCA + Mahalanobis.** Data-dependent linear dimension reduction as a competitive classical baseline.
5. **DCMH** (Jiang & Li, CVPR 2017). Strongest cross-modal hashing baseline at matched bit budget ($m$ real dimensions ≈ $32m$ bits with 32-bit floats; compare at matched bit budgets).
6. **CCQ** (Long et al., SIGIR 2016). Composite correlation quantization.
7. **LoRA on CLIP projection head.** Rank-$r$ update on top of existing CLIP projection. Industry-standard parameter-efficient approach; parameter-matched to the rank-$r$ Mahalanobis variant.
8. **Product Quantization on CLIP features.** FAISS IVFPQ with matched bit budget.

### 5.3 Primary experiments

**E1. Performance vs. dimension.** For each dataset, plot Recall@{1,5,10} as a function of $m$ for JL+Mahalanobis, CLIP-head, and all baselines. Hypothesis: JL+Mahalanobis within 2 points of CLIP-head at $m \geq 256$, gap grows at $m = 64$.

**E2. Parameter efficiency.** Plot performance vs. trainable parameter count (across rank-$r$ variants). Hypothesis: JL + rank-16 Mahalanobis Pareto-dominates LoRA on CLIP head at matched parameter budget.

**E3. Width-complexity scaling.** Construct subsets of MS-COCO with controlled relationship-graph Gaussian width (by filtering to specific concept subsets, and by varying the caption-set diversity). Plot dimension-to-achieve-fixed-recall vs. estimated width. Hypothesis: linear-in-width scaling predicted by Claim 1.

**E4. Out-of-distribution retrieval.** Train on MS-COCO, evaluate on Flickr30K and Conceptual Captions without retraining. Hypothesis: JL's obliviousness gives better OOD generalization than learned heads that may overfit to MS-COCO's caption style.

**E5. Federated/privacy stress test.** Simulate federated training where clients share only JL-projected features but not raw CLIP features. Compare performance to centralized training; compare attack success rate of feature-inversion attacks (He et al. 2019) against JL-projected vs. raw features. Hypothesis: JL-projected features resist feature-inversion while preserving retrieval utility.

### 5.4 Diagnostic experiments

**D1. Expansion/shrinkage decomposition.** Following Gui–Chen–Liu, compute the singular-value spectrum of the learned CLIP projection head and the (random) JL. Measure how much of the learned head's expansion/shrinkage pattern is recovered by the Mahalanobis head. Visualize with eigenvalue-alignment plots.

**D2. Class-conditional distortion.** For each concept class in NUS-WIDE, measure the empirical JL distortion restricted to that class's pair set. Test whether theoretically-predicted width-adaptive scaling holds per-class.

**D3. Relationship-graph ablations.** Train on (i) full MS-COCO pairs, (ii) pairs filtered to high-semantic-agreement captions, (iii) pairs augmented with synthetic adversarial negatives. Measure how relationship-graph structure affects optimal $m$.

**D4. Backbone generalization.** Repeat core experiments with (a) ViT-L/14 backbones, (b) DINO-v2 + BGE text encoder, (c) audio–text with CLAP encoders. Hypothesis: JL+Mahalanobis advantage is backbone-agnostic.

### 5.5 Negative controls and sanity checks

- **Shuffle-label control.** Run with randomly permuted image–text pairings; performance should collapse to chance, confirming the pipeline genuinely learns from the relationship graph.
- **Zero-Mahalanobis control.** Mahalanobis frozen at identity; this measures pure JL performance and should be strictly worse than full pipeline (else the Mahalanobis isn't helping).
- **Random-seed variability.** Repeat with 5 different JL random seeds; report mean ± std. Low variance across seeds would empirically validate the obliviousness claim.

### 5.6 Metrics and statistical testing

- **Retrieval.** Recall@{1, 5, 10}; image-to-text and text-to-image directions reported separately and averaged.
- **Classification retrieval (NUS-WIDE, MIR-Flickr).** mAP at various cutoffs.
- **Parameter efficiency.** FLOPs-at-inference, trainable parameter count, training wall-clock.
- **Statistical testing.** Paired bootstrap confidence intervals over test set; report 95% CIs. Statistical significance tested with paired permutation test against CLIP-head baseline.

---

## 6. Implementation Details

### 6.1 Compute and software

- **Primary hardware.** Single NVIDIA A100 80GB sufficient for all experiments except CC3M scale test (which needs 4× A100).
- **Software stack.** PyTorch 2.x, HuggingFace Transformers for backbones, FAISS for efficient evaluation, `scipy.sparse` for sparse JL storage.
- **JL implementation.** Use the Kane–Nelson construction: for each column independently sample $s$ non-zero positions uniformly without replacement and assign $\pm 1/\sqrt{s}$ signs from a 4-wise-independent family (Cohen–Jayram–Nelson 2018 construction for rigor). Store as `scipy.sparse.csr_matrix`.

### 6.2 Data pipeline

1. Download MS-COCO 2017, Flickr30K, NUS-WIDE, MIR-Flickr.
2. Run each image through frozen CLIP-ViT-B/32 once; cache the 768-dim pooled output.
3. Run each caption/tag-set through frozen CLIP text encoder once; cache the 512-dim EOS output.
4. Build relationship-graph tensors indexing into the cache.
5. Training loop operates entirely on cached features — no image decoding — making each epoch ~30 seconds on MS-COCO.

### 6.3 Code organization

```
project/
├── theory/            # proofs, width-estimation utilities
├── models/
│   ├── jl.py          # Kane–Nelson sparse JL
│   ├── mahalanobis.py # full and rank-r Mahalanobis
│   └── pipeline.py    # composed pipeline
├── training/
│   ├── infonce.py
│   └── spectral.py    # for theoretical-transparency variants
├── eval/
│   ├── retrieval.py   # Recall@K, mAP
│   └── diagnostics.py # expansion/shrinkage analysis
├── data/              # dataset loaders, feature caches
├── experiments/       # one script per experiment E1–E5, D1–D4
└── notebooks/         # exploratory analysis, figure generation
```

### 6.4 Reproducibility

- All JL random seeds fixed and logged.
- All feature caches versioned by backbone checkpoint hash.
- All experiments runnable with a single `python experiments/run_Ei.py --config configs/Ei.yaml`.
- Code and feature caches released under MIT license; total reproduction cost ~$200 of cloud A100 time.

### 6.5 Timeline

| Phase | Duration | Deliverables |
|---|---|---|
| 1. Theory consolidation | Weeks 1–6 | Proofs of Claims 1–3; technical report draft |
| 2. Infrastructure | Weeks 3–8 | Feature cache, JL implementation, training loop, evaluation harness |
| 3. Core experiments (E1, E2) | Weeks 8–14 | Results on MS-COCO, Flickr30K |
| 4. Scaling and OOD (E3, E4) | Weeks 14–20 | Width-complexity plots, cross-dataset generalization |
| 5. Applications (E5) | Weeks 20–26 | Federated/privacy experiments |
| 6. Diagnostics (D1–D4) | Weeks 22–30 | Expansion/shrinkage analysis, backbone generalization |
| 7. Writing | Weeks 28–36 | Conference submission (NeurIPS/ICML-equivalent deadline) |

---

## 7. Related Literature

### 7.1 Johnson–Lindenstrauss foundations

The classical lemma (Johnson & Lindenstrauss, 1984; elementary proof by Dasgupta & Gupta 2003) and its sparse and fast variants (Achlioptas 2003; Ailon & Chazelle 2006; Kane & Nelson 2014; Fandina–Høgsgaard–Larsen 2023) establish that $m = \Theta(\varepsilon^{-2} \log n)$ suffices for preserving all pairwise distances in an $n$-point set. The Larsen–Nelson (FOCS 2017) lower bound shows this is tight even against nonlinear embeddings, meaning any improvement must come from exploiting structure in the point set.

### 7.2 Width-adaptive and structured-set JL

Gordon's theorem (1988) and Klartag–Mendelson (JFA 2005) show that for structured sets of Gaussian width $w$, $m = O(w^2 / \varepsilon^2)$ suffices — a strict improvement over $\log n$ when $w^2 \ll \log n$. Bourgain–Dirksen–Nelson (STOC 2015) extend this to sparse JL matrices. Narayanan–Nelson (STOC 2019) give optimal *terminal* embeddings that preserve distances from any ambient query to a fixed anchor set in $m = O(\varepsilon^{-2} \log n)$. These are the key technical tools for Claim 1.

### 7.3 Compressive metric learning

Karimi–Wong–Ghodsi (APBC 2006) and the LightOn supervised-random-projection line combine random projection with class-aware metric learning but without generalization theory. Harandi–Salzmann–Hartley (ICML 2017) jointly learn projection and metric on Riemannian manifolds. Palias–Kabán (IJCNN 2024) is the most direct precedent: they prove dimension-free generalization for Mahalanobis LMNN trained in a JL-projected space, with error scaling in the stable dimension of the data support. Extending this from single-modality LMNN to cross-modal InfoNCE is a primary theoretical contribution of the proposal.

### 7.4 Cross-modal hashing

A decade of work — CMSSH (Bronstein et al. 2010), CVH (Kumar & Udupa 2011), IMH (Song et al. 2013), CMFH (Ding et al. 2014), SCM (Zhang & Li 2014), SePH (Lin et al. 2015), DCMH (Jiang & Li 2017), CCQ (Long et al. 2016) — optimizes objectives that explicitly preserve user-specified cross-modal pair sets, usually in Hamming space with learned nonlinear projections. These methods are the closest existing precedent to the proposal's training objective, but none connects to JL distortion theory. The Wang–Shen–Song–Ji (TPAMI 2018) survey and the Wang–He (2016) survey provide comprehensive overviews.

### 7.5 Multi-modal foundation models

CLIP (Radford et al., ICML 2021), ALIGN (Jia et al., ICML 2021), and ImageBind (Girdhar et al., CVPR 2023) all use learned linear projection heads between frozen or jointly-trained backbones and a shared cosine-similarity space, trained with InfoNCE. SimCLR (Chen et al., ICML 2020) established the empirical importance of the projection head; Gui–Chen–Liu (NeurIPS 2023) provide the first theoretical account via an "expansion/shrinkage" phenomenon. Our Claim 3 and diagnostic experiment D1 directly engage with this theory.

### 7.6 Efficient-attention JL applications

Linformer (Wang et al. 2020) projects keys and values along the sequence axis using JL, with a correctness proof from the distributional JL lemma. Performer (Choromanski et al., ICLR 2021) uses positive orthogonal random features to approximate the softmax kernel. These demonstrate that JL-style projections are compatible with gradient-based training in transformer architectures, but neither addresses cross-modal alignment.

### 7.7 Random projections as training mechanisms

BEST-RQ (Chiu et al., ICML 2022) uses a frozen random-projection quantizer as the *target* in masked-prediction speech self-supervision. RanDumb (Prabhu et al. 2024) shows random Fourier features outperform learned features in online continual learning. Sui et al. (ICLR 2024) learn representations by reconstructing random data projections. These empirically validate the use of frozen random projections at training time, supporting feasibility of the proposal.

### 7.8 Provable contrastive learning

HaoChen–Wei–Gaidon–Ma (NeurIPS 2021) prove linear-probe generalization for spectral contrastive loss via the top-$k$ eigendecomposition of an augmentation graph. Their framework is the natural theoretical vehicle for the proposal: compose spectral-contrastive with Gordon-width and Palias–Kabán-style compressive-metric generalization to obtain Claim 2. Wen et al. (CILA, 2024) provide tighter bounds with data-augmentation-aware analysis.

---

## 8. Practical Implications

### 8.1 On-device multi-modal retrieval

A frozen $768 \to 256$ sparse JL with $s = 10$ non-zeros per column consumes ~7.5 KB of storage and requires ~2500 multiply-accumulates per query. Composed with a $256 \times 256$ Mahalanobis (65K parameters, 65K MACs), the total inference footprint fits comfortably in mobile-device caches. This would enable CLIP-quality multi-modal retrieval on smartphones without server round-trips — currently impractical due to CLIP's 512-dim projection heads requiring ~400K parameters *just for projection*.

### 8.2 Federated multi-modal learning

A major obstacle to federated learning of multi-modal models is that clients cannot share raw features due to privacy concerns (features can be inverted; He–Zhang–Lee 2019). A frozen JL is a one-way function in a meaningful sense: inversion requires solving an underdetermined linear system, and if $m \ll d$ the solution is not unique. Experiment E5 will test whether the feature-inversion attack success rate drops substantially when only JL-projected features are shared, while retrieval performance is preserved. Positive results would enable privacy-preserving federated training of multi-modal foundation models — a currently open problem.

### 8.3 Compressed cross-modal indexes

Modern vector-search systems (FAISS, Pinecone, Weaviate) index billions of CLIP embeddings. Replacing the 512-dim CLIP embedding with a 128-dim JL+Mahalanobis embedding gives a 4× storage reduction and a 4× search speedup with minimal recall loss. For a billion-scale index this translates to $O(100\text{GB})$ memory savings per replica and proportional cloud-cost reduction.

### 8.4 Rapid domain adaptation

Because only the Mahalanobis head is trained (~65K parameters at $m = 256$), adaptation to a new domain requires orders of magnitude less data and compute than fine-tuning the CLIP projection head (~1.3M parameters). This makes it feasible to maintain domain-specialized multi-modal retrievers (medical imaging + reports, legal documents + evidence, satellite imagery + captions) without repeatedly fine-tuning large models.

### 8.5 Audit-friendly explainability

The pipeline decomposes cleanly into a *data-independent* JL step (fully specified by a random seed) and a *data-dependent* Mahalanobis step (whose spectrum can be audited). For regulated domains (medical, legal) this separation is valuable: one can certify the oblivious step never discards information specific to any protected subgroup, while the Mahalanobis step's eigendecomposition provides transparent insight into which directions the model emphasizes.

---

## 9. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| JL+Mahalanobis gap to CLIP-head never closes below 3 Recall@1 points | Medium | High | Results are still publishable as a principled negative result identifying what learned heads capture beyond JL; D1 diagnostic isolates which directions matter |
| Gaussian width estimation on real data is intractable | High | Medium | Use upper-bound proxies (effective rank, $\log n$); verify theoretical scaling on synthetic data with controlled width |
| InfoNCE spectral rewrite is not tight enough for dimension-free bound | Medium | Medium | Fall back to pair-based contrastive loss for theory; use InfoNCE for empirics with weaker but still meaningful bounds |
| Expansion/shrinkage phenomenon (Gui–Chen–Liu) is irrecoverable by linear Mahalanobis | Low | High | Diagnostic D1 identifies this; if confirmed, extend pipeline to include a small MLP between JL and similarity computation; proves informative either way |
| Theoretical results exist but with bad constants | Medium | Low | Empirical constants are what matter for practice; report both and discuss gap honestly |
| Cross-modal scale mismatch between image and text features breaks JL width analysis | Medium | Medium | Normalize per-modality before JL; theoretical analysis explicitly handles anisotropic case via Mendelson (GAFA 2007) extensions |

---

## 10. Expected Contributions

1. **Theoretical.** First formal generalization bound for oblivious-projection + learned-Mahalanobis cross-modal contrastive learning, scaling dimension-freely in the relationship-graph's statistical dimension.
2. **Empirical.** First demonstration that frozen JL + small learned metric approaches (or matches) CLIP-style learned projection head on standard cross-modal retrieval benchmarks.
3. **Methodological.** Open-source reproducible pipeline with 20–100× fewer trainable parameters than standard CLIP-style heads, and feature caches that reduce the cost of future multi-modal research by an order of magnitude.
4. **Practical.** Privacy-preserving multi-modal representation sharing via JL-projected features with measured inversion-attack resistance.
5. **Diagnostic.** Empirical characterization of *exactly what* learned projection heads capture beyond oblivious distance preservation, informing future architecture design.

---

## 11. References

**JL foundations.** Johnson & Lindenstrauss (1984); Dasgupta & Gupta (Random Structures & Algorithms 2003); Achlioptas (JCSS 2003); Ailon & Chazelle (STOC 2006, SICOMP 2009); Kane & Nelson (JACM 2014); Cohen–Jayram–Nelson (SOSA 2018); Fandina–Høgsgaard–Larsen (ICML 2023).

**Lower bounds.** Larsen & Nelson (ICALP 2016, FOCS 2017); Alon (Disc. Math. 2003).

**Width-adaptive JL.** Gordon (1988); Klartag & Mendelson (JFA 2005); Mendelson–Pajor–Tomczak-Jaegermann (GAFA 2007); Bourgain–Dirksen–Nelson (STOC 2015, GAFA 2015); Dirksen (FoCM 2016); Liaw–Mehrabian–Plan–Vershynin (2017); Amelunxen–Lotz–McCoy–Tropp (2014); Oymak & Tropp (2017).

**Terminal embeddings.** Mahabadi–Makarychev–Makarychev–Razenshteyn (STOC 2018); Narayanan & Nelson (STOC 2019); Cherapanamjeri & Nelson (TheoretiCS 2024).

**Compressive metric learning.** Karimi–Wong–Ghodsi (APBC 2006, SRP 2018); Harandi–Salzmann–Hartley (ICML 2017); Palias & Kabán (IJCNN 2024); Reeve & Kabán (Machine Learning 2022).

**Cross-modal hashing.** Bronstein et al. (CMSSH, CVPR 2010); Kumar & Udupa (CVH, IJCAI 2011); Zhen & Yeung (CRH, NeurIPS 2012); Song et al. (IMH, SIGMOD 2013); Ding et al. (CMFH, CVPR 2014); Zhang & Li (SCM, AAAI 2014); Lin et al. (SePH, CVPR 2015); Jiang & Li (DCMH, CVPR 2017); Long et al. (CCQ, SIGIR 2016); Wang–Shen–Song–Ji survey (TPAMI 2018).

**Multi-modal foundation models.** Radford et al. (CLIP, ICML 2021); Jia et al. (ALIGN, ICML 2021); Girdhar et al. (ImageBind, CVPR 2023); Chen–Kornblith–Norouzi–Hinton (SimCLR, ICML 2020); Gui–Chen–Liu (NeurIPS 2023).

**Efficient attention.** Wang et al. (Linformer, 2020); Choromanski et al. (Performer, ICLR 2021); Yu et al. (Orthogonal Random Features, NeurIPS 2016).

**Random projections as training mechanisms.** Chiu et al. (BEST-RQ, ICML 2022); Prabhu et al. (RanDumb, 2024); Sui et al. (ICLR 2024); Daniely–Frostig–Singer (ICLR 2017).

**Provable contrastive learning.** HaoChen–Wei–Gaidon–Ma (NeurIPS 2021); Wen et al. (CILA, 2024).

**Learned sketches.** Hsu–Indyk–Katabi–Vakilian (ICLR 2019); Indyk–Vakilian–Yuan (NeurIPS 2019); Liu–Li–Razenshteyn–Woodruff (ICLR 2023); Tsikouras et al. (NeurIPS 2024).

**Privacy attacks on features.** He–Zhang–Lee (CCS 2019).

**Software.** Johnson–Douze–Jégou (FAISS, 2017).
