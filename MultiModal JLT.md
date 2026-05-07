# Shared Embedding Spaces in Many-Modality Contrastive Learning, and the Case for a Johnson–Lindenstrauss Bottleneck

## Overview

This report has two halves. **Part A** surveys how existing "omnimodal" representation-learning systems handle the *single shared embedding space* problem when extending beyond text+image to audio + video + text + image (and depth, IMU, 3D, thermal, etc.). **Part B** analyses, from theoretical and empirical angles, whether inserting a Johnson–Lindenstrauss-Transform (JLT) bottleneck — a fixed (random or learned) low-rank linear projection placed *immediately after each modality encoder, present at training and inference* — would scale **better, worse, or neutrally** to this many-modality regime. Throughout, "JLT-in-the-loop" is taken to be the proposal under analysis: the encoder learns under the constraint of having to survive a small target dimension k, and at inference only the k-dimensional vector is stored.

Where reported numbers, dimensions, or design choices appear, they are sourced inline. Where the literature speculates rather than confirms, this is flagged.

---

## PART A — Existing Many-Modality Shared Embedding Spaces

### A.1 ImageBind (Girdhar et al., FAIR, CVPR 2023)

ImageBind learns a *single joint embedding* across **six modalities**: image/video, text, audio, depth, thermal, and IMU. The architecture uses a separate encoder per modality (ViT-H for image/video and the visually-treated modalities, the CLIP text tower for text, a 1D-conv-then-Transformer for IMU); each encoder is followed by a **modality-specific linear projection head** that yields a fixed-size d-dimensional embedding, which is L2-normalized and used in an InfoNCE loss against image embeddings ([Girdhar et al., CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Girdhar_ImageBind_One_Embedding_Space_To_Bind_Them_All_CVPR_2023_paper.pdf); [Levi blog summary of architecture](https://gillevi.github.io/posts/ImageBind)). The shared embedding **dimension is 1024** for the released `imagebind_huge` model (initialized from OpenCLIP ViT-H), with the image and text encoders frozen and only the new modality encoders + projection heads trained ([GitHub facebookresearch/ImageBind](https://github.com/facebookresearch/ImageBind); [implementation walkthrough](https://itnext.io/imagebind-one-embedding-space-to-bind-them-all-b48c8623d39b)).

The key trick is the **image-anchor strategy**: because (image, X) pairs exist abundantly for many X (audio via Audioset 2M pairs, depth via SUN-RGBD 5K, thermal via LLVIP 12K, IMU via Ego4D 7.5K), they only train ImageBind on each (image, X) pair separately. Cross-modal alignments not seen during training (e.g., audio↔text, depth↔audio) emerge implicitly because every modality is bound to image, and image is bound to text via the OpenCLIP backbone ([CVPR paper Sec. 3](https://openaccess.thecvf.com/content/CVPR2023/papers/Girdhar_ImageBind_One_Embedding_Space_To_Bind_Them_All_CVPR_2023_paper.pdf); [MYRIAD blog](https://creatis-myriad.github.io/2024/03/20/ImageBind.html)). The reported emergent zero-shot accuracies match or exceed AudioCLIP-style trinity supervision; ImageBind sets new emergent SOTA on Audioset, ESC, Clotho, AudioCaps, LLVIP and SUN-D ([CVPR paper Table 1–3](https://openaccess.thecvf.com/content/CVPR2023/papers/Girdhar_ImageBind_One_Embedding_Space_To_Bind_Them_All_CVPR_2023_paper.pdf)).

**Problems with image-anchored binding**: (i) the image is privileged, producing an *unbalanced* embedding space where modalities far from image (audio, IMU) are pushed outward; (ii) no constraint that modality A and modality B align well except *via* image; (iii) the modality gap (see A.11) is replicated and compounded for each new modality.

### A.2 LanguageBind (Zhu et al., ICLR 2024)

LanguageBind replaces the image anchor with the **language anchor**, freezing the OpenCLIP text encoder and training video/audio/depth/infrared encoders to align to it via contrastive learning, with LoRA-only fine-tuning for efficiency ([Zhu et al., ICLR 2024](https://arxiv.org/pdf/2310.01852)). They release **VIDAL-10M**, a 10M-sample dataset of video–infrared–depth–audio–language quintuplets ([VIDAL repo](https://github.com/PKU-YuanGroup/LanguageBind)). The shared embedding dimension follows the underlying CLIP backbone — typically **512 (ViT-B) or 768 (ViT-L)** depending on which OpenCLIP variant is used. LanguageBind reports SOTA zero-shot text-to-video on MSR-VTT (+1.9%), MSVD (+8.8%), DiDeMo (+6.3%), and ActivityNet (+4.4%) over InternVideo, and beats ImageBind by 23.8% top-1 on ESC-50 audio classification ([ICLR paper](https://arxiv.org/pdf/2310.01852)).

The language-anchor strategy avoids the privileged-vision criticism but introduces its own problem: modalities like raw audio may have little natural alignment with descriptive text (sound textures and music are notoriously hard to caption), so audio↔language pairs require synthetic captioning pipelines.

### A.3 AudioCLIP (Guzhov et al., ICASSP 2022)

AudioCLIP extends CLIP into a **(image, text, audio) trinity** by replacing the image encoder slot of CLIP with a parallel branch using **ESResNeXt-fbsp** as the audio encoder and training contrastively on AudioSet ([ICASSP 2022 IEEE](https://resourcecenter.ieee.org/conferences/icassp-2022/spsicassp22vid1820); [AudioCLIP code](https://github.com/AndreyGuzhov/AudioCLIP)). It uses CLIP's text-encoder hidden size of 512 as the joint dimension, inheriting CLIP's vocabulary and projection ([DCASE technical report](https://dcase.community/documents/challenge2022/technical_reports/DCASE2022_Wu_100_t6b.pdf)). AudioCLIP achieves **97.15% on ESC-50 and 90.07% on UrbanSound8K**, and zero-shot **69.4% / 68.78%** on the same datasets — first to set zero-shot ESC baselines ([ICASSP record](https://resourcecenter.ieee.org/conferences/icassp-2022/spsicassp22vid1820)).

### A.4 CLAP (Microsoft / LAION variants)

CLAP (Contrastive Language–Audio Pretraining) is a CLIP-style two-tower model for audio↔text. The LAION-CLAP variant uses **HTS-AT** or PANN as audio encoder and CLIP's text transformer (12 layers, 8 heads, 77 context, 512-dim hidden) as text encoder, with a **shared projection dimension of 512** ([LAION-CLAP DCASE report](https://dcase.community/documents/challenge2022/technical_reports/DCASE2022_Wu_100_t6b.pdf); [HuggingFace CLAP config](https://huggingface.co/docs/transformers/en/model_doc/clap); [ailia CLAP overview](https://medium.com/axinc-ai/clap-feature-extraction-model-for-searching-audio-from-text-dcfd4c93756e)). LAION-Audio-630K (633,526 audio–text pairs) is the released training corpus; LAION-CLAP raises ESC-50 zero-shot from Microsoft-CLAP's 82.6% to **89.1%** ([HAL paper](https://hal.science/hal-04766539v1/file/clap.pdf)). CLAP is the de-facto audio tower for downstream omnimodal systems (e.g., OmniBind uses it as one of 14 expert spaces).

### A.5 Video–Text: VideoCLIP, X-CLIP, InternVideo2, VideoCoCa

Video-text alignment historically uses CLIP-derived towers: **VideoCLIP** (Xu et al., 2021) and **X-CLIP** add temporal attention; **VideoCoCa** (Yan et al., 2022) inflates CoCa to videos. The most recent and largest is **InternVideo2** (Wang et al., ECCV 2024), which scales to a **6B-parameter video encoder**, uses BERT-Large (19 layers as encoder + 5 cross-attention layers as decoder) for text, and a 12-layer BEATs-initialized 90M audio encoder with 64-dim log-Mel filterbank features ([InternVideo2 paper](https://arxiv.org/pdf/2403.15377); [HTML version](https://arxiv.org/html/2403.15377v1)). InternVideo2's training is staged: spatiotemporal token reconstruction → video–audio–speech–language contrastive learning → joint LLM training. The contrastive embedding follows CLIP/CoCa convention; ViT-1B uses CoCa-1B's hyperparameters and ViT-6B uses InternViT-6B's. It was previously SOTA on most video benchmarks; trained on ~100M video–text pairs and 50M video–audio–speech–text quadruples ([VidVec related work](https://arxiv.org/html/2602.08099)).

### A.6 Meta-Transformer (Zhang et al., 2023)

Meta-Transformer takes a different stance: a **single frozen transformer encoder** processes 12 modalities (text, image, point cloud, audio spectrogram, video, infrared, hyperspectral, X-ray, IMU, tabular, graph, time-series) by mapping them all into a *shared token space* with modality-specific tokenizers, then using one frozen LAION-2B-pretrained encoder to extract features ([Zhang et al. 2023 arXiv](https://arxiv.org/pdf/2307.10802); [project page](https://kxgong.github.io/meta_transformer/)). Base-scale uses **embedding dim 768, MLP dim 3072, 12 blocks/heads** (B16); Large uses **dim 1024, 24 blocks, 16 heads** (L14) ([Meta-Transformer GitHub](https://github.com/invictus717/MetaTransformer)). It is the first framework to perform unified learning across 12 modalities with unpaired data, and a precursor to **OneLLM** ([HuggingFace papers](https://huggingface.co/papers/2307.10802)). Note: Meta-Transformer is *not* a contrastive multimodal model in the CLIP sense — it is more of a unified perception backbone; the shared-embedding-space property is achieved by token-space sharing rather than InfoNCE alignment.

### A.7 OmniBind (Wang et al., 2024)

OmniBind is the current SOTA "omni" representation. Rather than training one encoder per modality from scratch, it **binds 14 pre-trained spaces together** (CLIP/EVA-CLIP-18B, CLAP, Wavcaps, Uni3D, etc.) via lightweight projection MLPs and **learned routers** that dynamically weight each expert space, inspired by Mixture-of-Experts ([OmniBind arXiv](https://arxiv.org/abs/2407.11895); [HTML Sec. 1](https://arxiv.org/html/2407.11895v1); [Moonlight literature review](https://www.themoonlight.io/en/review/omnibind-large-scale-omni-multimodal-representation-via-binding-spaces)). Three model scales — **7B, 14B, and 30B parameters** — all support 3D point cloud, audio, image, and language (later extended to video). The 30B model trains in 3 days on a single 8×4090 node from unpaired data, using a cross-modal alignment loss and a **language representation decoupling** loss to prevent text embeddings aligned to different modalities from conflicting ([OmniBind GitHub](https://github.com/zehanwang01/OmniBind)). The output dimension is dictated by the largest underlying expert (EVA-CLIP-18B has 1024-d projections for the largest configuration). OmniBind sets SOTA across 13 benchmarks for any-modality-pair retrieval; recent work (EBind, [arXiv 2511.14229](https://arxiv.org/html/2511.14229)) confirms OmniBind is currently the strongest model that simultaneously embeds text+image+audio+video+point clouds.

### A.8 UniBind, FreeBind, OneLLM, NExT-GPT

- **UniBind** (Lyu et al., CVPR 2024) addresses ImageBind's *unbalanced embedding space* by making alignment centers **modality-agnostic**, using LLM-generated class-wise embedding centers as anchors. It binds seven modalities (image, text, audio, point cloud, thermal, video, event), achieving an average **+6.36% zero-shot recognition** over ImageBind, +3.83% on N-Caltech with E-CLIP ([UniBind arXiv](https://arxiv.org/pdf/2403.12532); [CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Lyu_UniBind_LLM-Augmented_Unified_and_Balanced_Representation_Space_to_Bind_Them_CVPR_2024_paper.pdf)).
- **FreeBind** (Wang et al., ICML 2024) treats representation spaces as basic units and uses **"space bonds"** (Displacement Bond and Combination Bond) to integrate expert spaces into a pre-trained unified space without catastrophic forgetting ([FreeBind paper](https://arxiv.org/pdf/2405.04883); [ICML 2024](https://proceedings.mlr.press/v235/wang24co.html)). It is a precursor to OmniBind from the same group.
- **OneLLM** (Han et al., CVPR 2024) aligns **eight modalities** (image, audio, video, point cloud, depth/normal, IMU, fMRI, text) to language using a frozen CLIP universal encoder + a **Universal Projection Module** (mixture of expert projection heads + routers) + frozen LLM ([OneLLM paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Han_OneLLM_One_Framework_to_Align_All_Modalities_with_Language_CVPR_2024_paper.pdf); [project page](https://onellm.csuhan.com/)). The modality token dimension is **30×1024**.
- **NExT-GPT** (Wu et al., ICML 2024) builds an "any-to-any" MM-LLM by reusing **ImageBind as the front-end encoder** for all input modalities, then connecting to Vicuna and to Stable-Diffusion / Zeroscope / AudioLDM diffusion decoders for output. Only ~1% of parameters (the projection layers) are tuned ([NExT-GPT arXiv](https://arxiv.org/abs/2309.05519); [project page](https://next-gpt.github.io/)). Embedding dimension is inherited from ImageBind (1024).

### A.9 Composition of Pre-Trained Spaces: C-MCR and Ex-MCR

- **C-MCR** (Wang et al., NeurIPS 2023): given two pre-trained MCRs sharing an overlapping modality (e.g., CLIP shares text with CLAP, and shares image with ULIP), C-MCR learns two simple projectors that connect the spaces using the overlapping modality as positive anchor pairs. No new paired data is required. Audio-visual SOTA on retrieval, source localization, counterfactual recognition; advanced 3D zero-shot on ModelNet40 ([C-MCR paper](https://arxiv.org/pdf/2305.14381); [NeurIPS 2023 page](https://proceedings.neurips.cc/paper_files/paper/2023/hash/46362971bfc3a97e6a271f2eb90fba17-Abstract-Conference.html)).
- **Ex-MCR** (Wang et al., NeurIPS 2024) addresses C-MCR's main flaw — that connecting two MCRs forgets the original alignments — by **extending one MCR space into another's** rather than mapping both into a brand-new space ([Ex-MCR paper](https://arxiv.org/pdf/2310.08884); [NeurIPS 2024 poster](https://neurips.cc/virtual/2024/poster/95280); [Code](https://github.com/MCR-PEFT/Ex-MCR)). Without paired data, Ex-MCR achieves SOTA on 3D-image, audio-text, audio-visual, and 3D classification; produces emergent audio↔3D alignment.

### A.10 3D Modality: PointBind, PointCLIP, ULIP-2

- **PointCLIP** projects point clouds to multi-view 2D depth maps and feeds them through CLIP's image encoder ([PointCLIP CVPR 2022](https://arxiv.org/html/2212.05171)).
- **ULIP** (Xue et al., CVPR 2023) and **ULIP-2** (Xue et al., 2024) train a 3D encoder to **align with the frozen CLIP image–text space**, using triplets (point cloud, rendered image, text) on ShapeNet55. ULIP outperforms PointCLIP by **+28.8% on ModelNet40 zero-shot** ([ULIP paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Gao_ULIP_Learning_a_Unified_Representation_of_Language_Images_and_Point_Clouds_for_3D_Understanding.pdf); [Salesforce blog](https://www.salesforce.com/blog/ulip/)). Embedding dim = CLIP's (512 for ViT-B/32 backbone).
- **Point-Bind** (Guo et al., 2023) brings 3D point clouds into ImageBind's space by training a 3D encoder against image, language, audio, and video embeddings simultaneously, enabling any-to-3D generation, 3D embedding arithmetic, and 3D zero-shot ([Point-Bind arXiv](https://arxiv.org/abs/2309.00615); [GitHub](https://github.com/ZiyuGuo99/Point-Bind_Point-LLM)). It uses ImageBind's 1024-d projection space.

### A.11 The Modality Gap (Liang et al., NeurIPS 2022) and follow-ups

Liang et al. demonstrated that even after CLIP-style training, embeddings from different modalities occupy **disjoint cones** in the shared space, separated by an "arm's length" gap ([NeurIPS 2022 paper PDF](https://arxiv.org/pdf/2203.02053); [project site](https://modalitygap.readthedocs.io/en/latest/)). The cause is two-fold: (a) the **cone effect** in deep-network initialization (representations live in a narrow cone before any training), and (b) optimization with a contrastive loss + a learnable temperature. Key recent results refine this picture:

- **It's Not a Modality Gap** (Fahim et al., 2024): the gap is in fact the **contrastive gap** — a generic property of NT-Xent, not unique to modalities; closing it slightly via projection improves downstream performance ([arXiv 2405.18570](https://arxiv.org/pdf/2405.18570)).
- **Decipher the Modality Gap** (Yi et al., Oct 2025): proves that under the **subspace constraint** (i.e., each modality occupies a low-rank subspace due to dimension collapse), the modality gap converges to the smallest angle between hyperplanes — identifying **dimension collapse as the fundamental origin** of the gap ([arXiv 2510.03268](https://arxiv.org/pdf/2510.03268)).
- **Yaras et al., Feb 2026**: characterize gradient-flow dynamics that show *mismatched data pairs* and a learnable temperature drive the gap ([arXiv 2412.07909](https://arxiv.org/pdf/2412.07909)).
- **Closing the Modality Gap Aligns Group-Wise Semantics**: extends the analysis from bimodal to trimodal benchmarks and shows reducing the gap improves clustering metrics while preserving retrieval ([arXiv 2601.18525](https://arxiv.org/pdf/2601.18525)).

The implication for many-modality learning is severe: with m modalities, one can have up to **m disjoint cones**, each "arm's length" from the others, *amplifying* cross-modal misalignment and reducing retrieval accuracy unless explicit gap-closing or modality-symmetric losses are used (UniBind, GRAM, Symile).

### A.12 SigLIP, SigLIP 2

- **SigLIP** (Zhai et al., ICCV 2023) replaces the global softmax of CLIP's InfoNCE with a **pairwise sigmoid loss** on each image–text pair, eliminating the need to all-gather similarities across devices ([arXiv 2303.15343](https://arxiv.org/abs/2303.15343); [ICCV 2023 PDF](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhai_Sigmoid_Loss_for_Language_Image_Pre-Training_ICCV_2023_paper.pdf)). This decouples batch size from the loss, enables 1M-batch experiments, and performs better at small batch sizes; ImageNet zero-shot saturates around 32k batch. The embedding dimension is preserved from the encoder backbone (768 for ViT-B, 1024 for ViT-L, 1152 for SoViT-400m).
- **SigLIP 2** (Tschannen et al., Feb 2025) augments SigLIP with captioning-based pretraining, self-distillation, masked prediction, and online data curation; supports multilingual (109 languages, multilingual Gemma tokenizer) and dynamic-resolution variants ([arXiv 2502.14786](https://arxiv.org/abs/2502.14786); [HuggingFace blog](https://huggingface.co/blog/siglip2)).

For **many-modality** settings, the sigmoid loss is structurally appealing because it operates per-pair, so adding modalities multiplies pair counts but does not require all-gathering across modalities. However, no paper has yet published an omnimodal SigLIP with audio + video + text + image; the closest is the LAION community's experimentation with HTSAT-fused CLAP. Theoretical analysis ([Lee et al. 2024 PMLR](https://proceedings.mlr.press/v238/lee24a/lee24a.pdf)) shows the sigmoid loss converges to a **double-Constant Embedding Model (CCEM)** — a structured generalization of the simplex equiangular tight frame — which suggests it may resist some pathologies of softmax-InfoNCE in the multimodal regime.

### A.13 Other relevant recent omnimodal architectures

- **VAST** (Chen et al., NeurIPS 2023) trains a vision–audio–subtitle–text model on a curated **VAST-27M** dataset of 27M video clips, each with 5 vision, 5 audio, and 1 omni-modality caption; uses EVA-CLIP-g and BEATs encoders, achieving 22 new SOTAs on cross-modality benchmarks ([VAST paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/e6b2b48b5ed90d07c305932729927781-Paper-Conference.pdf); [Code](https://github.com/TXH-mercury/VAST)).
- **GRAM** (Cicchetti et al., ICLR 2025) replaces pairwise cosine similarity with the **Gramian volume** of the parallelotope spanned by m modality vectors, ensuring *simultaneous* alignment across all modalities. It outperforms ImageBind/LanguageBind by 5–10 points on video–audio–text retrieval and audio-video classification ([GRAM ICLR 2025](https://iris.uniroma1.it/retrieve/46499d8a-ab88-4b94-9f3e-46da88ea5a4c/Cicchetti_Gramian_2025.pdf); [GRAM page](https://ispamm.github.io/GRAM/)). Critically, GRAM is geometric, not pairwise, so it is inherently better suited to many modalities.
- **Symile** (Saporta et al., NeurIPS 2024) generalizes contrastive learning by maximizing a lower bound on **total correlation** across all modalities using a **multilinear inner product (MIP)**. Symile achieves near-perfect 1.0 retrieval where CLIP achieves 0.5 (random) on synthetic three-modality tasks with strict higher-order dependencies ([Symile NeurIPS](https://papers.nips.cc/paper_files/paper/2024/file/6828259348d99d5e8994028bfdf15d09-Paper-Conference.pdf); [PyPI page](https://pypi.org/project/symile/)). However, recent work [arXiv 2604.05834](https://arxiv.org/abs/2604.05834) finds Symile is **fragile** to a single misaligned modality due to multiplicative interactions, motivating "Gated Symile". Note: this latter paper has a 2026 preprint date and should be treated as preliminary.
- **EBind** (arXiv 2511.14229, Nov 2025) shows that, surprisingly, *one frozen encoder + one MLP projector per modality* trained on a small curated dataset can recover most of OmniBind-30B's performance, suggesting **the projection head is doing most of the heavy lifting** in space-binding ([EBind HTML](https://arxiv.org/html/2511.14229)). This is highly relevant to the JLT proposal: the projection head is essentially what JLT-in-the-loop modifies.

### A.14 Summary Table (Part A)

| System | Modalities | Anchor | Embedding dim | Training data | Backbone(s) |
|---|---|---|---|---|---|
| CLIP | 2 (image, text) | symmetric | 512–1024 | 400M pairs | ViT + transformer |
| AudioCLIP | 3 (image, text, audio) | symmetric trinity | 1024 | AudioSet | CLIP + ESResNeXt-fbsp |
| LAION-CLAP | 2 (audio, text) | symmetric | 512 | LAION-Audio-630K | HTS-AT + RoBERTa |
| ImageBind | 6 (image, text, audio, depth, thermal, IMU) | image | 1024 | image–X pairs only | OpenCLIP-H + ViT/Conv |
| LanguageBind | 5 (video, audio, depth, infrared + language) | language | 768 / 1024 | VIDAL-10M | OpenCLIP + LoRA |
| Meta-Transformer | 12 | shared token space | 768 / 1024 | LAION-2B (images) | frozen ViT |
| OmniBind | 4 → 5 (3D, audio, image, language [+video]) | multi-anchor MoE | ~1024 | unpaired, 14 expert spaces | EVA-CLIP-18B + CLAP + Uni3D + … |
| OneLLM | 8 | language | 1024 | 2M instruction items | frozen CLIP + LLM + UPM (MoE) |
| NExT-GPT | 4 (text, image, video, audio) | language (LLM) | 1024 | inherits ImageBind | ImageBind + Vicuna + diffusion |
| C-MCR | 3 (audio-visual / 3D-language) | overlapping modality | small projector | unpaired | CLIP + CLAP / ULIP |
| Ex-MCR | 4 (3D-image-text-audio) | base MCR | inherits CLIP | unpaired | CLIP + CLAP + ULIP |
| ULIP / ULIP-2 | 3 (image, text, point cloud) | image–text via CLIP | 512 | ShapeNet55 triplets | CLIP + Point-BERT/PointMLP |
| Point-Bind | 5 (3D + ImageBind's 4) | image (via ImageBind) | 1024 | ULIP-style triplets | ImageBind + 3D encoder |
| UniBind | 7 | LLM-augmented modality-agnostic centers | follows backbone | LLM-generated text knowledge base | CLIP-style backbones |
| GRAM | n (video, audio, text+) | volumetric | follows backbone | omni-modality captions | EVA-CLIP + BEATs + BERT |
| Symile | n | total-correlation MIP | follows backbone | Symile-M3 (33M), Symile-MIMIC | architecture-agnostic |
| InternVideo2 | 4 (video, audio, speech, text) | text | follows backbone | ~100M video–text + 50M quadruples | ViT-6B + BEATs + BERT-Large |
| VAST | 4 (vision, audio, subtitle, text) | text | follows backbone | VAST-27M | EVA-CLIP-g + BEATs |

The emergent picture: **almost all current omnimodal systems use 512–1024-dimensional embeddings**, regardless of modality count. None systematically explore the trade-off curve between embedding dimension and many-modality accuracy. None deliberately use a **fixed (random) bottleneck** in training; the closest is the lightweight linear "projection head" (modality-specific) used everywhere.

---

## PART B — Theoretical and Analytical Comparison: Would a JLT Bottleneck Scale Better, Worse, or Neutrally to Many Modalities?

### B.1 Intrinsic dimension differs by modality

The intrinsic dimension (ID) of representation manifolds varies sharply across modalities:

- **Image manifolds** are remarkably low-dimensional: standard datasets (CIFAR-10, ImageNet, MS-COCO) have intrinsic dimension on the order of **26–43**, even though each image lives in pixel space of dimension >10⁵ ([Pope et al., ICLR 2021](https://openreview.net/pdf?id=XJk19XzGq2J); [Pope et al. arXiv](https://arxiv.org/abs/2104.08894)).
- In trained networks, last-hidden-layer ID is "orders of magnitude smaller than the number of units" and **predicts test accuracy** ([Ansuini et al., NeurIPS 2019](https://arxiv.org/abs/1905.12784)).
- **Audio / speech** manifold IDs are small in the tens — multi-class disordered voice analysis finds local intrinsic dimensions in the single to low double digits ([Liu et al., 2018 J Voice](https://doi.org/10.1177/0003489418780439); [acoustic scene classification work](https://arxiv.org/pdf/2204.00555) shows rapid singular-value decay in intermediate audio layers indicating low-D structure). Speech audio embeddings exhibit **layerwise ID peaks** in the middle layers, with semantic abstraction increasing with ID ([arXiv 2602.04081](https://arxiv.org/pdf/2602.04081)).
- **Text / language model** representations have a layerwise ID that peaks in middle layers and is correlated with semantic richness ([same paper](https://arxiv.org/pdf/2602.04081)); LM fine-tuning ID estimates by Aghajanyan et al. give task-specific IDs in the **hundreds to low thousands** ([HuggingFace papers comment](https://huggingface.co/papers/2012.13255)).
- **Video** manifolds are higher-dimensional than images because they add temporal degrees of freedom; no clean ID number exists in the literature, but InternVideo2's 6B-parameter encoder and the consistent use of 1024-d projection suggest empirical d∈[256, 1024] is required for SOTA accuracy.

**JLT implication**: a single fixed k that satisfies all modalities must respect the *largest* per-modality ID. If image ID ≈ 40 and video ID ≈ 200–300 and text ID ≈ 100, then k ≈ 256 should suffice for inner-product preservation. Empirically, MRL and DirectCLR work well with d≈64–256 across image/text retrieval ([Marqo MRL+CLIP results](https://www.marqo.ai/blog/matryoshka-representation-learning-with-clip-for-multimodal-retrieval-and-ranking)). A *forced* shared k (say k=128) is therefore likely to over-compress video and under-utilize image; this argues for **modality-specific JLT widths into a shared space** — i.e., k_video ≥ k_image after a learned alignment, but with the *output* dimension (after final projection) shared. This generalizes the standard CLIP-style "modality-specific projection head", but with the JL guarantee added.

### B.2 JL Union Bound Across Modalities

The standard distributional JL lemma states: for n points and target distortion ε, k = O(log n / ε²) is sufficient to preserve all pairwise inner products with high probability ([Wikipedia random projection](https://en.wikipedia.org/wiki/Random_projection); [scikit-learn random projection docs](https://scikit-learn.org/stable/modules/random_projection.html); [Freksen survey](https://arxiv.org/pdf/2103.00564); [Optimization can learn JL embeddings](https://openreview.net/pdf?id=w3JCTBRduf)). With m modalities of n_m points each, the union bound across all O((Σn_m)²) cross-modal pairs gives k = O(log Σn_m / ε²).

**Quantitative example**: for n = 10⁹ items per modality across m = 5 modalities (≈5×10⁹ total), with ε = 0.1:
- log₂(5×10⁹) ≈ 32.2; k ≥ 4·log(N)/(ε²/2 − ε³/3) ≈ 4·32/0.0467 ≈ **2740** for ε-embedding guarantee in the *worst case* (per scikit-learn's `johnson_lindenstrauss_min_dim`).
- For ε = 0.2 (a more practical "effective rank"), k ≈ **600**.

**Critically, the dependence on the number of modalities m is logarithmic**, not linear. Adding audio + video + IMU + thermal to a text+image system multiplies n by a constant (say ×5), increasing k by log₂(5) ≈ 2.3 dimensions. This is the strongest argument that **JLT should scale gracefully with modality count**: the extra geometric capacity needed is essentially negligible relative to the intrinsic encoder capacity. The catch: the JL guarantee is for *Euclidean inner-product preservation under random orthogonal projection*; the multimodal contrastive objective requires **cross-modal pairs** to obey specific separations (positives near, negatives far), not arbitrary distances. So the JL bound is a *necessary* but not *sufficient* condition — it tells you the geometry can fit, not that the optimizer will find it.

### B.3 Modality Gap × JLT Interaction

Three relevant facts:
1. The modality gap is fundamentally caused by *dimension collapse into disjoint subspaces* per modality ([Yi et al. 2025](https://arxiv.org/pdf/2510.03268)).
2. A JLT does not (on its own) close the gap — random projection preserves inner products, so disjoint cones in d remain disjoint cones in k.
3. However, a *bottleneck* k smaller than the natural per-modality cone dimension forces the encoder to discover a more shared structure during training: there is no longer enough capacity for each modality to "live in its own corner".

**Prediction**: a JLT bottleneck during training will *reduce* the modality gap because it geometrically couples modalities — the gradient signal forces each modality to use the *same* k coordinates, not its own preferred subspace. The MissModal and PROMISE results ([MIT Press TACL paper](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00628/118797/MissModal-Increasing-Robustness-to-Missing); [PROMISE arXiv](https://arxiv.org/html/2511.10997)) suggest that *any* alignment-encouraging architectural constraint reduces the gap. This is consistent with the EBind finding that lightweight projectors recover most of OmniBind's performance — the projection is doing alignment work, and a JL projection is a special case.

**Caveat**: the gap is sometimes *desirable* — closing it post-hoc can degrade downstream performance ([Yi et al. 2025 Sec. A.1](https://arxiv.org/pdf/2510.03268); [Liang et al. 2022 follow-ups](https://github.com/Weixin-Liang/Modality-Gap)). So an aggressively low k may collapse the gap in a way that hurts retrieval. The right operating point is "small enough to force coupling, large enough to preserve modality-specific information".

### B.4 Alignment–Uniformity–Compression Trilemma

For m modalities, the InfoNCE objective has up to m(m−1)/2 pairwise constraints (cross-modal positive pairs) plus uniformity constraints. With more modalities:
- **Alignment** (positives close): becomes harder because the same anchor must be close to multiple modalities simultaneously.
- **Uniformity** (negatives spread on the unit sphere): in d=k dimensions, uniformity is easier with larger k (more room for neural-collapse-style simplex equiangular tight frames).
- **Compression** (storage): demands smaller k.

A JLT bottleneck **enforces compression by construction**; the question is whether the encoder can simultaneously satisfy alignment and uniformity at small k when there are many modalities. The neural-collapse / ETF literature ([Lu & Steinerberger 2022 referenced in Lee et al. 2024 PMLR](https://proceedings.mlr.press/v238/lee24a/lee24a.pdf)) shows the optimal contrastive-loss embedding *lives* on a simplex ETF in d ≥ N − 1 (where N is number of classes/labels). For N ≈ 10⁹ items this is unbounded, but in practice well-trained encoders satisfy near-ETF structure with d in the low hundreds. So **a JLT bottleneck at k ≈ 256–512 should be sufficient** for many-modality contrastive learning, possibly with slight degradation versus k ≈ 1024 because of reduced uniformity headroom.

### B.5 Modality Collapse

There is now a literature documenting **modality collapse** in multimodal contrastive distillation and dataset distillation: representations from one modality become over-concentrated, cross-modal alignment degrades, and one modality "dominates" ([Beyond Modality Collapse / RepBlend, arXiv 2505.14705](https://arxiv.org/pdf/2505.14705); [Decipher modality gap, arXiv 2510.03268](https://arxiv.org/pdf/2510.03268)). DirectCLR ([Jing et al., ICLR 2022](https://arxiv.org/pdf/2110.09348)) demonstrates that **dimensional collapse** (the embedding spans a strictly lower-dimensional subspace than k) arises in contrastive learning from two mechanisms: strong augmentation that exceeds intra-class variance, and implicit low-rank regularization in deep networks ([Meta AI blog](https://ai.meta.com/blog/understanding-dimensional-collapse/)).

A **JLT bottleneck during training** is a double-edged sword here:
- **Mitigation**: a fixed JLT distributes the gradient signal across all k coordinates, preventing any single modality from "capturing" a disjoint subspace. DirectCLR's success — applying InfoNCE only to a sub-vector of the representation while letting the residual connection back-propagate full-rank updates — is essentially the same idea as JLT-in-the-loop, and it *outperforms* SimCLR with a learned linear projector on ImageNet ([DirectCLR repo](https://github.com/facebookresearch/directclr); [openreview](https://openreview.net/forum?id=YevsQ05DEN7)).
- **Amplification risk**: if k is *too* small relative to the number of modalities, the model may collapse all modalities onto a degenerate low-rank embedding. Modality collapse evidence in MMR/MSA ([MMR Robustness arXiv 2510.05839](https://arxiv.org/html/2510.05839)) shows that contrastive learning is sensitive to imbalance and dimensional headroom.

### B.6 Scaling with Number of Modalities m

Three regimes:

**Strategy A (current SOTA): increase d as m grows.** ImageBind, LanguageBind, and OmniBind all default to d=1024, with no clear ablation on what happens as m → 10. This is conservative but storage-expensive.

**Strategy B (JLT-in-the-loop): keep d small (k=128–256), enforce JL bottleneck during training and inference.** Training cost is essentially unchanged; storage cost shrinks 4× to 8×. Theoretical capacity (B.2) is sufficient. Risk: empirical accuracy at full-batch contrastive training has not been measured for m ≥ 4.

**Strategy C (MRL-style nested): train at multiple k simultaneously.** Matryoshka Representation Learning (Kusupati et al., NeurIPS 2022) proves that a single model can produce useful embeddings at d ∈ {8, 16, 32, …, 2048} **without accuracy loss** versus independently-trained models, and gives 14× embedding-size reduction at equal ImageNet accuracy ([MRL arXiv](https://arxiv.org/abs/2205.13147); [HuggingFace blog](https://huggingface.co/blog/matryoshka)). MRL has been applied to CLIP-style multimodal retrieval (e.g., GCL in Marqo's evaluation) with strong results: MRL-trained CLIP retains performance at 64-d while non-MRL CLIP loses substantial nDCG ([Marqo blog](https://www.marqo.ai/blog/matryoshka-representation-learning-with-clip-for-multimodal-retrieval-and-ranking)). **fMRLRec** (EMNLP 2024 Findings) extends MRL to multimodal recommendation across language + visual features ([fMRLRec paper](https://aclanthology.org/2024.findings-emnlp.786.pdf); [GitHub](https://github.com/yueqirex/fMRLRec)); SMEC (EMNLP 2025) further refines compression in the multimodal regime ([SMEC paper](https://aclanthology.org/2025.emnlp-main.1332.pdf)).

**Strategy B vs. C**: a JLT bottleneck is a *single-target-dimension* version of MRL. If the deployment requires only one fixed k, JLT-in-the-loop is simpler and may give marginally better accuracy at that specific k (no nested-loss interference). If the deployment needs flexible k, MRL is superior. The two are not mutually exclusive: one could train with MRL nested losses where each level is itself a JL projection — this is, to our knowledge, **unexplored in the literature**.

### B.7 Storage Scaling

For a database of N items with m modalities, storing all embeddings requires **m·N·k·4 bytes** (float32). For N = 1B and m = 5, k = 1024 yields **20 TB**; k = 128 yields **2.5 TB**; k = 64 yields **1.25 TB**. Combined with binary or int8 quantization (standard practice in vector databases), JLT can take this from ~20 TB to ~300 GB. **This is the strongest pragmatic argument for a JLT bottleneck**: storage cost scales linearly in m, so the multiplicative savings from a smaller k compound across modalities.

### B.8 Robustness to Missing Modalities

Most omnimodal benchmarks involve missing modalities at inference (medical AI, sensor failures, partial uploads). The literature is unanimous:
- Missing-modality performance drops 10–43% in vanilla contrastive systems ([C-MAM paper, ACM TIST 2024](https://dl.acm.org/doi/10.1145/3746456); [PROMISE arXiv](https://arxiv.org/html/2511.10997)).
- Methods that **align representations to a common low-dimensional space** (MissModal, single-branch SRMM, PROMISE) are systematically more robust to missing modalities ([SRMM arXiv 2408.07445](https://arxiv.org/html/2408.07445v1); [survey arXiv techrxiv](https://www.techrxiv.org/users/1000208/articles/1362573/master/file/data/Multimodel_Survey_2025/Multimodel_Survey_2025.pdf)).

A JLT bottleneck **explicitly forces all modalities through the same k-dimensional aperture** during training, which by analogy with SRMM should *improve* missing-modality robustness. The intuition: the encoder learns to encode each modality in an interchangeable form, since the bottleneck is shared. This is consistent with Symile's finding that pre-trained representations form a **sufficient statistic** for predicting missing modalities ([Symile NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/file/6828259348d99d5e8994028bfdf15d09-Paper-Conference.pdf)).

### B.9 Connections to MoE / Modality-Specific Projection Heads

Recent omnimodal methods increasingly use **mixture-of-experts (MoE) routers** at the projection layer:
- **OneLLM**'s Universal Projection Module is explicitly a MoE of image-projection experts with a per-modality router ([OneLLM CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Han_OneLLM_One_Framework_to_Align_All_Modalities_with_Language_CVPR_2024_paper.pdf)).
- **OmniBind**'s router learns dynamic weights over 14 expert spaces ([OmniBind arXiv](https://arxiv.org/abs/2407.11895)).
- **EBind** ([arXiv 2511.14229](https://arxiv.org/html/2511.14229)) shows one MLP projector per modality is sufficient.

Relation to JLT: **a MoE projection head is a modality-conditioned function f_m: R^d → R^k**, while a JLT is a single fixed function f: R^d → R^k. The two compose: one could have **per-modality nonlinear heads** that each end with a **shared JLT projection into the joint k-space**. This generalizes EBind's MLP+projector setup with a JL guarantee on the final dimension. The JL bound applies cleanly here because the JLT operates linearly on the post-MLP features, preserving inner products.

### B.10 Predictions

Based on the above, concrete predictions for JLT-in-the-loop in the audio+video+text+image regime:

1. **At equal k**, JLT-in-the-loop will **match or slightly beat** standard CLIP-style learned linear projection heads, because the JL property is automatic for random matrices and the encoder learns to use the surviving dimensions optimally (analogous to DirectCLR's ImageNet result). *Confidence: moderate–high*.
2. **As k decreases below ~128 with m ≥ 4**, JLT-in-the-loop will **outperform** non-bottlenecked baselines, because the latter incur catastrophic dimension collapse, while the JLT's full-rank random structure preserves inner products by construction. *Confidence: moderate*.
3. **For storage at fixed accuracy**, JLT-in-the-loop will give **2–4× compression** versus current ImageBind/OmniBind defaults (1024-d) at equal zero-shot accuracy, similar to MRL's 14× claim on ImageNet but more modest because contrastive multimodal training requires more headroom for uniformity. *Confidence: moderate*.
4. **Missing-modality robustness will improve** by 5–10 percentage points relative to vanilla architectures, because the bottleneck forces all modalities through a shared aperture (analogous to SRMM's results). *Confidence: lower*.
5. **The modality gap will shrink** with smaller k (because dimension collapse into disjoint subspaces is geometrically harder), but at extreme k (k < 32) retrieval may degrade because the gap *partially supports* discrimination. *Confidence: moderate*.
6. **A learned JL projection (optimization-based per [Tsikouras et al. NeurIPS 2024](https://openreview.net/pdf?id=w3JCTBRduf))** will likely outperform a fixed random one at k ≤ 64, because data-aware structure helps; at k ≥ 256 the difference will vanish. *Confidence: moderate*.
7. **JLT + MRL composition (untested)**: applying JLT at each MRL nesting level could give the union of both benefits — flexible k with JL-guaranteed inner-product preservation. We are not aware of any paper testing this. *Confidence: low — pure speculation*.

### B.11 Concrete Experimental Protocols to Validate or Refute

A clean experimental program:

**E1 — Scaling-curve comparison**: train ImageBind-style four-modality contrastive models (image + text + audio + video) at k ∈ {32, 64, 128, 256, 512, 1024}, three architectures: (i) baseline learned linear head, (ii) fixed random JLT after encoder, (iii) learned JLT (orthogonal-initialized linear, frozen during training). Evaluate emergent zero-shot on AudioCaps, MSR-VTT, ESC-50, ImageNet, and cross-modal retrieval (audio↔image, audio↔video) on Audioset-VGGSound.

**E2 — Modality-gap measurement**: for each variant in E1, measure the centroid distance between modalities (Liang et al.'s gap metric) and the singular-value spectrum of each modality's embedding (DirectCLR's collapse metric). Predicted: JLT-in-the-loop reduces gap and increases effective rank.

**E3 — Missing-modality stress test**: at inference, randomly drop 1, 2, or 3 modalities; measure accuracy degradation. Compare to MissModal/SRMM.

**E4 — JLT × MRL composition**: train MRL-style nested losses where each level uses a JLT projection rather than truncation. Compare to vanilla MRL on the same benchmarks.

**E5 — Modality-specific intrinsic-dimension matching**: estimate per-modality ID using Levina-Bickel or TwoNN; pick k_m per modality so it just exceeds its ID, then compose into a shared k via an aligning JL projection. Compare to a single forced k.

**E6 — Composition with SigLIP loss**: replace InfoNCE with sigmoid loss in all of the above. The sigmoid loss's per-pair structure may interact differently with bottlenecks.

### B.12 Open Questions

1. **Does the modality gap have a "sweet spot" size that is unattainable without an explicit bottleneck?** Yi et al. 2025 imply yes; no paper has measured this in the m ≥ 4 regime.
2. **What is the right per-modality k vs. shared k trade-off?** Audio ID ≈ 30, image ID ≈ 40, video ID ≈ 200, text ID ≈ 100 — but no paper has estimated joint ID for omnimodal datasets.
3. **Does JLT-in-the-loop affect downstream tasks (zero-shot generation via diffusion decoders, retrieval-augmented LLMs)?** ImageBind, NExT-GPT, and OneLLM all produce 1024-d embeddings consumed by DALLE-2, SD, AudioLDM; would these decoders be robust to k=128 inputs?
4. **Is there a fundamental lower bound on k that depends on m?** The JL bound says k = O(log(mN)/ε²); is this tight for *contrastive* multimodal learning?
5. **How does a JLT bottleneck interact with SigLIP's geometric structure?** SigLIP converges to a CCEM with explicit dimensional structure ([Lee et al. 2024 PMLR](https://proceedings.mlr.press/v238/lee24a/lee24a.pdf)); would JLT be redundant or complementary?

### B.13 Bottom-Line Synthesis

A JLT bottleneck **should scale equally well or slightly better than current omnimodal approaches** as the number of modalities grows, for the following reasons:

- **JL union bound is logarithmic in modality count and item count**, so capacity requirements barely grow with m.
- **Modality gap is caused by per-modality dimension collapse**, which a bottleneck mitigates by forcing all modalities through shared coordinates.
- **DirectCLR has empirically demonstrated** that a low-rank/diagonal projection in the contrastive loss *outperforms* a learned full-rank linear projector on ImageNet — this is the unimodal evidence that JLT-in-the-loop is competitive.
- **MRL has empirically demonstrated** that nested low-d losses preserve accuracy across dimensions in CLIP, image classification, and (recently) multimodal recommendation — this is the closest published precedent to JLT-in-the-loop in multimodal training.
- **Storage and inference cost scale linearly in m·k**, so any reduction in k from 1024 to 128 yields an 8× compounding savings as m grows.
- **EBind's 2025 finding** that simple per-modality MLP projectors recover most of OmniBind-30B's accuracy implies the projection layer is doing the bulk of "binding" work; a JLT is a structurally constrained projection that should retain most of this benefit.

The main risks are (a) extreme compression (k < 32) collapsing the modality gap in a way that hurts retrieval, (b) cross-modal alignment failure when the bottleneck is too tight relative to the joint intrinsic dimension, and (c) interaction effects with downstream decoders that expect 1024-d input.

The strongest version of the proposal is therefore a **per-modality nonlinear MLP head + a shared learned-orthogonal JL projection into k=128–256-d, trained end-to-end with an InfoNCE or sigmoid loss, possibly with MRL-style nested objectives at multiple k values for flexible deployment**. To our knowledge, this exact configuration has not been published, and would constitute a novel and likely impactful contribution to the omnimodal representation-learning literature — particularly on the efficient-storage and missing-modality-robustness Pareto frontiers, which existing 1024-d ImageBind/LanguageBind/OmniBind systems do not address systematically.

---

## Sources Cited (selected, in addition to inline citations)

- Girdhar et al., "ImageBind: One Embedding Space To Bind Them All", CVPR 2023 — [PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Girdhar_ImageBind_One_Embedding_Space_To_Bind_Them_All_CVPR_2023_paper.pdf)
- Zhu et al., "LanguageBind: Extending Video-Language Pretraining to N-modality", ICLR 2024 — [arXiv 2310.01852](https://arxiv.org/abs/2310.01852)
- Guzhov et al., "AudioCLIP", ICASSP 2022 — [DFKI listing](https://www.dfki.de/en/web/research/projects-and-publications/publication/12157)
- Wu et al., "Large-scale Contrastive Language-Audio Pretraining" (LAION-CLAP), ICASSP 2023 — [HAL PDF](https://hal.science/hal-04766539v1/file/clap.pdf)
- Wang et al., "OmniBind: Large-scale Omni Multimodal Representation via Binding Spaces", 2024 — [arXiv 2407.11895](https://arxiv.org/abs/2407.11895)
- Liang et al., "Mind the Gap", NeurIPS 2022 — [arXiv 2203.02053](https://arxiv.org/pdf/2203.02053)
- Yi et al., "Decipher the Modality Gap", 2025 — [arXiv 2510.03268](https://arxiv.org/pdf/2510.03268)
- Wang et al., "C-MCR", NeurIPS 2023 — [arXiv 2305.14381](https://arxiv.org/pdf/2305.14381)
- Wang et al., "Ex-MCR", NeurIPS 2024 — [arXiv 2310.08884](https://arxiv.org/pdf/2310.08884)
- Wang et al., "FreeBind", ICML 2024 — [arXiv 2405.04883](https://arxiv.org/pdf/2405.04883)
- Han et al., "OneLLM", CVPR 2024 — [arXiv 2312.03700](https://arxiv.org/abs/2312.03700)
- Wu et al., "NExT-GPT", ICML 2024 — [arXiv 2309.05519](https://arxiv.org/abs/2309.05519)
- Lyu et al., "UniBind", CVPR 2024 — [arXiv 2403.12532](https://arxiv.org/abs/2403.12532)
- Wang et al., "InternVideo2", ECCV 2024 — [arXiv 2403.15377](https://arxiv.org/abs/2403.15377)
- Chen et al., "VAST", NeurIPS 2023 — [arXiv 2305.18500](https://arxiv.org/abs/2305.18500)
- Zhang et al., "Meta-Transformer", 2023 — [arXiv 2307.10802](https://arxiv.org/abs/2307.10802)
- Xue et al., "ULIP", CVPR 2023 — [arXiv 2212.05171](https://arxiv.org/abs/2212.05171)
- Guo et al., "Point-Bind & Point-LLM", 2023 — [arXiv 2309.00615](https://arxiv.org/abs/2309.00615)
- Cicchetti et al., "GRAM", ICLR 2025 — [arXiv 2412.11959](https://arxiv.org/abs/2412.11959)
- Saporta et al., "Symile", NeurIPS 2024 — [arXiv 2411.01053](https://arxiv.org/abs/2411.01053)
- Zhai et al., "SigLIP", ICCV 2023 — [arXiv 2303.15343](https://arxiv.org/abs/2303.15343)
- Tschannen et al., "SigLIP 2", 2025 — [arXiv 2502.14786](https://arxiv.org/abs/2502.14786)
- Kusupati et al., "Matryoshka Representation Learning", NeurIPS 2022 — [arXiv 2205.13147](https://arxiv.org/abs/2205.13147)
- Jing et al., "Understanding Dimensional Collapse / DirectCLR", ICLR 2022 — [arXiv 2110.09348](https://arxiv.org/pdf/2110.09348)
- Pope et al., "Intrinsic Dimension of Images", ICLR 2021 — [OpenReview](https://openreview.net/pdf?id=XJk19XzGq2J)
- Ansuini et al., "Intrinsic Dimension of Data Representations", NeurIPS 2019 — [arXiv 1905.12784](https://arxiv.org/abs/1905.12784)
- Tsikouras et al., "Optimization Can Learn Johnson-Lindenstrauss Embeddings", NeurIPS 2024 — [OpenReview](https://openreview.net/pdf?id=w3JCTBRduf)
- "EBind: a practical approach to space binding", Nov 2025 — [arXiv 2511.14229](https://arxiv.org/html/2511.14229)
- Wang et al., "fMRLRec", EMNLP 2024 Findings — [ACL](https://aclanthology.org/2024.findings-emnlp.786.pdf)