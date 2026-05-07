# NeurIPS-Gap Closure Suite (Implemented)

This directory implements the full requested track:

- Karpathy-standard split builder and validation.
- COCO val2017 availability check/download.
- Separate Karpathy-compatible cache artifacts.
- E7 Karpathy retrain suite with seeds 0-4 and m in {64,128,256,512}.
- Baselines: CLIP head, random JL + Mahalanobis, orthogonal trainable JL-style head,
  learned sparse-JL head, MRL nested head, DirectCLR proxy, random JL only control.
- Statistical reports: mean/std/95% CI, paired deltas vs CLIP head, paired t-tests,
  Holm-Bonferroni correction.
- AudioCaps+CLAP third-modality cache and tri-modal mixed-pair training.
- Tri-modal metrics: image-text and audio-text retrieval plus modality/rank diagnostics.
- GPU2/GPU3 queue orchestration via tmux with resumable stage markers and aggregate summary.
