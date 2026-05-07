# MultiModal NeurIPS-Gap Closure Suite

This folder contains an isolated, end-to-end implementation of the full experimental extension plan:

1. Karpathy-standard data protocol and caches.
2. E7 Karpathy retrain with 5 seeds, CI/significance reporting, and expanded baselines.
3. Third-modality (AudioCaps + CLAP audio) tri-modal experiments.
4. Queue orchestration for GPUs 2 and 3 with tmux, logging, checkpoint resume, and aggregation.

## Folder Layout

- `configs/`: experiment configuration files for stages 1-4 and smoke tests.
- `multimodal/`: implementation package.
- `scripts/`: environment bootstrap, queue runners, and tmux launcher.
- `results/full_suite/`: outputs, markers, and provenance snapshots.
- `logs/`: stage logs.

## Run

```bash
cd "/jumbo/lisp/f004ndc/CLIP JST"
bash MultiModal/scripts/launch_tmux_gpu23.sh
```

## Monitor

```bash
tmux ls
tmux attach -t mm_gpu2
tmux attach -t mm_gpu3
```

## Stage Markers

- `MultiModal/results/full_suite/markers/stage1_prepare.done.json`
- `MultiModal/results/full_suite/stage2_e7/markers/stage2_e7_karpathy.done.json`
- `MultiModal/results/full_suite/stage3_trimodal/markers/stage3_trimodal.done.json`
- `MultiModal/results/full_suite/markers/stage4_aggregate.done.json`

Each stage writes JSON provenance with config/environment/git hash and elapsed wall-time.
