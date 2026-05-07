# PREREG_STAGE69

## Locked Design
- Triple: Image–Speech–Text via `mteb/SpeechCoco`
- Methods: ['modular_shared_jl', 'modular_separate_jl', 'audio_linear_probe']
- Dims: [64, 128, 256, 512]
- Seeds: [0, 1, 2, 3, 4]
- Grid size: 60

## Locked Prediction Rule
- alpha_locked: 0.282002205907 (from Stage43 fit_train.alpha)
- Rule: `R_ia_pred = alpha_locked * sqrt(R_it_obs * R_at_obs)`

## Success Criteria
- cell_mean_r >= 0.85
- cell_mean_MAE <= 0.01
- geometric mean ranks top-2 by CV-R2 in Stage72

## Failure Policy
- max retries per failed unit: 2
- retries must use identical configs
- failed units remain explicit; no silent replacement
