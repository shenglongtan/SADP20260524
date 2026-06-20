# SADP Performance Root-Cause Audit

Date: 2026-06-20

## Scope

This audit focuses on why the current SADP implementation can train and infer successfully, but obtains weak anomaly detection metrics on SWaT/WADI-style industrial multivariate time-series data.

The objective is not to rewrite the model, but to identify high-probability causes of low Precision, Recall, F1-score, and AUC-PR.

## Executive Findings

### P0. Curriculum learning does not reach the full prediction horizon

Current training uses curriculum learning:

`pred_loss = loss(pred_y[..., :pred_step], true_y[..., :pred_step])`

with default:

- `horizon_size = 12`
- `pred_step = 1` at initialization
- `cl_update_num = 2500`

In the formal Kaggle run, each epoch has only about 46 batches and early stopping occurs at epoch 27. Therefore, the total number of optimizer steps is about 1242, which is below `cl_update_num=2500`.

This means the model is very likely trained only on the first prediction step, while final residual export and anomaly scoring use all 12 prediction steps.

Impact:

- Later horizons are weakly supervised or effectively unsupervised.
- Multi-step residuals become inflated during validation/test.
- Point-level aggregation over horizons can turn horizon error into many false positives.
- This directly depresses Precision, F1-score, and AUC-PR.

Recommended fix:

- For the current experiment, disable curriculum learning with `--cl False`.
- Or reduce `cl_update_num` so that `pred_step` reaches `horizon_size` well before early stopping.
- Log `pred_step` at each epoch and verify that the best checkpoint is trained on all horizons used for scoring.

### P0. Normal validation distribution is far from the training distribution

Evidence from the Kaggle training log:

- TrainPred decreases to about `0.025`.
- ValPred remains around `0.305`.
- The best checkpoint appears at epoch 17, then validation loss deteriorates.

This gap is too large for a healthy normal-only train/validation split. It means the model learns the first 70% of the `Train==1` normal segment but does not generalize to the last 30% normal segment.

Current code behavior:

- `Train==1` rows are split chronologically into 70% train and 30% validation.
- `Train==0` rows are used as test.
- The scaler is fit only on the train split.

This is scientifically leakage-safe, but may create a strong normal-operation distribution shift. In SWaT, plant dynamics can change substantially across the normal period, especially after startup and regime transitions.

Impact:

- Validation residual distribution is high even though validation labels are normal.
- The model-selected best epoch is dominated by this distribution mismatch.
- Thresholds calibrated on validation scores may not represent stable normal operation.
- Test scores become inflated, causing high false positives.

Recommended next checks:

- Compare train/val/test feature means and standard deviations by sensor.
- Plot train/val/test raw sensor trajectories for the top residual contributors.
- Try a normal-only validation split closer to the test boundary only after ensuring it is truly normal, or use multiple normal validation blocks.
- Consider training on the full confirmed normal segment and using a small normal calibration subset for thresholding.

### P1. Postprocess residual z-score can amplify near-zero residual variance

Current scoring computes:

`norm_error = (error_matrix - train_mu) / (train_sigma + 1e-8)`

where `train_sigma` is computed from training residuals per node.

This is mathematically valid, but risky when the model fits some sensors almost perfectly on train. If a node has very small training residual variance, even modest normal validation/test residuals become huge z-scores.

Impact:

- Excessive false positives.
- Test anomaly ratio becomes unrealistically high.
- Mean pooling may still be dominated by multiple inflated channels.

Evidence from result:

- `test_anomaly_ratio_raw = 0.3707`
- `test_anomaly_ratio_pa = 0.6719`

These ratios are far above expected attack prevalence and indicate an over-sensitive scoring pipeline.

Recommended next checks:

- Inspect min/percentiles of `residual_sigma.npy`.
- Rank sensors by average normalized test residual.
- Compare `joint_error` vs `mtgnn_pred_error` scoring.
- Add a robust lower bound for residual sigma, such as percentile floor or MAD/IQR-based scaling.

### P1. Default threshold percentile is too permissive for this score distribution

Current route A uses:

`threshold = percentile(score_val, 99.0)`

The reported threshold is only:

`0.6080`

With this threshold, the model flags 37% of test points as anomalous. This is not a reasonable operating point for industrial anomaly detection unless the test interval is mostly under attack, which SWaT is not.

Impact:

- Precision collapses.
- Recall may look acceptable only because the detector fires too frequently.

Recommended next checks:

- Sweep percentiles: 99.0, 99.3, 99.5, 99.7, 99.8, 99.9, 99.95.
- Report Precision-Recall curve and maximum test F1 only as diagnostic, not as deployment threshold selection.
- Consider mean+std or robust quantile thresholding on normal calibration scores.

### P1. Point-adjustment is currently unsuitable as the main metric

Current PA expands each positive prediction by `horizon_size - 1`.

With frequent raw positives, PA makes the prediction almost continuously anomalous:

- raw anomaly ratio: `0.3707`
- PA anomaly ratio: `0.6719`

Impact:

- PA recall increases, but precision and F1 degrade.
- PA masks whether the raw detector is well localized.

Recommendation:

- Use raw point-level metrics as the primary debugging metric.
- Keep PA only as a secondary paper-comparison metric after raw false positives are controlled.

### P2. Joint residual mixes L1 prediction error with squared AE reconstruction error

Current residual:

`joint_error = abs(y_true - y_pred) + beta * (sq_rec_real + lambda * sq_rec_pred)`

Training objective:

`L_total = L_pred + beta * (L_rec_real + lambda * L_rec_pred)`

This is internally consistent in structure, but the scales differ:

- MTGNN residual is absolute error.
- AE residuals are squared errors.

Impact:

- Depending on score distribution, AE terms may either vanish or dominate.
- A single fixed `beta=0.2`, `lambda=0.5` may not be optimal for anomaly scoring even if it is acceptable for training.

Recommendation:

- Evaluate `--score-source mtgnn` and `--score-source joint` side by side.
- Export separate MTGNN, AE-real, AE-pred score curves.
- Consider score-level normalization before fusion.

### P2. Dynamic graph is dense and batch-averaged

The current MTGNN uses `GetAttMatrix` to infer an adjacency matrix from input sensor windows. This satisfies the algorithm requirement that the graph is generated from raw sensor data, but the current V1 graph has these risks:

- No explicit top-k sparsification in the active model path.
- The returned adjacency is batch-level/global for the batch, not point-specific.
- Dense softmax attention can smooth unrelated sensors together.

Impact:

- Graph may not reflect sparse industrial process topology.
- Dense message passing can dilute local fault signatures and create correlated false positives.

Recommendation:

- Save and inspect learned adjacency matrices.
- Compare with `top_k` sparse graph variants or process-informed constraints.
- Treat graph quality as a separate ablation axis.

### P3. Sliding step reduces training sample diversity

Current 10s data uses:

- `window_size=36`
- `horizon_size=12`
- `sliding_step=12`

This means training samples advance every 120 seconds. It improves speed but reduces training windows from the normal segment to only 2895 samples.

Impact:

- The model sees fewer normal temporal contexts.
- The detector may be brittle under normal regime changes.

Recommendation:

- Try `sliding_step=6` first.
- If memory allows, try `sliding_step=3`.
- Keep `sliding_step=12` as a fast-debug setting, not necessarily final paper setting.

## Priority Action Plan

1. Re-train once with curriculum learning disabled, or ensure `pred_step` reaches all 12 horizons before early stopping.
2. Diagnose residual distributions before changing model layers.
3. Run postprocess ablations:
   - `score-source=mtgnn`
   - `score-source=joint`
   - `time-aggregate=mean`
   - `var-reduce=mean/max/topk_mean/p95`
   - threshold percentiles from 99.0 to 99.95
4. Inspect train/val/test normal distribution shift by sensor.
5. Try robust residual standardization.
6. Re-train with `sliding_step=6`.
7. Only then revise graph sparsity or model architecture.

## Current Judgment

The most likely root cause is not a single runtime bug. It is a combination of:

1. curriculum learning not reaching the full prediction horizon,
2. train/validation normal distribution shift,
3. over-sensitive residual z-score standardization,
4. low percentile threshold,
5. PA amplification,
6. possible dense-graph smoothing.

The first next step should be diagnostic experiments on saved residuals, not immediate model rewriting.
