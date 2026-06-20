# GDN-Style Prediction-Branch Anomaly Scoring Protocol

## Purpose

This document defines the current prediction-branch post-processing protocol used in SADP. The protocol is designed for evaluating the anomaly detection capability of the MTGNN forecasting branch under a literature-grounded scoring rule.

This protocol is not the final joint SADP anomaly score. The full SADP model is still trained with the joint objective:

```text
L_total = L_pred + beta * (L_rec_real + lambda * L_rec_pred)
```

The scoring protocol in this document only uses the forecasting residual produced by the MTGNN branch. AE reconstruction residuals are not used in this specific diagnostic score.

## Literature Basis

The scoring rule follows the graph deviation scoring idea in:

- Deng, A. and Hooi, B. Graph Neural Network-Based Anomaly Detection in Multivariate Time Series. AAAI, 2021.
- Official implementation: https://github.com/d-ailin/GDN

The MTGNN backbone is based on:

- Wu, Z., Pan, S., Long, G., Jiang, J., Chang, X., and Zhang, C. Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks. KDD, 2020.

MTGNN itself is a forecasting model and does not define an anomaly score. Therefore, SADP uses MTGNN as the forecasting branch and borrows the GDN-style residual scoring rule for prediction-branch anomaly detection.

## Mathematical Definition

Let \(y_{t,i}\) be the observed value of sensor \(i\) at time \(t\), and let \(\hat{y}_{t,i}\) be the MTGNN prediction.

### 1. Forecasting Residual

```text
e_{t,i} = |y_{t,i} - y_hat_{t,i}|
```

This residual is the local point-wise source of the prediction loss \(L_pred\). During training, \(L_pred\) aggregates these errors over time, sensors, horizons, and batches. During post-processing, the residual matrix is retained for anomaly scoring.

### 2. Robust Residual Normalization

For each sensor \(i\), compute the median and inter-quartile range from training residuals only:

```text
median_i = median_t(e_{t,i}^{train})
IQR_i = Q75_t(e_{t,i}^{train}) - Q25_t(e_{t,i}^{train})
```

Then normalize validation and test residuals:

```text
z_{t,i} = (e_{t,i} - median_i) / (|IQR_i| + eps)
```

The GDN official implementation uses `eps = 1e-2`. The SADP `--score-preset gdn` uses the same value through `--residual-scale-eps 1e-2`.

SADP uses training residual statistics for validation and testing to maintain strict test-set isolation. This is a deliberate industrial anomaly detection protocol adaptation.

### 3. Sensor-Wise Aggregation

The normalized sensor residuals are aggregated by maximum:

```text
S_t = max_i z_{t,i}
```

This follows the GDN rationale that an industrial attack may affect only a small subset of sensors or even one sensor.

### 4. Temporal Smoothing

A causal simple moving average is applied:

```text
S_bar_t = mean(S_{t-3}, S_{t-2}, S_{t-1}, S_t)
```

This corresponds to `before_num = 3` in the GDN official implementation, i.e. current point plus the previous three points.

### 5. Thresholding

The threshold is selected from the validation scores:

```text
tau = max_t(S_bar_t^val)
```

The test decision is:

```text
label_t = 1 if S_bar_t^test >= tau else 0
```

Test labels are never used for normalization or threshold selection. They are only used for final metric calculation.

## Implementation Mapping

The protocol is implemented in:

```text
Test/anomaly_scoring_threshold_pa.py
```

Recommended command:

```bash
python Test/anomaly_scoring_threshold_pa.py \
  --run-dir "$RUN_DIR" \
  --data-path "$DATA_PATH" \
  --score-preset gdn \
  --eval-granularity point \
  --time-aggregate mean \
  --save-subdir postprocess_gdn_robust_iqr_max_valmax_smooth4
```

The `--score-preset gdn` expands to:

```text
score_source = mtgnn
residual_norm_method = robust_iqr
residual_scale_eps = 1e-2
var_reduce = max
score_smooth_window = 4
score_smooth_method = mean
score_smooth_direction = causal
threshold_method = val_max
```

## Scope and Limitation

This protocol evaluates the MTGNN prediction branch only. It should be used to answer:

```text
Can the forecasting branch produce anomaly-separable residuals?
```

It should not be described as the final SADP joint anomaly score because it does not use:

```text
L_rec_real
L_rec_pred
AE reconstruction residuals
joint residuals
```

The final SADP scoring rule should be designed and justified separately if AE reconstruction residuals are included in anomaly decision making.
