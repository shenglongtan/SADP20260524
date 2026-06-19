# 后处理严格协议审查与修订说明

日期：2026-06-19

本文档记录 `Test/anomaly_scoring_threshold_pa.py` 按照新的后处理规范完成的审查与修订结果。R011 曾按用户给定的 `F1_val` 监督阈值公式实现默认逻辑；随后根据 SWaT 数据事实确认主实验应采用路线 A，即正常验证集无监督阈值。因此当前默认阈值为 `percentile`，`f1_val` 保留为可选校准模式。

## 1. 用户给定的目标规范

用户最初给定的监督阈值数学流程为：

```text
e_{t,i} = |y_{t,i} - yhat_{t,i}|
e_norm_{t,i} = (e_{t,i} - mu_i) / (sigma_i + 1e-8)
S_t = mean_i(e_norm_{t,i})
tau_best = argmax_tau F1_val(tau)
Label_test(t) = 1 if S_test(t) >= tau_best else 0
```

该监督阈值版本的关键约束：

- `mu_i/sigma_i` 必须来自历史训练集残差，不能从验证集或测试集实时估计。
- 阈值搜索只能使用验证集分数和验证集标签。
- 测试集标签只能在最终指标计算阶段使用。
- AUC-PR 必须使用测试集连续分数 `S_test`，不能使用二元预测标签。

## 2. 原脚本与规范的不一致

修订前主要不一致包括：

1. 标准化统计量来自验证集残差：

```python
center, scale = compute_norm_stats(val_err_var, args.norm_method)
```

这与“历史训练集常量”要求不一致。

2. 默认阈值策略为无监督分位数：

```text
threshold-method = percentile
```

这与当时给定的 `argmax F1_val` 要求不一致。但在路线 A 中，验证集是正常段，因此无监督分位数阈值反而是主实验应采用的协议。

3. 默认空间聚合为 `max`：

```text
var-reduce = max
```

这与 Mean-pooling 规范不一致。

4. 原脚本执行窗口级评价，尚未实现滑动窗口到原始时间点的点级反投影。

5. 旧的扫描逻辑可能用测试集指标挑选最佳扫描结果，不适合作为严格实验协议。

## 3. 当前修订后的后处理协议

当前脚本默认配置为：

```text
--eval-granularity point
--score-source auto
--stats-source train
--time-aggregate max
--var-reduce mean
--threshold-method percentile
--threshold-percentile 99.0
```

含义：

- 默认执行点级评价。
- 默认优先使用 `joint_error`，如果缺失则回退到 MTGNN 残差。
- 标准化统计量从训练残差计算。
- 同一原始时间点被多个窗口覆盖时，默认用 `max` 反投影聚合。
- 传感器维度使用 `mean` 聚合。
- 阈值通过正常验证集分数分布的分位数确定，不使用验证集异常标签。
- `f1_val` 仍可通过命令行显式启用，但只用于有监督阈值校准实验。

## 4. 点级反投影

当前点级评价使用：

```text
error[sample, node, horizon]
y_time[sample, horizon]
y_attack_point[sample, horizon]
```

反投影为：

```text
point_error[time, node]
point_label[time]
```

如果多个窗口覆盖同一个 `time`，默认聚合：

```text
point_error[time, node] = max(errors covering the same time and node)
point_label[time] = max(labels covering the same time)
```

## 5. 训练脚本配套修订

为了让后处理严格使用训练残差统计量，`train_attention.py` 已同步导出：

```text
train_joint_error.npy
train_mtgnn_pred_error.npy
train_ae_rec_real_error.npy
train_ae_rec_pred_error.npy
y_train_true.npy
y_train_pred.npy
train_y_time.npy
train_y_attack_window.npy
train_y_attack_point.npy
```

同时继续导出验证集和测试集对应文件。

训练集残差导出前会将 `train_loader.indices` 恢复为原始顺序，避免训练过程 shuffle 后导致：

```text
train_joint_error 与 train_y_time 错位
```

## 6. 指标计算

当前测试集指标包括：

- Precision
- Recall
- F1-score
- ROC-AUC
- AUC-PR / Average Precision

其中 AUC-PR 调用：

```python
average_precision_score(y_true, scores)
```

这里的 `scores` 是连续异常分数，不是二元标签。

## 7. 数据隔离保证

当前 `summary.json` 中记录：

```json
"test_labels_used_for_threshold": false
```

测试集标签只在以下阶段使用：

```text
pred_test_raw 和 pred_test_pa 已经生成之后
```

因此测试集标签不参与：

- 训练残差统计量计算
- 验证集阈值搜索
- 测试集异常分数生成

## 8. 验证情况

已完成语法检查：

```powershell
python -m py_compile train_attention.py Test\anomaly_scoring_threshold_pa.py
```

结果：

```text
通过，无语法错误。
```

已完成合成点级后处理测试：

```text
selected_residual_source = joint_error
eval_granularity = point
threshold_method = f1_val
test_labels_used_for_threshold = False
raw_f1 = 1.0
pr_auc = 1.0
```

该测试验证了：

- 训练残差统计量可正常广播到验证/测试残差。
- 点级反投影可运行。
- 验证集 F1 阈值搜索可运行。
- AUC-PR 使用连续分数。

## 9. 后续注意

如果显式启用 `threshold-method=f1_val`，但验证集标签只有单一类别，脚本会主动报错。原因是 F1 阈值搜索要求验证集同时包含正常点和异常点。路线 A 主实验默认使用 `threshold-method=percentile`，不会触发该问题。

若未来需要使用 `f1_val`，必须准备含标签异常的验证/校准集，并在论文中说明这是 supervised threshold calibration。
