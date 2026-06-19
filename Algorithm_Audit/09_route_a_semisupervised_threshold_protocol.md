# 路线 A：半监督异常检测数据划分与无监督阈值协议

日期：2026-06-19

本文档记录当前 SADP 主实验推荐采用的路线 A 协议。

## 1. 数据事实

对 `Data/SWat/swat_data_10s.pkl` 的检查结果为：

```text
Train=1:
2015-12-22 16:00:00 到 2015-12-28 09:59:50
49680 行
Attack=0，完全正常

Train=0:
2015-12-28 10:00:00 到 2016-01-02 14:59:50
44993 行
包含攻击，Attack=1 共 5496 行

第一个攻击采样点:
2015-12-28 10:29:10
```

因此 `Train` 列不是模型特征，而是 SWaT 正常训练段/攻击测试段的实验协议标记。

## 2. 算法性质

当前 SADP 是半监督工业时序异常检测算法：

```text
训练阶段：只使用正常传感器数据，不使用 Attack 标签。
检测阶段：通过预测残差 + AE 重构残差度量偏离。
评价阶段：Attack 标签只用于最终指标计算。
```

异常标签不进入：

- 模型输入
- MTGNN loss
- AE loss
- 训练残差标准化统计量
- 无监督阈值选择

## 3. 当前推荐划分

保留 `data_maker.py` 当前 Train 列协议：

```text
Train==1 正常段
  -> 前 70% 作为 train
  -> 后 30% 作为 val

Train==0 含攻击段
  -> 全部作为 test
```

该划分得到：

```text
train: 2015-12-22 16:00:00 到 2015-12-26 16:35:50，正常
val:   2015-12-26 16:36:00 到 2015-12-28 09:59:50，正常
test:  2015-12-28 10:00:00 到 2016-01-02 14:59:50，含攻击
```

## 4. 后处理阈值

由于验证集来自 `Train==1` 正常段，验证集标签只有正常类。因此主实验不能使用：

```text
tau_best = argmax_tau F1_val(tau)
```

主实验默认使用无监督阈值：

```text
tau = percentile(S_val, 99)
```

其中 `S_val` 是正常验证集异常分数。

当前 `Test/anomaly_scoring_threshold_pa.py` 默认配置改为：

```text
--eval-granularity point
--score-source auto
--stats-source train
--time-aggregate max
--var-reduce mean
--threshold-method percentile
--threshold-percentile 99.0
```

## 5. F1 阈值的定位

`--threshold-method f1_val` 保留为可选模式，但它不属于路线 A 主协议。

只有当专门构造含异常标签的 validation/calibration 集时，才能使用：

```text
--threshold-method f1_val
```

论文中应将其称为：

```text
supervised threshold calibration
```

而不是半监督主实验默认设置。

## 6. 当前程序影响

本次修订只改变后处理默认阈值策略：

- `percentile` 成为默认阈值方法。
- `f1_val` 保留，但需要验证集同时包含正常和异常。
- `summary.json` 新增：

```json
"route": "A_semisupervised_unsupervised_threshold",
"val_labels_used_for_threshold": false,
"test_labels_used_for_threshold": false
```

测试集标签仍只用于最终指标计算。

