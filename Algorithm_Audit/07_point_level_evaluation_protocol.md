# 滑动窗口结果回到原始点级评价的协议说明

日期：2026-06-19

本文档说明在当前滑动窗口多步预测框架下，如何从窗口级预测结果恢复到原始时间序列点级异常评价。

## 1. 当前窗口样本的时间含义

在 `data_maker.py::create_input_data()` 中，每个窗口样本由锚点 `t` 生成：

```text
X_i = x[t-L+1 : t]
Y_i = x[t+1 : t+H]
```

其中：

- `L = window_size`
- `H = horizon_size`
- `t` 每次按 `sliding_step` 前进

当前程序同时保存：

```text
y_attack_window[i] = max(label[t+1 : t+H])
y_attack_point[i, h] = label[t+h+1]
y_time[i, h] = timestamp[t+h+1]
```

因此，虽然模型训练和后处理首先以“窗口样本”为单位工作，但每个预测步 `h` 都能通过 `y_time[i,h]` 对应回原始时间序列中的一个真实时间点。

## 2. 为什么可以回到点级

模型输出和残差的基本形状为：

```text
error[i, n, h]
```

含义是：

- 第 `i` 个滑动窗口样本
- 第 `n` 个传感器节点
- 第 `h` 个未来预测步

而 `y_time[i,h]` 记录了这个误差对应的原始时间点。因此可以建立映射：

```text
(i, h) -> original_time = y_time[i, h]
```

再把同一个原始时间点上来自不同窗口的多个分数进行聚合，就能得到点级异常分数：

```text
point_score[time] = Aggregate({ window_score[i,h] | y_time[i,h] = time })
```

## 3. 多个窗口覆盖同一点时如何处理

如果 `sliding_step < horizon_size`，同一个原始时间点会被多个预测窗口覆盖。例如 `H=12, sliding_step=1` 时，一个时间点最多可能获得 12 个预测残差。

常见聚合策略：

```text
max   : 对任一窗口高度异常都敏感，适合攻击检测召回优先。
mean  : 更稳定，但可能稀释短暂强异常。
last  : 只使用最近预测起点给出的分数，强调最短预测距离。
first : 使用最早能预测到该点的窗口分数，强调更长提前量。
```

对工业异常检测，建议默认使用：

```text
point_score[time] = max(scores covering the same time)
```

原因是攻击往往表现为局部强偏离，`max` 更符合“任一预测视角发现异常即可告警”的检测逻辑。为了论文严谨性，可以在附录或消融中比较 `max/mean/last`。

## 4. 节点和 horizon 的推荐顺序

当前窗口级后处理是：

```text
[Samples, Nodes, Horizon]
  -> horizon_reduce
  -> node normalization
  -> var_reduce
  -> [Samples]
```

若要做点级评价，更推荐先保留 horizon 维并反投影：

```text
[Samples, Nodes, Horizon]
  -> 对每个 horizon 点映射到原始 time
  -> 在同一 time 上聚合同节点残差
  -> 得到 [Time, Nodes]
  -> 用验证集点级残差统计量标准化每个节点
  -> 节点维聚合为 point_score[Time]
  -> 阈值判定
  -> 与原始 point label 比较
```

这样做的优点是：评价单位真正回到原始时间点，而不是把一个预测窗口压缩成一个样本标签。

## 5. 标签选择

窗口级标签：

```text
y_attack_window[i] = max(y_attack_point[i, :])
```

点级标签：

```text
point_label[time] = ground truth attack label at that original time
```

当前 `y_attack_point[i,h]` 已经包含点级标签，配合 `y_time[i,h]` 即可反投影得到 `point_label[time]`。

当多个窗口覆盖同一个时间点时，`y_attack_point` 对同一 `time` 的标签理论上应完全一致；实际程序可用 `max` 聚合，作为安全处理：

```text
point_label[time] = max(labels covering the same time)
```

## 6. 与当前代码的关系

当前 `Test/anomaly_scoring_threshold_pa.py` 主评估仍使用：

```text
y_attack_window
```

这属于窗口级评价。

要实现最终点级评价，需要在后处理脚本中增加一个点级分支：

```text
--eval-granularity window | point
```

推荐默认目标：

```text
--eval-granularity point
```

保留窗口级评价作为对照或调试输出。

## 7. 当前算法口径

根据用户确认，当前算法口径固定为：

- AE 训练重构完整窗口，异常评分只取未来预测段，保持现状。
- 动态图由原始传感器数据生成，当前实现使用第 0 个核心传感器数值通道，不使用 `Attack/Train` 元数据。
- AE 只承担节点独立的时域重构。
- 最终评价目标应回到原始时间序列点级。
- 不设计 AE-only 评分模式，因为 MTGNN 是基础分支；消融只需要 MTGNN-only 与 MTGNN+AE joint 等。

## 8. 后续建议修订

建议后续对 `Test/anomaly_scoring_threshold_pa.py` 做一次独立修订：

1. 读取 `val_y_time/test_y_time` 与 `val_y_attack_point/test_y_attack_point`。
2. 对 `val_joint_error/test_joint_error` 保留 horizon 维进行反投影。
3. 生成 `point_score_val.npy`、`point_score_test.npy`。
4. 生成 `point_label_test.npy`。
5. 输出点级 `raw` 与 `PA` 指标。
6. 同时保留现有窗口级指标，便于论文补充对照。

