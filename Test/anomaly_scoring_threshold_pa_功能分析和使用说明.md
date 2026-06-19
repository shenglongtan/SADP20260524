# anomaly_scoring_threshold_pa.py 功能分析与使用说明

## 1. 文件概述

`anomaly_scoring_threshold_pa.py` 是一个轻量级的异常检测后处理流水线。与 `advanced_anomaly_detection.py` 相比，它更专注于核心功能，提供了：

- **异常分值计算**：支持多种时间与变量维度的降维策略
- **阈值自适应选择**：无监督的百分位与均值±k·std 策略
- **点调整（Point Adjustment, PA）**：针对时序异常传播延迟的工业级容错
- **丰富的 CLI 参数接口**：无需修改代码即可在终端动态指定超参数
- **自动化参数扫描**：网格扫参寻找最优阈值与延迟组合
- **安全的数据隔离**：使用验证集统计量防止测试集数据泄漏

该脚本优先考虑工程实用性与参数灵活性，适合在模型评估与工业部署中快速迭代调参。

---

## 2. 主要功能与模块说明

### 模块一：CLI 命令行接口（`parse_args()`）

提供以下参数控制：

| 参数                     | 类型  | 默认值              | 说明                                             |
| ------------------------ | ----- | ------------------- | ------------------------------------------------ |
| `--run-dir`              | str   | 必需                | 包含 y_val_true.npy、y_val_pred.npy 等的输出目录 |
| `--data-path`            | str   | None                | 数据集根目录，包含 val.npz/test.npz（可选）      |
| `--norm-method`          | str   | zscore              | 特征归一化方法：zscore 或 robust（四分位距）     |
| `--horizon-reduce`       | str   | max                 | 时间步降维：max（最大）或 mean（均值）           |
| `--var-reduce`           | str   | max                 | 传感器降维：max/mean/topk_mean/p95               |
| `--var-topk`             | int   | 10                  | topk_mean 时的 K 值                              |
| `--threshold-method`     | str   | percentile          | 阈值选择方法：percentile 或 mean_std             |
| `--threshold-percentile` | float | 99.0                | 百分位数阈值的百分比（0-100）                    |
| `--threshold-k`          | float | 3.0                 | mean_std 方法中的 k 值                           |
| `--scan-percentiles`     | str   | ""                  | 扫参列表（逗号分隔），如 '98,99,99.5,99.9'       |
| `--scan-pa-delays`       | str   | ""                  | 点调整延迟扫参列表，如 '3,5,8,11'                |
| `--pa-delay`             | int   | None                | 固定点调整延迟；缺省为 horizon_size-1            |
| `--save-subdir`          | str   | anomaly_postprocess | 输出子目录名                                     |

### 模块二：异常分值计算（`compute_scores()`）

三阶段管道：

1. **时间步降维**（Horizon Reduction）
   - 输入：[N, V, H] 的绝对残差矩阵
   - max：提取每个变量在预测窗口内最大的偏离
   - mean：计算平均偏离程度
   - 输出：[N, V]

2. **空间变量归一化**（Normalization）
   - 使用验证集统计量计算中心（center）与尺度（scale）
   - 应用归一化公式：`norm_err = (err_var - center) / (scale + eps)`
   - 防止量纲差异导致某些传感器主导结果

3. **变量维度降维**（Variable Reduction）
   - max：取最坏传感器（适合安全关键场景）
   - mean：全局均值（平衡视图）
   - topk_mean：Top-K 最严重的传感器均值（兼顾局部与鲁棒性）
   - p95：95 分位数（过滤极值噪点）
   - 输出：[N] 一维全局异常分数

### 模块三：标准化策略（`compute_norm_stats()`）

- **zscore**：基于均值与标准差（易受极值干扰）
- **robust**：基于中位数与四分位距 IQR（抗极值）

### 模块四：阈值与点调整

- **`choose_threshold()`**：支持百分位与均值±k·std 两种无监督策略
- **`point_adjust_binary()`**：核心 PA 算法
  - 输入：二分类预测向量与延迟步数
  - 过程：对每个被判定为异常的点，向前后各延展 delay 步
  - 输出：扩展后的异常覆盖区间
  - 应用：工业异常由于传播延迟，只要在邻域内命中任一异常，整个区间均判定为报警成功

### 模块五：分类评估（`safe_binary_metrics()`）

安全计算：

- Precision、Recall、F1
- ROC-AUC、PR-AUC（当标签含有两类时）
- 自动处理全 0/全 1 标签的边界情况（返回 NaN 或 0）

### 模块六：主控流水线（`main()`）

执行顺序：

1. 加载验证集与测试集预测数据
2. 计算验证集的尺度锚点（中心与缩放）
3. 计算验证集与测试集异常分值
4. 基于验证集选定阈值
5. 对测试集执行硬切割 + PA 扩展
6. 保存中间结果（npy 格式）
7. 若有标签则进行客观评估
8. 若指定扫参列表，则执行网格搜索找最优工作点
9. 输出汇总 JSON 文件

---

## 3. 必需输入文件

位于 `--run-dir` 目录：

- `y_val_true.npy`：验证集真实观测，形状 [N_val, Nodes, Horizon]
- `y_val_pred.npy`：验证集模型预测，形状 [N_val, Nodes, Horizon]
- `y_true.npy`：测试集真实观测，形状 [N_test, Nodes, Horizon]
- `y_pred.npy`：测试集模型预测，形状 [N_test, Nodes, Horizon]

可选输入（若指定 `--data-path`）：

- `val.npz`、`test.npz`：包含 `y_attack_window` 或 `y_attack_point` 的数据集文件（用于客观评估）

---

## 4. 输出文件

保存在 `--run-dir/anomaly_postprocess` 目录：

- `score_val.npy`：验证集异常分数
- `score_test.npy`：测试集异常分数
- `norm_err_val.npy`：验证集归一化残差矩阵 [N, V]
- `norm_err_test.npy`：测试集归一化残差矩阵 [N, V]
- `pred_test_raw.npy`：测试集硬切割预测（0/1）
- `pred_test_pa.npy`：测试集 PA 调整后的预测（0/1）
- `summary.json`：汇总参数与指标
- `scan_results.json`（可选）：扫参结果，按 pa_f1 降序排列

---

## 5. 使用示例

### 基础用法（最小配置）

```bash
python anomaly_scoring_threshold_pa.py \
  --run-dir /path/to/model/output
```

### 指定策略（有标签进行评估）

```bash
python anomaly_scoring_threshold_pa.py \
  --run-dir /path/to/model/output \
  --data-path /path/to/dataset \
  --norm-method robust \
  --var-reduce topk_mean \
  --var-topk 15 \
  --threshold-percentile 98.5 \
  --pa-delay 8
```

### 自动扫参（寻最优工作点）

```bash
python anomaly_scoring_threshold_pa.py \
  --run-dir /path/to/model/output \
  --data-path /path/to/dataset \
  --scan-percentiles "98,98.5,99,99.5,99.9" \
  --scan-pa-delays "3,5,8,11,15"
```

此模式会尝试所有组合（25 种），并根据测试集真值标签计算每种组合的 F1，最终在 `scan_results.json` 中按 F1 降序输出。

---

## 6. 参数组合推荐

#### 场景 A：偏向高召回（尽量检出异常）

```bash
--norm-method robust \
--var-reduce topk_mean --var-topk 10 \
--horizon-reduce max \
--threshold-percentile 95 \
--pa-delay 8
```

#### 场景 B：偏向高精度（尽量避免误报）

```bash
--norm-method zscore \
--var-reduce max \
--horizon-reduce mean \
--threshold-percentile 99.5 \
--pa-delay 3
```

#### 场景 C：生产部署（平衡模式）

```bash
--norm-method robust \
--var-reduce p95 \
--horizon-reduce max \
--threshold-method mean_std --threshold-k 2.5 \
--pa-delay 5
```

---

## 7. 关键设计细节

#### 数据泄漏防护

- 所有归一化参数（center、scale）仅从 **验证集** 计算
- 测试集在完全见不到任何训练/验证统计信息下进行评分
- 确保离线评估与线上部署的一致性

#### 点调整（PA）的作用

- 工业异常传播有延迟：触发时间点 t 之后，故障后果可能在 t+1, t+2, ... 才逐步显现
- PA 策略允许预测偏移，只要在 [t-delay, t+delay] 窗口内命中真实攻击点，即认为检测成功
- 典型 delay = horizon_size - 1，可调小以提高精度或调大以保证召回

#### 为什么使用验证集确立阈值而非测试集

- 测试集阈值必须在"完全盲法"下确立，否则存在循环论证（用测试集指标调参又用该参数评估测试集）
- 验证集是中间代理，假设其数据分布与测试集相近但不含反馈信息

---

## 8. 常见错误与排查

| 错误信息                                 | 原因                 | 解决方案                                   |
| ---------------------------------------- | -------------------- | ------------------------------------------ |
| FileNotFoundError: Missing required file | 输入文件不存在       | 检查 `--run-dir` 路径，确认 npy 文件已生成 |
| ValueError: ... shape mismatch           | 真值与预测维度不一致 | 重新生成预测张量，确保形状完全对齐         |
| ValueError: 3D array expected            | 输入不是三维张量     | 检查 y_true/y_pred 的形状，应为 [N, V, H]  |
| roc_auc_score: undefined when...         | 标签全为 0 或全为 1  | 该数据集无法计算 AUC，脚本已自动处理为 NaN |

---

## 9. 依赖环境

```
numpy
scikit-learn
```

可选：

```
pandas  # 若需手工后处理输出结果
matplotlib  # 若需绘制自定义图表
```

---

## 10. 进阶用法

#### 在 Python 代码中按函数调用

```python
from pathlib import Path
from anomaly_scoring_threshold_pa import compute_scores, point_adjust_binary

# 自定义组合调用各功能函数
center, scale = ...  # 从验证集计算
scores, norm_err = compute_scores(y_test_true, y_test_pred, center, scale, ...)
pred_raw = (scores > tau).astype(np.int8)
pred_pa = point_adjust_binary(pred_raw, delay=8)
```

#### 批量扫参脚本包装

若需对多个模型/数据集进行系统化评估，建议：

1. 为脚本新增配置文件支持（JSON/YAML）
2. 或将 `main()` 改为可被外部循环多次调用的函数
3. 使用 `joblib.Parallel` 并行化扫参

---

## 11. 总结

`anomaly_scoring_threshold_pa.py` 是一款"即插即用"的后处理工具，特别适合：

- 快速验证不同后处理策略的效果
- 工业部署前的超参数网格搜索
- 与多种主模型（STGNN、LSTM、GRU 等）的集成
- 需要 CLI 参数灵活性而非复杂分析的场景

若需更深入的根因追溯、异常片段聚类、详细可视化，请参考 `advanced_anomaly_detection.py`。
