# 02 全项目程序功能地图

本轮目标：先不修改算法代码，先完整识别项目中每个程序的角色、调用关系和当前主执行链路，避免把历史版本、辅助脚本和当前主算法混在一起判断。

## 1. 项目代码规模

当前主要 Python 程序约 16 个，核心代码约 5200 行左右，不含 notebook。

主要代码文件：

| 文件 | 角色初判 | 行数级别 |
| --- | --- | --- |
| `train_attention.py` | 当前主训练入口 | 约 500 行 |
| `data_maker.py` | 通用在线数据制作 | 约 280 行 |
| `data_loader.py` | 数据加载与 batch 迭代 | 约 230 行 |
| `tools.py` | 参数、指标、绘图、保存、测试工具 | 约 760 行 |
| `attention_matrix.py` | 动态注意力构图模块 | 约 500 行 |
| `GNN/model_attention.py` | 当前 attention 版 MTGNN 主模型 | 约 320 行 |
| `GNN/trainer_attention.py` | 当前 attention 版训练器 | 约 520 行 |
| `GNN/model.py` | 原始/备用 MTGNN 模型 | 约 480 行 |
| `GNN/layer.py` | 基础 GNN 层、GraphConstructor、LayerNorm | 约 480 行 |
| `GNN/ae.py` | 可选窗口自编码器分支 | 约 70 行 |
| `Test/anomaly_scoring_threshold_pa.py` | 推荐后处理：异常分数、阈值、PA | 约 320 行 |
| `Test/advanced_anomaly_detection.py` | 另一套后处理 pipeline | 约 450 行 |
| `Test/visualize_attack_pred_vs_true.py` | 攻击事件附近预测可视化 | 约 360 行 |
| `DataMaker/WADI/WADI_DataMaker_V1.py` | WADI 数据制作参考实现 | 约 210 行 |
| `DataMaker/WADI/DownsampledVisualization.py` | WADI 下采样可视化 | 约 260 行 |
| `DataMaker/Compare.py` | 实验结果对比绘图 | 约 140 行 |

## 2. 当前主训练链路

当前最可能的主链路如下：

```text
train_attention.py
  -> tools.get_input_paras()
  -> data_loader.load_local_dataset()
       -> 若 data_path 是目录：读取 train/val/test.npz + scaler.pkl
       -> 若 data_path 是文件：调用 data_maker.DataMaker(...).run()
  -> GNN.model_attention.MTGNN
       -> attention_matrix.GetAttMatrix
       -> GNN.layer.LayerNorm / linear
       -> 内部自定义 DilatedInceptionLayer / MixHopLayer
  -> GNN.trainer_attention.Trainer
       -> masked_mae / masked_mape / masked_rmse
       -> 可选 GNN.ae.WindowAE
  -> tools.test_model()
  -> 保存 Result.pkl / TrainLoss.pkl / ValidLoss.pkl / TestError.pkl 等
  -> 如果 save_pred_result=True，保存 y_true.npy / y_pred.npy
```

主链路证据：

- `train_attention.py` 导入 `GNN.model_attention.MTGNN`。
- `train_attention.py` 导入 `GNN.trainer_attention.Trainer as MTGNNTrainer`。
- `train_attention.py` 调用 `load_local_dataset()`。
- `train_attention.py` 使用 `tools.test_model()` 做最终验证/测试。

## 3. 模型相关文件关系

### 3.1 当前主模型：`GNN/model_attention.py`

该文件定义了一个“精简版/纯净版” MTGNN：

- 内部定义 `DilatedInceptionLayer`
- 内部定义 `GraphConvLayer`
- 内部定义 `MixHopLayer`
- 主类 `MTGNN`
- 使用 `GetAttMatrix` 做动态图构建
- 前向输入：`[Batch, Channel, Node, Time]`
- 前向输出：`x.transpose(1, 3)`，预期为 `[Batch, 1, Node, Horizon]`

初步判断：这是当前训练入口真正使用的模型。

### 3.2 备用/历史模型：`GNN/model.py`

该文件更接近原始 MTGNN，支持：

- embedding graph constructor
- attention
- attention_v2
- 外部静态邻接矩阵

但当前 `train_attention.py` 没有导入它，因此它不是当前主链路。它适合作为历史版本或备用实现审查。

### 3.3 基础层：`GNN/layer.py`

包含较完整的 GNN 基础层：

- `nconv`
- `dy_nconv`
- `linear`
- `prop`
- `mixprop`
- `dy_mixprop`
- `dilated_1D`
- `dilated_inception`
- `GraphConstructor`
- `LayerNorm`

当前 `model_attention.py` 只显式使用 `LayerNorm` 和 `linear`。

## 4. 动态构图文件关系

`attention_matrix.py` 定义：

- `GetAttMatrix`
- `GetAttMatrixV2`

当前主模型 `model_attention.py` 只使用 `GetAttMatrix`。

已确认接口级硬问题：

```python
# attention_matrix.py
def __init__(self, node_num: int, window_size: int, ...)
```

但当前主模型中：

```python
# GNN/model_attention.py
self.latent_correlation = GetAttMatrix(series_num=node_num, window_size=window_size)
```

这两个接口不一致。`GetAttMatrix` 接收的是 `node_num`，不是 `series_num`。如果没有其他隐藏兼容层，这会导致 `TypeError: unexpected keyword argument 'series_num'`。

同样问题也出现在备用模型 `GNN/model.py` 中，`GetAttMatrix` 和 `GetAttMatrixV2` 都以 `series_num=` 调用。

当前无法通过运行实例化验证，因为命令行 Python 环境缺少 `torch`，但源码层面的接口不一致已经明确。

## 5. 数据相关文件关系

### 5.1 当前通用数据制作：`data_maker.py`

作用：

- 读取 `.csv` / `.pkl` / `.h5`
- 如果有 `Timestamp`，设置为索引
- 创建 `CustomScaler`
- 创建滑动窗口
- 70% / 10% / 20% 顺序划分 train/val/test
- 可保存 `.npz` 和 `scaler.pkl`

已在 `01_data_pipeline_audit.md` 中记录严重问题：

- `Attack` / `Train` 可能进入模型输入。
- 创建了 scaler 但没有实际 transform。
- 没有按 `Train==1 / Train==0` 划分。

### 5.2 当前数据加载：`data_loader.py`

作用：

- 如果 `data_path` 是目录，读取 `train.npz` / `val.npz` / `test.npz`。
- 如果 `data_path` 是文件，调用 `DataMaker(..., save=False)` 在线生成。
- 构造 `DataLoader`。

已确认问题：

- `load_local_dataset()` 不接收 `sliding_step`，所以 `tools.py` 中的 `--sliding_step` 当前不会传入在线数据制作流程。
- train/val/test 默认都会 padding。

### 5.3 WADI 参考制作脚本：`DataMaker/WADI/WADI_DataMaker_V1.py`

该脚本比当前通用 `data_maker.py` 更接近正确工业异常检测协议：

- 明确 `label_cols = ["Timestamp", "Train", "Attack"]`
- `feature_cols` 排除标签列
- 使用 `Train==1` / `Train==0` 区分训练验证与测试
- scaler 只在训练特征上拟合
- 对特征执行 `scaler.transform()`
- 状态列可设置为不标准化

初步判断：后续修复 `data_maker.py` 时，可以参考此脚本的逻辑，但要抽象成 SWaT/WADI 通用版本。

## 6. 训练器与评估关系

### 6.1 `GNN/trainer_attention.py`

当前训练器：

- `train()`：训练单个 batch，返回 loss/mape/rmse 和动态邻接矩阵。
- `eval()`：验证单个 batch。
- 支持课程学习 `cl`。
- 支持可选 AE 重构分支。

初步风险点：

- 如果数据没有标准化，但 `eval()` 又执行 `inverse_transform()`，评估尺度会错误。
- 如果 `use_ae=True`，`true_win = torch.cat([hist, true_y], dim=3)` 依赖 `true_y` 和 `hist` 的通道/时间维完全匹配，需要单独审查。

### 6.2 `tools.test_model()`

作用：

- 遍历 loader。
- 输入转为 `[B, C, N, L]`。
- 模型输出后 `squeeze()` 并拼接。
- 将 `true_y` 和 `pred_y` 反归一化。
- 计算 `calc_error()`。

初步风险点：

- `output_y.squeeze()` 如果 batch 或维度为 1，可能导致维度意外坍缩，需要审查。
- 如果验证/测试 loader padding，函数用 `[:true_y.size(0)]` 裁剪，能部分防止 padding 污染，但仍需确认 `true_y` 原始长度是否未 padding。

## 7. 后处理与可视化文件关系

### 7.1 推荐后处理：`Test/anomaly_scoring_threshold_pa.py`

该脚本逻辑较清晰：

- 读取 `y_val_true.npy`、`y_val_pred.npy`、`y_true.npy`、`y_pred.npy`
- 用验证集误差估计标准化中心和尺度
- 计算测试异常分数
- 基于验证集分数选阈值
- 执行 Point Adjustment
- 如果 `.npz` 中有 `y_attack_window` 或 `y_attack_point`，则计算指标

当前主训练脚本只保存 `y_true.npy` 和 `y_pred.npy`，没有保存 `y_val_true.npy` 和 `y_val_pred.npy`，因此与该后处理脚本存在输出格式不匹配问题。

### 7.2 另一套后处理：`Test/advanced_anomaly_detection.py`

该脚本：

- 读取 `y_true.npy` / `y_pred.npy`
- 可选读取标签 pkl
- 计算异常分数
- 可有监督选阈值或无监督选阈值
- 支持可视化与保存

风险：

- 该脚本在测试集自身误差上计算归一化均值和方差，可能引入测试分布信息泄漏。相比之下，`anomaly_scoring_threshold_pa.py` 用验证集估计尺度更合理。

### 7.3 攻击段可视化：`Test/visualize_attack_pred_vs_true.py`

用途：

- 读取预测结果和 `test.npz`
- 需要 `test.npz` 中包含 `y_time`
- 按攻击事件绘制真值/预测曲线

当前 `data_maker.py` 保存的 `.npz` 只有 `x` 和 `y`，不包含 `y_time`，因此该脚本与当前通用数据制作输出不完全兼容。

## 8. 输出文件格式不一致问题

当前训练入口：

```python
if args.save_pred_result:
    np.save(os.path.join(args.save, 'y_true'), true_y.detach().cpu().numpy())
    np.save(os.path.join(args.save, 'y_pred'), pred_y.detach().cpu().numpy())
```

这会生成：

- `y_true.npy`
- `y_pred.npy`

但 `Test/anomaly_scoring_threshold_pa.py` 要求：

- `y_val_true.npy`
- `y_val_pred.npy`
- `y_true.npy`
- `y_pred.npy`

因此当前训练输出与推荐后处理脚本之间存在断点。

## 9. 当前主链路初步结论

当前项目不是单一干净流水线，而是由多个阶段性版本组合而成：

1. `train_attention.py + model_attention.py + trainer_attention.py` 是当前训练主链路。
2. `data_maker.py` 是当前主链路的数据来源之一，但其逻辑落后于 `DataMaker/WADI/WADI_DataMaker_V1.py`。
3. `GNN/model.py` 是旧版/备用模型，当前不直接参与训练。
4. `Test/anomaly_scoring_threshold_pa.py` 是较合理的后处理方向，但当前训练输出与它不完全对接。
5. 可视化脚本要求的数据元信息比当前 `.npz` 保存内容更丰富。

## 10. 下一步审查建议

下一步应进入“主链路硬错误审查”，优先检查：

1. `GetAttMatrix(series_num=...)` 接口不一致。
2. 数据列污染与标准化缺失。
3. `save_pred_result` 输出不足，无法支撑后处理。
4. 模型输出维度与 `Trainer.train()`、`tools.test_model()` 是否严格一致。
5. `DataLoader` padding 和 `test_model()` 裁剪是否可靠。

在这些硬错误确认和修复前，不建议讨论模型创新点是否成立。

