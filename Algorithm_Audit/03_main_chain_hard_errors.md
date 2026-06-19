# 03 主训练链路硬错误与断点审查

本文件记录当前主链路中已经确认的硬错误、断点和高风险不一致。这里的“硬错误”指的是可能导致程序无法运行、训练/测试不使用预期逻辑、后处理无法衔接，或实验结果科学性失效的问题。

## H1. `GetAttMatrix` 构造参数不一致，主模型可能无法初始化

证据位置：

- `attention_matrix.py`
- `GNN/model_attention.py`
- `GNN/model.py`

`attention_matrix.py` 中 `GetAttMatrix` 的构造函数为：

```python
def __init__(self,
             node_num: int,
             window_size: int,
             dropout_rate: float = 0.5,
             leaky_rate: float = 0.2):
```

但当前主模型 `GNN/model_attention.py` 中调用为：

```python
self.latent_correlation = GetAttMatrix(series_num=node_num, window_size=window_size)
```

问题：

`GetAttMatrix` 不接收 `series_num` 关键字参数。除非运行环境中有另一个同名兼容版本，否则这里会触发：

```text
TypeError: __init__() got an unexpected keyword argument 'series_num'
```

同类问题也出现在备用模型 `GNN/model.py` 中：

```python
GetAttMatrix(series_num=node_num, window_size=window_size)
GetAttMatrixV2(series_num=node_num, ...)
```

而 `GetAttMatrixV2` 的构造函数同样使用 `node_num`，不是 `series_num`。

影响：

这是当前主链路最高优先级硬错误。若未修复，模型无法实例化，训练无法开始。

建议修复方向：

统一接口命名。最小修复是把调用端改为：

```python
GetAttMatrix(node_num=node_num, window_size=window_size)
```

备用模型中的 `GetAttMatrixV2(series_num=...)` 也应改为 `node_num=...`。

## H2. 保存了最佳模型，但最终验证/测试前没有加载最佳权重

修复状态：已修复，见 `Revision_Log.md` 中 `R002`。

证据位置：

- `train_attention.py`

训练中，当验证损失刷新历史最优时会执行：

```python
torch.save(engine.model.state_dict(), os.path.join(args.save, "model.pth"))
```

但训练循环结束后，进入最终验证和测试前，没有发现：

```python
engine.model.load_state_dict(torch.load(...))
```

问题：

最终 `test_model()` 使用的是当前内存中的模型参数。若早停触发或最后若干轮已经退化，则最终测试结果不一定对应验证集最优模型。

影响：

这会导致：

- 保存的 `model.pth` 与最终报告指标不一致。
- “巅峰模型”叙述与实际测试对象不一致。
- 论文表格中的测试结果可能不是验证集选择出的最佳模型。

建议修复方向：

在最终验证/测试前加载最佳权重：

```python
best_path = os.path.join(args.save, "model.pth")
engine.model.load_state_dict(torch.load(best_path, map_location=device))
```

并明确日志说明“正在使用验证集最佳模型进行最终测试”。

## H3. 推荐后处理脚本需要验证集预测，但训练脚本只保存测试集预测

修复状态：已修复，见 `Revision_Log.md` 中 `R005`。

证据位置：

- `train_attention.py`
- `Test/anomaly_scoring_threshold_pa.py`

训练脚本当前只保存：

```python
np.save(os.path.join(args.save, 'y_true'), true_y.detach().cpu().numpy())
np.save(os.path.join(args.save, 'y_pred'), pred_y.detach().cpu().numpy())
```

实际生成：

- `y_true.npy`
- `y_pred.npy`

但 `Test/anomaly_scoring_threshold_pa.py` 要求：

```python
required = ["y_val_true.npy", "y_val_pred.npy", "y_true.npy", "y_pred.npy"]
```

问题：

当前训练输出不能直接支撑推荐后处理脚本。

影响：

异常分数阈值选择无法严格基于验证集完成。若改用测试集自身估计阈值或归一化尺度，会造成测试集信息泄漏。

建议修复方向：

训练阶段同时保存验证集预测：

- `y_val_true.npy`
- `y_val_pred.npy`
- `y_true.npy`
- `y_pred.npy`

也可以调整 `test_model()` 返回格式或增加 `predict_loader()` 函数，避免重复逻辑。

## H4. `data_maker.py` 数据处理与训练器反归一化逻辑不一致

修复状态：已修复，见 `Revision_Log.md` 中 `R003`。

证据位置：

- `data_maker.py`
- `tools.py`
- `GNN/trainer_attention.py`

`data_maker.py` 创建：

```python
scalers = CustomScaler(df_train.values)
```

但滑动窗口生成使用：

```python
data = np.expand_dims(self.df.values, axis=-1)
```

没有实际执行：

```python
scalers.transform(...)
```

而 `tools.test_model()` 和 `Trainer.eval()` 会执行：

```python
pred_y = scalers.inverse_transform(pred_y)
true_y = scalers.inverse_transform(true_y)
```

问题：

如果训练数据没有被标准化，评估阶段再执行反归一化，尺度会被错误放大/平移。

影响：

回归指标 MAE/RMSE/MAPE 可能不代表真实物理误差，论文结果不可采信。

建议修复方向：

要么：

1. 数据进入模型前必须标准化，评估再反归一化；

要么：

2. 如果决定用原始物理量训练，则评估阶段不能再 `inverse_transform()`。

从工业多变量时间序列建模角度，建议采用方案 1。

## H5. `Attack` / `Train` 可能进入模型输入，属于严重标签泄漏

修复状态：已修复，见 `Revision_Log.md` 中 `R003`。

证据位置：

- `data_maker.py`
- `Docs/swat_data_raw_profile.md`

SWaT 数据包含：

- `Timestamp`
- `Attack`
- `Train`
- 51 个左右物理传感器/执行器变量

`data_maker.py` 只把 `Timestamp` 设置为索引，没有删除 `Attack` 和 `Train`。随后 `create_input_data()` 直接使用：

```python
self.df.values
```

问题：

`Attack` 真实异常标签和 `Train` 划分标记可能成为模型节点。

影响：

这是异常检测中最严重的数据污染之一。模型可能直接或间接学习标签本身，而不是工业过程状态。

建议修复方向：

明确：

```python
label_cols = ["Timestamp", "Train", "Attack"]
feature_cols = [c for c in df.columns if c not in label_cols]
```

训练、预测只使用 `feature_cols`。标签列只用于评估和可视化。

## H6. 命令行 `--sliding_step` 未传入在线数据制作流程

修复状态：已修复，见 `Revision_Log.md` 中 `R004`。

证据位置：

- `tools.py`
- `data_loader.py`
- `data_maker.py`

`tools.py` 定义：

```python
parser.add_argument('--sliding_step', type=int, default=12, ...)
```

`DataMaker.__init__()` 支持：

```python
sliding_step: int = 1
```

但 `data_loader.py` 调用为：

```python
maker = DataMaker(path, window_size, horizon_size, save=False)
```

问题：

命令行设置不会影响在线数据制作，实际总是使用 `sliding_step=1`。

影响：

样本量、重叠率、训练耗时和复现实验参数都可能与记录不一致。

建议修复方向：

`load_local_dataset()` 增加 `sliding_step` 参数，并从 `train_attention.py` 传入 `args.sliding_step`。

## H7. 可视化脚本导入不存在的 `ATTACK_EVENTS`

修复状态：已修复，见 `Revision_Log.md` 中 `R010`。

证据位置：

- `tools.py`
- `Test/visualize_attack_pred_vs_true.py`
- `DataMaker/WADI/DownsampledVisualization.py`

`tools.py` 当前定义的是：

```python
ATTACK_EVENTS_SWAT = [...]
ATTACK_EVENTS_WADI = [...]
```

但脚本中使用：

```python
from tools import ATTACK_EVENTS
```

问题：

`ATTACK_EVENTS` 这个别名当前不存在。

影响：

相关可视化脚本会在 import 阶段失败。

建议修复方向：

根据数据集显式选择：

```python
from tools import ATTACK_EVENTS_SWAT as ATTACK_EVENTS
```

或在 `tools.py` 中添加清晰的默认别名，但不建议模糊默认，因为 SWaT/WADI 攻击事件不同。

## H8. `tools.test_model()` 使用 `output_y.squeeze()`，存在维度坍缩风险

修复状态：已修复，见 `Revision_Log.md` 中 `R007`。

证据位置：

- `tools.py`

当前：

```python
pred_y.append(output_y.squeeze())
```

问题：

如果某些维度为 1，`squeeze()` 会删除所有长度为 1 的维度。当前预期输出可能是 `[B, 1, N, H]`，正常希望只去掉通道维，得到 `[B, N, H]`。

影响：

当 batch size、节点数或 horizon 某个维度为 1 时，可能导致后续 `torch.cat()` 或 `calc_error()` 维度不一致。

建议修复方向：

改为只压缩通道维：

```python
pred_y.append(output_y.squeeze(1))
```

并用断言确认输出维度。

## H9. `masked_mae` 和 `masked_mape` 对全掩码情况不如 `masked_mse` 稳健

修复状态：已修复，见 `Revision_Log.md` 中 `R009`。

证据位置：

- `tools.py`

`masked_mse()` 中有：

```python
if torch.sum(mask) == 0:
    return torch.tensor(0.0, device=pred_y.device)
```

但 `masked_mae()` 和 `masked_mape()` 直接执行：

```python
mask /= torch.mean(mask)
```

问题：

如果某个 batch 的有效标签全被 mask 掉，可能出现除零或 NaN。虽然当前 SWaT 数据没有缺失值，这个问题不一定触发，但函数设计不完全一致。

影响：

通用性和鲁棒性不足。若未来处理含缺失值的数据集，可能污染训练/评估。

建议修复方向：

让 `masked_mae()`、`masked_mape()` 与 `masked_mse()` 使用一致的安全逻辑。

## 优先级建议

第一优先级，必须先修：

1. H1：注意力模块接口不一致。（已修复，见 R001）
2. H2：最终测试未加载最佳模型。（已修复，见 R002）
3. H4/H5：数据标准化缺失与标签泄漏。（已修复，见 R003）
4. H3：训练输出与后处理脚本不匹配。（已修复，见 R005）

第二优先级：

5. H6：`sliding_step` 参数未生效。（已修复，见 R004）
6. H8：`squeeze()` 维度风险。（已修复，见 R007）
7. H7：可视化攻击事件别名。
8. H9：mask 指标鲁棒性。

## 当前结论

当前代码已经完成 `H1/H2/H3/H4/H5/H6/H7/H8/H9` 的关键修复，主模型初始化、最佳权重回载、训练输出后处理接口、数据标准化、标签泄漏、滑动步长参数传递、可视化攻击事件表导入、最终评估预测张量维度风险和掩码指标全掩码鲁棒性均已有明确处理。后续仍建议通过一次小规模真实训练验证完整长流程产物落盘、显存占用和后处理结果一致性。
