# 01 数据流水线审查

已审查文件：

- `data_maker.py`
- `data_loader.py`
- `tools.py` 中的 `CustomScaler` 和参数解析部分
- `DataMaker/WADI/WADI_DataMaker_V1.py` 作为参考实现

## 预期数据流程

从代码注释和模型目标看，当前项目预期采用“预测误差驱动”的异常检测流程：

1. 加载 SWaT/WADI 原始数据表。
2. 移除非传感器特征字段，例如 `Timestamp`、`Train`、`Attack`。
3. 仅使用训练集拟合标准化器。
4. 使用该标准化器转换训练集、验证集和测试集的物理传感器特征。
5. 生成 Seq2Seq 滑动窗口：
   - `x`: `[样本数, 历史窗口长度, 节点数, 特征维度]`
   - `y`: `[样本数, 预测窗口长度, 节点数, 特征维度]`
6. 训练 MTGNN 预测未来传感器状态。
7. 使用预测残差作为异常证据。

## 已确认问题

### F1. `Attack` 和 `Train` 可能被当作模型输入特征

位置：

- `data_maker.py` 的 `get_df()`：只把 `Timestamp` 设置为索引。
- `data_maker.py` 的 `create_input_data()`：直接使用 `self.df.values`。

问题：

如果源数据表包含 `Attack` 和 `Train` 两列，这两列会一起进入模型，成为“传感器节点”。当前 SWaT 数据画像已经确认数据中包含 `Timestamp`、`Attack` 和 `Train`。

这意味着模型可能在训练和测试时直接看到真实异常标签 `Attack`，以及训练/测试划分标记 `Train`。

影响：

这是异常检测任务中的严重标签泄漏/数据污染问题。如果 `Attack` 参与模型输入，则检测性能不具备科学可信度，论文中也不能作为有效实验结果报告。

建议方向：

显式区分：

- 特征列：仅保留真实物理传感器/执行器变量。
- 元信息列：`Timestamp`、`Train`、`Attack`。

模型输入 `x/y` 只能由特征列构建，而窗口级异常标签应单独保存，用于后处理评估。

### F2. `CustomScaler` 被创建，但在 `data_maker.py` 中没有真正应用

位置：

- `data_maker.py` 中创建标准化器：`scalers = CustomScaler(df_train.values)`
- `data_maker.py` 的 `create_input_data()`：直接使用原始 `self.df.values`

问题：

代码创建并返回/保存了标准化器，但没有在滑动窗口生成前调用 `scalers.transform()`。

影响：

模型很可能直接在原始物理量纲上训练。SWaT/WADI 中不同变量量纲差异很大，例如水位、流量、压力、阀门状态混在一起，会影响训练稳定性。

更严重的是，如果评估阶段又调用 `inverse_transform()`，则可能出现尺度逻辑不一致：模型输出本来就是原始尺度，却又被反归一化一次。

参考：

`DataMaker/WADI/WADI_DataMaker_V1.py` 中的实现更合理：

```python
data_raw = _df[feature_cols].values.astype(np.float64)
data_std = scaler.transform(data_raw).astype(np.float32)
```

建议方向：

在生成窗口前，对特征列执行 `scalers.transform()`。并确保只有当预测值和真实值处于标准化空间时，评估阶段才执行 `inverse_transform()`。

### F3. `--sliding_step` 被解析，但没有传入 `DataMaker`

位置：

- `tools.py`：定义了 `--sliding_step`，默认值为 `12`。
- `data_loader.py` 的 `load_local_dataset()`：调用 `DataMaker(path, window_size, horizon_size, save=False)`。

问题：

`DataMaker.__init__()` 支持 `sliding_step` 参数，但 `load_local_dataset()` 没有暴露或传入该参数。因此在线生成数据时，实际使用的是 `DataMaker` 默认值 `sliding_step=1`，而不是命令行中的设置。

影响：

实验可能在不知情的情况下使用了大量高度重叠的窗口，导致：

- 样本数量远大于预期。
- 训练耗时和显存压力增加。
- 实验复现时参数记录与真实行为不一致。
- 相邻样本高度重叠，可能影响验证/测试独立性判断。

建议方向：

给 `load_local_dataset()` 增加 `sliding_step` 参数，并在 `train_attention.py` 中传入 `args.sliding_step`。

### F4. 训练/验证/测试划分基于生成后的窗口，而不是原始 `Train` 列

位置：

- `data_maker.py` 的 `run()`：对生成后的窗口采用 70% / 10% / 20% 顺序划分。

问题：

SWaT/WADI 通常有官方或半官方划分方式：`Train==1` 表示正常训练段，`Train==0` 表示测试/攻击段。当前通用划分逻辑忽略了该列。

如果 `swat_data_raw.pkl` 或 `swat_data_10s.pkl` 合并了正常段和攻击段，那么当前划分可能把攻击窗口放入训练集或验证集。

影响：

这会破坏与 Anomaly Transformer、DAGMM、LSTM-VAE、PatchTST、TimesNet 等方法的公平对比，也不符合工业异常检测中“仅用正常数据训练”的常见设定。

建议方向：

需要明确实验协议：

- 官方 SWaT/WADI 协议：从 `Train==1` 中划分训练/验证，从 `Train==0` 中构造测试集。
- 通用时间顺序协议：删除 `Train` 和 `Attack` 后按时间顺序划分，但必须在论文中明确说明。

对于 SCI 论文，通常建议优先采用官方/常用协议。

### F5. 验证集和测试集 padding 会复制样本

位置：

- `data_loader.py` 的 `DataLoader.__init__()`：默认对所有数据加载器使用 `add_pad=True`，通过复制最后一个样本补齐 batch。

问题：

训练阶段补齐 batch 是可以接受的。但验证/测试阶段如果也复制样本，后续保存预测值或计算指标时，可能包含人工重复样本。

影响：

可能造成：

- 测试指标轻微偏差。
- 预测数组长度与真实时间戳/标签长度不一致。
- 后处理脚本需要额外裁剪。

建议方向：

训练集可以保留 `add_pad=True`。验证集和测试集建议使用 `add_pad=False`，或者记录原始样本数并在推理后裁剪掉 padding 部分。

## 待确认问题

1. 当前项目是否应严格采用 SWaT/WADI 的 `Train==1` 训练、`Train==0` 测试协议？
2. 离散执行器/状态变量是否应该标准化？还是保留原始 0/1/2 状态值？
3. 模型预测目标是否应包含所有物理变量？还是只预测连续传感器，把执行器状态作为辅助协变量？
4. 是否需要在 `.npz` 中保存窗口级异常标签，例如 `y_attack_window` 和 `y_attack_point`，以保证后处理可复现？

## 初步结论

当前数据流水线还不适合作为论文级实验基线。最优先需要解决的问题是：

1. 去除 `Attack` / `Train` 等标签或元信息列，避免标签泄漏。
2. 在滑动窗口生成前真正执行标准化。
3. 明确并实现符合 SWaT/WADI 异常检测任务的训练/验证/测试划分协议。

在这些问题修复之前，不建议先判断模型结构优劣，因为模型性能可能被数据处理问题严重干扰。

