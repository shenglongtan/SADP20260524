# WADI_DataMaker_V1.py 功能分析与使用说明

## 1. 文件概述

`WADI_DataMaker_V1.py` 是用于 WADI 数据集预处理和样本生成的脚本。主要功能包括：

- 加载原始 WADI 数据集
- 检查缺失值
- 划分特征列、状态列与连续列
- 根据 `Train` 字段划分训练、验证和测试集
- 可选下采样处理
- 训练/验证集按时间顺序切分
- 构造标准化器 (`CustomScaler`)
- 生成滑动窗口样本对 `(x, y)`
- 保存处理后的数据和元信息

---

## 2. 主要功能模块

### 2.1 参数配置

- `WINDOW_SIZE`：历史窗口长度，默认为 48。
- `HORIZON_SIZE`：预测窗口长度，默认为 12。
- `TRAIN_RATIO`：训练集比例，默认 0.7。
- `DOWNSAMPLE_SEC`：下采样时间间隔，默认 10 秒；设置为 `None` 时不下采样。
- `save_path`：输出保存目录，可根据工程路径修改。

### 2.2 数据加载与检查

- 从相对路径 `../../Data/WADI/data.pkl` 加载原始数据。
- 统计并打印缺失值总数。

### 2.3 列类型划分

- `label_cols`：`Timestamp`、`Train`、`Attack`。
- `feature_cols`：除上述标签列以外的所有列。
- `status_cols`：名称包含 `STATUS` 的离散/开关类列。
- `cont_cols`：连续变量列（除状态列之外的特征列）。

### 2.4 状态列分布检查

- 遍历 `status_cols`，打印每个状态列的取值分布。

### 2.5 训练、验证、测试集切分

- `Train==1` 的数据划归 `df_train_val`，用于训练和验证。
- `Train==0` 的数据划归 `df_test`，用于测试。
- 保留 `Timestamp`, `Train`, `Attack` 和所有特征列。

### 2.6 下采样处理（可选）

- 若 `DOWNSAMPLE_SEC` 不为 `None`，则对 `df_train_val` 和 `df_test` 调用 `downsample_block()`。
- 下采样后打印数据维度变化。

### 2.7 训练集与验证集切分

- 按时间顺序切分 `df_train_val`。
- `df_train` 为前 `TRAIN_RATIO` 部分；`df_val` 为剩余部分。

### 2.8 构造标准化器

- 使用 `CustomScaler` 对训练集特征拟合。
- 对状态列保持原始比例：将其 `mean=0, std=1`，避免状态列被归一化。

### 2.9 滑动窗口样本生成

函数 `build_xy_from_df(_df)`：

- 输入：原始 DataFrame
- 输出：
  - `x` 形状为 `[num_samples, WINDOW_SIZE, num_nodes, 1]`
  - `y` 形状为 `[num_samples, HORIZON_SIZE, num_nodes, 1]`
- 处理流程：
  1. 提取特征列并转为 `np.float64`
  2. 通过 `scaler.transform()` 标准化
  3. 增加最后一维特征通道
  4. 根据窗口大小滑动切分样本

### 2.10 保存结果

- 保存 `train/val/test` 三类数据为压缩 NPZ 文件：`train.npz`、`val.npz`、`test.npz`。
- 保存标准化器 `scaler.pkl`。
- 保存元信息 `meta.pkl`，包括窗口长度、预测长度、下采样配置、列顺序等。

---

## 3. 使用说明

### 3.1 环境依赖

需要安装以下 Python 库：

- `numpy`
- `pandas`
- `hues`

并确保 `tools.py` 中包含：

- `CustomScaler`
- `check_folder`
- `downsample_block`

### 3.2 运行方式

直接执行脚本：

```bash
python WADI_DataMaker_V1.py
```

如果需要在 Jupyter 中执行，可按单元顺序运行；如果在脚本模式下执行，脚本会直接输出结果并保存文件。

### 3.3 结果文件说明

- `train.npz`：训练样本数据
- `val.npz`：验证样本数据
- `test.npz`：测试样本数据
- `scaler.pkl`：用于后续数据反归一化或推理时的标准化器
- `meta.pkl`：元信息字典，包含数据处理配置和列序

### 3.4 推荐修改

- 若希望使用不同窗口或预测长度，可修改 `WINDOW_SIZE` 和 `HORIZON_SIZE`。
- 若希望不下采样，设置 `DOWNSAMPLE_SEC = None`。
- 若希望保存到其他目录，可修改 `save_path`。
- 若希望将数据线路配置为命令行参数，可进一步封装此脚本。

---

## 4. 典型场景

- 作为 WADI 异常检测模型训练的数据预处理脚本。
- 生成可直接输入 STGNN / GNN 模型训练的标准化滑动窗口数据。
- 用于对 WADI 数据集进行下采样实验和数据扩增对比。

---

## 5. 注意事项

1. `data.pkl` 的路径是相对位置 `../../Data/WADI/data.pkl`，请根据实际目录结构确认该文件存在。
2. `save_path` 目录不会自动创建，脚本使用 `check_folder()` 进行创建。
3. `status_cols` 不做归一化，仅保留原值；如果需要对状态列做 embedding 或 one-hot，可在后续模型输入中处理。
4. 若输入数据时间步不足，将触发 `ValueError`，提示 `Not enough timesteps`。

---

## 6. 扩展建议

- 将 `build_xy_from_df()` 提取为可复用函数模块，方便其他数据集调用。
- 如果需要支持多种数据格式，可增加 `csv` / `parquet` 的加载逻辑。
- 若希望保存更丰富的元信息，可将 `Attack`、`Timestamp` 等列的取值范围也一并保存。
