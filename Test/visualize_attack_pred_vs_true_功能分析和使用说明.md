# visualize_attack_pred_vs_true.py 功能分析与使用说明

## 1. 文件概述

`visualize_attack_pred_vs_true.py` 是攻击事件可视化专用工具。它将模型预测的 3D 滑窗张量 `[样本数, 节点数, 预测步长]` 还原为时序 DataFrame，然后围绕已知的工业异常事件生成**上下双联对比图表**，展示直接受击节点与间接波及节点的真实值vs预测值曲线。

**核心能力：**

- 时序降维：滑窗预测张量 → 连续时间序列 DataFrame
- 受损评估：基于分布对比（Z-score）识别每个事件的根因与次要波及节点
- 双层可视化：第一层(上)展示直接受击节点，第二层(下)展示Top-K波及节点
- 灰度底纹标记：攻击时间区间高亮，便于定位异常窗口
- 批量导出：支持按事件 ID 单独渲染或全量处理 16 个攻击事件，输出 PNG/TIF 高清图

特别适用于：

- 模型效果验证：直观观察真实与预测的一致性
- 根因分析：发现异常如何从主目标传播到相邻传感器
- 论文插图：高质量 300dpi 图表便于出版

---

## 2. 主要模块与功能

### 模块一：环境依赖与工程寻址

**函数：** `_find_project_root(start: Path) -> Path`

- 动态定位包含 `tools.py` 的工程根目录
- 无论从哪个层级启动脚本，都能正确导入 SWaT/WADI 显式攻击事件列表
- 跨平台兼容（使用 `pathlib.Path`）

**依赖导入：**

```python
json, pickle, sys, pathlib.Path  # 文件操作与路径管理
numpy, pandas, matplotlib, IPython.display  # 数据与绘图
tools.ATTACK_EVENTS_SWAT / tools.ATTACK_EVENTS_WADI  # 工业异常事件的已知时间与受击节点
```

**配置输出：**

- 全局字体设置（支持中英文）
- 打印工程根目录路径便于调试

---

### 模块二：全局超参数配置

可编辑的运行时参数表：

| 参数           | 类型      | 默认值             | 说明                                                |
| -------------- | --------- | ------------------ | --------------------------------------------------- |
| `RUN_DIR`      | Path      | E:\DataForCode\... | 模型输出目录，包含 y_true.npy、y_pred.npy           |
| `DATA_PATH`    | Path/None | None               | 数据集根目录；None 时从 Parameters.json 自动读取    |
| `EVENT_ID`     | int/None  | 16                 | 指定事件编号 1-16；None 则绘制全部 16 个事件        |
| `PRE_MINUTES`  | int       | 10                 | 攻击前观察窗口宽度（分钟）                          |
| `POST_MINUTES` | int       | 10                 | 攻击后观察窗口宽度（分钟）                          |
| `TOPK`         | int       | 3                  | 除直接受击点外，额外展示受波及最严重的 Top-K 个节点 |
| `PV_ONLY`      | bool      | True               | 业务过滤：仅分析 Process Variable (后缀 \_PV) 变量  |
| `COLOR_LIST`   | list[str] | 预定义调色板       | 自定义 RGB 离散色，保证各节点曲线区分度             |
| `SAVE_DIR`     | Path/None | ./Export/          | 图表导出目录；None 则保存至 RUN_DIR/attack_viz      |

**配置示例：**

```python
EVENT_ID = 5          # 仅分析第5次攻击
PRE_MINUTES = 20      # 前向看20分钟
TOPK = 5              # 展示Top5波及节点
PV_ONLY = False       # 包含所有节点（设备状态+过程变量）
```

---

### 模块三：数据 I/O 与特征读取

**函数组：**

#### `find_params_json(run_dir: Path) -> Path`

- 在运行目录及其父目录寻找 `Parameters.json`
- 用于从配置文件反向解析数据集路径

#### `resolve_data_path(run_dir: Path, data_path_arg) -> Path`

- 优先使用显式传入的 `data_path_arg`
- 否则从 Parameters.json 中提取 `data_path` 字段
- 验证路径存在性，缺失则抛出异常

#### `load_feature_names(data_path: Path, node_num: int) -> list[str]`

- 从 `data_path/meta.pkl` 读取 pickle 化的特征列表（传感器名）
- 若 pickle 不存在或维度不匹配，生成占位符 `['node_0', 'node_1', ...]`
- 确保后续 DataFrame 列名可读性高（如 `1_AIT_001_PV` 而非 `node_0`）

---

### 模块四：时序降维与连续重构

**函数：** `aggregate_windows_to_timeseries(arr, y_time, columns, first_step_only=True) -> pd.DataFrame`

**输入：**

- `arr`：模型预测/真实张量，形状 [samples, nodes, horizon]
- `y_time`：时间戳矩阵，形状 [samples, horizon]，元素为日期字符串或 unix 秒
- `columns`：特征名列表，长度 = nodes
- `first_step_only`：降维策略选择

**两种降维策略：**

1. **first_step_only=True（推荐）**
   - 仅提取每个预测窗口的第一步（h=0）
   - 避免多个窗口在同一时刻的重叠造成的模糊
   - 对重复时间戳自动求均值
2. **first_step_only=False（备选）**
   - 展平整个视界，对所有时间步都进行拼接
   - 当时间戳完全无重叠时适用
   - 对重复时间戳暴力求均值（可能引入噪点）

**输出：**

- 标准 Pandas DataFrame
- 行索引：DatetimeIndex（时间戳）
- 列：传感器特征名
- 已排序且无重复时间

**关键细节：**

```python
# Z-score 转换用于后续根因分析
s = pd.Series(vals, index=ts_index).groupby(level=0).mean().sort_index()
```

---

### 模块五：受波及节点效应评估

**函数：** `calc_event_effect(true_df, st, ed, pre, candidate_cols) -> pd.Series`

**功能：** 对标准化效应指标评分，识别受波及最深的节点

**过程：**

1. **基准期提取**

   ```python
   baseline = true_df.loc[max(st-pre, min_time): st, candidate_cols]
   base_mean = baseline.mean(axis=0)
   base_std = baseline.std(axis=0).replace(0, np.nan)
   ```

2. **攻击期提取**

   ```python
   attack = true_df.loc[st:ed, candidate_cols]
   atk_mean = attack.mean(axis=0)
   ```

3. **标准化偏移量（Z-score 思想）**
   ```python
   effect = ((atk_mean - base_mean).abs() / base_std).dropna().sort_values(ascending=False)
   ```

   - 分子：攻击期均值与基准期均值的绝对差
   - 分母：基准期标准差（去掉 0 以避免除零）
   - 结果按严重程度降序排列

**输出：**

- 索引为节点名，值为效应 Z-score
- Top 节点即网络传播的主要受害者

---

### 模块六：可视化绘图引擎

**函数：** `plot_event(event_idx_1based, st, ed, attack_cols, true_df, pred_df, top_nodes, pre, post) -> (Figure, DataFrame)`

**双层结构：**

```
┌──────────────────────────────────────┐
│  上层 (ax1) - 直接受击节点           │  height_ratio=1
│  展示 attack_cols 的真实vs预测曲线   │
├──────────────────────────────────────┤
│  下层 (ax2) - Top-K 波及节点         │  height_ratio=2
│  展示 top_nodes 的真实vs预测曲线     │
│  并计算 MAE 与 RMSE                  │
└──────────────────────────────────────┘
```

**绘图细节：**

- **真实值**：实线 `-`
- **预测值**：虚线 `--`
- **攻击区间**：灰色半透明底纹 `[st, ed]`
- **色彩映射**：支持自定义 RGB 列表，循环分配以保证节点区分度
- **图例**：去重后统一放置在图表下方外侧，自适应换行

**误差计算（下层 ax2）：**

```python
err = (atk_pred - atk_true).astype(float)
mae = np.nanmean(np.abs(err))   # 忽略 NaN
rmse = np.sqrt(np.nanmean(err**2))
```

**输出文件：**

- 按 `build_output_stem()` 函数生成标准化文件名
- 格式：`event{:02d}_pre10m_post10m_top3_pvonly.png/tif`
- 分别保存 PNG (预览) 与 TIF (出版级 300dpi 无损)

---

### 模块七：主控执行流水线

分为 4 个阶段：

#### **Part 1 - 数据加载**

```python
y_true = np.load(run_dir / 'y_true.npy')       # [N, V, H]
y_pred = np.load(run_dir / 'y_pred.npy')       # [N, V, H]
test_npz = np.load(data_path / 'test.npz')
y_time = test_npz['y_time']                    # [N, H]
```

- 验证文件存在性
- 检查维度匹配与形状一致性
- 拦截依赖文件缺失

#### **Part 2 - 重构降维**

```python
true_df = aggregate_windows_to_timeseries(y_true, y_time, feature_cols, first_step_only=True)
pred_df = aggregate_windows_to_timeseries(y_pred, y_time, feature_cols, first_step_only=True)
```

- 从 3D 张量生成连续时序 DataFrame
- 加载特征名，实现可读的列标签

#### **Part 3 - 事件队列组装**

```python
if EVENT_ID is not None:
    event_iter = [(EVENT_ID, ATTACK_EVENTS[EVENT_ID-1])]  # 单一事件
else:
    event_iter = [(i+1, ev) for i, ev in enumerate(ATTACK_EVENTS)]  # 全部16事件
```

#### **Part 4 - 批量渲染与指标收集**

```python
for event_id, (st, ed, attack_cols) in event_iter:
    effect = calc_event_effect(...)  # 评估波及
    top_nodes = [*attack_cols, *effect.index[:TOPK]]  # 汇总焦点节点
    fig, m_df = plot_event(...)  # 引擎渲染
    plt.show()  # 弹出窗口
    all_metrics.append(m_df)  # 累积指标
```

---

## 3. 必需输入文件

**位于 `RUN_DIR` 目录：**

- `y_true.npy`：测试集真实观测，形状 [N_test, Nodes, Horizon]
- `y_pred.npy`：测试集模型预测，形状 [N_test, Nodes, Horizon]

**位于 `DATA_PATH` 目录（通过 Parameters.json 或显式指定）：**

- `test.npz`：包含 `y_time` 字段（时间戳矩阵，形状 [N_test, Horizon]）
- `meta.pkl`（可选）：包含 `feature_cols` 列表的 pickle 文件

**配置文件（自动查找）：**

- `Parameters.json`：在 RUN_DIR 或其父目录，包含 `data_path` 字段

---

## 4. 输出文件

保存到 `SAVE_DIR` 目录（默认 `./Export/`）：

| 文件名                       | 格式 | 说明                               |
| ---------------------------- | ---- | ---------------------------------- |
| `event{:02d}_..._pvonly.png` | PNG  | 预览图，300dpi，适合屏幕展示       |
| `event{:02d}_..._pvonly.tif` | TIFF | 出版级无损图，300dpi，适合论文插图 |

**文件名规则：**

```
event{EVENT_ID:02d}_pre{PRE_MINUTES}m_post{POST_MINUTES}m_top{TOPK}_{pv_part}.{fmt}
例：event05_pre10m_post10m_top3_pvonly.png
```

若 `EVENT_ID=None`（绘全部事件），则生成 16 个文件。

---

## 5. 使用示例

### 基础用法（分析单一事件）

```bash
python visualize_attack_pred_vs_true.py
```

需提前编辑脚本中的配置单元（模块二），指定 `RUN_DIR` 与 `EVENT_ID`。

### 修改配置后重新运行

假设要分析第 3 个攻击事件，观察时间窗口各 20 分钟，显示 Top5 波及节点：

**在脚本中修改：**

```python
RUN_DIR = Path(r'E:\DataForCode\5_AnomalyDetection\WADI_STGNN_Output\20260426_160928\0')
EVENT_ID = 3          # 改为第3事件
PRE_MINUTES = 20      # 改为前20分钟
POST_MINUTES = 20     # 改为后20分钟
TOPK = 5              # 改为Top5节点
```

然后运行脚本，自动生成并弹出对比图。

### 全量可视化（所有 16 个攻击事件）

```python
EVENT_ID = None       # 为 None 时遍历所有
PRE_MINUTES = 15
POST_MINUTES = 15
SAVE_DIR = './Export_Full/'
```

会生成 16 个 PNG 和 16 个 TIF，每个对应一个攻击事件。

### 高级：仅关注设备状态节点

```python
PV_ONLY = False       # 包括所有节点，不只是 _PV 后缀
```

### 提高图表质量（论文用）

```python
SAVE_DIR = '/path/to/paper/figures/'
# 脚本已固定 dpi=300，TIF 格式可直接用于出版
```

---

## 6. 场景应用指南

### 场景 A：快速验证模型效果（开发阶段）

```python
EVENT_ID = 1          # 仅看第1个事件
PRE_MINUTES = 10
POST_MINUTES = 10
TOPK = 3
PV_ONLY = True        # 仅关键变量
```

**预期输出：**

- 单张对比图，快速判断预测的基本一致性
- 若预测曲线贴近真实，说明模型训练有效

### 场景 B：根因分析（诊断阶段）

```python
EVENT_ID = 8          # 重点审查第8个异常事件
PRE_MINUTES = 30      # 放宽时间窗口
POST_MINUTES = 30
TOPK = 10             # 显示更多波及节点
PV_ONLY = False       # 追踪异常如何在全网传播
```

**预期输出：**

- 观察直接受击节点 (attack_cols) 何时开始偏离
- 观察波及节点 (top_nodes) 的滞后时间与传播规律
- 根因分析：Z-score 最高的节点即根本故障源

### 场景 C：论文插图（出版阶段）

```python
EVENT_ID = 12         # 代表性事件（宜选有明显特征的事件）
PRE_MINUTES = 25
POST_MINUTES = 25
TOPK = 5
PV_ONLY = True
SAVE_DIR = '/path/to/paper/figures/'
# 注意脚本已固定 figsize=(15,10), dpi=300，满足出版要求
```

**生成的 TIF 文件：**

- 300dpi，无损压缩，可直接嵌入 LaTeX/Word 论文
- 自动调整字体大小与图例排版，适应排版布局

---

## 7. 常见问题与排查

| 问题                                        | 原因                 | 解决方案                                                       |
| ------------------------------------------- | -------------------- | -------------------------------------------------------------- |
| FileNotFoundError: Cannot find project root | 未找到 tools.py      | 检查 RUN_DIR 的上级目录是否包含 tools.py                       |
| FileNotFoundError: y_true.npy / y_pred.npy  | 预测文件缺失         | 确保模型推断已完成并保存 .npy 文件                             |
| FileNotFoundError: test.npz                 | 数据集不存在         | 检查 Parameters.json 中的 data_path，或显式设置 DATA_PATH 参数 |
| KeyError: 'y_time' not in test.npz          | 数据集缺少时间戳     | 重新生成 test.npz，确保包含 y_time 字段                        |
| ValueError: shape mismatch                  | 真实与预测维度不一致 | 检查模型输出形状是否与数据集节点数一致                         |
| ValueError: expected 3D arrays              | 输入维度错误         | y_true/y_pred 应为 [N, V, H]，不能是 2D 或 4D                  |
| no 'meta.pkl' found                         | 特征列名读取失败     | 脚本会自动回退为 node_0/node_1/..., 不影响功能                 |
| EVENT_ID out of range                       | 事件编号超界         | WADI 数据集共 16 个事件，EVENT_ID 应在 1-16 范围               |
| 输出图表为空白                              | 时间戳格式不兼容     | 检查 y_time 数据类型，确保是可转换为 Timestamp 的格式          |

---

## 8. 关键参数调优建议

### 时间窗口大小

- **PRE_MINUTES = 5-10**：快速扫描，突出攻击瞬间
- **PRE_MINUTES = 20-30**：详细分析，看长期趋势
- 默认 10 分钟平衡两端

### Top-K 选择

- **TOPK = 1-2**：仅焦点节点，简洁清晰
- **TOPK = 5-10**：完整传播链，根因分析
- 默认 3 为折中

### PV_ONLY 过滤

- **True**：工业优先，仅关键过程变量
- **False**：学术完整，包括所有传感器与开关量

### 色彩与字体

- 预定义 10 色循环调色板，支持自定义 RGB
- 字体大小已动态调整（开发 14pt，渲染 25pt），按需修改 `plt.rcParams['font.size']`

---

## 9. 依赖环境

```
numpy
pandas
matplotlib
```

可选（若需数据处理扩展）：

```
scikit-learn
```

---

## 10. 代码集成用例

#### 在其他脚本中调用可视化函数

```python
from visualize_attack_pred_vs_true import (
    load_feature_names, aggregate_windows_to_timeseries,
    calc_event_effect, plot_event
)

# 自定义组合
y_true = np.load('y_true.npy')
y_pred = np.load('y_pred.npy')
y_time = np.load('y_time.npy')

feature_cols = load_feature_names(Path('data'), y_true.shape[1])
true_df = aggregate_windows_to_timeseries(y_true, y_time, feature_cols)
pred_df = aggregate_windows_to_timeseries(y_pred, y_time, feature_cols)

# 对某个自定义事件时间段进行可视化
st = pd.Timestamp('2017-10-09 12:30:00')
ed = pd.Timestamp('2017-10-09 13:15:00')
fig, m_df = plot_event(
    event_idx_1based=99,
    st=st, ed=ed,
    attack_cols=['1_AIT_001_PV', '2_AIT_002_PV'],
    true_df=true_df,
    pred_df=pred_df,
    top_nodes=['1_AIT_001_PV', '2_AIT_002_PV', '3_FIT_003_PV'],
    pre=pd.Timedelta(minutes=10),
    post=pd.Timedelta(minutes=10)
)
plt.show()
```

#### 批量输出多个事件的 PNG/TIF

```python
for event_id in range(1, 17):
    # 编程方式动态修改全局配置
    EVENT_ID = event_id
    # 重新加载并执行 main() 逻辑
    # ...
```

---

## 11. 总结

`visualize_attack_pred_vs_true.py` 是攻击事件**对标分析工具**，特别适合：

- **快速验证**：通过直观图表判断模型预测质量
- **根因挖掘**：基于 Z-score 效应评分找出网络中的故障传播源和次生波及
- **论文呈现**：输出 300dpi 高清图表，支持 PNG 屏幕展示与 TIF 出版级插图
- **生产部署**：可视化工业异常与模型诊断结果，增强工程师信心

与 `anomaly_scoring_threshold_pa.py` 的协作：后者完成后处理（异常分值 → 二分类），本脚本则对二分类结果进行深度诊断和可视化论证。
