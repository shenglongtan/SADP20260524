# DownsampledVisualization.py 功能分析与使用说明

## 1. 文件简介

`DownsampledVisualization.py` 是用于 WADI 数据集中多尺度下采样攻击事件可视化的脚本。它主要展示不同采样频率下，测试集中攻击时间段的分布情况，并通过断轴方式压缩无攻击长时间段，以提高可读性。

## 2. 核心功能

- 读取原始 WADI 测试集数据
- 划分特征列、状态列与连续列
- 对测试集数据执行多尺度下采样
- 根据攻击事件时间段生成可视化片段
- 使用断轴布局展示不同采样频率下的攻击时间点
- 绘制统一图例，方便不同攻击事件对比

## 3. 代码结构说明

### 3.1 导入依赖

- `numpy`, `pandas`：数据处理
- `matplotlib.pyplot`, `matplotlib.dates`, `matplotlib.lines`：绘图与时间轴格式化
- `datetime.timedelta`：控制时间段前后扩展和合并逻辑
- `downsample_block`：自定义下采样函数
- `ATTACK_EVENTS_WADI as ATTACK_EVENTS`：WADI 预定义攻击事件时间段列表

### 3.2 数据加载与预处理

1. 从 `../../Data/WADI/data.pkl` 加载数据。
2. 打印数据形状与头部样本。
3. 识别特征列，排除 `Timestamp`、`Train`、`Attack` 标签列。
4. 将包含 `STATUS` 的列视为状态列，其余视为连续列。
5. 仅保留 `Train==0` 的测试集数据。

### 3.3 下采样准备

- 定义 `DOWNSAMPLE_LIST = [10, 30, 60, 120]`，表示秒级下采样间隔。
- 通过 `downsample_block()` 生成不同采样尺度的数据。
- `df_test_dict` 保存原始数据和各尺度下采样后的测试集。

### 3.4 断轴可视化逻辑

- 定义 `PAD_BEFORE`、`PAD_AFTER`：在每个攻击段前后保留额外正常区间。
- 定义 `MERGE_GAP`：将相邻时间段合并，避免图段过于碎片化。
- 对攻击事件时间段进行合并得到最终展示片段 `segments`。
- `y_levels` 映射各采样尺度在图中的垂直位置。

### 3.5 绘图布局与样式

- 设置全局字体为 `Times New Roman` 和中文后备 `SimSun`。
- 采用 `plt.subplots()` 创建按攻击段数量的子图窗口，并共享 y 轴。
- 每个子图展示一个时间段内的攻击点，横向断轴排列。

### 3.6 绘制攻击点

- 遍历每个展示片段和每个攻击事件。
- 仅绘制与当前片段有交集的攻击时间点。
- 对不同采样尺度使用不同 y 轴高度，并对原始数据点绘制较小点。
- 使用 `attack_colors` 为每个攻击事件统一配色。

### 3.7 断轴装饰与图例

- 通过 `ax.plot()` 在子图之间绘制斜线 `//`，强化断轴视觉效果。
- 在画布底部添加统一图例，展示攻击事件编号。

## 4. 使用说明

### 4.1 运行前准备

- 确保 Python 环境已安装：
  - `numpy`
  - `pandas`
  - `matplotlib`
- 确保脚本能访问 `../../Data/WADI/data.pkl` 数据文件。
- 确保 `tools.py` 中定义了 `downsample_block` 和 `ATTACK_EVENTS_WADI`。

### 4.2 运行方式

```bash
python DownsampledVisualization.py
```

运行后将直接弹出图形窗口，并显示按不同下采样尺度展示的攻击事件分布。

### 4.3 结果说明

- 图中的每一行代表一种采样频率：
  - 最底层 `orig` 表示原始数据频率
  - 其余分别表示 `10s`, `30s`, `60s`, `120s` 下采样
- 每个绘制点表示该时间点属于攻击事件
- 断轴结构使得无攻击时间段被压缩，仅保留与攻击相关的时间片段

## 5. 常见修改建议

- 修改 `DOWNSAMPLE_LIST`：调整展示的下采样尺度。
- 修改 `PAD_BEFORE` / `PAD_AFTER`：控制每个攻击事件前后保留的正常时间长度。
- 修改 `MERGE_GAP`：控制合并片段的紧密程度。
- 如需保存图像，可在 `plt.show()` 之前添加：

```python
fig.savefig('DownsampledVisualization.png', dpi=300, bbox_inches='tight')
```

## 6. 注意事项

- `ATTACK_EVENTS` 列表必须包含按时间顺序的攻击时间段。
- 若原始测试数据中时间戳格式不一致，可能导致数据过滤失败。
- 若 `Attack` 列没有明确标记 `1` 的攻击点，图中将无法显示攻击事件。
- 如果画布中子图过多，可适当增加 `figsize` 或减少展示片段数量。

## 7. 适用场景

- 分析 WADI 数据集中不同采样频率下攻击事件的时间分布差异
- 为下采样实验结果制作可视化对比图
- 为论文或报告准备多尺度攻击时间序列图示
