# advanced_anomaly_detection.py 功能分析与使用说明

## 1. 文件概述

`advanced_anomaly_detection.py` 实现了一个完整的“异常检测后处理与评估管道”（AnomalyDetectionPipeline）。该脚本用于：

- 从磁盘加载模型输出（y_true.npy / y_pred.npy）与模型参数（Parameters.json）
- 计算基于重构误差的样本级异常分数（score）
- 自动/半自动选择判定阈值（支持有监督 F1 优化与无监督百分位/3-sigma 策略）
- 将阈值转化为二分类预测并应用时域“点调整（Point Adjustment, PA）”容错策略
- 统计并报告分类性能、变量贡献度与异常片段（segment）聚类
- 绘制三段式对比图（分值曲线、判定前后覆盖对比），并将结果落盘（NumPy/CSV）

该脚本既可以作为命令行工具运行，也可以作为库在上层程序中按需调用 `AnomalyDetectionPipeline` 类的 API。

---

## 2. 主要功能与方法（类接口）

`AnomalyDetectionPipeline(save_folder: Path, label_path: Optional[Path]=None)`

- 初始化：指定包含预测文件与参数文件的 `save_folder`，可选 `label_path` 用于有监督阈值选择与评估。
- 内部会自动调用：`_load_predictions()`、`_load_parameters()`、`_load_labels()`。

核心公开方法：

- `compute_anomaly_scores(horizon_reduce='max', var_reduce='max') -> np.ndarray`
  - 输入：无（使用已加载的 `y_true` 与 `y_pred`）
  - 输出：一维异常分数 `scores`（长度 = 样本数 N）
  - 过程：计算绝对残差 → 时间维降维（max/mean）→ 对每个变量做 z-score 归一化 → 变量维降维（max/mean）

- `select_threshold(method='auto', percentile=99.0, grid_size=200) -> (threshold, metrics)`
  - 支持策略：
    - `auto`：若提供 `labels_df`（有标签），使用 F1 网格搜索优化阈值；否则退回 `percentile`。
    - `percentile`：按分位数切分（无监督）
    - `mean_std`：均值 + k\*std（脚本中默认 k=3）
  - 返回：阈值与本次选择的统计信息/指标字典。

- `apply_point_adjustment(pred_labels, delay=None) -> adjusted_labels`
  - 点调整（PA）：对每个预测点在左右 `delay` 步内若有任一命中则将整个区间标为异常（1），用于工业场景的容错判定。
  - `delay` 默认为 `horizon_size - 1`。

- `cluster_anomaly_segments(pred_labels) -> List[(start, end)]`
  - 将离散异常点合并为连续片段（左闭右开），便于后续事件检索与可视化。

- `analyze_variable_contribution() -> pd.DataFrame`
  - 根据归一化残差矩阵，统计每个变量作为 "最大贡献者" 的次数及百分比，用于根因定位。

- `generate_evaluation_report(pred_labels_before, pred_labels_after) -> pd.DataFrame`
  - 在有标签时输出点调整前后的精确率/召回/F1/ROC-AUC/PR-AUC/TP/FP/FN 等汇总表格，并返回 DataFrame 便于保存或进一步处理。

- `visualize_comparison(pred_labels_before, pred_labels_after, figsize=(18,10)) -> matplotlib.figure.Figure`
  - 绘制三行图：分值与阈值、调整前的判定覆盖（与真实标签叠加）、调整后的判定覆盖（与真实标签叠加）。返回 Figure 对象。

- `save_results(output_prefix='advanced_')`
  - 将 `scores`, `label_before`, `label_after` 保存为 `.npy` 和一份汇总 `.csv`。

- `run_pipeline(threshold_method='auto', output_prefix='advanced_') -> dict`
  - 门面函数：按顺序调用计算分值→选择阈值→生成原始标签→应用 PA→根因分析→聚类分段→生成报告→绘图→保存，并返回结果字典。

---

## 3. 所需输入文件与输出文件

必备输入（位于 `save_folder`）：

- `y_true.npy`：numpy 数组，形状 `[N, V, H]`（真实观测/标签张量）
- `y_pred.npy`：numpy 数组，形状 `[N, V, H]`（模型预测张量）
- `Parameters.json`：包含 `window_size` 等超参数的 JSON 文件

可选输入：

- `label_path`（如 `data_10s.pkl`）：用于有监督阈值优化与评估的标签表（Pandas DataFrame 或一维 numpy 数组）

主要输出（保存到 `save_folder`）：

- `advanced_anomaly_score.npy`
- `advanced_anomaly_label_before_adjustment.npy`
- `advanced_anomaly_label_after_adjustment.npy`
- `advanced_anomaly_results.csv`（样本级汇总，包含 score 与两路标签）
- 可选：生成的 Figure 对象（脚本中返回，但未直接保存为图片，用户可根据需要保存）

---

## 4. 运行方法（示例）

1. 以脚本方式运行（默认示例）：

```bash
python advanced_anomaly_detection.py
```

脚本默认在 `main()` 中使用示例路径：

- `save_folder = Path(r'E:\PyDeepLearningLab\Code\07_RA-STGNN\Test')`
- `label_path = Path(r'E:\PyDeepLearningLab\Code\07_RA-STGNN\Data\WADI\data_10s.pkl')`

请根据本机路径修改为实际位置。

2. 在 Python 中按 API 调用：

```python
from pathlib import Path
from Test.advanced_anomaly_detection import AnomalyDetectionPipeline

pipeline = AnomalyDetectionPipeline(save_folder=Path('path/to/results'), label_path=Path('path/to/labels.pkl'))
res = pipeline.run_pipeline(threshold_method='auto', output_prefix='exp1_')
# res 是包含 scores、threshold、labels_before/after、segments、report、figure 的字典
```

---

## 5. 配置建议与参数说明

- `compute_anomaly_scores`：
  - `horizon_reduce`: 'max'（更偏重峰值异常）或 'mean'（更偏重持续偏差）。通常对突发异常选择 'max'。
  - `var_reduce`: 'max' 或 'mean'，'max' 强调最坏传感器（适合安全场景）。

- `select_threshold`：
  - 当有可靠标签时，使用 `method='auto'` 会自动进入 F1 网格搜索优化。
  - 无标签时，`method='percentile'`（如 99%）或 `method='mean_std'`（3-sigma）是常用无监督策略。

- `apply_point_adjustment`：
  - `delay` 设置为 `horizon_size - 1` 是合理默认，若需要更严格或更宽松的匹配，可调整为更小或更大整数。

---

## 6. 依赖环境

建议 Python 环境安装：

- numpy
- pandas
- matplotlib
- scikit-learn

可用 requirements.txt（示例）：

```
numpy
pandas
matplotlib
scikit-learn
```

---

## 7. 常见问题与排查

- 错误：FileNotFoundError: Missing prediction files
  - 检查 `save_folder` 路径是否正确，确认 `y_true.npy` 与 `y_pred.npy` 是否存在且形状一致。

- 错误：Label length mismatch
  - 提示标签文件与预测样本数量不匹配，检查 `labels_df` 的行数与 `y_true` 的样本数 N 是否一致或可截断对齐。

- 指标异常（ROC/PR 为 0 或报错）
  - 说明真实标签中异常或正常类为单一类别，需使用其他评价方式或确保样本含有至少两类标签。

---

## 8. 扩展建议

- 自动保存 Figure：在 `run_pipeline` 中增加 `fig.savefig(...)` 以便产出固定的图片文件。
- 命令行参数：使用 `argparse` 支持用户指定 `save_folder`、`label_path`、`threshold_method` 与 `output_prefix`。
- 指标时间序列：将 `norm_var_error` 等中间结果保存为 `.npz` 便于后续根因分析和可视化交叉比对。

---

## 9. 我可以为你做的下一步

- 将 `run_pipeline` 的默认路径改为相对路径并添加 `argparse` 支持（便于在任意目录运行）。
- 将最终 Figure 自动保存为高分辨率 PNG/TIFF 并将报告输出到 `report.xlsx`。

如果需要我现在替你实现其中一项，请告诉我首选项。
