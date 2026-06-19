# 人工 + AI 联合逐步复核路线

日期：2026-06-19

本文档用于指导后续逐步检查程序功能与数学算法。建议每次只检查一个阶段：你在 VSCode 中打开对应文件，我负责解释代码、核对张量形状、指出数学含义和潜在风险，并在需要时直接修改程序与记录修订日志。

## 阶段 0：环境与运行入口确认

目标：确认使用同一个 Python、同一个工作目录、同一套保存结构。

人工检查：

- 在 VSCode 中确认当前文件夹为 `SADP20260524`。
- 确认运行入口为 `train_attention.py`。
- 确认实验输出进入 `Save/Experiments/`。

AI 辅助：

- 检查 `requirements.txt` 与当前 Python 环境。
- 检查 torch 是否能稳定 import、是否能完成最小前向测试。
- 记录环境异常，例如 Python/Torch 运行时退出。

通过标准：

- `py_compile` 通过。
- 最小 MTGNN 前向输出形状为 `[B, 1, N, H]`。
- 最小 AE 联合训练一步能返回有限损失。

## 阶段 1：数据协议与标签语义

目标：确认模型没有读入攻击标签或划分标签，且训练/验证/测试划分符合异常检测科学性。

重点文件：

- `data_maker.py`
- `data_loader.py`
- `tools.py::CustomScaler`

人工检查：

- 查看原始数据列名，确认 `Attack`、`Train` 的含义。
- 确认哪些列是物理传感器特征。
- 确认验证集是否应该完全正常。

AI 辅助：

- 逐行解释 `_separate_features_and_metadata()`。
- 逐行解释 `_split_dataframe()`。
- 统计 `y_attack_window` 与 `y_attack_point` 的正例数量。
- 检查 `CustomScaler` 是否只在训练段拟合。

通过标准：

- `Attack/Train` 不进入 `feature_cols`。
- scaler 只由训练集物理特征计算。
- 训练、验证、测试样本数和攻击标签数量清楚可解释。

## 阶段 2：滑动窗口与张量形状

目标：确认每个样本的历史窗口、预测窗口、标签对齐方式正确。

重点文件：

- `data_maker.py::create_input_data`
- `data_loader.py::DataLoader`
- `train_attention.py` 中 tensor transpose 部分

人工检查：

- 确认 `window_size`、`horizon_size`、`sliding_step` 是论文实验需要的设置。
- 确认 `y_true/y_pred` 表示传感器数值，不是攻击标签。

AI 辅助：

- 用一个小数组手工演示 `x_offset` 和 `y_offset`。
- 画出 `[Samples, Window, Nodes, Features] -> [Batch, Features, Nodes, Window]` 的转换。
- 检查 padding 后最终保存时是否已裁掉伪样本。

通过标准：

- 任意样本 `X` 覆盖 `t-L+1:t`，`Y` 覆盖 `t+1:t+H`。
- 标签 `y_attack_window` 和模型输出样本数量一致。

## 阶段 3：动态图构建数学检查

目标：确认 `GetAttMatrix` 的输入、输出、邻接矩阵含义与 MTGNN 使用方式一致。

重点文件：

- `attention_matrix.py::GetAttMatrix`
- `GNN/model_attention.py::forward`

人工检查：

- 明确动态图是否只基于第 0 通道。
- 确认论文中是否要声称动态图使用所有通道。

AI 辅助：

- 分解 GRU 输入 `[N, B, L]` 的意义。
- 解释 attention 矩阵 `[N, N]` 如何进入 MixHop。
- 检查对称化、softmax、dropout 对动态图含义的影响。

通过标准：

- `attention_input` 形状和 `GetAttMatrix` 期望完全一致。
- 邻接矩阵维度为 `[Nodes, Nodes]`。
- 构图口径在论文中可被准确描述。

## 阶段 4：MTGNN 主预测分支

目标：确认时间卷积、图传播、残差、skip 和输出层维度完全闭合。

重点文件：

- `GNN/model_attention.py`

人工检查：

- 确认 `conv_channels/residual_channels/skip_channels/end_channels` 取值。
- 确认是否需要子图训练 `split_num`。

AI 辅助：

- 跟踪每层输入输出张量维度。
- 检查感受野 `receptive_field` 与 `window_size` 的关系。
- 增加必要参数校验，例如 `conv_channels % 4 == 0`。

通过标准：

- 前向输出为 `[B, 1, N, H]`。
- 动态图与当前节点子集维度一致。
- 参数取值不会引发隐藏维度不匹配。

## 阶段 5：AE 分支与联合损失

目标：确认 AE 与 MTGNN 的并列关系、联合训练方式、重构目标符合你的算法定义。

重点文件：

- `GNN/ae.py`
- `GNN/trainer_attention.py`
- `train_attention.py::collect_module_residual_outputs`

人工检查：

- 决定 AE 重构损失是否应覆盖完整窗口还是仅未来段。
- 决定 AE 是否保持节点独立时域重构，还是后续加入空间重构。

AI 辅助：

- 对照公式 `L_total = L_pred + beta*(L_rec_real + lambda*L_rec_pred)`。
- 分析 `true_win=[history,true_future]` 和 `pred_win=[history,pred_future]`。
- 核对训练损失与最终残差导出的差异。

通过标准：

- `use_ae=True` 是默认算法主设定。
- 最佳模型选择依据 `L_total`。
- AE 权重与 MTGNN 权重同时保存、同时回载。
- 训练公式与异常评分公式在论文中口径一致。

## 阶段 6：检查点、预测与残差落盘

目标：确认最终测试使用的是验证集最佳权重，且保存文件足够支撑后处理和消融实验。

重点文件：

- `train_attention.py`
- `Save/README.md`

人工检查：

- 确认实验目录命名是否便于论文实验管理。
- 确认需要保留哪些残差文件用于后续消融。

AI 辅助：

- 检查 `model.pth`、`ae_model.pth`、`joint_checkpoint.pth`。
- 检查 `predictions/` 下所有 `.npy` 的形状和语义。
- 检查 `residual_meta.json` 是否足够解释标签和数值文件。

通过标准：

- 最终验证/测试前已回载最佳 MTGNN 和 AE。
- `joint_error`、MTGNN 残差、AE 残差均可独立读取。

## 阶段 7：异常评分与评价协议

目标：确认异常分数、阈值、PA 和真实标签评价符合工业异常检测论文规范。

重点文件：

- `Test/anomaly_scoring_threshold_pa.py`
- `Test/visualize_attack_pred_vs_true.py`

人工检查：

- 决定默认评分使用 `joint_error` 还是某种消融残差。
- 决定窗口级标签还是点级标签作为主表指标。
- 决定是否保留 PA 指标，以及是否同时报告 raw 指标。

AI 辅助：

- 解释验证集定标与阈值选择为何不泄漏测试集。
- 检查 `horizon_reduce`、`var_reduce` 的数学含义。
- 检查 PA 是否符合目标期刊对异常检测评价的常见要求。

通过标准：

- 主实验默认 `score-source=auto` 并选择 `joint_error`。
- 阈值只由验证集决定。
- 评价标签语义清晰，不混淆传感器值与攻击标签。

## 阶段 8：消融实验设计

目标：为论文实验表格预留可解释的消融路线。

建议消融：

- MTGNN-only：关闭 AE 或后处理使用 `--score-source mtgnn`。
- Joint：默认完整模型。
- AE-only：后续需要扩展 `--score-source ae_real/ae_pred/ae_combined`。
- 无动态图：后续可设计静态图或全连接/单位图对照。
- 不同残差聚合：`max`、`mean`、`topk_mean`、`p95`。

通过标准：

- 每个消融只改变一个核心因素。
- 每个实验保存路径、参数、残差、指标都可追溯。

## 建议执行顺序

下一轮建议从阶段 1 开始，而不是直接看模型。原因是只要数据协议没有完全确认，后续所有模型性能解释都可能失去意义。

推荐下一步：

```text
阶段 1：数据协议与标签语义
```

我们可以从 `data_maker.py` 的 `_separate_features_and_metadata()` 和 `_split_dataframe()` 开始，你在 VSCode 打开对应代码，我逐行解释其数学和实验含义，并同步检查真实数据生成结果。

