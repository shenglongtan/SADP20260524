# 算法完整性审查底稿

日期：2026-06-19

本文档记录当前 `SADP20260524` 项目的算法级完整性审查结果。审查目标不是再次列出代码功能，而是确认数据协议、模型结构、联合损失、异常分数和后处理评估之间是否形成闭环，并标出后续需要人工与 AI 共同确认的数学口径。

## 0. 已确认的算法口径

根据用户在 2026-06-19 的确认，当前算法口径固定如下：

1. AE 重构损失与异常评分的窗口口径保持当前实现：训练阶段 AE 重构 `history + future` 完整窗口，最终异常评分只取未来预测段。
2. 动态图由原始传感器数据生成。当前实现使用第 0 个核心传感器数值通道构图；`Attack/Train` 元数据不进入构图或预测。
3. AE 只承担节点独立的时域重构，不负责节点间空间混合。
4. 最终评价目标希望回到原始时间序列点级；当前窗口级评价后续需要扩展点级反投影评价。
5. 后续消融实验不需要 AE-only 评分模式，因为 MTGNN 是基础分支；保留 MTGNN-only 与 MTGNN+AE joint 对照即可。

## 1. 当前主算法链路

当前推荐主链路为：

```text
data_maker.py
  -> data_loader.py
  -> train_attention.py
       -> GNN/model_attention.py
       -> attention_matrix.py
       -> GNN/ae.py
       -> GNN/trainer_attention.py
       -> tools.py::test_model
  -> Test/anomaly_scoring_threshold_pa.py
```

其中：

- `GNN/model_attention.py` 是当前实际训练使用的 MTGNN 主模型。
- `GNN/model.py` 是旧版/备用模型，不在当前 `train_attention.py` 主链路中直接使用。
- `Test/anomaly_scoring_threshold_pa.py` 是当前推荐后处理入口。
- `Test/advanced_anomaly_detection.py` 仍按旧保存结构和 `y_true/y_pred` 预测残差工作，尚未升级到 AE+MTGNN 综合残差协议，不建议作为当前主实验评价入口。

## 2. 数据协议完整性

当前数据生成协议已经具备异常检测实验所需的基本科学性：

- `Attack` 和 `Train` 会作为元数据列排除，不进入模型输入特征。
- 标准化器 `CustomScaler` 只在训练段物理传感器矩阵上拟合，避免验证集/测试集分布污染。
- 若存在 `Train` 列，优先采用 `Train==1` 作为训练/验证候选段、`Train==0` 作为测试段。
- 若不存在可用 `Train` 列，则回退为时间顺序 `70%/10%/20%` 划分。
- 滑动窗口额外保存 `y_attack_window`、`y_attack_point`、`y_time`，后处理阶段可以读取真实攻击标签。

主张量约定为：

```text
DataMaker 输出 X: [Samples, Window, Nodes, Features]
DataMaker 输出 Y: [Samples, Horizon, Nodes, Features]
训练入口 input_x: [Batch, Features, Nodes, Window]
训练目标 true_y: [Batch, 1, Nodes, Horizon]
MTGNN 输出 pred_y: [Batch, 1, Nodes, Horizon]
后处理残差: [Samples, Nodes, Horizon]
```

需要注意：当前预测目标固定为第 0 个特征通道。若启用 `hour_as_feature` 或 `week_as_feature`，这些时间特征会进入 MTGNN 的 `start_conv`，但不会作为动态图构图输入，也不会作为 AE 重构目标。

## 3. 模型结构与数学目标

### 3.1 MTGNN 预测分支

MTGNN 分支实现的是多变量传感器序列的多步预测：

```text
X_{t-L+1:t} -> \hat{X}_{t+1:t+H}
```

主要结构为：

- `GetAttMatrix` 从输入窗口第 0 通道中学习当前 batch 的动态图邻接矩阵。
- `DilatedInceptionLayer` 提取多尺度时间特征。
- `MixHopLayer` 使用动态图及其转置图执行双向多跳空间传播。
- 输出层生成未来 `H` 步预测。

当前动态图构图只使用目标通道：

```python
attention_input = input_m[:, 0, :, :].permute(0, 2, 1)
```

这是可以成立的设计选择，但论文中应表述为“基于目标传感器历史轨迹的动态功能依赖构图”，不能宣称动态图显式融合了所有辅助通道。

### 3.2 AE 重构分支

AE 分支是与 MTGNN 并列训练的时间窗口重构模块：

```text
WindowAE: [B, 1, N, L+H] -> [B, 1, N, L+H]
```

当前 AE 使用 `Conv2d(kernel_size=(1,3))`，只在时间维卷积，节点维卷积核为 1。因此它主要学习每个节点自身时间窗口的重构规律，并不直接混合节点间空间信息。若论文中称 AE 捕获“spatio-temporal reconstruction”，需要谨慎；更准确的表述是“temporal window reconstruction branch over node-wise target-channel trajectories”。

### 3.3 联合损失

当前训练实现为：

```text
L_total = L_pred + beta * (L_rec_real + lambda * L_rec_pred)
```

其中：

- `L_pred`：MTGNN 多步预测损失，当前为 `masked_mae`。
- `L_rec_real`：AE 对真实窗口 `[history, future_true]` 的 MSE 重构误差。
- `L_rec_pred`：AE 对生成窗口 `[history, future_pred]` 的 MSE 重构误差。
- 启用课程学习时，`L_pred` 可能只覆盖当前解锁的部分预测步；但 AE 重构损失仍覆盖完整 `[history + future]` 窗口。

这是当前最重要的数学口径之一：训练阶段 AE 损失覆盖完整窗口，而最终异常残差导出阶段只取未来预测段的 AE 重构误差。该设计可以成立，因为历史段用于学习正常动态背景，未来段用于检测当前预测窗口异常；但后续论文公式和代码说明必须明确。

## 4. 异常分数闭环

当前最终测试阶段会导出：

```text
y_val_true.npy / y_val_pred.npy
y_true.npy / y_pred.npy
val_mtgnn_pred_error.npy / test_mtgnn_pred_error.npy
val_ae_rec_real_error.npy / test_ae_rec_real_error.npy
val_ae_rec_pred_error.npy / test_ae_rec_pred_error.npy
val_joint_error.npy / test_joint_error.npy
residual_meta.json
```

其中 `y_true/y_pred` 表示真实/预测的传感器数值，不是攻击标签。攻击标签来自数据集中的 `y_attack_window` 或 `y_attack_point`。

综合残差为：

```text
joint_error = mtgnn_pred_error
              + beta * (ae_rec_real_error + lambda * ae_rec_pred_error)
```

后处理脚本默认 `--score-source auto`，优先使用 `joint_error`，否则回退到 MTGNN 预测误差。异常分数生成过程为：

```text
[Samples, Nodes, Horizon]
  -> 按 horizon 降维
  -> 用验证集统计量归一化每个节点残差
  -> 按节点维聚合为 [Samples]
  -> 用验证集分数选择阈值
  -> 测试集二分类
  -> Point Adjustment
```

这一闭环目前是完整的。

## 5. 已通过的基础检查

- 主路径文件 `py_compile` 已通过：
  - `tools.py`
  - `data_maker.py`
  - `data_loader.py`
  - `attention_matrix.py`
  - `train_attention.py`
  - `GNN/model_attention.py`
  - `GNN/trainer_attention.py`
  - `GNN/ae.py`
  - `Test/anomaly_scoring_threshold_pa.py`
- 之前已验证默认参数中 `use_ae=True`、`save_pred_result=True`。
- 之前已验证综合残差文件导出与后处理 `joint_error` 自动选择逻辑。

本轮尝试执行一个 CPU 小型 MTGNN+AE 前向与训练冒烟测试时，本机 Python/Torch 运行时发生异常退出，未得到可靠数值结果。该问题先记录为环境验证风险，不直接等同于算法代码错误。

## 6. 需要后续确认或修复的完整性问题

### C1. AE 训练重构窗口与异常评分窗口口径不同

训练阶段：

```text
L_rec_real, L_rec_pred 覆盖 history + future 完整窗口
```

残差导出阶段：

```text
ae_rec_real_error, ae_rec_pred_error 只取 future 段
```

用户已确认维持当前实现。论文中建议表述为：AE 使用完整窗口学习正常动态背景与生成窗口一致性，异常检测阶段只对未来预测段计分，因为该段才是当前样本的检测目标。

### C2. 课程学习与 AE 损失覆盖范围不同

开启课程学习时，`L_pred` 可能只训练前若干个预测步，但 AE 仍重构完整未来段。这会使早期训练阶段 AE 分支比预测分支更早接触完整 horizon。该设计可能增强稳定性，也可能改变联合目标权重，需要后续实验确认。

### C3. 动态图只使用第 0 通道

用户已确认动态图由原始传感器数据生成。当前实现使用第 0 个核心传感器数值通道，不使用时间辅助特征，也不使用 `Attack/Train` 元数据。该口径应在论文中明确为“基于原始目标传感器历史轨迹的动态图构建”。

### C4. AE 不做节点间空间混合

用户已确认 AE 只承担节点独立时域重构。当前实现合理，论文中不应把 AE 分支表述为负责跨节点空间重构；跨节点依赖由 MTGNN 动态图预测分支承担。

### C5. `conv_channels` 必须能被 4 整除

`DilatedInceptionLayer` 将 `conv_channels` 均分给 4 个时间卷积分支。默认 `32` 没问题，但如果用户改成不能被 4 整除的数，实际拼接通道数会小于期望值，后续 `1x1` 卷积可能维度不匹配。建议后续增加参数校验。

### C6. `split_num` 需要约束不超过 `node_num`

若 `split_num > node_num`，则 `sub_num = int(node_num / split_num)` 可能为 0，导致前面若干子图为空。默认 `split_num=1` 没问题，但建议后续增加参数校验。

### C7. `CustomScaler.inverse_transform` 的 NumPy 3D 广播存在潜在风险

当前主链路中 3D 反归一化主要走 torch 分支或 `_inverse_transform_3d`，因此不直接触发。但 `CustomScaler.inverse_transform(np.ndarray)` 对 `[Samples, Nodes, Horizon]` 形状的数组直接执行：

```python
data * self.std + self.mean
```

当 `Horizon != Nodes` 时，NumPy 广播可能沿最后一维错误匹配。建议后续修复为显式按节点维广播。

### C8. 后处理当前使用窗口级标签

`anomaly_scoring_threshold_pa.py` 当前评估时使用 `y_attack_window`，属于窗口级评价。用户已确认最终希望回到原始时间序列点级评价。当前 `data_maker.py` 已保存 `y_attack_point` 和 `y_time`，具备从窗口预测结果反投影回原始时间点的基础；后续需要扩展后处理脚本。

### C9. 旧后处理脚本与当前主协议不一致

`advanced_anomaly_detection.py` 当前更像旧版扩展分析脚本：

- 依赖旧保存结构。
- 主要从 `y_true/y_pred` 计算误差。
- 未默认使用 AE 综合残差。

建议暂时标记为 legacy，不参与主实验结论，除非后续专门升级。

### C10. 验证集是否绝对无攻击需要数据层确认

当前阈值和归一化统计量来自验证集，这避免了测试泄漏。但如果验证集中含有攻击段，则阈值会受到异常污染。对于 SWaT/WADI 常见协议，`Train==1` 通常应为正常训练段；但仍建议在真实数据上统计 `val_y_attack_window.sum()`，确认验证集攻击比例。

## 7. 当前综合判断

从程序完整性看，当前主算法已经形成了可解释闭环：

```text
正常训练段拟合标准化
  -> MTGNN 学习动态图多步预测
  -> AE 并列学习真实/生成窗口重构
  -> L_total 选择最佳联合模型
  -> 回载 MTGNN 与 AE 最优权重
  -> 导出 MTGNN 残差、AE 残差、综合残差
  -> 验证集定标与阈值
  -> 测试集异常判别与 PA 评价
```

但从 SCI 论文级严谨性看，当前仍需重点确认：

1. AE 重构损失与异常评分的窗口口径。
2. 动态图是否只由目标通道构建。
3. AE 是否只承担节点独立时域重构。
4. 最终评价采用窗口级还是点级攻击标签。
5. 点级反投影评价如何在后处理脚本中实现。
