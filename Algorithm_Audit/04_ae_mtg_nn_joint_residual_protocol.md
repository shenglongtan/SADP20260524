# AE 与 MTGNN 并列模块及综合残差协议

日期：2026-06-19

本文档用于固定当前 SADP 模型中 MTGNN 预测模块、AE 重构模块和最终异常评分阶段的术语与文件协议，避免后续把“传感器预测值”“攻击标签”“预测残差”“重构残差”混用。

## 1. 模块关系

当前算法中，AE 模块与 MTGNN 模块是并列的两个独立模块。

- MTGNN 模块：学习多变量工业传感器时间序列的时空依赖，并输出未来传感器预测值。
- AE 模块：对由历史窗口与未来窗口拼接得到的时间窗口进行重构，用重构误差补充异常判别信息。
- 训练阶段：二者通过联合损失共同优化。
- 测试与后处理阶段：二者分别导出模块级残差，再组合为综合残差。

## 2. 训练目标

启用 AE 时，最佳模型选择依据为：

```text
L_total = L_pred + beta * (L_rec_real + lambda * L_rec_pred)
```

其中：

- `L_pred`：MTGNN 预测损失，当前实现使用 masked MAE。
- `L_rec_real`：AE 对真实窗口的重构损失，当前实现使用 MSE。
- `L_rec_pred`：AE 对 MTGNN 生成窗口的重构损失，当前实现使用 MSE。
- `beta`：AE 重构项总权重，对应命令行参数 `--ae_beta`。
- `lambda`：生成窗口重构项权重，对应命令行参数 `--ae_lambda`。

当显式设置 `--use_ae false` 时，程序进入消融模式：

```text
L_total = L_pred
```

此时最佳模型选择、最终残差导出和后处理评分均退化为 MTGNN 预测误差。

## 3. 术语定义

必须严格区分以下概念：

- `y_true`：真实传感器数值，不是攻击标签。
- `y_pred`：MTGNN 预测传感器数值，不是攻击标签。
- `y_attack_window`：窗口级真实攻击标签，来自数据集。
- `y_attack_point`：点级真实攻击标签，来自数据集。
- `pred_test_raw`：后处理脚本基于异常分数和阈值得到的原始攻击预测标签。
- `pred_test_pa`：后处理脚本经过 Point Adjustment 后得到的攻击预测标签。

## 4. 残差导出协议

每个 `run_xx/predictions/` 目录下保存：

```text
y_val_true.npy
y_val_pred.npy
y_true.npy
y_pred.npy
val_mtgnn_pred_error.npy
test_mtgnn_pred_error.npy
val_mtgnn_pred_error_physical.npy
test_mtgnn_pred_error_physical.npy
val_ae_rec_real_error.npy
test_ae_rec_real_error.npy
val_ae_rec_pred_error.npy
test_ae_rec_pred_error.npy
val_joint_error.npy
test_joint_error.npy
residual_meta.json
```

含义：

- `*_mtgnn_pred_error.npy`：标准化空间中的 MTGNN 预测绝对误差。
- `*_mtgnn_pred_error_physical.npy`：物理量纲空间中的 MTGNN 预测绝对误差，用于工程解释，不作为默认综合评分输入。
- `*_ae_rec_real_error.npy`：标准化空间中真实窗口未来段的 AE 平方重构误差。
- `*_ae_rec_pred_error.npy`：标准化空间中 MTGNN 生成窗口未来段的 AE 平方重构误差。
- `*_joint_error.npy`：综合残差，默认用于后处理异常评分。

综合残差定义为：

```text
joint_error = mtgnn_pred_error + beta * (ae_rec_real_error + lambda * ae_rec_pred_error)
```

若 AE 关闭，则：

```text
joint_error = mtgnn_pred_error
```

## 5. 后处理评分协议

`Test/anomaly_scoring_threshold_pa.py` 新增 `--score-source`：

```text
auto    : 默认值，优先使用 joint_error；若旧实验没有该文件，则回退到 MTGNN 预测误差。
joint   : 强制使用 joint_error，缺失文件时报错。
mtgnn   : 使用 MTGNN 预测误差，用于消融对照。
ae_real : 使用 AE 对真实窗口未来段的重构残差。
ae_pred : 使用 AE 对 MTGNN 生成窗口未来段的重构残差。
ae_sum  : 使用 ae_rec_real_error + ae_sum_lambda * ae_rec_pred_error。
```

后处理仍只在验证集残差上估计归一化统计量与阈值，测试集只用于最终评分，避免测试集信息泄漏。

## 6. 论文实验影响

这次修订使主实验默认体现“MTGNN 预测 + AE 重构”的联合异常检测逻辑。未来消融实验可以至少设计三组：

- MTGNN-only：`--use_ae false` 或后处理 `--score-source mtgnn`。
- Joint：默认配置，训练和评分均使用 AE。
- AE residual ablation：后处理阶段使用 `--score-source ae_real`、`--score-source ae_pred` 或 `--score-source ae_sum` 单独考察 AE 重构残差的异常区分能力。

当前代码已经支持上述三类实验。需要注意，AE-only 后处理只是分析 AE 残差是否能补足预测分支漏检事件；它不代表主算法训练阶段关闭 MTGNN。
