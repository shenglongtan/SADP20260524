# Save Directory Structure

本目录用于保存由代码生成的派生数据、训练实验结果和后处理结果。原始数据仍建议保留在 `Data/` 下，不与派生产物混放。

## Datasets

`Save/Datasets/` 用于保存 `DataMaker(save=True)` 生成的预处理数据包。

推荐命名格式：

```text
{dataset}_w{window_size}_h{horizon_size}_s{sliding_step}_{split_protocol}_{timestamp}
```

示例：

```text
swat_data_10s_w36_h12_s12_traincol_20260619_130455
```

典型文件：

```text
train.npz
val.npz
test.npz
scaler.pkl
data_meta.pkl
```

## Experiments

`Save/Experiments/` 用于保存完整训练实验。

推荐命名格式：

```text
{timestamp}_{dataset}_{model}_w{window_size}_h{horizon_size}_s{sliding_step}_seed{seed}
```

示例：

```text
20260619_130455_swat_data_10s_MTGNN_w36_h12_s12_seed2024
```

典型结构：

```text
Parameters.json
aggregate/
run_00/
run_01/
```

`aggregate/` 保存跨重复实验的汇总指标和图表。

每个 `run_xx/` 保存一次独立重复实验：

```text
checkpoints/
predictions/
graphs/
postprocess/
```

`checkpoints/` 保存当前 run 的最优权重：

```text
model.pth
ae_model.pth
joint_checkpoint.pth
```

- `model.pth`：验证集最优时刻的 MTGNN 权重。
- `ae_model.pth`：验证集最优时刻的 AE 权重；仅在 `use_ae=True` 时生成。
- `joint_checkpoint.pth`：同时记录 MTGNN、AE、最佳 epoch、`L_total/L_pred/L_rec_*` 与 AE 权重系数的联合检查点。

`predictions/` 保存最终后处理需要的传感器数值和模块级残差：

```text
y_val_true.npy
y_val_pred.npy
y_train_true.npy
y_train_pred.npy
y_true.npy
y_pred.npy
train_mtgnn_pred_error.npy
val_mtgnn_pred_error.npy
test_mtgnn_pred_error.npy
train_mtgnn_pred_error_physical.npy
val_mtgnn_pred_error_physical.npy
test_mtgnn_pred_error_physical.npy
train_ae_rec_real_error.npy
val_ae_rec_real_error.npy
test_ae_rec_real_error.npy
train_ae_rec_pred_error.npy
val_ae_rec_pred_error.npy
test_ae_rec_pred_error.npy
train_joint_error.npy
val_joint_error.npy
test_joint_error.npy
train_y_time.npy
val_y_time.npy
test_y_time.npy
train_y_attack_window.npy
val_y_attack_window.npy
test_y_attack_window.npy
train_y_attack_point.npy
val_y_attack_point.npy
test_y_attack_point.npy
residual_meta.json
```

说明：

- `y_true/y_pred` 是传感器真实值和 MTGNN 预测值，不是攻击/正常标签。
- `*_joint_error.npy` 是默认后处理评分输入。
- `train_*_error.npy` 用于后处理阶段计算历史训练残差统计量 `mu/sigma`。
- `*_y_time.npy` 与 `*_y_attack_point.npy` 用于将滑动窗口结果反投影回原始时间序列点级评价。
- `residual_meta.json` 记录综合残差公式、AE 开关、`beta/lambda` 和标签语义。

## Postprocess

`Save/Postprocess/` 预留给跨实验后处理汇总，例如多组实验的阈值扫描对比表、最终论文汇总图等。单个 run 的后处理结果默认保存在：

```text
Save/Experiments/{experiment}/run_00/postprocess/
```
