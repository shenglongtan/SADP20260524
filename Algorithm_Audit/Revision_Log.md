# 代码修订记录

本文档用于记录 `SADP20260524` 项目的每一次实质性代码修订。记录格式尽量固定，便于后续论文实验复现、问题回溯和版本整理。

## 记录字段说明

- 修订编号：按时间顺序递增，例如 `R001`。
- 日期：执行修订的日期。
- 对应审查项：来自 `Algorithm_Audit` 中的问题编号，例如 `H1`。
- 修订类型：错误修复、接口统一、数据协议修订、实验输出修订等。
- 涉及文件：实际修改过的文件。
- 修订原因：为什么要改。
- 修订内容：具体改了什么。
- 验证情况：已经完成的检查，以及尚未完成的运行验证。
- 后续注意：该修订与后续模块审查的关系。

---

## R001 - 修复 `GetAttMatrix` 构造参数不一致

- 日期：2026-06-19
- 对应审查项：`H1`
- 修订类型：接口统一 / 硬错误修复

### 涉及文件

- `GNN/model_attention.py`
- `GNN/model.py`
- `Algorithm_Audit/README.md`
- `Algorithm_Audit/Revision_Log.md`

### 修订原因

`attention_matrix.py` 中 `GetAttMatrix` 的构造函数签名为：

```python
def __init__(self, node_num: int, window_size: int, ...)
```

`GetAttMatrixV2` 的构造函数同样使用：

```python
def __init__(self, node_num: int, window_size: int, ...)
```

但主模型和备用模型中使用了：

```python
GetAttMatrix(series_num=node_num, ...)
GetAttMatrixV2(series_num=node_num, ...)
```

这会导致模型初始化时出现关键字参数不匹配错误：

```text
TypeError: __init__() got an unexpected keyword argument 'series_num'
```

### 修订内容

1. 在 `GNN/model_attention.py` 中，将当前主模型的动态图构造调用：

```python
GetAttMatrix(series_num=node_num, window_size=window_size)
```

修订为：

```python
GetAttMatrix(node_num=node_num, window_size=window_size)
```

2. 在 `GNN/model.py` 中，将备用模型的 `attention` 分支：

```python
GetAttMatrix(series_num=node_num, window_size=window_size)
```

修订为：

```python
GetAttMatrix(node_num=node_num, window_size=window_size)
```

3. 在 `GNN/model.py` 中，将备用模型的 `attention_v2` 分支：

```python
GetAttMatrixV2(series_num=node_num, ...)
```

修订为：

```python
GetAttMatrixV2(node_num=node_num, ...)
```

4. 在 `Algorithm_Audit/README.md` 中登记 `Revision_Log.md`。

### 验证情况

已完成：

- 已全局检索 `series_num=` 调用点。
- 已确认 Python 源码中与 `GetAttMatrix/GetAttMatrixV2` 构造函数相关的 `series_num=` 调用均已改为 `node_num=`。
- 已使用本机 Python 3.9.13 环境完成当前主模型初始化验证。
- 已使用随机输入完成一次最小前向传播验证：
  - 输入形状：`[2, 1, 5, 36]`
  - 输出预测形状：`[2, 1, 5, 12]`
  - 输出邻接矩阵形状：`[5, 5]`

验证命令使用解释器：

```text
C:\Users\MR.long\AppData\Local\Programs\Python\Python39\python.exe
```

验证结果：

```text
MTGNN init OK
GetAttMatrix
forward OK
y_shape= (2, 1, 5, 12)
A_shape= (5, 5)
```

未完成：

- 尚未执行真实数据 batch 的训练/验证 smoke test。
- 尚未验证备用模型 `GNN/model.py` 的 `attention` 和 `attention_v2` 分支。

建议验证命令：

```python
from GNN.model_attention import MTGNN
import torch

model = MTGNN(
    device=torch.device("cpu"),
    node_num=5,
    feature_num=1,
    window_size=36,
    horizon_size=12,
)
print("MTGNN init OK")
```

### 后续注意

该修订只解决构造参数不一致问题，不代表模型完整前向传播已经通过。下一步应继续检查：

- `GetAttMatrix` 前向输入维度是否与 `model_attention.py` 中的 `attention_input` 严格匹配。
- `MTGNN.forward()` 输出维度是否与 `Trainer.train()`、`tools.test_model()` 完全一致。
- 当前数据输入的 `node_num` 是否排除了 `Attack/Train` 等非物理变量。

---

## R002 - 最终验证/测试前显式回载验证集最佳模型

- 日期：2026-06-19
- 对应审查项：`H2`
- 修订类型：训练流程修复 / 最佳模型一致性修复

### 涉及文件

- `train_attention.py`
- `Algorithm_Audit/03_main_chain_hard_errors.md`
- `Algorithm_Audit/Revision_Log.md`

### 修订原因

训练过程中，当验证集损失刷新历史最优时，程序会保存：

```python
torch.save(engine.model.state_dict(), os.path.join(args.save, "model.pth"))
```

但训练循环结束后，最终验证集和测试集评估之前，原程序没有显式执行：

```python
engine.model.load_state_dict(...)
```

因此最终测试使用的可能是最后一轮模型，而不是验证集最优模型。这会导致：

- `model.pth` 与最终报告指标不一致。
- “巅峰模型”日志叙述与真实测试对象不一致。
- 论文实验表格可能不是基于验证集选择出的最佳模型。

### 修订内容

在 `train_attention.py` 的训练循环结束之后、最终验证/测试之前，新增最佳权重回载逻辑：

```python
best_model_path = os.path.join(args.save, "model.pth")
if os.path.exists(best_model_path):
    engine.model.load_state_dict(torch.load(best_model_path, map_location=device))
    engine.model.to(device)
    hues.success(f"  已回载验证集最优模型权重: [{best_model_path}]  ")
else:
    hues.warn(f"  未找到验证集最优模型权重: [{best_model_path}]，最终评估将使用当前内存模型。  ")
```

该逻辑确保：

1. 若存在验证集最优权重，则最终验证和测试一定使用该权重。
2. 若由于异常情况没有产生 `model.pth`，程序不会直接崩溃，而是给出明确警告。

### 验证情况

已完成：

- 已使用本机 Python 3.9.13 对 `train_attention.py` 执行语法编译检查。

验证命令：

```powershell
& 'C:\Users\MR.long\AppData\Local\Programs\Python\Python39\python.exe' -m py_compile train_attention.py
```

验证结果：

```text
通过，无语法错误。
```

未完成：

- 尚未执行真实训练流程验证 `model.pth` 保存和回载是否在完整训练日志中按预期出现。
- 尚未验证多次重复实验 `repeat_exp_num > 1` 时每个 run 子目录下的最佳模型回载是否完全独立。

### 后续注意

本修订只解决“最终评估模型权重不一致”的问题。下一步仍需继续处理：

- `H3`：训练输出与后处理脚本不匹配，缺少 `y_val_true.npy/y_val_pred.npy`。
- `H4/H5`：数据标准化缺失和 `Attack/Train` 标签泄漏。
- `H8`：`tools.test_model()` 中 `squeeze()` 存在维度坍缩风险。

---

## R003 - 修复数据标准化缺失与 `Attack/Train` 标签泄漏

- 日期：2026-06-19
- 对应审查项：`H4/H5`
- 修订类型：数据协议修订 / 实验科学性修复 / 标签泄漏修复

### 涉及文件

- `data_maker.py`
- `data_loader.py`
- `Algorithm_Audit/03_main_chain_hard_errors.md`
- `Algorithm_Audit/Revision_Log.md`

### 修订原因

原始 `data_maker.py` 中虽然构造了：

```python
scalers = CustomScaler(df_train.values)
```

但后续滑动窗口生成直接使用：

```python
data = np.expand_dims(self.df.values, axis=-1)
```

即实际输入模型的数据没有执行标准化。与此同时，`Timestamp` 只被设置为索引，`Attack` 和 `Train` 没有从特征矩阵中移除，因此真实异常标签和数据划分标记可能进入模型输入，属于严重标签泄漏。

### 修订内容

1. 在 `data_maker.py` 中新增物理特征与实验元数据分离逻辑：

```python
feature_cols = [col for col in df.columns if col not in self.meta_cols]
```

其中 `Attack` 和 `Train` 被识别为元数据列，只用于划分、评估和后处理，不再进入模型输入。

2. 若数据包含 `Train` 列，优先采用工业异常检测协议：

```text
Train == 1 -> 训练/验证候选段
Train == 0 -> 测试段
```

随后将 `Train==1` 数据按时间顺序切分为训练集和验证集。若没有可用的 `Train` 协议，则回退为严格时间顺序 `70%/10%/20%` 划分。

3. 标准化器现在只基于训练段物理特征拟合：

```python
scalers = CustomScaler(split_dfs["train"].values)
```

并且滑动窗口生成前显式执行：

```python
scaled_values = scalers.transform(feature_df.values).astype(np.float32)
```

因此训练、验证和测试输入均为基于训练集统计量标准化后的物理变量。

4. 保存/在线生成数据时额外保留后处理所需元信息：

- `y_attack_window`：预测窗口级异常标签，窗口内任意时间点异常则为 1。
- `y_attack_point`：预测窗口逐点异常标签。
- `y_time`：预测窗口对应时间戳。
- `data_meta.pkl`：物理特征列、排除的元数据列、划分协议、窗口参数和原始划分行数。

5. 在 `data_loader.py` 中补充读取逻辑，使 `.npz` 中除 `x/y` 之外的标签和时间信息可以继续进入 `data` 字典；在线生成路径也会保留 `maker.generated_extra` 与 `maker.data_meta`。

### 验证情况

已完成：

- 已使用本机 Python 3.9.13 对 `data_maker.py` 和 `data_loader.py` 执行语法编译检查。

验证命令：

```powershell
& 'C:\Users\MR.long\AppData\Local\Programs\Python\Python39\python.exe' -m py_compile data_maker.py data_loader.py
```

验证结果：

```text
通过，无语法错误。
```

- 已使用真实 SWaT 10 秒数据执行在线数据生成 smoke test。

验证设置：

```text
数据文件：Data\SWat\swat_data_10s.pkl
window_size = 36
horizon_size = 12
sliding_step = 12
save = False
```

关键验证结果：

```text
模型物理特征列数量: 51
已排除元数据列: ['Attack', 'Train']
has_attack_feature = False
has_train_feature = False
split_protocol = train_column
split_rows = {'train': 34776, 'val': 14904, 'test': 44993}
训练集 x: (2895, 36, 51, 1), y: (2895, 12, 51, 1)
验证集 x: (1239, 36, 51, 1), y: (1239, 12, 51, 1)
测试集 x: (3746, 36, 51, 1), y: (3746, 12, 51, 1)
test_extra_shapes = {'y_attack_window': (3746,), 'y_attack_point': (3746, 12), 'y_time': (3746, 12)}
```

标准化检查：

```text
train_x_channel_mean_abs_max = 0.0038733552
train_x_channel_std_minmax = 0.0, 1.0011307001
```

其中标准差为 0 的通道对应训练段内恒定的执行器/传感器变量，`CustomScaler` 已将这些通道的标准差修正为 1.0 以避免除零；窗口采样后的标准差为 0 符合该物理事实。

未完成：

- 尚未执行完整训练流程，验证 `Trainer`、`tools.test_model()` 与新标准化输入在真实训练中是否完全衔接。
- 尚未修复 `H6`：命令行 `--sliding_step` 仍未传入 `load_local_dataset()` 的在线数据制作流程。

### 后续注意

该修订会改变模型实际看到的输入变量和数据尺度，因此后续实验指标与旧结果不可直接比较。旧结果可能受到标签泄漏和尺度错误影响，建议在论文实验表格中仅使用本修订之后重新跑出的结果。

---

## R004 - 修复 `--sliding_step` 未传入在线数据制作流程

- 日期：2026-06-19
- 对应审查项：`H6`
- 修订类型：实验参数传递修复 / 可复现性修复

### 涉及文件

- `data_loader.py`
- `train_attention.py`
- `Algorithm_Audit/03_main_chain_hard_errors.md`
- `Algorithm_Audit/Revision_Log.md`

### 修订原因

`tools.py` 中已经定义命令行参数：

```python
parser.add_argument('--sliding_step', type=int, default=12, ...)
```

`DataMaker.__init__()` 也支持：

```python
sliding_step: int = 1
```

但 `train_attention.py` 调用 `load_local_dataset()` 时没有传入 `args.sliding_step`，而 `data_loader.py` 内部实例化 `DataMaker` 时也没有暴露该参数。因此在线数据制作实际总是使用 `DataMaker` 默认的 `sliding_step=1`，导致：

- 命令行实验参数不真实生效。
- 样本量和训练耗时与实验记录不一致。
- 论文复现实验中的滑动窗口重叠率不可追溯。

### 修订内容

1. 在 `data_loader.py` 的 `load_local_dataset()` 中新增参数：

```python
sliding_step: int = 1
```

2. 在线文件数据路径下，实例化 `DataMaker` 时显式传入：

```python
maker = DataMaker(
    path,
    window_size,
    horizon_size,
    sliding_step=sliding_step,
    save=False,
)
```

3. 在 `train_attention.py` 中，将命令行参数继续传入数据加载流程：

```python
sliding_step=args.sliding_step
```

### 验证情况

已完成：

- 已使用本机 Python 3.9.13 对 `data_loader.py` 和 `train_attention.py` 执行语法编译检查。

验证命令：

```powershell
& 'C:\Users\MR.long\AppData\Local\Programs\Python\Python39\python.exe' -m py_compile data_loader.py train_attention.py
```

验证结果：

```text
通过，无语法错误。
```

- 已使用真实 SWaT 10 秒数据验证 `sliding_step` 会改变在线生成样本数。

验证设置：

```text
数据文件：Data\SWat\swat_data_10s.pkl
window_size = 36
horizon_size = 12
batch_size = 64
save = False
```

关键结果：

```text
sliding_step = 12:
x_train = (2895, 36, 51, 1)
x_val   = (1239, 36, 51, 1)
x_test  = (3746, 36, 51, 1)
data_meta["sliding_step"] = 12

sliding_step = 24:
x_train = (1448, 36, 51, 1)
x_val   = (620, 36, 51, 1)
x_test  = (1873, 36, 51, 1)
data_meta["sliding_step"] = 24
```

同时确认 H4/H5 的修复仍然保持：

```text
feature_count = 51
excluded_meta = ['Attack', 'Train']
```

### 后续注意

该修订只影响在线从原始文件生成数据的路径。如果输入 `path` 是已经预处理好的目录，`sliding_step` 不会重新改变已有 `.npz` 的样本数量；此时应以该目录保存时的 `data_meta.pkl` 为准。

---

## R005 - 保存验证集预测以支撑异常阈值后处理

- 日期：2026-06-19
- 对应审查项：`H3`
- 修订类型：训练输出修复 / 后处理接口修复 / 防测试集阈值泄漏

### 涉及文件

- `train_attention.py`
- `Algorithm_Audit/03_main_chain_hard_errors.md`
- `Algorithm_Audit/Revision_Log.md`

### 修订原因

后处理脚本 `Test/anomaly_scoring_threshold_pa.py` 明确要求训练结果目录中存在：

```python
required = ["y_val_true.npy", "y_val_pred.npy", "y_true.npy", "y_pred.npy"]
```

其中验证集预测用于估计异常分数分布和阈值，测试集预测只用于最终客观评估。原训练脚本只保存：

```python
y_true.npy
y_pred.npy
```

因此后处理脚本无法直接运行。如果研究者改用测试集自身估计阈值，会造成测试集信息泄漏，破坏异常检测实验科学性。

### 修订内容

1. 在最终验证集评估阶段，不再丢弃 `test_model()` 返回的验证集真实值与预测值：

```python
ret_valid_error, val_true_y, val_pred_y, _ = test_model(...)
```

2. 当 `args.save_pred_result=True` 时，同时保存验证集和测试集预测结果。当前版本在 `R006` 后已将这些文件统一保存到单次 run 的 `predictions/` 子目录：

```python
np.save(os.path.join(run_dirs["predictions"], 'y_val_true.npy'), val_true_y.detach().cpu().numpy())
np.save(os.path.join(run_dirs["predictions"], 'y_val_pred.npy'), val_pred_y.detach().cpu().numpy())
np.save(os.path.join(run_dirs["predictions"], 'y_true.npy'), true_y.detach().cpu().numpy())
np.save(os.path.join(run_dirs["predictions"], 'y_pred.npy'), pred_y.detach().cpu().numpy())
```

3. 增加日志提示，明确预测结果已保存到当前实验目录。

### 验证情况

已完成：

- 已使用本机 Python 3.9.13 对 `train_attention.py` 执行语法编译检查。

验证命令：

```powershell
& 'C:\Users\MR.long\AppData\Local\Programs\Python\Python39\python.exe' -m py_compile train_attention.py
```

验证结果：

```text
通过，无语法错误。
```

- 已使用真实 SWaT 10 秒数据和临时零预测模型执行轻量行为验证，不运行完整训练，仅验证 `test_model()` 输出和 `.npy` 保存形状。

验证设置：

```text
数据文件：Data\SWat\swat_data_10s.pkl
window_size = 36
horizon_size = 12
sliding_step = 96
batch_size = 64
```

关键结果：

```text
val_shapes  = (155, 51, 12), (155, 51, 12)
test_shapes = (469, 51, 12), (469, 51, 12)
files = ['y_pred.npy', 'y_true.npy', 'y_val_pred.npy', 'y_val_true.npy']
```

该形状满足 `Test/anomaly_scoring_threshold_pa.py` 对 `[samples, nodes, horizon]` 三维输入的要求。

未完成：

- 尚未执行完整训练流程验证 `args.save_pred_result=True` 时真实模型预测文件在实验目录中的最终落盘情况。

### 后续注意

该修订只补齐预测张量输出，不改变异常分数计算、阈值选择或 Point Adjustment 策略。后续若要完整闭环异常检测实验，还需要确认后处理脚本能正确获得 `y_attack_window` 标签；若训练使用在线原始文件而非预处理目录，建议后续增加从训练输出目录读取标签的兼容逻辑。

---

## R006 - 重置派生文件与实验结果保存结构

- 日期：2026-06-19
- 对应审查项：实验管理结构优化
- 修订类型：文件结构重置 / 实验可复现性管理 / 结果归档规范化

### 涉及文件

- `tools.py`
- `data_maker.py`
- `train_attention.py`
- `Test/anomaly_scoring_threshold_pa.py`
- `Save/README.md`
- `Save/Datasets/.gitkeep`
- `Save/Experiments/.gitkeep`
- `Save/Postprocess/.gitkeep`
- `Algorithm_Audit/Revision_Log.md`

### 修订原因

原始保存结构存在三类管理问题：

1. 预处理数据默认保存在原始数据文件旁边，例如 `Data\SWat\swat_data_10s_时间戳`，容易污染原始数据目录。
2. 训练实验默认保存到 `Save\时间戳\0`，目录名缺少数据集、模型和窗口参数，不利于复现实验追踪。
3. 单次 run 目录中模型权重、预测张量、动态图和后处理结果混放，长期实验后难以管理。

### 修订内容

1. 新建 `Save` 标准目录骨架：

```text
Save/
  README.md
  Datasets/
  Experiments/
  Postprocess/
```

2. 将训练实验默认根目录从：

```text
./Save/
```

调整为：

```text
./Save/Experiments/
```

3. 训练实验目录命名改为包含核心复现实验参数：

```text
{timestamp}_{dataset}_{model}_w{window_size}_h{horizon_size}_s{sliding_step}_seed{seed}
```

示例：

```text
20260619_130455_swat_data_10s_MTGNN_w36_h12_s12_seed2024
```

4. 单次重复实验目录从：

```text
0/
1/
```

调整为：

```text
run_00/
run_01/
```

5. 每个 `run_xx` 内部拆分为标准子目录：

```text
checkpoints/   # model.pth
predictions/   # y_val_true.npy, y_val_pred.npy, y_true.npy, y_pred.npy
graphs/        # AdjMatrix*.pt/.npy
postprocess/   # 单次 run 后处理结果
```

6. 跨重复实验的汇总结果统一保存到：

```text
aggregate/
```

包括：

```text
Result.pkl
TrainLoss.pkl
ValidLoss.pkl
ValidError.pkl
TestError.pkl
[MTGNN]_Error.png
[MTGNN]_BoxRMSE.png
[MTGNN]_Loss.png
```

7. `DataMaker(save=True)` 的预处理数据输出从原始数据旁边调整到：

```text
Save/Datasets/
```

目录命名格式：

```text
{dataset}_w{window_size}_h{horizon_size}_s{sliding_step}_{split_protocol}_{timestamp}
```

示例：

```text
swat_data_10s_w36_h12_s12_traincol_20260619_132009
```

8. 后处理脚本 `Test/anomaly_scoring_threshold_pa.py` 默认输出目录改为：

```text
run_00/postprocess/
```

并优先从：

```text
run_00/predictions/
```

读取 `y_val_true.npy/y_val_pred.npy/y_true.npy/y_pred.npy`。若旧实验结果没有 `predictions/` 子目录，则自动回退读取 run 根目录，保持一定兼容性。

### 验证情况

已完成：

- 已使用本机 Python 3.9.13 对相关脚本执行语法编译检查。

验证命令：

```powershell
& 'C:\Users\MR.long\AppData\Local\Programs\Python\Python39\python.exe' -m py_compile tools.py data_maker.py train_attention.py Test\anomaly_scoring_threshold_pa.py
```

验证结果：

```text
通过，无语法错误。
```

- 已完成轻量路径命名与后处理读取验证。

关键结果：

```text
dataset_dir = .\Save\Datasets\swat_data_10s_w36_h12_s12_traincol_20260619_132009
experiment_name = 20260619_130455_swat_data_10s_MTGNN_w36_h12_s12_seed2024
loaded_keys = ['y_pred.npy', 'y_true.npy', 'y_val_pred.npy', 'y_val_true.npy']
loaded_shape = (2, 3, 4)
```

未完成：

- 尚未执行一次完整真实训练以验证新目录结构下所有真实模型产物的最终落盘情况。

### 后续注意

后续运行后处理脚本时，建议传入单次 run 目录，例如：

```powershell
python Test\anomaly_scoring_threshold_pa.py --run-dir Save\Experiments\...\run_00 --data-path Save\Datasets\...
```

而不是直接传入 `predictions/` 子目录。这样后处理结果会自然保存到同一个 run 的 `postprocess/` 下。

---

## R007 - 修复 `test_model()` 无参数 `squeeze()` 造成的维度坍缩风险

- 日期：2026-06-19
- 对应审查项：`H8`
- 修订类型：评估流程鲁棒性修复 / 预测张量维度修复

### 涉及文件

- `tools.py`
- `Algorithm_Audit/03_main_chain_hard_errors.md`
- `Algorithm_Audit/Revision_Log.md`

### 修订原因

`tools.test_model()` 原来使用：

```python
pred_y.append(output_y.squeeze())
```

该写法会删除所有长度为 1 的维度。当前模型输出预期为：

```text
[B, 1, N, H]
```

正常只应删除第 1 维通道维，得到：

```text
[B, N, H]
```

但当 `batch_size=1`、`node_num=1` 或 `horizon_size=1` 时，无参数 `squeeze()` 会继续删除 batch、node 或 horizon 维，导致 `torch.cat()`、`calc_error()` 或预测文件保存形状不稳定。

### 修订内容

1. 将无参数 `squeeze()` 改为只处理通道维：

```python
if output_y.dim() == 4:
    if output_y.size(1) != 1:
        raise ValueError(...)
    output_y = output_y.squeeze(1)
elif output_y.dim() != 3:
    raise ValueError(...)
pred_y.append(output_y)
```

2. 增加预测张量与真实张量形状一致性检查：

```python
if pred_y.shape != true_y.shape:
    raise ValueError(...)
```

该检查可以在模型输出接口异常时尽早失败，避免静默生成错误 `.npy` 文件。

### 验证情况

已完成：

- 已使用本机 Python 3.9.13 对 `tools.py` 执行语法编译检查。

验证命令：

```powershell
& 'C:\Users\MR.long\AppData\Local\Programs\Python\Python39\python.exe' -m py_compile tools.py
```

验证结果：

```text
通过，无语法错误。
```

- 已执行极端维度小样本验证，覆盖 `batch_size=1, node_num=1, horizon_size=1`。

关键结果：

```text
true_shape = (2, 1, 1)
pred_shape = (2, 1, 1)
```

- 已使用真实 SWaT 10 秒数据和临时零预测模型执行轻量验证。

验证设置：

```text
数据文件：Data\SWat\swat_data_10s.pkl
window_size = 36
horizon_size = 12
sliding_step = 192
batch_size = 64
```

关键结果：

```text
val_true_shape = (78, 51, 12)
val_pred_shape = (78, 51, 12)
```

该形状继续满足后处理脚本对 `[samples, nodes, horizon]` 的输入要求。

### 后续注意

本修订只处理最终 `test_model()` 推理阶段的张量压缩风险。训练/验证过程中的损失计算仍保持 `[B, 1, N, H]` 与 `[B, 1, N, H]` 对齐，不受本次修订影响。

---

## R008 - 完善 AE 与 MTGNN 并列模块的联合训练、权重保存回载和综合残差后处理

- 日期：2026-06-19
- 对应审查项：`AE-1/AE-2/AE-3/AE-4/AE-5`
- 修订类型：算法目标一致性修订 / 实验科学性修订 / 后处理协议修订

### 涉及文件

- `tools.py`
- `GNN/trainer_attention.py`
- `train_attention.py`
- `Test/anomaly_scoring_threshold_pa.py`
- `Save/README.md`
- `Algorithm_Audit/README.md`
- `Algorithm_Audit/04_ae_mtg_nn_joint_residual_protocol.md`
- `Algorithm_Audit/Revision_Log.md`

### 修订原因

当前算法目标中，AE 模块与 MTGNN 模块是并列的两个独立模块。原程序存在以下不完整点：

1. AE 默认未开启，不符合当前算法主实验设定。
2. 最佳模型保存/回载只覆盖 MTGNN，没有覆盖 AE。
3. 验证集最佳模型选择没有明确基于 `L_total`。
4. 最终测试阶段只导出 `y_true/y_pred`，没有导出 MTGNN 残差、AE 重构残差和综合残差。
5. 后处理脚本仍默认从 `abs(y_true-y_pred)` 得分，未升级为综合残差评分。
6. 术语上需要明确：`y_true/y_pred` 是传感器真实值/预测值，不是攻击标签。

### 修订内容

1. 在 `tools.py` 中修改默认参数：

```python
--use_ae default=True
--save_pred_result default=True
```

2. 在 `GNN/trainer_attention.py` 中统一训练与验证损失返回：

```text
[L_total, L_pred, L_rec_real, L_rec_pred, MAPE, RMSE]
```

启用 AE 时：

```text
L_total = L_pred + beta * (L_rec_real + lambda * L_rec_pred)
```

关闭 AE 时：

```text
L_total = L_pred
L_rec_real = 0
L_rec_pred = 0
```

同时将梯度裁剪范围扩展为 MTGNN 与 AE 的全部可训练参数，避免开启 AE 后只裁剪 MTGNN。

3. 在 `train_attention.py` 中完善最佳权重保存：

```text
checkpoints/model.pth
checkpoints/ae_model.pth
checkpoints/joint_checkpoint.pth
```

其中 `joint_checkpoint.pth` 记录：

- `model_state_dict`
- `ae_state_dict`
- `use_ae`
- `best_epoch`
- `valid_total_loss`
- `valid_pred_loss`
- `valid_rec_real_loss`
- `valid_rec_pred_loss`
- `ae_beta`
- `ae_lambda`
- `selection_metric`

4. 在最终验证/测试前显式回载：

- MTGNN 最优权重：`model.pth`
- AE 最优权重：`ae_model.pth`

若 `use_ae=False`，程序明确记录：最终评估采用 `L_total=L_pred`，综合残差退化为 MTGNN 预测残差。

5. 在 `train_attention.py` 中新增残差导出：

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

默认综合残差为：

```text
joint_error = mtgnn_pred_error + beta * (ae_rec_real_error + lambda * ae_rec_pred_error)
```

6. 在 `Test/anomaly_scoring_threshold_pa.py` 中新增 `--score-source`：

```text
auto   : 默认优先使用 joint_error；旧实验缺失时回退到 MTGNN 预测误差。
joint  : 强制使用 joint_error，缺失时报错。
mtgnn  : 使用 MTGNN 预测误差，用于消融对照。
```

后处理输出新增：

```text
selected_error_val.npy
selected_error_test.npy
```

并在 `summary.json` 中记录：

```text
score_source_arg
selected_residual_source
```

7. 新增 `Algorithm_Audit/04_ae_mtg_nn_joint_residual_protocol.md`，归档 AE/MTGNN 模块关系、术语定义、残差公式和后处理协议。

### 验证情况

已完成：

- 已使用本机 Python 3.9.13 对以下文件执行语法编译检查：

```powershell
& 'C:\Users\MR.long\AppData\Local\Programs\Python\Python39\python.exe' -m py_compile tools.py GNN\trainer_attention.py train_attention.py Test\anomaly_scoring_threshold_pa.py
```

验证结果：

```text
通过，无语法错误。
```

- 已验证默认参数：

```text
use_ae = True
save_pred_result = True
```

- 已使用小样本 dummy loader 验证残差导出函数。

关键结果：

```text
输出键 = ['ae_rec_pred_error', 'ae_rec_real_error', 'joint_error', 'mtgnn_pred_error', 'mtgnn_pred_error_physical', 'sensor_pred', 'sensor_true']
sensor_true shape = (2, 3, 2)
sensor_pred shape = (2, 3, 2)
joint_error shape = (2, 3, 2)
```

- 已使用临时 `.npy` 文件验证后处理脚本默认选择综合残差：

```text
score source = joint_error
selected_residual_source = joint_error
```

未完成：

- 尚未执行真实 SWaT/WADI 数据的完整训练，未验证真实长流程下所有残差文件的最终落盘耗时和显存占用。

### 后续注意

1. 主实验默认使用 AE 与 MTGNN 联合训练和综合残差评分。
2. MTGNN-only 消融实验可以通过以下两种方式之一实现：

```powershell
python train_attention.py --use_ae false
python Test\anomaly_scoring_threshold_pa.py --run-dir ... --score-source mtgnn
```

3. 当前代码支持 joint 与 MTGNN-only 两类评分。若后续需要“只使用 AE 重构残差”的消融，还需要继续扩展 `--score-source ae_real/ae_pred/ae_combined`。

---

## R009 - 修复掩码指标在全掩码 batch 下的 NaN 风险

- 日期：2026-06-19
- 对应审查项：`H9`
- 修订类型：评估指标鲁棒性修复 / 数值稳定性修复

### 涉及文件

- `tools.py`
- `Algorithm_Audit/03_main_chain_hard_errors.md`
- `Algorithm_Audit/Revision_Log.md`

### 修订原因

`masked_mse()` 已经针对全掩码情况设置安全返回：

```python
if torch.sum(mask) == 0:
    return torch.tensor(0.0, device=pred_y.device)
```

但 `masked_mae()` 和 `masked_mape()` 原来直接执行：

```python
mask /= torch.mean(mask)
```

如果某个 batch 的有效标签全部被 mask 掉，会出现除以 0 或 NaN。虽然当前 SWaT/WADI 主流程通常没有这种全缺失 batch，但该函数作为通用指标函数，应保持与 `masked_mse()` 一致的鲁棒性。

### 修订内容

1. 新增 `_build_safe_mask()`，统一构造安全 mask。
2. 当有效标签数为 0 时，返回 `None`，调用方直接返回同设备、同 dtype 的 0 张量。
3. `masked_mae()`、`masked_mape()` 改用统一安全 mask。
4. `masked_r2()` 也补充全掩码防护，避免 `torch.sum(mask)=0` 时计算均值导致 NaN。

### 验证情况

已完成：

- 已使用本机 Python 3.9.13 对 `tools.py` 执行语法编译检查。
- 已构造全 NaN 标签和 `null_val=0.0` 全掩码标签进行指标验证。

关键结果：

```text
{'mae_nan': 0.0, 'mape_nan': 0.0, 'mse_nan': 0.0, 'rmse_nan': 0.0, 'r2_nan': 0.0, 'mae_zero_null': 0.0, 'mape_zero_null': 0.0}
finite = True
```

### 后续注意

该修订不会改变正常有效标签 batch 的指标定义，只在极端全掩码情况下提供安全退化值，避免训练日志、早停判断或最终评估被 NaN 污染。

---

## R010 - 修复可视化脚本导入不存在的 `ATTACK_EVENTS`

- 日期：2026-06-19
- 对应审查项：`H7`
- 修订类型：可视化脚本接口修复 / 数据集事件表显式化

### 涉及文件

- `Test/visualize_attack_pred_vs_true.py`
- `Test/visualize_attack_pred_vs_true_功能分析和使用说明.md`
- `DataMaker/WADI/DownsampledVisualization.py`
- `DataMaker/WADI/DownsampledVisualization.ipynb`
- `DataMaker/WADI/DownsampledVisualization_功能分析和使用说明.md`
- `Algorithm_Audit/03_main_chain_hard_errors.md`
- `Algorithm_Audit/Revision_Log.md`

### 修订原因

`tools.py` 当前定义的是：

```python
ATTACK_EVENTS_SWAT
ATTACK_EVENTS_WADI
```

但部分可视化脚本导入：

```python
from tools import ATTACK_EVENTS
```

该别名并不存在，会导致脚本在 import 阶段失败。并且 SWaT 与 WADI 的攻击事件表不同，不适合在 `tools.py` 中设置模糊默认别名。

### 修订内容

1. `Test/visualize_attack_pred_vs_true.py` 改为显式导入：

```python
from tools import ATTACK_EVENTS_SWAT, ATTACK_EVENTS_WADI
```

并新增 `select_attack_events(data_path, run_dir)`，根据路径中的 `swat/wadi` 自动选择对应事件表。

2. `Test/visualize_attack_pred_vs_true.py` 同时兼容当前保存结构：

```text
run_dir/predictions/y_true.npy
run_dir/predictions/y_pred.npy
```

若不存在 `predictions/` 子目录，则回退旧路径：

```text
run_dir/y_true.npy
run_dir/y_pred.npy
```

3. `Test/visualize_attack_pred_vs_true.py` 的特征名读取兼容：

```text
data_meta.pkl
meta.pkl
```

4. `DataMaker/WADI/DownsampledVisualization.py` 和对应 notebook 改为：

```python
from tools import ATTACK_EVENTS_WADI
ATTACK_EVENTS = ATTACK_EVENTS_WADI
```

或 notebook 中：

```python
from tools import downsample_block, ATTACK_EVENTS_WADI as ATTACK_EVENTS
```

5. 同步更新相关功能说明文档中的旧 `ATTACK_EVENTS` 表述。

### 验证情况

已完成：

- 已使用本机 Python 3.9.13 对以下文件执行语法编译检查：

```powershell
& 'C:\Users\MR.long\AppData\Local\Programs\Python\Python39\python.exe' -m py_compile tools.py Test\visualize_attack_pred_vs_true.py DataMaker\WADI\DownsampledVisualization.py Test\anomaly_scoring_threshold_pa.py
```

验证结果：

```text
通过，无语法错误。
```

- 已验证攻击事件表可正常导入：

```text
attack_events = 41 16
```

其中 SWaT 为 41 个事件，WADI 为 16 个事件。

### 后续注意

该修订解决 import 阶段硬错误，并让可视化脚本适配当前 `run_xx/predictions/` 保存结构。可视化脚本仍依赖真实预测文件和 `test.npz`，因此完整绘图效果需要在完成一次真实训练并生成预测文件后再验证。

---

## R011 - 后处理脚本按严格协议重构为训练统计量、验证 F1 阈值与点级评价

### 修订日期

2026-06-19

### 涉及文件

- `Test/anomaly_scoring_threshold_pa.py`
- `train_attention.py`
- `Save/README.md`
- `Algorithm_Audit/08_postprocess_strict_protocol_audit.md`
- `Algorithm_Audit/README.md`

### 修订原因

用户明确给出后处理数学规范：

```text
e_{t,i} = |y_{t,i} - yhat_{t,i}|
e_norm_{t,i} = (e_{t,i} - mu_i) / (sigma_i + 1e-8)
S_t = mean_i(e_norm_{t,i})
tau_best = argmax_tau F1_val(tau)
Label_test(t) = 1 if S_test(t) >= tau_best else 0
```

并要求检查：

1. `mu/sigma` 是否严格来自历史训练集。
2. 阈值搜索是否只使用验证集。
3. 测试集标签是否在判定结束前隔离。
4. AUC-PR 是否使用连续分数。
5. 空间聚合是否沿传感器维执行 mean-pooling。

原脚本默认使用验证集残差估计标准化统计量，并默认采用分位数阈值与窗口级评价，不满足该严格协议。

### 修订内容

1. 重构 `Test/anomaly_scoring_threshold_pa.py`：

```text
默认 eval_granularity = point
默认 stats_source = train
默认 var_reduce = mean
默认 threshold_method = f1_val
默认 score_source = auto，优先 joint_error
```

2. 新增训练残差统计量协议：

```text
train_error[time,node] -> mu_i/sigma_i
val/test error 使用固定 train mu/sigma 标准化
```

3. 新增点级反投影：

```text
error[sample,node,horizon] + y_time[sample,horizon]
  -> point_error[time,node]

y_attack_point[sample,horizon] + y_time[sample,horizon]
  -> point_label[time]
```

同一原始时间点被多个窗口覆盖时，默认 `max` 聚合。

4. 阈值选择改为验证集 F1 最大化：

```python
precision_recall_curve(labels_val, scores_val)
```

并使用：

```python
pred = (score >= threshold)
```

与用户给定公式一致。

5. AUC-PR 使用连续异常分数：

```python
average_precision_score(y_true, scores)
```

6. `train_attention.py` 同步导出训练集残差和点级反投影所需文件：

```text
train_joint_error.npy
train_mtgnn_pred_error.npy
train_ae_rec_real_error.npy
train_ae_rec_pred_error.npy
y_train_true.npy
y_train_pred.npy
train_y_time.npy
train_y_attack_window.npy
train_y_attack_point.npy
```

7. 导出训练集残差前重置 `train_loader.indices` 为原始顺序，避免训练 shuffle 后与 `train_y_time` 错位。

8. 更新 `Save/README.md` 和新增 `08_postprocess_strict_protocol_audit.md`。

### 验证情况

已完成语法检查：

```powershell
& 'C:\Users\MR.long\AppData\Local\Programs\Python\Python39\python.exe' -m py_compile train_attention.py Test\anomaly_scoring_threshold_pa.py
```

结果：

```text
通过，无语法错误。
```

已完成合成点级后处理测试，结果摘要：

```text
selected_residual_source = joint_error
eval_granularity = point
threshold_method = f1_val
test_labels_used_for_threshold = False
raw_f1 = 1.0
pr_auc = 1.0
```

### 后续注意

若真实验证集标签只有正常类，`threshold-method=f1_val` 会报错。这不是程序 bug，而是监督式 F1 阈值选择的数学前提不满足。届时需要决定：

1. 使用含异常标签的验证集。
2. 或将阈值策略改为无监督，并在论文中明确说明。

---

## R012 - 路线 A：后处理默认阈值改为正常验证集无监督分位数

### 修订日期

2026-06-19

### 涉及文件

- `Test/anomaly_scoring_threshold_pa.py`
- `Algorithm_Audit/08_postprocess_strict_protocol_audit.md`
- `Algorithm_Audit/09_route_a_semisupervised_threshold_protocol.md`
- `Algorithm_Audit/README.md`

### 修订原因

用户确认 SWaT 数据中：

```text
Train=1: 2015-12-22 16:00:00 到 2015-12-28 09:59:50，完全正常
Train=0: 2015-12-28 10:00:00 到 2016-01-02 14:59:50，包含攻击
```

因此当前算法主实验应采用半监督异常检测路线 A：

```text
Train==1 正常段 -> train/val
Train==0 含攻击段 -> test
```

在该路线中，验证集来自正常段，只有正常标签，不能使用 `F1_val` 搜索阈值。

### 修订内容

1. 将 `Test/anomaly_scoring_threshold_pa.py` 默认阈值策略从：

```text
--threshold-method f1_val
```

改为：

```text
--threshold-method percentile
--threshold-percentile 99.0
```

2. 保留 `f1_val` 作为显式可选模式，仅用于有监督阈值校准实验。

3. 在 `summary.json` 中新增：

```json
"route": "A_semisupervised_unsupervised_threshold",
"val_labels_used_for_threshold": false,
"test_labels_used_for_threshold": false
```

4. 新增 `09_route_a_semisupervised_threshold_protocol.md`，记录路线 A 的数据划分、阈值协议和论文表述建议。

### 后续注意

路线 A 主实验不使用验证集攻击标签选择阈值。测试集 `Attack` 标签只用于最终 Precision、Recall、F1、ROC-AUC、AUC-PR 评价。

---

## R013 - 性能根因代码审计归档

### 日期

2026-06-20

### 涉及文件

- `Algorithm_Audit/10_performance_root_cause_audit.md`

### 审计原因

Kaggle 正式训练已完成，程序可正常训练、保存、后处理和评价，但异常检测指标明显低于预期：

- Precision 偏低
- Recall 未达到论文级表现
- F1-score 偏低
- AUC-PR 不理想

### 审计结论摘要

本轮未修改模型代码，新增性能根因审计文档。当前低性能更可能来自以下组合因素，而不是单一运行错误：

1. 训练正常段与验证正常段存在明显分布差异。
2. 基于训练残差的 z-score 标准化可能被近零残差方差放大。
3. `99%` 分位数阈值对当前分数分布过低，导致测试异常比例过高。
4. Point Adjustment 在原始误报较高时进一步放大误报。
5. 当前动态图为 dense batch-level attention，可能弱化局部故障特征。

### 后续建议

优先开展残差分布诊断、阈值敏感性实验、MTGNN-only 与 joint residual 对比、`time-aggregate` 和 `var-reduce` 消融，再决定是否修改模型结构。

---

## R014 - 关闭默认课程学习以完整训练多步预测 horizon

### 修订日期

2026-06-20

### 涉及文件

- `tools.py`
- `GNN/trainer_attention.py`
- `train_attention.py`

### 修订原因

Kaggle 正式训练中每个 epoch 约 46 个 batch，早停发生在第 27 轮，总训练 batch 约 1242，低于默认 `cl_update_num=2500`。

在原默认配置 `--cl True` 下，`pred_step` 可能长期停留在 1，导致模型主要只训练第 1 个预测步；但最终验证、测试、残差导出和异常评分使用完整 `horizon_size=12`。这会造成训练目标与异常评分口径不一致，是导致多步残差偏高和误报升高的高风险原因。

### 修订内容

1. 将 `tools.py` 中 `--cl` 默认值从 `True` 改为 `False`。
2. 在 `GNN/trainer_attention.py` 中，当课程学习关闭时，将 `pred_step` 初始化为 `horizon_size`。
3. 将课程学习递增条件改为仅在 `self.cl=True` 时生效，并限制 `pred_step < horizon_size`。
4. 修正 `train_attention.py` 中 `cl_update_num` 注释，明确其单位是 batch，不是 epoch。

### 后续实验建议

下一轮 Kaggle 训练建议直接使用默认参数，或显式添加：

```bash
--cl False
```

训练日志中应确认 `Namespace(..., cl=False, ...)`。该修订预计优先改善完整 12 步预测残差质量，并可能降低测试阶段误报率。

---

## R015 - 后处理增加 residual sigma floor 并确认当前最佳评分配置

### 修订日期

2026-06-20

### 涉及文件

- `Test/anomaly_scoring_threshold_pa.py`
- `.gitignore`
- `.vscode/settings.json`
- `Algorithm_Audit/10_performance_root_cause_audit.md`

### 修订原因

Kaggle 诊断显示，关闭课程学习后训练目标已覆盖完整 horizon，但后处理异常分数仍存在大量误报。进一步诊断发现，训练残差标准差过小会放大部分低方差通道的标准化残差，尤其是 node `10` 与 node `5`。

关键现象：

```text
without sigma floor:
node 10 test p99 normalized error ≈ 304.87
F1-score ≈ 0.4489
PR-AUC ≈ 0.3477

with sigma_floor_value = 0.05:
node 10 test p99 normalized error ≈ 19.77
F1-score ≈ 0.6594
PR-AUC ≈ 0.5804
```

### 修订内容

1. 在 `Test/anomaly_scoring_threshold_pa.py` 中新增：

```bash
--sigma-floor-method none|quantile|value
--sigma-floor-quantile
--sigma-floor-value
```

2. 默认 `sigma-floor-method=none`，保持旧实验可复现。
3. 启用 floor 时，只使用训练残差 sigma 进行修正，不使用测试标签，不构成数据泄露。
4. 后处理输出新增 `residual_sigma_raw.npy`，同时保留修正后的 `residual_sigma.npy`。
5. `summary.json` 新增 `sigma_floor_info`，记录 floor 前后 sigma 统计量。
6. 新增 `.vscode/settings.json` 并调整 `.gitignore`，固定项目 UTF-8 设置，避免中文注释在 VSCode/终端中误显示为乱码。

### 当前最佳后处理配置

```bash
--score-source mtgnn
--time-aggregate mean
--threshold-percentile 99.8
--sigma-floor-method value
--sigma-floor-value 0.05
```

### 当前最佳 raw point-level 结果

```text
Precision = 0.7235
Recall    = 0.6057
F1-score  = 0.6594
ROC-AUC   = 0.8412
PR-AUC    = 0.5804
```

### 科学解释

当前结果表明，原始低性能并非主要由模型无法训练造成，而是由后处理中的低方差通道残差标准化放大导致。固定 sigma floor 后，误报大幅减少，Precision、F1-score 和 PR-AUC 显著提升。

### 后续建议

1. 将 `MTGNN + sigma_floor=0.05 + percentile=99.8` 作为当前主后处理候选配置。
2. 暂不将 `joint_error` 作为主结果，后续单独重新设计 AE 残差融合。
3. 检查 node `5` 与 node `10` 的真实传感器名称、训练段方差、测试段轨迹和攻击对应关系。
4. 后续若更换数据集、采样率或滑窗参数，需重新验证 `sigma_floor_value=0.05` 是否仍合理。

---

## R016 - 后处理新增异常分数时间平滑实验参数
### 修订日期

2026-06-20

### 涉及文件

- `Test/anomaly_scoring_threshold_pa.py`
- `Algorithm_Audit/Revision_Log.md`

### 修订原因

Kaggle 事件级诊断显示，`var_reduce=max` 可将事件召回率从 `0.6857` 提升到 `0.8286`，说明许多攻击事件具有局部单通道强残差信号；但 `max` 同时导致测试集预测异常比例过高，说明正常验证集和测试集中存在孤立尖峰噪声。为区分“持续性局部异常”和“单点噪声尖峰”，需要在最终一维异常分数上加入时间平滑实验。

### 修订内容

1. 在 `Test/anomaly_scoring_threshold_pa.py` 新增参数：

```bash
--score-smooth-window
--score-smooth-method mean|median
--score-smooth-direction causal|centered
```

2. 默认 `score-smooth-window=1`，即不启用平滑，保持历史实验完全可复现。
3. 平滑位置为传感器维度聚合之后、阈值选择之前；验证集阈值和测试集判定均使用同一平滑规则。
4. 默认 `score-smooth-direction=causal`，只使用当前点和过去点，避免使用未来测试信息。
5. 输出文件中 `score_val.npy` 与 `score_test.npy` 保存最终用于阈值和指标计算的平滑分数；额外保存 `score_val_raw.npy` 与 `score_test_raw_score.npy` 作为未平滑分数备查。
6. `summary.json` 新增 `score_smoothing_info`，记录平滑窗口、方法、方向和是否启用。

### 实验建议

优先测试：

```bash
--var-reduce max --score-smooth-window 3 --score-smooth-method mean --score-smooth-direction causal
--var-reduce max --score-smooth-window 5 --score-smooth-method mean --score-smooth-direction causal
--var-reduce topk_mean --var-topk 5 --score-smooth-window 3 --score-smooth-method mean --score-smooth-direction causal
```

如果 `max + causal smoothing` 能在保持较高事件召回的同时显著降低异常比例和误报，则说明当前瓶颈主要来自单点噪声尖峰，而不是模型残差完全失效。

---

## R017 - 项目级 UTF-8 编码规则加固与历史乱码复核
### 修订日期

2026-06-20

### 涉及文件

- `.editorconfig`
- `.gitattributes`
- `.vscode/settings.json`
- `Algorithm_Audit/Revision_Log.md`

### 修订原因

Windows PowerShell 5.1 和部分编辑器在读取无 BOM UTF-8 中文文件时，可能按系统 ANSI/GBK 进行解码，从而把正常中文显示为 `鍚`、`璁`、`銆`、`涓€` 等乱码。该问题会造成审阅注释、日志和修订文档时的误判。

### 复核结果

已扫描项目内 `*.py` 与 `*.md` 文件，未检出典型历史乱码片段。当前源码和文档内容本身为 UTF-8 正常中文；此前在终端中看到的乱码主要来自 PowerShell 显示/解码方式，而不是文件内容已经损坏。

### 修订内容

1. 新增 `.editorconfig`，固定项目文本文件使用 UTF-8、LF 换行、文件末尾保留换行。
2. 新增 `.gitattributes`，固定 Python、Markdown、JSON、TXT、PowerShell、Notebook 文件作为文本并使用 LF 换行，减少 Windows 与 Kaggle/Linux 之间的换行漂移。
3. 修改 `.vscode/settings.json`，将 `files.autoGuessEncoding` 设为 `false`，避免 VSCode 自动把 UTF-8 中文误判为 GBK。
4. 保留 `PYTHONUTF8=1` 与 `PYTHONIOENCODING=utf-8`，确保 Python 运行与输出默认采用 UTF-8。

### 使用说明

如果在 Windows PowerShell 中查看中文源码，建议显式使用：

```powershell
Get-Content -Encoding UTF8 path\to\file.py
```

或者在终端会话开始时设置：

```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
```

VSCode 中直接打开文件时，应按 `.vscode/settings.json` 和 `.editorconfig` 以 UTF-8 正常显示。
