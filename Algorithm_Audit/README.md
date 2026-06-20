# 算法审查记录

本文件夹用于记录 `SADP20260524` 项目的系统性算法审查过程，重点面向工业多变量时间序列异常检测任务。

## 审查原则

- 先审查数据处理，再审查模型结构。因为如果存在数据泄漏、标签污染或划分错误，后续模型结果都会失去可信度。
- 在每个模块记录输入输出张量形状。
- 区分“已确认问题”“设计选择”“论文写作影响”。
- 审查记录与代码修改分开管理。除非明确确认修复方案，否则先不直接改动核心代码。

## 模块审查顺序

1. `data_maker.py` 和 `data_loader.py`：数据划分、标准化、滑动窗口、标签处理。
2. `train_attention.py`：训练循环、验证流程、测试流程、保存逻辑。
3. `attention_matrix.py`：自适应动态图构建。
4. `GNN/model_attention.py`：MTGNN 主体前向传播与张量维度。
5. `GNN/trainer_attention.py`：损失函数、课程学习、反归一化、自编码器分支。
6. `Test/anomaly_scoring_threshold_pa.py`：异常分数、阈值选择、Point Adjustment。

## 当前文档

- `01_data_pipeline_audit.md`：数据生成与加载流程的第一轮审查。
- `02_project_function_map.md`：全项目程序功能地图与当前主链路识别。
- `03_main_chain_hard_errors.md`：当前主训练链路的硬错误、断点和高风险不一致。
- `04_ae_mtg_nn_joint_residual_protocol.md`：AE 与 MTGNN 并列模块、联合损失、综合残差和后处理评分协议。
- `05_algorithm_completeness_audit.md`：当前主算法链路、数学目标、异常评分闭环与待确认问题。
- `06_stepwise_manual_ai_review_plan.md`：后续人工 + AI 联合逐步复核路线。
- `07_point_level_evaluation_protocol.md`：滑动窗口预测结果反投影回原始点级评价的协议说明。
- `08_postprocess_strict_protocol_audit.md`：后处理严格数学协议、数据隔离和点级评价修订说明。
- `09_route_a_semisupervised_threshold_protocol.md`：路线 A 半监督数据划分与无监督阈值协议。
- `10_performance_root_cause_audit.md`：异常检测性能偏低的根因排查记录。
- `11_gdn_style_prediction_scoring_protocol.md`：基于 GDN 的 MTGNN 预测分支单一残差评分协议。
- `Revision_Log.md`：代码修订记录总表，后续每次实质修改都在这里登记。

## 结果保存结构

当前派生数据与实验结果的统一保存结构见：

- `Save/README.md`

核心约定：

- `Save/Datasets/`：预处理后的 `train/val/test` 数据包。
- `Save/Experiments/`：训练实验结果、模型权重、预测张量、动态图和汇总图表。
- `Save/Postprocess/`：跨实验后处理汇总结果预留目录。
