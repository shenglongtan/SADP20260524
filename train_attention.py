#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os  # 导入操作系统接口模块，用于处理文件路径和目录创建
import json  # 导入 JSON 模块，用于保存残差导出说明与检查点元信息
import hues  # 导入 hues 库，用于在终端输出高亮、彩色的日志信息，方便监控训练状态
import torch  # 导入 PyTorch 深度学习核心基础库
import numpy as np  # 导入 NumPy 库，用于高效的矩阵运算和多维数组处理
import pandas as pd  # 导入 Pandas 库，用于构建数据表格、序列化保存多步预测误差等
import torch.optim as optim  # 导入 PyTorch 的优化器模块（如 Adam 等，具体在 Trainer 中实例化）

from tqdm import tqdm  # 导入 tqdm 库，用于在控制台渲染直观的进度条
from datetime import datetime  # 导入 datetime 模块，用于获取当前时间以动态生成保存目录

# ----------------- 自定义模块导入 -----------------
from GNN.model_attention import MTGNN  # 导入修改后的基于内置 Attention 动态构图的 MTGNN 模型
from GNN.trainer_attention import Trainer as MTGNNTrainer  # 导入模型训练引擎，封装了前向传播、反向传播和优化步骤
from GNN.ae import WindowAE  # 导入时间窗口自编码器，用于异常检测的联合重构分支
from data_loader import DataLoader, load_local_dataset  # 导入数据加载器函数（需确保已配置为按需加载的 Lazy 模式）
from tools import get_input_paras, load_adj, load_node_feature, get_save_path, to_pickle, calc_error, plt_error, \
    plt_loss, check_folder, set_random_seed, get_adj_m, test_model, plt_box, save_args_to_json, create_loss_df
# 导入一系列工具函数：参数获取、误差计算、可视化绘图、结果序列化等


def safe_path_tag(value: str) -> str:
    """
    将数据集名、模型名等字符串转换为适合目录名的短标签。
    """
    safe = "".join(ch if ch.isalnum() or ch in ["-", "_"] else "_" for ch in str(value))
    safe = "_".join(part for part in safe.split("_") if part)
    return safe or "item"


def infer_dataset_tag(data_path: str) -> str:
    """
    从文件或目录路径中提取数据集标签。
    """
    norm_path = os.path.normpath(str(data_path))
    base = os.path.basename(norm_path)
    stem, ext = os.path.splitext(base)
    return safe_path_tag(stem if ext else base)


def build_experiment_name(args, timestamp: str) -> str:
    """
    构建包含核心复现实验参数的实验目录名。
    """
    dataset_tag = infer_dataset_tag(args.data_path)
    model_tag = safe_path_tag(args.model)
    return (
        f"{timestamp}_{dataset_tag}_{model_tag}"
        f"_w{args.window_size}_h{args.horizon_size}_s{args.sliding_step}_seed{args.seed}"
    )


def prepare_run_dirs(run_dir: str) -> dict:
    """
    为单次重复实验创建标准化子目录。
    """
    dirs = {
        "run": run_dir,
        "checkpoints": os.path.join(run_dir, "checkpoints"),
        "predictions": os.path.join(run_dir, "predictions"),
        "graphs": os.path.join(run_dir, "graphs"),
        "postprocess": os.path.join(run_dir, "postprocess"),
    }
    for path in dirs.values():
        check_folder(path)
    return dirs


def _normalize_model_output(output_y: torch.Tensor, true_y_4d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    将 MTGNN 输出统一为 [B, 1, N, H] 和 [B, N, H] 两种视图。
    """
    if output_y.dim() == 4:
        if output_y.size(1) != 1:
            raise ValueError(f"MTGNN 输出应为 [B, 1, N, H]，收到: {tuple(output_y.shape)}")
        pred_4d = output_y
        pred_3d = output_y[:, 0, :, :]
    elif output_y.dim() == 3:
        pred_3d = output_y
        pred_4d = output_y.unsqueeze(1)
    else:
        raise ValueError(f"MTGNN 输出应为 [B, 1, N, H] 或 [B, N, H]，收到: {tuple(output_y.shape)}")

    if pred_4d.shape != true_y_4d.shape:
        raise ValueError(f"预测张量与真实张量形状不一致: pred={tuple(pred_4d.shape)}, true={tuple(true_y_4d.shape)}")
    return pred_4d, pred_3d


def _inverse_transform_3d(tensor_3d: torch.Tensor, scalers) -> np.ndarray:
    """
    将 [Samples, Nodes, Horizon] 标准化张量还原为物理量纲数组。
    """
    data = tensor_3d.detach().cpu().numpy().astype(np.float32)
    if scalers is None:
        return data

    if hasattr(scalers, "mean") and hasattr(scalers, "std"):
        mean = np.asarray(scalers.mean, dtype=np.float32)
        std = np.asarray(scalers.std, dtype=np.float32)
        if mean.ndim == 0:
            return (data * std + mean).astype(np.float32)
        return (data * std.reshape(1, -1, 1) + mean.reshape(1, -1, 1)).astype(np.float32)

    transformed = scalers.inverse_transform(tensor_3d.detach().clone())
    if isinstance(transformed, torch.Tensor):
        return transformed.detach().cpu().numpy().astype(np.float32)
    return np.asarray(transformed, dtype=np.float32)


def collect_module_residual_outputs(
        loader,
        true_y_np: np.ndarray,
        scalers,
        model,
        ae_model,
        device,
        ae_beta: float,
        ae_lambda: float,
) -> dict:
    """
    导出异常检测阶段所需的 MTGNN 预测误差、AE 重构误差和综合误差。

    约定：
    - sensor_true / sensor_pred 为物理量纲传感器数值，不是攻击标签。
    - mtgnn_pred_error 为标准化空间中的 |真实传感器值 - MTGNN预测传感器值|。
    - ae_rec_*_error 为标准化空间中的逐元素平方重构误差，对齐训练阶段 MSE 重构损失。
    - joint_error = mtgnn_pred_error + beta * (ae_rec_real_error + lambda * ae_rec_pred_error)。
    """
    model.eval()
    use_ae = ae_model is not None
    if use_ae:
        ae_model.eval()

    true_std_parts, pred_std_parts = [], []
    mtgnn_error_parts, ae_real_error_parts, ae_pred_error_parts, joint_error_parts = [], [], [], []

    for x, y in loader.get_iterator():
        input_x = torch.Tensor(x).transpose(1, 3).to(device)
        true_y_4d = torch.Tensor(y).transpose(1, 3)[:, 0:1, :, :].to(device)

        with torch.no_grad():
            output_y, _ = model(input_x)
            pred_y_4d, pred_y_3d = _normalize_model_output(output_y, true_y_4d)
            true_y_3d = true_y_4d[:, 0, :, :]

            mtgnn_pred_error = torch.abs(true_y_3d - pred_y_3d)
            if use_ae:
                hist = input_x[:, 0:1, :, :]
                true_win = torch.cat([hist, true_y_4d], dim=3)
                pred_win = torch.cat([hist, pred_y_4d], dim=3)
                rec_true = ae_model(true_win)
                rec_pred = ae_model(pred_win)
                horizon_size = true_y_4d.size(-1)
                ae_rec_real_error = torch.square(rec_true[..., -horizon_size:] - true_y_4d).squeeze(1)
                ae_rec_pred_error = torch.square(rec_pred[..., -horizon_size:] - pred_y_4d).squeeze(1)
                joint_error = mtgnn_pred_error + ae_beta * (ae_rec_real_error + ae_lambda * ae_rec_pred_error)
            else:
                ae_rec_real_error = torch.zeros_like(mtgnn_pred_error)
                ae_rec_pred_error = torch.zeros_like(mtgnn_pred_error)
                joint_error = mtgnn_pred_error

        true_std_parts.append(true_y_3d.detach().cpu())
        pred_std_parts.append(pred_y_3d.detach().cpu())
        mtgnn_error_parts.append(mtgnn_pred_error.detach().cpu())
        ae_real_error_parts.append(ae_rec_real_error.detach().cpu())
        ae_pred_error_parts.append(ae_rec_pred_error.detach().cpu())
        joint_error_parts.append(joint_error.detach().cpu())

    target_size = int(true_y_np.shape[0])

    def cat_trim(parts):
        return torch.cat(parts, dim=0)[:target_size, ...]

    true_std = cat_trim(true_std_parts)
    pred_std = cat_trim(pred_std_parts)
    mtgnn_pred_error = cat_trim(mtgnn_error_parts).numpy().astype(np.float32)
    ae_rec_real_error = cat_trim(ae_real_error_parts).numpy().astype(np.float32)
    ae_rec_pred_error = cat_trim(ae_pred_error_parts).numpy().astype(np.float32)
    joint_error = cat_trim(joint_error_parts).numpy().astype(np.float32)

    sensor_true = _inverse_transform_3d(true_std, scalers)
    sensor_pred = _inverse_transform_3d(pred_std, scalers)

    return {
        "sensor_true": sensor_true,
        "sensor_pred": sensor_pred,
        "mtgnn_pred_error": mtgnn_pred_error,
        "mtgnn_pred_error_physical": np.abs(sensor_true - sensor_pred).astype(np.float32),
        "ae_rec_real_error": ae_rec_real_error,
        "ae_rec_pred_error": ae_rec_pred_error,
        "joint_error": joint_error,
    }


def save_residual_outputs(
        prediction_dir: str,
        val_outputs: dict,
        test_outputs: dict,
        args,
        use_ae: bool,
        train_outputs: dict = None,
) -> None:
    """
    保存最终异常检测评分需要的传感器数值与模块级残差文件。
    """
    split_map = {
        "val": val_outputs,
        "test": test_outputs,
    }
    if train_outputs is not None:
        split_map["train"] = train_outputs

    name_map = {
        "sensor_true": {"train": "y_train_true.npy", "val": "y_val_true.npy", "test": "y_true.npy"},
        "sensor_pred": {"train": "y_train_pred.npy", "val": "y_val_pred.npy", "test": "y_pred.npy"},
        "mtgnn_pred_error": {
            "train": "train_mtgnn_pred_error.npy",
            "val": "val_mtgnn_pred_error.npy",
            "test": "test_mtgnn_pred_error.npy",
        },
        "mtgnn_pred_error_physical": {
            "train": "train_mtgnn_pred_error_physical.npy",
            "val": "val_mtgnn_pred_error_physical.npy",
            "test": "test_mtgnn_pred_error_physical.npy",
        },
        "ae_rec_real_error": {
            "train": "train_ae_rec_real_error.npy",
            "val": "val_ae_rec_real_error.npy",
            "test": "test_ae_rec_real_error.npy",
        },
        "ae_rec_pred_error": {
            "train": "train_ae_rec_pred_error.npy",
            "val": "val_ae_rec_pred_error.npy",
            "test": "test_ae_rec_pred_error.npy",
        },
        "joint_error": {"train": "train_joint_error.npy", "val": "val_joint_error.npy", "test": "test_joint_error.npy"},
    }

    for split, outputs in split_map.items():
        for key, names in name_map.items():
            np.save(os.path.join(prediction_dir, names[split]), outputs[key])

    meta = {
        "use_ae": bool(use_ae),
        "ae_beta": float(args.ae_beta),
        "ae_lambda": float(args.ae_lambda),
        "best_model_selection": "L_total" if use_ae else "L_pred (AE disabled ablation)",
        "sensor_value_files": {
            "y_train_true.npy": "training true sensor values in physical units, used only for residual-stat export",
            "y_train_pred.npy": "training MTGNN predicted sensor values in physical units",
            "y_val_true.npy": "validation true sensor values in physical units, shape [samples, nodes, horizon]",
            "y_val_pred.npy": "validation MTGNN predicted sensor values in physical units",
            "y_true.npy": "test true sensor values in physical units",
            "y_pred.npy": "test MTGNN predicted sensor values in physical units",
        },
        "residual_files": {
            "mtgnn_pred_error": "standardized absolute prediction error |x_true - x_pred|",
            "mtgnn_pred_error_physical": "physical-unit absolute prediction error |x_true - x_pred|",
            "ae_rec_real_error": "standardized squared AE reconstruction error for the real window future segment",
            "ae_rec_pred_error": "standardized squared AE reconstruction error for the MTGNN-generated window future segment",
            "joint_error": "mtgnn_pred_error + ae_beta * (ae_rec_real_error + ae_lambda * ae_rec_pred_error); if AE is disabled, joint_error equals mtgnn_pred_error",
        },
        "label_semantics": {
            "y_true_y_pred": "sensor values, not normal/attack labels",
            "y_attack_window_y_attack_point": "ground-truth normal/attack labels when available in dataset npz files",
            "pred_test_raw_pred_test_pa": "postprocess-predicted normal/attack labels",
        },
    }
    with open(os.path.join(prediction_dir, "residual_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def save_prediction_extras(prediction_dir: str, dataloader: dict) -> None:
    """
    保存点级后处理反投影所需的时间索引与攻击标签。

    这些标签不参与模型训练，只供后处理阶段执行验证集阈值搜索与测试集最终评价。
    """
    for split in ["train", "val", "test"]:
        for key in ["y_time", "y_attack_window", "y_attack_point"]:
            data_key = f"{split}_{key}"
            if data_key in dataloader:
                np.save(os.path.join(prediction_dir, f"{split}_{key}.npy"), dataloader[data_key])


def main(args):
    """
    模型训练、验证和测试的主控函数。
    :param args: 从命令行或配置文件解析得到的超参数集合
    """
    # 1. 设定全局随机种子，确保实验在相同的软硬件环境下结果绝对可复现
    set_random_seed(args.seed)
    
    # 2. 获取计算硬件：自动检测是否支持 CUDA（GPU），否则降级使用 CPU 计算
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hues.info(f"  当前计算硬件已挂载至: [{device}]  ")
    run_dirs = prepare_run_dirs(args.save)
    
    # 3. 加载本地数据集，生成返回包含训练集、验证集、测试集、数据标准化缩放器
    dataloader = load_local_dataset(
    path=args.data_path,               # [1] 数据源路径：读取本地已序列化的工业时间序列数据集
    batch_size=args.batch_size,        # [2] 训练集批次大小：由参数解析器动态传入的极限保命配置
    valid_batch_size=args.batch_size,  # [3] 验证集批次大小：与训练集严格同步，防止评估时爆显存
    test_batch_size=args.batch_size,   # [4] 测试集批次大小：与训练集严格同步，保证最终测试的安全
    window_size=args.window_size,      # [5] 历史观测窗口 (L)：模型每次提取时空特征的历史时间步数
    horizon_size=args.horizon_size,    # [6] 未来预测视野 (p)：模型单次需要预测的未来时间步长
    sliding_step=args.sliding_step     # [7] 滑动窗口步长：确保命令行实验参数真实作用于在线数据制作
    )
    # ==============================================================================
    # 4. 数据自适应尺寸探测与参数动态覆写 (Dynamic Parameter Overriding)
    # 尽管用户在 argparse 中预设了 window_size, feature_num 等维度参数，
    # 但实际清洗可能导致维度改变。为绝对杜绝 "Tensor Shape Mismatch"，
    # 此处强制覆写全局 args 参数，作为实例化深层神经网络的唯一尺寸金标准。
    # ==============================================================================
    # 从特征张量 X 提取真实形状：预期结构通常为 [Batch_size, Window_size, Node_num, Feature_num]
    # 下划线 '_' 表示我们不需要提取 Batch_size，只抓取后三个定义模型架构维度的核心参数
    _, args.window_size, args.node_num, args.feature_num = dataloader['train_loader'].get_x_shape()
    
    # 从标签张量 Y 提取真实形状：预期结构通常为 [Batch_size, Horizon_size, Node_num, Feature_num]
    # 我们只关心预测的未来步长 (Horizon_size) 是否与模型期望对齐
    _, args.horizon_size, _, _ = dataloader['train_loader'].get_y_shape()

    # (注：基于物理距离或传感器连通性的旧版静态先验图加载逻辑已被注释弃用，
    # 当前网络已全面由自带的 Attention 模块进行纯数据驱动的动态拓扑结构学习与推断)

    # ==============================================================================
    # 5. 挂载量纲逆归一化器 (Inverse Scaler) 与调度器占位
    # 深度学习模型只能在无量纲的归一化空间 (如 0~1 或 均值0方差1) 内进行高效且稳定的梯度下降。
    # 然而，在撰写论文和评估工业检测精度时，MAE/RMSE 误差指标必须具备实际的工程物理意义
    # （例如：误差是 0.05 米的水位，还是 2 帕斯卡的压力）。
    # 提取的 scaler 将被传入 Trainer 中，确保在评估(Eval)阶段能将模型输出还原回真实物理量纲后再计算误差。
    # ==============================================================================
    scalers = dataloader['scaler']
    
    # 学习率调度器 (Learning Rate Scheduler) 占位符。在当前框架中，监控验证集性能并触发学习率阶梯衰减的核心逻辑，
    # 已经高度内聚并封装在了自定义的 Optim 优化器或 Trainer 引擎内部。因此，在外层主控脚本中，不再需要显式干预和维护 PyTorch 原生的 scheduler 实例。
    scheduler = None    
    
    # 6. 根据命令行参数选择并实例化核心深度学习模型
    if args.model == 'MTGNN':
        # 实例化 MTGNN 网络结构
        model = MTGNN(
            device=device,                     # 指定模型张量存放的硬件设备
            node_num=args.node_num,            # 拓扑图的节点总数 (N)
            feature_num=args.feature_num,      # 变量的特征通道数 (C)
            mix_hop_depth=args.mix_hop_depth,  # GCN 空间多跳传播的最大深度 K
            drop_rate=args.drop_rate,          # Dropout 神经元失活率，防止模型过拟合
            dl_exp=args.dl_exp,                # 时间卷积的空洞扩张率底数（控制感受野）
            conv_channels=args.conv_channels,  # 时空卷积模块中的隐藏层特征通道数
            residual_channels=args.residual_channels,  # 残差连接主干道的通道数
            skip_channels=args.skip_channels,  # 跨层跳跃连接汇总时的通道数
            end_channels=args.end_channels,    # 最终输出预测层前的全连接层特征数
            window_size=args.window_size,      # 观测序列长度 (L)
            horizon_size=args.horizon_size,    # 预测序列长度 (p)
            layer_num=args.layer_num,          # 堆叠的 (时域+空域) 特征提取模块总层数
            prop_alpha=args.prop_alpha,        # MixHop 空间传播中保留自身特征的比例系数
            layer_norm_affline=True            # 启用 LayerNorm 的可学习仿射变换参数
            )

        ae_model = None
        # 检查是否启用了自编码器辅助分支（常用于异常检测的重构误差计算）
        if args.use_ae:
            # 实例化处理时间窗口的自编码器模型
            ae_model = WindowAE(in_channels=1, hidden_channels=args.ae_channels)
            hues.info("  AE 重构模块已启用：训练目标采用 L_total = L_pred + beta * (L_rec_real + lambda * L_rec_pred)。  ")
        else:
            hues.warn("  AE 重构模块未启用：当前为消融模式，最佳模型选择与综合残差均退化为 MTGNN 预测误差。  ")

        # 7. 组装训练引擎（Trainer），将模型、优化器配置、损失函数等统一封装，简化主循环代码
        engine = MTGNNTrainer(
            model,                             # 传入刚才实例化的预测模型
            args.horizon_size,                 # 传入预测步长(p)，用于计算误差
            scalers,                           # 传入数据逆归一化器
            device,                            # 传入计算设备
            ae_model=ae_model,                 # 传入自编码器模型（可能为 None）
            use_ae=args.use_ae,                # 布尔标识：是否在引擎中激活 AE 前向传播
            ae_beta=args.ae_beta,              # Loss 权重：AE 纯重构损失的比例
            ae_lambda=args.ae_lambda,          # Loss 权重：AE 结合预测结果的混合重构比例
            cl=args.cl,                        # 布尔标识：是否启用课程学习 (Curriculum Learning) 策略
            cl_update_num=args.cl_update_num,  # 课程学习参数：每隔多少个 batch 增加一次预测任务难度
            gd_clip=args.gd_clip,              # 梯度裁剪阈值，防止 RNN/GNN 训练中梯度爆炸
            opt_lr=args.opt_lr,                # 优化器（如 Adam）的初始学习率
            opt_wd=args.opt_wd,                # 优化器的权重衰减系数（L2 正则化项）
        )
    else:
        # 如果输入的 args.model 不受支持，则抛出异常阻断程序
        raise ValueError('Model type not supported!')  

    # 打印最终确认传入的所有模型与环境参数，留存日志
    hues.info('当前执行配置参数列表:', args)  

    # ======================================  1. Train 训练阶段  ======================================
    # 用于记录每个 Epoch 训练集与验证集的平均 Loss，最终用于绘制 Loss 下降曲线图
    train_l, valid_l = [], []  
    # 历史最小验证误差阈值，初始化为正无穷 (infinity)，用于对比寻找模型性能最强时刻的权重
    min_loss = np.inf          
    
    # --- 新增：早停机制 (Early Stopping) 核心参数初始化 ---
    # getattr 确保即使 args 中未显式传递该参数，程序也能拥有安全的后备默认值
    patience = getattr(args, 'patience', 10)     # [容忍度] 允许模型连续多少轮在验证集上不取得有效进步（默认 10 轮）
    min_delta = getattr(args, 'min_delta', 1e-4) # [阈值] 判定为“有效进步”的最小降低幅度，防止极微小单调下降导致无效等待
    epochs_no_improve = 0                        # [计数器] 连续未取得“有效进步”的 Epoch 轮数累加器
    best_epoch = 0                               # [指针] 记录诞生历史最佳成绩 (min_loss) 的确切 Epoch 轮次

    # --- 计算进度条满格长度 ---
    # args.epoch_num (外层总轮数) × dataloader['train_loader'].get_x_shape()[0] (单次 Epoch 包含的 Batch 批次数)
    # 这个总数用于精确定位 tqdm 进度条的 100% 终点
    total_bt_num = args.epoch_num * dataloader['train_loader'].get_x_shape()[0]
    
    # 开启 tqdm 可视化控制台进度条，实时监控模型在每个 Batch 上的推进状态
    with tqdm(total=total_bt_num, desc="⏳️ 模型训练中", ncols=100, unit="Batch") as pbar:
        
        # ---------------- 核心训练外层大循环 (Epoch 级别) ----------------
        # 每次遍历完整的一遍数据称为 1 个 Epoch
        for cur_epoch in range(args.epoch_num):
            
            # 【防止记忆陷阱 (Overfitting)】：每个 Epoch 启动前，强制打乱时间窗口样本的排列顺序。
            # 让模型无法依赖数据的出场顺序来作弊，逼迫其学习时间序列间的深层逻辑映射。
            dataloader['train_loader'].shuffle()  
            
            # 临时列表：用于存放本 Epoch 内产生的成百上千个 Batch 的具体训练损失值
            cur_train_l = []  

            # 动态关系图收集器初始化
            if args.save_adj:
                A_sum = None  # 零时开辟空间：用于累加本 Epoch 内所有 Batch 推断出来的 Attention 矩阵
                A_cnt = 0     # 累加动作执行次数的计数器，最后用 A_sum / A_cnt 即可算出平滑的平均拓扑图

            # ---------------- 核心训练内层微循环 (Batch 级别) ----------------
            # 从训练加载器中，迭代获取批次特征张量 x 和批次标签张量 y
            for iter_num, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
                
                # 【张量特征维度倒换 (Tensor Permutation)】:
                # Dataloader 传出的 x 原生形状为: [B, L, N, C]
                # PyTorch 的 (时/空/图) 卷积算子通常约定【特征通道层】必须紧挨着 Batch 维度。
                # 通过 transpose(1, 3) 翻转第 1 维 (L) 和第 3 维 (C)。
                # 转换后送入 GPU 的 input_x 形状为: [B, C, N, L]
                input_x = torch.Tensor(x).transpose(1, 3).to(device)
                
                # 标签数据 y 原生形状为: [B, p, N, C]
                # 同样经过转换后，真实目标 true_y 形状变为: [B, C, N, p]
                true_y = torch.Tensor(y).transpose(1, 3).to(device)

                # ---------------- 显存防爆与子图截断机制 (Sub-graph Mechanism) ----------------
                # 算法原理：若节点数 N 非常庞大，让 Attention 层计算全局两两节点的注意力图会产生 O(N^2) 的爆炸显存占用。
                # 这里的解决思路是“大图化小图”：每次只挑选部分节点构成子图，在子图内部训练关联。
                
                # 触发条件：每经过 args.shuffle_num 个 Batch，重新洗牌生成全局节点 (N) 的随机排列索引序列
                if iter_num % args.shuffle_num == 0:  
                    shuffle_i = np.random.default_rng(seed=cur_epoch * iter_num).permutation(args.node_num)

                # 计算等分后的每个子图中，应该包含多少个独立节点
                sub_num = int(args.node_num / args.split_num)  

                # 遍历这几个切分好的区块 (j 代表当前处理的是第几个区块)
                for j in range(args.split_num):  
                    # 摘取当前子区块的节点 ID 索引（如果是最后一块，则兜底截取所有剩余节点，防止不能整除的遗漏）
                    if j != args.split_num - 1:
                        node_id = shuffle_i[j * sub_num:(j + 1) * sub_num]
                    else:
                        node_id = shuffle_i[j * sub_num:]
                    
                    # 将这一小批节点 ID 转为符合 PyTorch 数据规范的 long 类型张量，并送至显存
                    node_id = torch.tensor(node_id, dtype=torch.long).to(device)

                    # 【切片数据】：利用 Python 优雅的切片机制，在第三维 (节点维度 N) 摘出这批属于 node_id 的特征和标签
                    split_x = input_x[:, :, node_id, :]
                    split_y = true_y[:, :, node_id, :]
                    
                    # 【送入引擎前向与反向传播】：
                    # 参数说明：split_y[:, 0, :, :] 中的 '0' 表示我们只抽取出第 0 个特征通道 (主要预测目标变量) 来计算误差。
                    # engine.train 将处理模型的前向推断、损失结算以及利用 Optimizer 优化调整模型权重。
                    # 返回的 errors: 是本次反向传播优化后的该子图 Loss。
                    # 返回的 adj_m_save: 是模型内部的 Attention 层针对当前子图数据临时推断出来的关联权重矩阵。
                    errors, adj_m_save = engine.train(split_x, split_y[:, 0, :, :], node_id)
                    cur_train_l.append(errors)  

                    # 【网络拓扑结构图隐式收集】：收集模型通过数据驱动方式学会的节点关联
                    if args.save_adj:
                        # 1. 维度压缩：若 Attention 输出是对每一个 Batch 样本都有一张独立图 [B, N, N]，则取平均，降维融合成该批次的总体代表图 [N, N]
                        A_b = adj_m_save.mean(dim=0) if adj_m_save.ndim == 3 else adj_m_save
                        
                        # 2. 【防爆核心】：必须使用 .detach() 斩断这张图与 PyTorch 动态计算图（Autograd）的历史依赖关系。
                        # 若不 detach，历史的几千次图运算都会强行挂载在显存中以备求导，会导致内存灾难性溢出。
                        A_b = A_b.detach()  
                        
                        # 3. 稳态累加：将清洗好的二维关系图直接累加进内存变量中
                        if A_sum is None:
                            A_sum = A_b.clone()
                        else:
                            A_sum += A_b
                        A_cnt += 1

                # 当前这 1 个大 Batch (包含了所有子图的分批训练) 全部运算完毕，推动一次进度条
                pbar.update(1)  

            # ---------------- Epoch 级别统计与收尾阶段 ----------------
            # 当前 Epoch 的训练周期结束，求出这几千个子图所产生的误差均值，作为代表本轮训练水平的最终分数
            train_metrics = np.array(cur_train_l)
            train_epoch_loss = np.mean(train_metrics[:, 0])
            train_pred_loss = np.mean(train_metrics[:, 1])
            train_rec_real_loss = np.mean(train_metrics[:, 2])
            train_rec_pred_loss = np.mean(train_metrics[:, 3])
            train_l.append(train_epoch_loss) # 装载进全局列表

            # ---------------- 验证阶段 (Validation Phase) ----------------
            # 原理：模型必须在从未见过的验证集中参加“模拟大考”，这是避免模型对训练集死记硬背的唯一鉴定标准。
            cur_valid_l = []  
            for _, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
                # 在验证集中，执行和训练集完全相同的张量流转规范: 将 [B, L, N, C] 转为 [B, C, N, L]
                input_x = torch.Tensor(x).transpose(1, 3).to(device)
                true_y = torch.Tensor(y).transpose(1, 3).to(device)
                
                # 【调用验证管线】：执行 engine.eval。
                # 引擎内会开启 model.eval() (关闭 BatchNorm 和 Dropout 等随机干扰成分)，
                # 并且处于 torch.no_grad() 保护下，模型不会在此阶段更新任何大脑权重，只专心纯推断。
                errors = engine.eval(input_x, true_y[:, 0, :, :])
                cur_valid_l.append(errors)

            # 统计汇总本次验证集考试的总体平均错误率
            valid_metrics = np.array(cur_valid_l)
            valid_epoch_loss = np.mean(valid_metrics[:, 0])
            valid_pred_loss = np.mean(valid_metrics[:, 1])
            valid_rec_real_loss = np.mean(valid_metrics[:, 2])
            valid_rec_pred_loss = np.mean(valid_metrics[:, 3])
            valid_l.append(valid_epoch_loss)

            # 更新 tqdm 进度条最右侧的监控面板，让实验者直观追踪 Train Loss 和 Val Loss
            pbar.set_postfix({
                'TrainTotal': f'{train_epoch_loss:.4f}',
                'ValTotal': f'{valid_epoch_loss:.4f}',
                'ValPred': f'{valid_pred_loss:.4f}',
            })

            # ---------------- 动态图落盘持久化 ----------------
            # 若配置了开启保存功能，并且成功累加到了矩阵数据
            if args.save_adj and (A_sum is not None and A_cnt > 0):
                # 用总累加图除以有效累加次数，提炼出极其平滑的，属于该 Epoch 专属的拓扑学习图
                # 并将其送出 GPU，搬运回普通 CPU 内存准备落盘，释放显存空间
                A_epoch = (A_sum / A_cnt).detach().cpu()  
                
                # 同时存为 PT 格式(便于后续二次被 PyTorch 加载) 以及 NPY 格式(便于 Python 做热力可视化分析)
                torch.save(A_epoch, os.path.join(run_dirs["graphs"], f"AdjMatrix_epoch{cur_epoch:03d}.pt"))
                np.save(os.path.join(run_dirs["graphs"], f"AdjMatrix_epoch{cur_epoch:03d}.npy"), A_epoch.numpy())

            # ---------------- 【核心】早停判决与巅峰权重抢救机制 ----------------
            # 判断逻辑 1: 本次大考成绩 (valid_epoch_loss) 是否打破了历史最好的记录 (min_loss)？
            # 只要破了纪录，不管降了多少，它就是目前物理上“最准的一版模型”，立即备份权重！
            if valid_epoch_loss < min_loss:
                best_epoch_candidate = cur_epoch + 1
                best_model_path = os.path.join(run_dirs["checkpoints"], "model.pth")
                best_ae_path = os.path.join(run_dirs["checkpoints"], "ae_model.pth")
                best_joint_path = os.path.join(run_dirs["checkpoints"], "joint_checkpoint.pth")
                # 抢救式保存最珍贵的参数字典字典 state_dict()
                torch.save(engine.model.state_dict(), best_model_path)
                if engine.use_ae:
                    torch.save(engine.ae_model.state_dict(), best_ae_path)
                torch.save({
                    "model_state_dict": engine.model.state_dict(),
                    "ae_state_dict": engine.ae_model.state_dict() if engine.use_ae else None,
                    "use_ae": bool(engine.use_ae),
                    "best_epoch": int(best_epoch_candidate),
                    "valid_total_loss": float(valid_epoch_loss),
                    "valid_pred_loss": float(valid_pred_loss),
                    "valid_rec_real_loss": float(valid_rec_real_loss),
                    "valid_rec_pred_loss": float(valid_rec_pred_loss),
                    "ae_beta": float(args.ae_beta),
                    "ae_lambda": float(args.ae_lambda),
                    "selection_metric": "L_total" if engine.use_ae else "L_pred",
                }, best_joint_path)
                
                # 将此刻这张最能体现正确因果拓扑关系的注意力矩阵，挂名为 _best 并独立留存
                if args.save_adj and A_sum is not None and A_cnt > 0:
                    torch.save(A_epoch, os.path.join(run_dirs["graphs"], "AdjMatrix_best.pt"))
                    np.save(os.path.join(run_dirs["graphs"], "AdjMatrix_best.npy"), A_epoch.numpy())
                
                # 判断逻辑 2: 虽然破了记录，但降幅是不是太小了，只是微弱的单调下降？(防止挤牙膏)
                if (min_loss - valid_epoch_loss) > min_delta:
                    # 只有降幅大于设定阈值(1e-4)，才认可其取得了“实质性有效进步”。此时耐心值瞬间清零。
                    epochs_no_improve = 0  
                else:
                    # 如果只是微小提升，判定为模型已经陷入了优化死胡同 (瓶颈期)，耐心惩罚计数器残酷 +1
                    epochs_no_improve += 1 
                
                # 更新历史水标，并且将当前光荣的 Epoch 登记在案
                min_loss = valid_epoch_loss  
                best_epoch = best_epoch_candidate
            else:
                # 若本次大考成绩相比历史变差了或毫无寸进，耐心惩罚计数器直接 +1
                epochs_no_improve += 1  
            
            # ---------------- 内存清洗防爆安全阀 ----------------
            # CUDA 虽然会自动回收部分显存，但在超大型图计算频繁切片的极端工况中，
            # 残留的显存碎片 (Fragmentation) 非常容易引起突发性的 Out-Of-Memory 异常。
            # 这句强制清空指令是长周期、大规模工业 AI 模型训练的一把绝对安全锁。
            torch.cuda.empty_cache()

            # 将本轮的详细赛况记录在控制台终端
            hues.log(
                f"Epoch [{cur_epoch+1:03d}/{args.epoch_num:03d}] | "
                f"TrainTotal: {train_epoch_loss:.4f} | TrainPred: {train_pred_loss:.4f} | "
                f"TrainRecReal: {train_rec_real_loss:.4f} | TrainRecPred: {train_rec_pred_loss:.4f} | "
                f"ValTotal: {valid_epoch_loss:.4f} | ValPred: {valid_pred_loss:.4f} | "
                f"ValRecReal: {valid_rec_real_loss:.4f} | ValRecPred: {valid_rec_pred_loss:.4f} | "
                f"停滞惩罚: {epochs_no_improve}/{patience}"
            )

            # ---------------- 早停熔断制裁器 (Early Stopping Trigger) ----------------
            # 如果模型连续数轮 (超过了耐心上限 patience) 都没有取得让阈值满意的有效进步...
            if epochs_no_improve >= patience:
                # 触发警告，宣告本网络结构已彻底收敛，不再具备被继续训练的价值
                hues.warn(f"  [早停触发] 模型连续 {patience} 轮未取得突破性进展(Δ < {min_delta})，已充分收敛！  ".center(100, '-'))
                hues.warn(f"  -> 训练在第 [{cur_epoch+1}] 轮提前终止。巅峰权重来自第 [{best_epoch}] 轮。  ".center(100, ' '))
                break  # 手动用 break 暴力斩断耗时的 Epoch 大循环，迫使程序迅速过渡至对保留下来的巅峰模型的最终客观评价环节

    # 如果模型表现坚挺，一次早停都没触发，平稳跑满了所有的设定轮次，则予以祝贺展示
    if epochs_no_improve < patience:
        hues.success(f"  [训练完成] 跑满设定轮次 {args.epoch_num} 轮。巅峰权重来自第 [{best_epoch}] 轮。  ".center(100, '-'))

    # ---------------- 巅峰模型回载 (Best Checkpoint Reload) ----------------
    # 训练过程中，每当验证集 Loss 刷新历史最优，都会将对应权重保存为 model.pth。
    # 最终验证/测试必须显式回载这份权重，否则实际评估对象可能是最后一轮模型，而非验证集最优模型。
    best_model_path = os.path.join(run_dirs["checkpoints"], "model.pth")
    best_ae_path = os.path.join(run_dirs["checkpoints"], "ae_model.pth")
    if os.path.exists(best_model_path):
        engine.model.load_state_dict(torch.load(best_model_path, map_location=device))
        engine.model.to(device)
        hues.success(f"  已回载验证集最优模型权重: [{best_model_path}]  ")
    else:
        hues.warn(f"  未找到验证集最优模型权重: [{best_model_path}]，最终评估将使用当前内存模型。  ")

    if engine.use_ae:
        if os.path.exists(best_ae_path):
            engine.ae_model.load_state_dict(torch.load(best_ae_path, map_location=device))
            engine.ae_model.to(device)
            hues.success(f"  已回载验证集最优 AE 权重: [{best_ae_path}]  ")
        else:
            hues.warn(f"  当前启用了 AE，但未找到最优 AE 权重: [{best_ae_path}]，最终评估将使用当前内存 AE。  ")
    else:
        hues.warn("  AE 未启用：最终测试与后处理残差将采用 MTGNN 预测误差，L_total = L_pred。  ")

    # 收尾工作：将最新的残留状态矩阵以通用的 AdjMatrix 名字再备一份
    if args.save_adj:
        save_path = os.path.join(run_dirs["graphs"], "AdjMatrix.pt")
        hues.success(f'保存关系矩阵至: [{save_path}].')  
        torch.save(adj_m_save, save_path)  

    # ======================================  2. Valid 验证集最终评估  ======================================
    # 使用巅峰状态的模型重跑一次全部验证集，返回详尽的验证指标结果
    hues.info("  正在基于巅峰模型计算验证集最终总体得分...  ")
    ret_valid_error, val_true_y, val_pred_y, _ = test_model(
        dataloader['val_loader'],
        dataloader['y_val'],
        scalers,
        engine.model,
        device,
    )

    # ======================================  3. TEST 测试集评估  ======================================
    # 在完全未参与训练或验证的纯净测试集上评估模型泛化能力
    hues.info("  正在基于巅峰模型计算测试集终极客观跑分...  ")
    ret_test_error, true_y, pred_y, batch_graphs = test_model(dataloader['test_loader'], dataloader['y_test'], scalers,
                                                              engine.model,
                                                              device)

    # =============================  4. Horizon Predict Results 各预测步长误差分析  ============================
    horizon_test_error = []  
    # 按照设定的未来预测序列长度 (horizon_size: p) 逐层拆解。
    # 因为随着向未来的步数深入，模型预测准确度通常会发生恶化，此处按独立时间步去计算误差以形成走势折线图。
    for hor_num in range(args.horizon_size):
        # 沿着最后一个时间步维度 (dim=2) 切片
        cur_pred_y = pred_y[:, :, hor_num]
        cur_true_y = true_y[:, :, hor_num]
        
        # 计算针对特定时间跨度上的物理真实误差 (比如第 10 步专属的 MAE/RMSE)
        errors = calc_error(cur_pred_y, cur_true_y)
        horizon_test_error.append(errors)  

    # 导出完整的真实与预测张量，供后续独立脚本（如异常检测阶段）做二次核验验证
    if args.save_pred_result:
        hues.info("  正在导出 MTGNN、AE 与综合异常评分所需残差文件...  ")
        # 训练集在训练循环中被 shuffle 过。导出训练残差前必须恢复原始顺序，
        # 这样 train_joint_error 与 train_y_time 才能逐样本对齐，用于计算历史 mu/sigma。
        for loader_key in ["train_loader", "val_loader", "test_loader"]:
            loader = dataloader.get(loader_key)
            if loader is not None and hasattr(loader, "indices") and hasattr(loader, "sample_size"):
                loader.indices = np.arange(loader.sample_size)

        train_residual_outputs = collect_module_residual_outputs(
            dataloader['train_loader'],
            dataloader['y_train'],
            scalers,
            engine.model,
            engine.ae_model if engine.use_ae else None,
            device,
            args.ae_beta,
            args.ae_lambda,
        )
        val_residual_outputs = collect_module_residual_outputs(
            dataloader['val_loader'],
            dataloader['y_val'],
            scalers,
            engine.model,
            engine.ae_model if engine.use_ae else None,
            device,
            args.ae_beta,
            args.ae_lambda,
        )
        test_residual_outputs = collect_module_residual_outputs(
            dataloader['test_loader'],
            dataloader['y_test'],
            scalers,
            engine.model,
            engine.ae_model if engine.use_ae else None,
            device,
            args.ae_beta,
            args.ae_lambda,
        )
        save_residual_outputs(
            run_dirs["predictions"],
            val_residual_outputs,
            test_residual_outputs,
            args,
            engine.use_ae,
            train_outputs=train_residual_outputs,
        )
        save_prediction_extras(run_dirs["predictions"], dataloader)
        hues.success(f"  已保存训练/验证/测试集传感器值、MTGNN 残差、AE 重构残差、综合残差与点级标签至: [{run_dirs['predictions']}]  ")

    # 训练流程结束，向外层返回全套指标记录数据
    return train_l, valid_l, ret_valid_error, ret_test_error, horizon_test_error


# ==============================================================================
# 模型训练主控入口 (Main Execution Entry Point)
# ==============================================================================
if __name__ == "__main__":
    # 打印终端日志横幅，标识多步预测任务启动
    hues.info("  开始训练多步预测模型  ".center(100, '='))

    # ==========================================================================
    # 1. 实验配置与环境初始化 (Configuration & Environment Setup)
    # ==========================================================================
    hues.info("  正在读取 tools.py 中的 get_input_paras() 预设参数...  ")
    
    # 解析命令行输入的超参数或加载工具类中的默认配置
    args = get_input_paras()  

    hues.info("  参数读取完毕！即将进入下一步：构建实验专属存储目录与环境...  ")

    # 提取当前精确的系统时间戳，并将核心复现实验参数写入目录名。
    task_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_name = build_experiment_name(args, task_timestamp)
    
    # 构建当前实验的绝对物理存储路径 (Root Directory)
    args.save = os.path.join(args.save, task_name)
    root_save_dir = args.save  # 缓存最外层根目录指针
    check_folder(args.save)    # 检查并自动创建存储路径
    aggregate_save_dir = os.path.join(root_save_dir, "aggregate")
    check_folder(aggregate_save_dir)
    hues.info(f"  成功构建实验专属存储根目录: [{root_save_dir}]  ")
    
    # 【科研复现性保障】：将当前模型的所有超参数、网络结构配置序列化为 JSON 格式并永久归档
    # 这对于后续撰写论文消融实验、回溯最优模型参数至关重要
    param_save_path = os.path.join(args.save, 'Parameters.json')
    save_args_to_json(args, param_save_path)
    hues.info(f"  模型超参数已成功归档保存至: [{param_save_path}]  ")

    # ==========================================================================
    # 2. 全局数据追踪器初始化 (Global Tracker Initialization)
    # ==========================================================================
    hues.info("  正在初始化全局数据追踪器 (DataFrames)...  ")
    
    # 初始化 Pandas DataFrame 用于聚合多次独立重复实验的结果，方便后续计算 Mean±Std
    res_df = pd.DataFrame()         # 核心误差指标全局汇总表 (MAE, RMSE, MAPE 等)
    train_loss_df = pd.DataFrame()  # 训练集动态 Loss 曲线追踪表
    valid_loss_df = pd.DataFrame()  # 验证集动态 Loss 曲线追踪表
    ret_valid_error, ret_test_error = [], [] # 缓存历次实验的最终误差列表

    hues.info("  追踪器初始化完毕！空数据表已就绪，将为您自动记录并聚合每次实验的误差与 Loss 曲线。  ")

    # ==========================================================================
    # 3. 独立重复实验微小循环 (Repeat Experiments Loop)
    # 【工程意义】：深度学习存在参数随机初始化与 GPU 浮点计算偏差。
    # 通过执行多次重复实验并取平均，可以有效消除偶然性，确保科研结论具备严格的统计学显著性。
    # ==========================================================================
    for run_num in range(args.repeat_exp_num):
        try:
            # 打印当前独立实验轮次的视觉分割线及详细中文说明
            hues.info(f"  正在启动第 [{run_num}] 轮独立重复实验  ".center(100, '-'))
            hues.info("  [运行内容]: 从零初始化模型权重，载入滑动窗口数据执行前向推演与梯度反向传播训练。")
            hues.info("  [实现功能]: 学习节点间的时空拓扑关联，最终计算并输出本轮最优模型的多步预测误差指标。")

            # 为当前这单次 Run 创建专属的子工作目录 (Sub-directory)
            args.save = os.path.join(root_save_dir, f"run_{run_num:02d}")
            check_folder(args.save)

            # ----- 核心拉起：启动主训练与验证管线 -----
            # 【全生命周期主引擎启动】：调用本文件上方定义的 main(args) 函数，执行深度学习完整闭环：
            # 1. 搭建 (Build)：根据 args 动态构建 Attention 网络架构与优化器。
            # 2. 训练 (Train)：载入滑动窗口数据，前向推演并利用反向传播 (Backpropagation) 更新网络权重。
            # 3. 选优 (Validate)：在验证集上实时计算误差，自动拦截并保存表现最好的“巅峰”模型权重 (best_model)。
            # 4. 测试 (Test)：用保存的巅峰模型对纯净测试集做最终评估，得出论文汇报所需的客观跑分。
            #
            # 【五大核心返回指标解析】：
            # - loss_l        : [过程心电图] 训练集各 Epoch 的 Loss 列表，观察模型学习进度。
            # - valid_l       : [过程心电图] 验证集各 Epoch 的 Loss 列表，监控是否死记硬背(过拟合)。
            # - v_error       : [验证总成绩] 巅峰模型在验证集上的综合大考成绩 (MAE, MAPE, RMSE, R^2)。
            # - t_error       : [最终总成绩] 巅峰模型在纯净测试集上的终极客观得分，直接填入论文对比表格！
            # - horizon_error : [高阶衰减分析] 按未来预测步长(1到H)细分的测试集误差，用于画时间衰减柱状图。
            loss_l, valid_l, v_error, t_error, horizon_error = main(args)
            
            # 将当前轮次的评估指标推入全局汇总池
            ret_valid_error.append(v_error)  
            ret_test_error.append(t_error)  

            # ==================================================================
            # 4. 局部数据结构化与全局汇入 (Data Structuring & Aggregation)
            # ==================================================================
            # 定义论文表格中常需要汇报的量纲与无量纲误差评价指标
            error_indices = ['MAE', 'MAPE', 'RMSE', 'R^2']
            
            # 构建单次运行的指标字典 (横向拼接 Run_ID 与 误差类别)
            cur_df = pd.DataFrame([
                [run_num for _ in range(len(error_indices))],  # 注入当前实验编号作为聚类标签
                error_indices                                  # 注入指标名称
            ]).T
            
            # 将按预测步长 (Horizon) 细分的误差矩阵拼接到主表右侧，形成完整的性能快照
            cur_df = pd.concat([cur_df, pd.DataFrame(horizon_error).T], axis=1)
            # 将单次运行快照无缝汇入全局总表
            res_df = pd.concat([res_df, cur_df])

            # 通过工具类抽取该轮次的 Loss 曲线下降轨迹，并存入全局 Loss 表
            train_loss_df = create_loss_df(run_num, loss_l, train_loss_df)
            valid_loss_df = create_loss_df(run_num, valid_l, valid_loss_df)

        except KeyboardInterrupt:
            # 【防御性编程】：优雅拦截用户的 Ctrl+C 手动强制中断信号
            # 避免直接崩溃导致已计算数小时的历史实验数据彻底丢失，保障高昂算力成本的投入不被浪费
            hues.warn(f"  训练在第 [{run_num}] 轮被手动中断。正在安全退出循环...  ".center(100, '='))
            break  # 直接跳出重复实验循环，无缝进入下方的数据持久化阶段

    # ==========================================================================
    # 5. 实验落幕与数据持久化 (Finalization & Data Serialization)
    # ==========================================================================
    # 【原理说明 - 数据宽表化】：
    # 深度学习多步预测的核心分析点在于“误差随时间的衰减”。
    # 这里通过列表推导式，将列名重构为 ['Run_Id', 'Type', 1, 2, ..., H] 的形式。
    # 这种宽表 (Wide Table) 结构非常适合直接塞进绘图库，画出横轴为预测步长，纵轴为误差的折线图。
    res_df.columns = ['Run_Id', 'Type'] + [i + 1 for i in range(args.horizon_size)]
    
    # 【原理说明 - 为什么用 Pickle 落盘？】：
    # 很多初学者喜欢存为 CSV 文件，但 CSV 会丢失 Python 数据类型（例如将 numpy 数组变成纯字符串），
    # 且读写速度极慢。Pickle (pkl) 是 Python 专属的二进制序列化格式，
    # 它能 100% 完美冻结内存中的 DataFrame 和复杂张量列表。
    # 这样后续在写 Jupyter Notebook 专门做论文数据分析时，读取加载是瞬间完成的，且数据结构零损耗。
    res_df.to_pickle(os.path.join(aggregate_save_dir, 'Result.pkl'))               # 包含多步预测的明细误差汇总表
    train_loss_df.to_pickle(os.path.join(aggregate_save_dir, 'TrainLoss.pkl'))     # 训练集 Loss 下降轨迹表
    valid_loss_df.to_pickle(os.path.join(aggregate_save_dir, 'ValidLoss.pkl'))     # 验证集 Loss 下降轨迹表
    to_pickle(ret_valid_error, os.path.join(aggregate_save_dir, 'ValidError.pkl')) # 巅峰模型验证集总体大考成绩
    to_pickle(ret_test_error, os.path.join(aggregate_save_dir, 'TestError.pkl'))   # 巅峰模型测试集终极客观跑分
    # 添加落盘成功后的详细提示
    hues.info(f"  [数据持久化成功] 核心评估数据已全部安全保存至: [{aggregate_save_dir}]  ")
    hues.info("  -> 包含文件: Result.pkl, TrainLoss.pkl, ValidLoss.pkl, ValidError.pkl, TestError.pkl  ")

    # ==========================================================================
    # 6. 自动化论文图表渲染 (Automated Plotting for Paper Drafts)
    # 【功能说明】：无需等待后续手动处理，训练一旦结束，系统立刻调用内置的科研级绘图引擎，
    # 自动生成符合高水平 SCI 期刊排版标准的高清 PNG 图片，实现“所见即所得”。
    # ==========================================================================
    
    # 图 1 (Error Line Plot): 误差衰减折线图。横轴是未来步长 1~H，纵轴是 MAE/RMSE 等。直观展示预测越远越难的物理规律。
    plt_error(res_df, save_path=os.path.join(aggregate_save_dir, f'[{args.model}]_Error.png'))  
    
    # 图 2 (Boxplot): 误差分布箱线图。由于我们做了多次独立实验 (repeat_exp_num)，箱线图能展示每次结果的波动范围。
    # 箱体越窄，说明您的模型鲁棒性 (Robustness) 极高，受随机初始化影响小，这是审稿人极其看重的一点。
    plt_box(res_df, which='RMSE', save_path=os.path.join(aggregate_save_dir, f'[{args.model}]_BoxRMSE.png'))  
    
    # 图 3 (Loss Curve): 训练/验证损失曲线图。也就是我们讨论过的“心电图”，用于证明模型已收敛且未发生严重过拟合。
    plt_loss(train_loss_df, valid_loss_df, save_path=os.path.join(aggregate_save_dir, f'[{args.model}]_Loss.png'))  

    # 打印全局任务圆满完成的终点标记，给漫长的算力等待画上一个完美的句号。
    hues.success('  任务圆满完成！所有训练与测试进程已安全终止。  '.center(100, '='))
