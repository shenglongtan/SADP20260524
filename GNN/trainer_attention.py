# # import torch  # 导入 PyTorch 核心库
# # import torch.optim as optim  # 导入 PyTorch 优化器模块，用于参数更新
# # import torch.nn.functional as nn_f  # 导入 PyTorch 的函数式神经网络接口（如 mse_loss 等）
# # from typing import Tuple, Any, List  # 导入类型提示，用于代码的静态类型检查
# # # 从自定义的 tools 模块中导入带掩码（屏蔽缺失值）的误差计算函数
# # from tools import masked_mae, masked_mape, masked_rmse, masked_mse
# # class Trainer:
# #     """
# #     模型训练器类 (The model's trainer)。
# #     封装了模型的前向传播、损失计算、反向传播、自编码器联合训练以及课程学习等核心训练逻辑。
# #     """
# #     def __init__(
# #             self,
# #             model,
# #             horizon_size: int,
# #             scalers,
# #             device,
# #             ae_model=None,
# #             use_ae: bool = False,
# #             ae_beta: float = 0.2,
# #             ae_lambda: float = 0.5,
# #             cl: bool = True,
# #             cl_update_num: int = 2500,
# #             gd_clip: int = None,
# #             opt_lr: float = 1e-3,
# #             opt_wd: float = 1e-4
# #     ):
# #         """
# #         类对象初始化函数。
# #         @param model: 待训练的主预测模型（如 MTGNN）。
# #         @param horizon_size: 预测的未来时间步长总数。
# #         @param scalers: 数据归一化/反归一化工具对象，用于在验证时还原真实误差。
# #         @param device: 训练所使用的计算设备（CPU 或 GPU）。
# #         @param ae_model: 用于联合训练的自编码器模型（用于异常检测的重构路径）。
# #         @param use_ae: 是否开启自编码器联合训练。
# #         @param ae_beta: 主预测损失与自编码器重构损失之间的权重平衡系数。
# #         @param ae_lambda: 自编码器中，真实窗口重构损失与生成（预测）窗口重构损失之间的权重比例。
# #         @param cl: 是否开启课程学习 (Curriculum Learning)，即从预测单步开始，逐渐增加预测步长。
# #         @param cl_update_num: 在课程学习中，每经过多少次训练迭代 (batches) 增加一步预测步长。
# #         @param gd_clip: 梯度裁剪的阈值。
# #                         梯度裁剪主要用于防止反向传播过程中梯度变得过大（梯度爆炸问题），
# #                         这会导致数值不稳定并使训练变得困难甚至无法收敛。
# #         @param opt_lr: 优化器的初始学习率。
# #         @param opt_wd: 优化器的权重衰减值（L2 正则化系数）。
# #         """
# #         self.scalers = scalers  # 挂载归一化工具
# #         self.model = model  # 挂载主预测模型
# #         self.model.to(device)  # 将主模型迁移到指定的计算设备上
# #         self.ae_model = ae_model  # 挂载自编码器模型
# #         # 严格判断：只有设置了 use_ae=True 且确实传入了 ae_model，才真正开启联合训练
# #         self.use_ae = use_ae and (ae_model is not None)
# #         self.ae_beta = ae_beta  # 挂载重构损失权重系数
# #         self.ae_lambda = ae_lambda  # 挂载预测窗重构占比系数
# #         if self.ae_model is not None:
# #             self.ae_model.to(device)  # 如果有自编码器，同样迁移到计算设备上
# #         # 收集主模型的所有可学习参数
# #         params = list(self.model.parameters())
# #         if self.use_ae:
# #             # 如果开启联合训练，将自编码器的参数也加入到优化列表中，实现端到端联合更新
# #             params += list(self.ae_model.parameters())
# #         # 实例化 Adam 优化器，传入合并后的参数列表、学习率和权重衰减
# #         self.optimizer = optim.Adam(params, lr=opt_lr, weight_decay=opt_wd)
# #         # 定义训练过程中使用的主目标损失函数
# #         # self.loss = masked_mse  # (注释掉的代码) 可选：屏蔽缺失值的均方误差
# #         self.loss = masked_mae  # 当前启用的：屏蔽缺失值的平均绝对误差 (L1 Loss)，对异常离群点更鲁棒
# #         # self.loss = torch.nn.HuberLoss(delta=0.01)  # (注释掉的代码) 可选：Huber Loss，兼具 L1 和 L2 的优点
# #         self.gd_clip = gd_clip  # 挂载梯度裁剪阈值
# #         self.train_num = 1  # 初始化全局训练步数（迭代次数）计数器
# #         self.pred_step = 1  # 课程学习初始化：最开始只计算预测第 1 步的 Loss
# #         self.horizon_size = horizon_size  # 保存总预测步长
# #         self.cl = cl  # 挂载课程学习开启标志
# #         self.cl_update_num = cl_update_num  # 挂载课程学习步长递增的频率
# #     def train(
# #             self,
# #             input_x: torch.tensor,
# #             true_y: torch.tensor,
# #             sample_indices: torch.tensor = None
# #     ) -> tuple[list[Any], Any]:
# #         """
# #         单次训练迭代函数。
# #         @param input_x: 特征矩阵张量。
# #                         形状预期为: [batch_size, feature_num, len(node_id), window_size]。
# #         @param true_y: 目标标签矩阵张量。
# #                        形状预期为: [batch_size, len(node_id), horizon_size]。

# #         @return: Tuple[list, Any]:
# #                     - 包含三个浮点数的列表: [计算得到的loss值, MAPE指标, RMSE指标]。
# #                     - adj_m_save: 模型在前向传播中生成的邻接矩阵（用于可视化或分析）。
# #         """
# #         self.model.train()  # 将主模型设置为训练模式（启用 Dropout 和 BatchNorm 等特性）
# #         if self.use_ae:
# #             self.ae_model.train()  # 将自编码器也设置为训练模式
# #         self.optimizer.zero_grad()  # 在每次反向传播前，清空（重置）之前累积的梯度
# #         # 执行主模型的前向传播
# #         # pred_y 的形状为 [batch_size, 1, node_num, horizon_size]
# #         pred_y, adj_m_save = self.model(input_x)
# #         # 调整真实标签 true_y 的形状以对齐预测值：
# #         # 从 [batch_size, len(node_id), window_size] 扩充通道维度
# #         # 变为 [batch_size, 1, len(node_id), window_size]
# #         true_y = torch.unsqueeze(true_y, dim=1)
# #         # ==================== Loss 计算与课程学习逻辑 ====================
# #         # 每训练 cl_update_num 个 batch，并且当前预测步数还没有达到最大 horizon_size 时
# #         if self.train_num % self.cl_update_num == 0 and self.pred_step <= self.horizon_size:
# #             self.pred_step += 1  # 增加预测步长，增加训练难度
# #         if self.cl:  # 如果开启了课程学习
# #             # 仅截取从第 0 步到当前 pred_step 步的预测值和真实值计算 Loss (由易到难)
# #             loss = self.loss(pred_y[..., :self.pred_step], true_y[..., :self.pred_step])
# #         else:
# #             # 否则，直接计算全部 horizon_size 长度的 Loss
# #             loss = self.loss(pred_y, true_y)
# #         # ==================== 自编码器联合训练逻辑 ====================
# #         if self.use_ae:
# #             # 仅提取输入特征的第 0 个通道（通常是预测目标本身的历史数据）进行重构
# #             hist = input_x[:, 0:1, :, :]  # [batch_size, 1, node_num, window_size]
# #             # 拼接历史序列和真实未来序列，形成完整的"真实时空窗口"
# #             true_win = torch.cat([hist, true_y], dim=3)
# #             # 拼接历史序列和模型预测的未来序列，形成包含预测偏差的"生成时空窗口"
# #             pred_win = torch.cat([hist, pred_y], dim=3)
# #             # 自编码器分别对真实窗口和生成窗口进行特征压缩与解码重构
# #             rec_true = self.ae_model(true_win)
# #             rec_pred = self.ae_model(pred_win)
# #             # 计算自编码器的重构误差（MSE）
# #             rec_loss_true = nn_f.mse_loss(rec_true, true_win)  # 真实窗口的重构误差
# #             rec_loss_pred = nn_f.mse_loss(rec_pred, pred_win)  # 生成窗口的重构误差
# #             # 将自编码器的重构损失联合加回主目标损失中
# #             # 总损失 = 预测损失 + beta * (真实重构损失 + lambda * 预测重构损失)
# #             loss = loss + self.ae_beta * (rec_loss_true + self.ae_lambda * rec_loss_pred)
# #         # 执行反向传播，自动计算并累积参数梯度
# #         loss.backward()
# #         # 如果设置了梯度裁剪阈值，则执行裁剪操作，防止梯度爆炸
# #         if self.gd_clip is not None:
# #             torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gd_clip)
# #         # 优化器根据计算出的梯度更新模型参数
# #         self.optimizer.step()
# #         self.train_num += 1  # 更新全局训练步数计数器
# #         # 返回当前 batch 的损失以及原始输出（未逆归一化）的 MAPE 和 RMSE 评估指标
# #         return [
# #             loss.item(),  # 将张量损失转换为 Python 浮点数
# #             masked_mape(pred_y, true_y).item(),
# #             masked_rmse(pred_y, true_y).item()
# #         ], adj_m_save
# #     def eval(self, input_x: torch.Tensor, true_y: torch.tensor) -> Tuple[float, float, float]:
# #         """
# #         验证/评估模型函数。在此阶段不需要计算梯度。
# #         @param input_x: 特征矩阵张量。
# #                         形状预期为: [batch_size, feature_num, len(node_id), window_size]。
# #         @param true_y: 目标标签矩阵张量。
# #                        形状预期为: [batch_size, len(node_id), window_size]。
# #         @return: Tuple[float, float, float]:
# #                     - loss (float): 验证集上的预测损失值。
# #                     - mape (float): 逆归一化还原量纲后计算的 MAPE 指标。
# #                     - rmse (float): 逆归一化还原量纲后计算的 RMSE 指标。
# #         """
# #         self.model.eval()  # 将模型设置为评估模式（关闭 Dropout，固定 BatchNorm 的均值和方差）
# #         # 执行前向传播（此调用外层通常需配合 with torch.no_grad(): 避免内存泄漏）
# #         output_y, _ = self.model(input_x)
# #         # 将真实标签增加通道维度以对齐预测张量
# #         true_y = torch.unsqueeze(true_y, dim=1)
# #         # 计算验证集的预测 Loss（使用初始化时指定的损失函数，通常为 MAE）
# #         loss = self.loss(output_y, true_y).item()
# #         # 为了计算具有实际物理意义的 MAPE 和 RMSE 误差指标，需要执行反归一化
# #         if self.scalers is not None:  # 如果输入数据经过了标准化
# #             # 使用 scaler 将预测值还原为真实的物理量纲
# #             pred_y = self.scalers.inverse_transform(output_y)
# #             # 使用 scaler 将缩放后的标签也还原为真实物理量纲
# #             true_y = self.scalers.inverse_transform(true_y)
# #         else:
# #             # 如果没有进行标准化，则直接使用模型输出
# #             pred_y = output_y
# #         # 返回验证集 Loss 及具有实际物理意义的真实误差
# #         return (
# #             loss,
# #             masked_mape(pred_y, true_y).item(),
# #             masked_rmse(pred_y, true_y).item(),
# #         )
# # class Optim(object):
# #     """
# #     自定义优化器封装类。
# #     用于管理不同的优化器策略（SGD, Adam 等）、学习率衰减以及手动梯度裁剪。
# #     (注：此模块在当前的代码体系中可能是一个早期的实现版本，在 Trainer 类中并未直接使用，
# #      Trainer 类使用的是原生的 optim.Adam。但为了保持代码完整性，保留并注释。)
# #     """
# #     def __init__(self, params, method, lr, clip, lr_decay=1, start_decay_at=None):
# #         """
# #         初始化自定义优化器。
# #         """
# #         self.params = params  # 保存需要优化的参数生成器或列表
# #         self.last_ppl = None  # 记录上一次验证集的性能指标（如困惑度或Loss），用于学习率衰减判定
# #         self.lr = lr  # 当前学习率
# #         self.gd_clip = clip  # 梯度裁剪阈值
# #         self.method = method  # 优化器字符串标识 ('sgd', 'adam' 等)
# #         self.lr_decay = lr_decay  # 每次衰减时学习率乘以的衰减因子
# #         self.start_decay_at = start_decay_at  # 指定在达到第几个 Epoch 时强制开始衰减
# #         self.start_decay = False  # 标记当前 Epoch 是否需要执行衰减操作
# #         self._create_optimizer()  # 根据指定的 method 初始化具体的 PyTorch 优化器实例
# #     def step(self):
# #         """
# #         执行一次参数更新步骤，并返回裁剪前的梯度范数。
# #         """
# #         # 计算梯度的 L2 范数，用于后续可能的监控
# #         grad_norm = 0
# #         if self.gd_clip is not None:
# #             # 使用 PyTorch 提供的标准方法进行梯度裁剪
# #             torch.nn.utils.clip_grad_norm_(self.params, self.gd_clip)
# #         # 以下注释掉的代码为早期手动计算和执行梯度裁剪的逻辑
# #         # for param in self.params:
# #         #     grad_norm += math.pow(param.grad.data.norm(), 2)
# #         #
# #         # grad_norm = math.sqrt(grad_norm)
# #         # if grad_norm > 0:
# #         #     shrinkage = self.max_grad_norm / grad_norm
# #         # else:
# #         #     shrinkage = 1.
# #         #
# #         # for param in self.params:
# #         #     if shrinkage < 1:
# #         #         param.grad.data.mul_(shrinkage)
# #         # 调用底层 PyTorch 优化器更新参数
# #         self.optimizer.step()
# #         return grad_norm
# #     def _create_optimizer(self):
# #         """
# #         工厂模式方法，根据初始化时的字符串 method 创建对应的 PyTorch 优化器。
# #         """
# #         if self.method == 'sgd':
# #             self.optimizer = optim.SGD(self.params, lr=self.lr, weight_decay=self.lr_decay)
# #         elif self.method == 'adagrad':
# #             self.optimizer = optim.Adagrad(self.params, lr=self.lr, weight_decay=self.lr_decay)
# #         elif self.method == 'adadelta':
# #             self.optimizer = optim.Adadelta(self.params, lr=self.lr, weight_decay=self.lr_decay)
# #         elif self.method == 'adam':
# #             # 注意：此处的 weight_decay 参数接收了 self.lr_decay，这可能是逻辑上的一个易混淆点
# #             self.optimizer = optim.Adam(self.params, lr=self.lr, weight_decay=self.lr_decay)
# #         else:
# #             # 如果提供了不支持的优化器类型，抛出异常
# #             raise RuntimeError("Invalid optim method: " + self.method)
# #     # decay learning rate if val perf does not improve or we hit the start_decay_at limit
# #     def updateLearningRate(self, ppl, epoch):
# #         """
# #         根据验证集性能或达到特定 Epoch 来动态衰减学习率。
# #         @param ppl: 当前 Epoch 在验证集上的性能指标（数值越大代表性能越差，如 Loss）。
# #         @param epoch: 当前的 Epoch 索引。
# #         """
# #         # 如果设置了衰减起始点，并且当前 epoch 已达到或超过该点，则触发衰减
# #         if self.start_decay_at is not None and epoch >= self.start_decay_at:
# #             self.start_decay = True
# #         # 如果当前性能指标 ppl 大于上一次记录的性能指标（说明验证集性能没有提升甚至恶化），则触发衰减
# #         if self.last_ppl is not None and ppl > self.last_ppl:
# #             self.start_decay = True
# #         # 执行衰减逻辑
# #         if self.start_decay:
# #             self.lr = self.lr * self.lr_decay  # 学习率乘以衰减因子
# #             print("Decaying learning rate to %g" % self.lr)  # 打印提示
# #         # 衰减标志复位，保证只在本 epoch 衰减一次
# #         self.start_decay = False
# #         self.last_ppl = ppl  # 记录当前性能，供下一个 Epoch 比较
# #         # 重新创建优化器实例以应用新的学习率
# #         # (较新版本的 PyTorch 推荐直接修改 optimizer.param_groups 中的 'lr' 属性，而不是重新创建对象)
# #         self._create_optimizer()


# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
import torch  # 导入 PyTorch 深度学习核心基础库
import torch.optim as optim  # 导入 PyTorch 优化器模块，用于参数更新与梯度下降
import torch.nn.functional as nn_f  # 导入 PyTorch 的函数式神经网络接口（包含常用损失函数与重构函数）
from typing import Tuple, Any, List  # 导入类型提示组件，提升代码阅读规范性与静态检查健壮性
# 
# 从项目自定义的 tools 模块中导入带有掩码（能自适应屏蔽缺失值 NaN）的专业评估指标函数
from tools import masked_mae, masked_mape, masked_rmse, masked_mse
# 
class Trainer:
    """
    时空图神经网络注意力模型训练器类 (The attention-based model's trainer)。
    高度封装了模型的前向传播、多维损失计算、反向传播、自编码器联合异常检测训练以及课程学习等核心控制流。
    """
    def __init__(
            self,
            model,
            horizon_size: int,
            scalers,
            device,
            ae_model=None,
            use_ae: bool = False,
            ae_beta: float = 0.2,
            ae_lambda: float = 0.5,
            cl: bool = True,
            cl_update_num: int = 2500,
            gd_clip: int = None,
            opt_lr: float = 1e-3,
            opt_wd: float = 1e-4
    ):
        """
        训练引擎的初始化构造函数，完成核心网络、优化器、损失函数以及各类训练策略的绑定。
        # 
        :param model: 待训练的核心时空预测网络（当前传入基于自适应注意力的 MTGNN 模型）
        :param horizon_size: 预测的未来时间步长总数（即多步预测的目标步数）
        :param scalers: 工业数据标准化/逆归一化器，用于在评估阶段将无量纲特征还原回真实物理量纲
        :param device: 硬件计算设备上下文（绑定为 CPU 或特定的 CUDA GPU 显卡）
        :param ae_model: 用于联合训练的时间窗口自编码器（选配，常用于结合重构残差的工业异常检测）
        :param use_ae: 是否激活自编码器联合训练的分支
        :param ae_beta: 联合 Loss 权重系数：控制多步预测损失与自编码器纯重构损失之间的业务平衡
        :param ae_lambda: 联合 Loss 权重系数：在自编码器内部，控制真实窗口重构与包含预测偏差的生成窗口重构的比例
        :param cl: 是否开启课程学习 (Curriculum Learning) 策略，即由易到难，从单步预测逐渐过渡到全步长预测
        :param cl_update_num: 课程学习控制步频：每经历多少个 Batch 迭代，就自动解锁下一个预测时间步长
        :param gd_clip: 梯度裁剪 (Gradient Clipping) 阈值，用于彻底杜绝深层 GNN 频繁发生的梯度爆炸隐患
        :param opt_lr: 优化器（Adam）的初始学习率
        :param opt_wd: 优化器的权重衰减系数（L2 正则化惩罚项，用于削弱模型过拟合）
        """
        self.scalers = scalers  # 挂载全局数据量纲缩放器
        self.model = model  # 挂载核心预测网络模型
        self.model.to(device)  # 强制将预测网络的所有权重参数推送到指定的 GPU 或 CPU 设备上
        # 
        self.ae_model = ae_model  # 挂载辅助自编码器模型
        # 严格的依赖防御：只有当外部明确要启用且传入的自编码器模型非空时，才真正激活重构分支
        self.use_ae = use_ae and (ae_model is not None)
        self.ae_beta = ae_beta  # 业务损失权重项挂载
        self.ae_lambda = ae_lambda  # 重构残差权重项挂载
        # 
        if self.ae_model is not None:
            self.ae_model.to(device)  # 同样强制将自编码器的权重矩阵部署到相同的计算设备上
            # 
        # 提取主模型中所有需要计算梯度并进行修正的可学习权重参数
        params = list(self.model.parameters())
        if self.use_ae:
            # 如果激活了端到端联合异常检测训练，将自编码器的参数追加至同一个参数列表中，统一进行反向传播
            params += list(self.ae_model.parameters())
            # 
        self.trainable_params = params
        # 实例化经典的 Adam 优化器，负责在反向传播时高效、动态地更新合并后的参数表
        self.optimizer = optim.Adam(self.trainable_params, lr=opt_lr, weight_decay=opt_wd)
        # 
        # 确定主任务的损失函数。在工业多变量时间序列（如 SWaT）的预测回归任务中，
        # Masked MAE (平均绝对误差) 相比 MSE 能够更好地防御异常攻击点引发的异常离群值对网络梯度的污染。
        self.loss = masked_mae  
        self.gd_clip = gd_clip  # 梯度裁剪阈值注入
        self.train_num = 1  # 全局训练步数迭代计数器，用来作为课程学习步长的升级依据
        self.pred_step = 1 if cl else horizon_size  # 关闭课程学习时，训练目标直接覆盖完整 horizon
        self.horizon_size = horizon_size  # 记录最终需要达到的最大多步预测总长度
        self.cl = cl  # 课程学习开关标志
        self.cl_update_num = cl_update_num  # 课程学习难度升级周期阈值

    def _calc_joint_losses(
            self,
            input_x: torch.Tensor,
            pred_y: torch.Tensor,
            true_y: torch.Tensor,
            pred_loss: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算 MTGNN 预测损失与 AE 重构损失构成的联合目标。

        启用 AE 时：
            L_total = L_pred + beta * (L_rec_real + lambda * L_rec_pred)
        未启用 AE 时：
            L_total = L_pred，两个 AE 重构损失置为 0。
        """
        zero_loss = pred_loss.new_tensor(0.0)
        if not self.use_ae:
            return pred_loss, pred_loss, zero_loss, zero_loss

        hist = input_x[:, 0:1, :, :]
        true_win = torch.cat([hist, true_y], dim=3)
        pred_win = torch.cat([hist, pred_y], dim=3)

        rec_true = self.ae_model(true_win)
        rec_pred = self.ae_model(pred_win)
        rec_loss_true = nn_f.mse_loss(rec_true, true_win)
        rec_loss_pred = nn_f.mse_loss(rec_pred, pred_win)
        total_loss = pred_loss + self.ae_beta * (rec_loss_true + self.ae_lambda * rec_loss_pred)
        return total_loss, pred_loss, rec_loss_true, rec_loss_pred
# 
    def train(
            self,
            input_x: torch.tensor,
            true_y: torch.tensor,
            sample_indices: torch.tensor = None
    ) -> tuple[list[Any], Any]:
        """
        核心单次 Batch 的前向计算与梯度反向更新函数。
        # 
        :param input_x: 输入的时序特征张量，形状为 [Batch, Channel, Node, Time]
        :param true_y: 目标的真实标签张量（未来步长），形状为 [Batch, Node, Horizon_Size]
        :param sample_indices: 兼容性占位符（在新版自适应注意力构图模型中，已无需外部显式传入节点ID）
        # 
        :return: 
            - 包含当前 Batch 综合表现的指标列表: [总损失值 Loss.item(), 未逆归一化前的 MAPE, 未逆归一化前的 RMSE]
            - adj_m_save: 模型在此刻前向传播中自发推演生成的动态邻接拓扑矩阵（用于提取和分析传感器关系图）
        """
        self.model.train()  # 显式将主模型切换为训练状态（激活神经网络的 Dropout 层与 BatchNorm 层的动态行为）
        if self.use_ae:
            self.ae_model.train()  # 若配置了联合训练，同步将自编码器拉入训练模式
            # 
        self.optimizer.zero_grad()  # 极其关键的起手式：清空并重置历史残留下来的权重梯度，防止梯度形成错误累加
        # 
        # --- 1. 执行主预测模型前向传播 ---
        # 此时调用的是修改后的新版模型，去除了 node_id 的硬约束，直接输入 input_x。
        # 返回当前批次的时序预测张量 pred_y [B, 1, N, Horizon] 以及注意力构图器算出的动态邻接矩阵 adj_m_save [N, N]
        pred_y, adj_m_save = self.model(input_x)
        # 
        # 在第 1 维（Channel 维度）上为真实标签 true_y 强行扩充一个维度
        # 使其形状从 [Batch, Node, Horizon] 完美对齐预测值的 [Batch, 1, Node, Horizon]
        true_y = torch.unsqueeze(true_y, dim=1)
        # 
        # ==================== 2. Loss 计算与课程学习控制逻辑 ====================
        # 当累积迭代步数达到了升级周期，且当前难度还没有超过多步预测的总上限时
        if self.cl and self.train_num % self.cl_update_num == 0 and self.pred_step < self.horizon_size:
            self.pred_step += 1  # 自动解锁下一阶段难度：命令模型开始多看一步未来的变化趋势
            # 
        if self.cl:  
            # 课程学习激活：截断张量，仅对未来第 0 步到当前 pred_step 步范围内的时序片段进行损失评估（由易到难）
            pred_loss = self.loss(pred_y[..., :self.pred_step], true_y[..., :self.pred_step])
        else:
            # 课程学习未激活：一上来就要求模型对全长 horizon_size 负责，直接算全段 Loss
            pred_loss = self.loss(pred_y, true_y)
            # 
        # ==================== 3. MTGNN + AE 联合目标损失 ====================
        loss, pred_loss, rec_loss_true, rec_loss_pred = self._calc_joint_losses(
            input_x,
            pred_y,
            true_y,
            pred_loss,
        )
            # 
        # ==================== 4. 反向传播与显存防爆控制 ====================
        loss.backward()  # 引发 PyTorch 底层 C++ 自动微分引擎计算，开始沿计算图反向计算所有权重的梯度值
        # 
        # 梯度裁剪防御机制：如果用户设置了裁剪阈值，就在优化器跨步之前强行阻断并缩放过大的梯度梯度范数
        if self.gd_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.trainable_params, self.gd_clip)
            # 
        self.optimizer.step()  # 核心大跨步：Adam 优化器根据刚才反向传播得到的纯净梯度更新模型的所有参数矩阵
        self.train_num += 1  # 步进累加器自增 1，为下一次课程学习进阶做准备
        # 
        # 【关键显存优化修订】：在这里返回评估指标时，必须对 pred_y 执行 .detach() 操作！
        # 这样可以将带着沉重反向梯度图的历史张量彻底与当前内存解绑，防止算误差时显存瞬间碎片化溢出。
        return [
            loss.item(),  # L_total；若未启用 AE，则等于 L_pred
            pred_loss.item(),
            rec_loss_true.item(),
            rec_loss_pred.item(),
            masked_mape(pred_y.detach(), true_y).item(),
            masked_rmse(pred_y.detach(), true_y).item()
        ], adj_m_save.detach()  # 图邻接矩阵同步执行 .detach() 脱离计算图，保护 GPU 显存安全
# 
    def eval(self, input_x: torch.Tensor, true_y: torch.tensor) -> Tuple[float, float, float]:
        """
        在独立的验证集（Validation Set）上验证模型当前 Epoch 的泛化和收敛效果。
        # 
        :param input_x: 验证集特征输入，形状为 [Batch, Channel, Node, Window_Size]
        :param true_y: 验证集真实未来标签，形状为 [Batch, Node, Window_Size]
        :return: 
            - loss (float): 模型在验证集上产生的标量预测误差（通常为无量纲的归一化 MAE）。
            - mape (float): 经过逆归一化器将数据映射回工业现场真实工程物理量纲后，计算出的绝对百分比误差。
            - rmse (float): 同样映射回真实工程量纲后，算出的物理尺度下的均方根误差。
        """
        self.model.eval()  # 显式将模型调整为评估/测试状态（全面冻结 Dropout 与 BatchNorm 行为，保证推理确定性）
        # 
        # 【关键显存优化修订】：在这里强行拉起 torch.no_grad() 上下文管理器锁死梯度。
        # 它能命令 PyTorch 引擎在前向计算时完全不构建任何历史反向图，这是 8GB 显存跑全量验证集的生命线。
        with torch.no_grad():
            output_y, _ = self.model(input_x)  # 前向传播纯推理，拿到单向预测输出
            # 
            # 将真实标签增加通道轴，对齐预测张量的 4D 架构
            true_y = torch.unsqueeze(true_y, dim=1)
            # 
            # 计算归一化刻度下的联合验证目标。启用 AE 时用于最佳模型选择的是 L_total。
            pred_loss = self.loss(output_y, true_y)
            loss, pred_loss, rec_loss_true, rec_loss_pred = self._calc_joint_losses(
                input_x,
                output_y,
                true_y,
                pred_loss,
            )
            # 
            # --- 量纲反演：为了让科研论文中的 MAE / RMSE 具备实际工业现场的工程物理意义，必须执行反归一化 ---
            if self.scalers is not None:  
                # 调用工具类的 inverse_transform，将 0~1 的缩放数字还原回实际的水位米数、压力帕斯卡等量纲
                pred_y = self.scalers.inverse_transform(output_y)
                true_y = self.scalers.inverse_transform(true_y)
            else:
                pred_y = output_y  # 若没有标准化器，则保持原样
                # 
            # 返回具有高度可读性的真实物理量纲下的工程误差四元组，提供给主循环记录日志
            return (
                loss.item(),
                pred_loss.item(),
                rec_loss_true.item(),
                rec_loss_pred.item(),
                masked_mape(pred_y, true_y).item(),
                masked_rmse(pred_y, true_y).item(),
            )
# 
# 
class Optim(object):
    """
    自定义高阶优化器包装器。
    用于对历史继承代码库提供向下兼容支持，集成了多种梯度寻优算法选择、手工梯度裁剪以及灵活的学习率阶梯衰减控制。
    （注：当前最新的架构主体已默认搭载在 Trainer 内部高效运作的 Adam 引擎，此项作为历史沉淀资产，保持原样注释并兼容存留。）
    """
    def __init__(self, params, method, lr, clip, lr_decay=1, start_decay_at=None):
        self.params = params  # 绑定传入的神经网络权重参数生成器
        self.last_ppl = None  # 历史性能指标监控哨兵（如用于判定验证集损失是否恶化）
        self.lr = lr  # 当前运行期的活动学习率
        self.gd_clip = clip  # 手动梯度拦截阈值
        self.method = method  # 策略字符串标识 ('sgd', 'adam', 'adagrad' 等)
        self.lr_decay = lr_decay  # 学习率每次衰减的惩罚底数（乘法因数）
        self.start_decay_at = start_decay_at  # 指定从第几个固定的 Epoch 开始启动强制性衰减
        self.start_decay = False  # 是否触发学习率下调动作的内部状态标识位
        self._create_optimizer()  # 触发内部的工厂函数，拉起对应的经典 PyTorch 优化器
# 
    def step(self):
        """
        手工调用执行单次跨步更新。
        """
        grad_norm = 0
        if self.gd_clip is not None:
            # 触发权重空间的梯度硬裁剪，防范梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.params, self.gd_clip)
        self.optimizer.step()  # 参数迭代调整
        return grad_norm
# 
    def _create_optimizer(self):
        """
        工厂设计模式：根据配置的控制标识，动态映射构建底层的 PyTorch 原始优化器。
        """
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr, weight_decay=self.lr_decay)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)
# 
    def updateLearningRate(self, ppl, epoch):
        """
        自适应学习率阶梯衰减算法：当监测到验证集性能开始停滞不前、或者达到特定的科研约束纪元时，自动降低学习率。
        """
        # 条件一：若设置了时间边界，且当前实验所在的 Epoch 已经跨过该边界，触发衰减
        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True
        # 条件二：若上一轮的验证集损失指标存在，且当前轮次算出的 ppl 大于上一轮（说明网络表现退步或产生震荡），触发衰减
        if self.last_ppl is not None and ppl > self.last_ppl:
            self.start_decay = True
            # 
        if self.start_decay:
            self.lr = self.lr * self.lr_decay  # 学习率等比下调
            print("Decaying learning rate to %g" % self.lr)  # 终端打印警告提示
            # 
        self.start_decay = False  # 状态复位，确保单个 Epoch 内仅调整一次
        self.last_ppl = ppl  # 哨兵交接，将当前表现记录为下一轮对比的基准
        self._create_optimizer()  # 重塑底层优化器以应用更新后的学习率
