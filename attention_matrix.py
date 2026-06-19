#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time: 2024/12/12 - 20:43
@Project: STGNN - attention_matrix
@Author: Yacan Man(曼亚灿)
@Email: manyacan@qq.com
@Website: https://www.manyacan.com

=============================================================================
模块功能说明：（大论文使用了V1版本的核心算法，V2版本是后续迭代升级的产物）
本模块是 STGNN（时空图神经网络）的【自适应动态图拓扑推断核心】。
在复杂的工业场景（如水处理厂 SWaT、WADI）中，往往没有现成的物理拓扑连接图。
本模块利用数据驱动的方法，通过 GRU 提取每个传感器的时间序列特征，
然后利用自注意力机制（Self-Attention）计算任意两个传感器之间的“隐藏功能关联度”，
最终动态推演出一张有向/无向的邻接矩阵，并计算其高阶切比雪夫拉普拉斯矩阵，
为后续的图卷积（GCN / MixHop）铺平道路。
=============================================================================
"""
 
#  输入的 shape：[batch_size, window_size, node_num, feature_num]
#  输出的 shape：[batch_size, horizon_size, node_num, feature_num]
#  - batch_size  (B): 训练/推理时的样本批次大小
#  - feature_num (C): 变量的特征通道数（多维异构信号特征维度）
#  - node_num  (N): 节点总数（空间拓扑图中的节点总数 Nodes）
#  - window_size (L): 观测序列长度（观测滑动窗口长度）
#  - horizon_size (p): 预测序列长度（预测滑动窗口长度）

import torch
import torch.nn.functional as F
from torch import nn


class GetAttMatrix(nn.Module):
    """
    【初代版本 V1】：基于自注意力机制构建动态图和拉普拉斯矩阵
    
    核心执行管线：
    1. 特征编码：通过 GRU 将一段历史时间序列特征转化为每个节点的隐表征(Latent Representation)。
    2. 相似度打分：通过可学习参数 Key 和 Query，计算节点间的自注意力权重（Self-Attention）。
    3. 拓扑对称化：强制将注意力矩阵对称化，得到无向图的邻接矩阵 A。
    4. 谱图归一化：计算图的规范化拉普拉斯矩阵 Laplacian = D^{-1/2}(D-A)D^{-1/2}。
    5. 多阶空间扩散：生成多阶切比雪夫多项式，用于后续替代昂贵的特征值分解进行图卷积。
    """
    def __init__(self,
                 node_num: int,
                 window_size: int,
                 dropout_rate: float = 0.5,
                 leaky_rate: float = 0.2):
        super(GetAttMatrix, self).__init__()
        self.node_num = node_num      # N: 节点总数（例如 SWaT 里的 51 个传感器变量）
        self.window_size = window_size    # L: 时间窗口大小/序列长度（送入网络的长程历史步数）

        # ========== Self-Attention 核心可学习参数矩阵 ==========
        # 为系统中的每一个节点分配一个可学习的 Key 权重和 Query 权重向量，维度为 [N, 1]
        self.weight_key = nn.Parameter(torch.zeros(size=(self.node_num, 1)))
        self.weight_query = nn.Parameter(torch.zeros(size=(self.node_num, 1)))
        
        # 使用 Xavier 均匀分布初始化注意力权重，确保训练初期的梯度既不消失也不爆炸
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)

        # 定义防止注意力过度拟合特定边的 Dropout 层
        self.dropout = nn.Dropout(p=dropout_rate)          
        
        # 核心时间特征提取器：将每个节点长度为 window_size 的一维时序信号，
        # 映射压缩为一个长度为 node_num 的隐藏表征向量
        self.GRU = nn.GRU(self.window_size, self.node_num)  
        
        # LeakyReLU 激活函数，保留微弱的负向连接信号，防止节点间完全断开成为孤岛
        self.leaky_relu = nn.LeakyReLU(leaky_rate)        

    def latent_correlation_layer(self, x):
        """
        潜在相关层 (Latent Correlation Layer) 主干计算流。
        
        参数:
            x: 输入的原始多变量时间序列张量, 形状 [batch_size, window_size, node_num]
               即 [样本批次 B, 序列长度 L, 传感器节点数 N]
        
        返回:
            mul_l: 用于图卷积的多阶切比雪夫拉普拉斯矩阵簇, 形状 [4, N, N]
            attention: 数据驱动生成的图邻接矩阵 A, 形状 [N, N]
        """
        # ========== 步骤1: GRU 特征提取 ==========
        # PyTorch 的标准 GRU 期望的输入形状是 [seq_len, batch, input_size]
        # 在此处空间 GRU 逻辑中，视作 [N, B, L]，所以先对输入 x 进行维度置换：[B, L, N] -> [N, B, L]
        # 将 permute 后的张量显式地再次调用 .contiguous()，确保输入 GRU 前绝对连续
        gru_x, _ = self.GRU(x.permute(2, 0, 1).contiguous().contiguous())  # gru_x 形状: [N, B, N]
        
        # 将 GRU 提取的隐变量转置回常规理解的批次优先格式: [B, N, N]
        gru_x = gru_x.permute(1, 0, 2).contiguous()
        
        # ========== 步骤2: 计算自注意力权重 (构图核心) ==========
        # 这一步计算得到了当前 Batch 下每一个样本的图结构
        attention = self.self_graph_attention(gru_x)  # 形状: [B, N, N]
        
        # 降维：将同一个 Batch 内所有样本学到的瞬时图结构求均值，
        # 得到一张平滑且宏观的、能够代表该 Batch 整体拓扑关联的图矩阵 A
        attention = torch.mean(attention, dim=0)  # 形状: [N, N]

        # ========== 步骤3: 构建规范化拉普拉斯矩阵 (Spectral Graph Theory) ==========
        # 1. 计算对角度矩阵 D (Degree Matrix)
        degree = torch.sum(attention, dim=1)  # 对邻接矩阵的每一行求和，获取每个节点的出度，[N]
        degree_l = torch.diag(degree)         # 将一维出度向量转为对角矩阵形式，[N, N]
        
        # 2. 强制对称化邻接矩阵：A = 0.5 * (A + A^T)
        # 物理意义：将有向图拉平为无向图，以满足标准谱图卷积对矩阵正定性和对称性的严格数学要求
        attention = 0.5 * (attention + attention.T)
        
        # 3. 计算规范化度矩阵 D^{-1/2}
        # 加入 1e-7 的极小扰动防止除以 0 导致梯度崩溃产生 NaN
        diagonal_degree_hat = torch.diag(1 / (torch.sqrt(degree) + 1e-7))
        
        # 4. 组装规范化拉普拉斯矩阵: laplacian = D^{-1/2} * (D - A) * D^{-1/2}
        # 这一步将图上的拉普拉斯算子变换到 [-1, 1] 区间，保证切比雪夫多项式的收敛性
        laplacian = torch.matmul(diagonal_degree_hat,
                                 torch.matmul(degree_l - attention, diagonal_degree_hat))

        # ========== 步骤4: 生成多阶空间扩散感受野 ==========
        # 生成 K=4 的多阶切比雪夫矩阵，使图网络具有 4 跳 (4-hops) 的空间节点通信能力
        mul_l = self.cheb_polynomial(laplacian)  # 返回形状: [4, N, N]

        return mul_l, attention

    def self_graph_attention(self, x):
        """
        基于加性注意力(Additive Attention)的节点相似度评分网络。
        实现逻辑: score(i,j) = LeakyReLU(Key_i + Query_j)
        
        参数:
            x: GRU 输出的节点隐表征，形状 [B, N, GRU_hidden_size]
        """
        # [B, N, N]
        x = x.permute(0, 2, 1).contiguous()
        batch_size, series_num_tr, embedded_dim = x.size()

        # ========== 计算全局 Key 和 Query 映射 ==========
        # 对于每个节点 i，将其隐向量投影到 Key 空间: [B, N, N] @ [N, 1] = [B, N, 1]
        key = torch.matmul(x, self.weight_key)
        # 对于每个节点 j，将其隐向量投影到 Query 空间: [B, N, 1]
        query = torch.matmul(x, self.weight_query)

        # ========== 交叉配对打分矩阵组装 ==========
        # 目的: 生成所有节点对 (i, j) 之间的关联得分。
        # 利用 PyTorch 的 repeat 和 view 实现高效的矩阵广播加法。
        # 结果 data 包含了所有 NxN 个可能连接的原始评分。
        # data = (key.repeat(1, 1, series_num_tr).view(batch_size, series_num_tr * series_num_tr, 1) +
                # query.repeat(1, series_num_tr, 1))
        # 
        # data = data.squeeze(2)  # 降维为 [B, N*N]
        # data = data.view(batch_size, series_num_tr, -1)  # 重新塑形为邻接矩阵的格式: [B, N, N]
        
        # 【核心修订】：利用广播加法
        # key 形状: [B, N, 1], query 形状: [B, N, 1]
        # key + query.transpose(1, 2) 会自动广播成 [B, N, N]
        # 这样就不需要 repeat 产生大量的冗余显存空间
        data = key + query.transpose(1, 2)
        
        # ========== 非线性激活与归一化 ==========
        data = self.leaky_relu(data)  # 削弱但不完全剔除负相关边的权重
        
        # 沿第 2 维 (目标节点) 施加 softmax，将边缘权重转化为概率分布
        # 含义：节点 i 分发到所有相邻节点 j 的信息量之和为 1
        attention = F.softmax(data, dim=2)  # [B, N, N]
        
        # 随机丢弃部分连接，增强模型对异常缺失数据的鲁棒性
        attention = self.dropout(attention)
        
        return attention 

    @staticmethod
    def cheb_polynomial(laplacian):
        """
        切比雪夫多项式 (Chebyshev Polynomials) 逼近器。
        
        图信号处理中的痛点：对拉普拉斯矩阵做特征值分解（求特征向量矩阵和对角特征矩阵）的时间复杂度极高 O(N^3)。
        解决方案：根据数学证明，使用切比雪夫多项式 T_k(laplacian) 可以递归地、高效地逼近空间卷积。
        
        递归公式:
        - T_0(laplacian) = I (单位矩阵)
        - T_1(laplacian) = laplacian (拉普拉斯矩阵)
        - T_k(laplacian) = 2 * laplacian * T_{k-1}(laplacian) - T_{k-2}(laplacian)
        """
        node_num = laplacian.size(0)  # 获取节点总数 N
        laplacian = laplacian.unsqueeze(0)  # 升维至 [1, N, N] 以便后续 cat 堆叠
        
        # 第 0 阶：T_0 = I (单位矩阵)，代表节点保留自身信息不向外传播
        first_laplacian = torch.zeros([1, node_num, node_num], 
                                      device=laplacian.device, dtype=torch.float)
        # 初始化对角线为 1
        first_laplacian[0, torch.arange(node_num), torch.arange(node_num)] = 1.0  
        
        # 第 1 阶：T_1 = laplacian，代表信息只在一跳（直接邻居）范围内传播
        second_laplacian = laplacian  
        
        # 第 2 阶：T_2 = 2 * laplacian * T_1 - T_0 = 2 * laplacian^2 - I
        # 矩阵乘法模拟两跳信息汇聚
        third_laplacian = (2 * torch.matmul(laplacian, second_laplacian)) - first_laplacian
        
        # 第 3 阶：T_3 = 2 * laplacian * T_2 - T_1
        # 模拟三跳深层信息汇聚
        forth_laplacian = 2 * torch.matmul(laplacian, third_laplacian) - second_laplacian
        
        # 将 0到3 阶传播矩阵在第 0 维纵向拼接，返回一个形状为 [4, N, N] 的三维张量
        multi_order_laplacian = torch.cat([first_laplacian, second_laplacian, 
                                           third_laplacian, forth_laplacian], dim=0)
        return multi_order_laplacian 

    def forward(self, x):
        """模型调用入口包装函数"""
        mul_l, attention = self.latent_correlation_layer(x)
        return mul_l, attention


# =========================================================================================


class GetAttMatrixV2(nn.Module):
    """
    【升级进化版 V2】：针对多通道异构工业信号的动态构图模块
    
    相比 V1 的重大架构升级：
    1. 自适应通道融合：支持处理 [B, C, N, L] 格式的多维复合信号输入。
        [batch_size, feature_num, node_num, window_size]
        - batch_size  (B): 训练/推理时的样本批次大小
        - feature_num (C): 变量的特征通道数（多维异构信号特征维度）
        - node_num  (N): 传感器节点总数（空间拓扑图中的节点总数 Nodes）
        - window_size (L): 历史观测序列长度（时间窗口长度）

    2. 温度缩放 (Temperature Scaling)：引入 tau 控制注意力权重的尖锐程度（Sharpness）。
    3. 稀疏正则化 (Sparsification)：工业系统大多是稀疏网络。V2 支持 Top-K 截断和 SparseMax，
       直接把弱关联边彻底斩断归零，大幅度提升图神经网络的抗噪性能。
    4. 指数移动平均 (EMA Smoothing)：训练初期生成的动态图容易随 Batch 发生剧烈抖动。
       V2 引入动量 EMA 平滑图结构，推理测试时能提供极其稳固的物理骨架。
    """

    def __init__(self,
                 node_num: int,
                 window_size: int,
                 feature_num: int = None,    # C: 变量通道数 (如第一通道是水位，第二通道是水温)
                 K: int = 4,                 # 切比雪夫多项式阶数，即信息传播的跳数，默认为 4
                 dropout_rate: float = 0.1,
                 leaky_rate: float = 0.2,
                 tau: float = 0.7,           # softmax 温度缩放系数
                 add_self_loop: bool = True, # 是否显式添加自环
                 symmetrize: bool = True,    # 是否强制对称化矩阵
                 normalize: str = "sym",     # 拉普拉斯归一化策略: 选填 "sym", "row", "none"
                 sparsify: str = "topk",     # 稀疏化策略: 选填 "topk", "sparsemax", "none"
                 topk: int = 20,             # 稀疏化时，对于每个节点，强制斩断排名 20 名开外的微弱连接
                 ema_momentum: float = 0.9,  # EMA 动量系数，0.9 表示当前图拓扑有 90% 继承自历史长效图
                 use_ema_during_train: bool = False, # 训练期是否应用 EMA 迟滞效应
                 device: str = "cpu"):
        super().__init__()

        self.N = node_num          
        self.L = window_size         # L: 序列长度（时间窗口长度）
        self.C = feature_num         
        
        # ========== 超参数固化 ==========
        self.K = K                  
        self.dropout = nn.Dropout(p=dropout_rate)
        self.leaky_relu = nn.LeakyReLU(leaky_rate)
        self.tau = tau
        self.add_self_loop = add_self_loop
        self.symmetrize = symmetrize
        self.normalize = normalize
        self.sparsify = sparsify
        self.topk = topk
        self.ema_momentum = ema_momentum
        self.use_ema_during_train = use_ema_during_train

        # ========= 表示学习与通道融合层 =========
        if self.C is not None:
            # 如果存在多通道，分配一组可学习的通道权重参数
            # 初始化时让各通道贡献均衡 (1/C)
            self.channel_weight = nn.Parameter(torch.ones(self.C) / max(1, self.C))
        else:
            self.channel_weight = None

        # GRU 编码器，参数设置 batch_first=False 意味着期望第一维是序列特征尺寸 L
        self.GRU = nn.GRU(input_size=self.L, hidden_size=self.N, batch_first=False)

        # 同样保留 V1 中的可学习节点映射评分权重
        self.weight_key = nn.Parameter(torch.zeros(size=(self.N, 1)))
        self.weight_query = nn.Parameter(torch.zeros(size=(self.N, 1)))
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)

        # ========== 定义 EMA 图拓扑缓冲池 ==========
        # register_buffer 的作用是向 PyTorch 注册一个状态变量。
        # 它是模型的一部分（会随模型保存在 .pth 里），但它【不参与反向传播求导】。
        self.register_buffer("ema_A", torch.eye(self.N, device=device))

        self.to(device)

    # ======== 数学与图论工具函数 ========
    @staticmethod
    def _sparsemax(logits, dim=-1, eps=1e-12):
        """
        SparseMax 激活函数 (Softmax 的硬截断替代品)。
        在求解投影距离时，能将低于某个动态阈值的对数值严格映射为 0（绝对稀疏），
        这是构建稀疏图网络避免内存 OOM 的利器。
        """
        z = logits
        # 1. 降序排列得分
        z_sorted, _ = torch.sort(z, dim=dim, descending=True)
        dim_sz = z.size(dim)
        
        range_vals = torch.arange(1, dim_sz + 1, device=z.device, dtype=z.dtype).view(
            *([1] * (z.dim() - 1)), dim_sz
        ).transpose(-1, dim)
        
        # 2. 判断激活截断阈值 K
        cumsum = torch.cumsum(z_sorted, dim=dim)
        k = torch.sum((1 + range_vals * z_sorted) > cumsum, dim=dim, keepdim=True)
        
        # 3. 计算支持集的补偿变量 tau
        tau = (torch.gather(cumsum, dim, k - 1) - 1) / k.clamp(min=1)
        
        # 4. 强制裁剪：得分低于 tau 的边直接抹零
        p = torch.clamp(z - tau, min=0)
        
        # 5. 为了数值稳定性进行再次归一化
        p = p / (p.sum(dim=dim, keepdim=True) + eps)
        return p

    @staticmethod
    def _row_normalize(A, eps=1e-12):
        """
        行归一化：A_{ij} = A_{ij} / sum_k(A_{ik})
        使马尔可夫状态转移概率矩阵的每行和为 1，多用于有向图。
        """
        d = A.sum(dim=1, keepdim=True)
        return A / (d + eps)

    @staticmethod
    def _sym_normalize(A, eps=1e-12):
        """
        对称谱归一化：A_{sym} = D^{-1/2} * A * D^{-1/2}
        拉平高出度枢纽节点(Hub)的影响力，多用于无向图 GCN。
        """
        d = A.sum(dim=1)
        d_hat = torch.diag(1.0 / torch.sqrt(d + eps))
        return d_hat @ A @ d_hat

    def _normalize_adj(self, A):
        """工厂函数：根据配置返回对应的归一化矩阵"""
        if self.normalize == "none":
            return A
        elif self.normalize == "row":
            return self._row_normalize(A)
        elif self.normalize == "sym":
            return self._sym_normalize(A)
        else:
            raise ValueError("normalize 参数配置错误")

    def _cheb_polynomial(self, laplacian, K):
        """
        [高级张量版] 生成 K 阶切比雪夫多项式
        支持任意阶 K (K=1时直接退化为 GCN 基本范式)
        """
        N = laplacian.size(0)
        T_list = []
        T0 = torch.eye(N, device=laplacian.device, dtype=laplacian.dtype)
        T1 = laplacian
        T_list.append(T0)
        if K == 1:
            return torch.stack(T_list, dim=0)
        T_list.append(T1)
        
        for _ in range(2, K):
            T_next = 2 * (laplacian @ T1) - T0
            T_list.append(T_next)
            T0, T1 = T1, T_next
            
        return torch.stack(T_list, dim=0)

    def _ema_update(self, A):
        """
        指数移动平均图更新。
        在 with torch.no_grad() 不求导的环境下，以动量混合历史图结构 ema_A 与新生成的当前图 A。
        使得最终学到的图结构具备历史记忆和平滑衰减特性。
        """
        with torch.no_grad():
            self.ema_A = self.ema_momentum * self.ema_A + (1 - self.ema_momentum) * A

    # ======== V2 改进型主工作流 ========
    def _unify_input(self, x):
        """
        维度与通道塌缩融合器。
        
        如果传入了多通道异构特征 [B, C, N, L]（比如：[批次, (压力,液位), 节点, 时间窗口长度]），
        利用学习到的通道权重参数，通过 tensordot 执行带权矩阵内积，
        将其无损地塌缩降维为标准分析格式 [B, N, L]。
        """
        if x.dim() == 3:
            # 基础格式 [B, L, N] 直接变轴为 [B, N, L]
            assert x.size(1) == self.L and x.size(2) == self.N, \
                f"维度校验不通过: 期望 [B, L={self.L}, N={self.N}], 实际 {list(x.shape)}"
            x_use = x.permute(0, 2, 1).contiguous()
            
        elif x.dim() == 4:
            # 复合通道格式 [B, C, N, L]
            B, C, N, L_dim = x.shape
            assert N == self.N and L_dim == self.L, f"尺寸错位"
            assert self.C is None or C == self.C, f"通道数错位"
            
            if self.channel_weight is None:
                w = torch.ones(C, device=x.device) / C  # 若未初始化权重则简单平均
            else:
                w = torch.softmax(self.channel_weight, dim=0)  # softmax 归一通道权值
                
            # 执行高级张量点积收缩：消除维度1 (C 通道)，得到形状 [B, N, L] 的综合特征场
            x_use = torch.tensordot(x, w, dims=([1], [0]))  
        else:
            raise ValueError("不支持的张量深度")
        return x_use  

    def _attention(self, H):
        """
        加入温度控制和稀疏裁剪机制的改良版注意力矩阵推演算法。
        """
        B, N, _ = H.shape
        # Key / Query 映射，[B, N, N] @ [N, 1] = [B, N, 1]
        key = H @ self.weight_key  
        query = H @ self.weight_query  

        # 使用广播机制计算所有可能的节点配对得分
        # query 先转置再 repeat: 形成交叉对应网格
        data = key.repeat(1, 1, N) + query.transpose(1, 2).repeat(1, N, 1)  
        data = self.leaky_relu(data)
        
        # 引入 tau 温度参数：
        # tau < 1 会放大得分间的差距，使最终的邻接概率矩阵变得极其两极分化（尖锐化）
        data = data / max(1e-6, self.tau)  

        # 判断并执行用户指定的稀疏图构建策略
        if self.sparsify == "sparsemax":
            A_b = self._sparsemax(data, dim=2)  # 使用截断非线性函数
        else:
            A_b = F.softmax(data, dim=2)  # 经典 Softmax
            
            # 开启 Top-K 硬裁剪策略
            if self.sparsify == "topk" and self.topk is not None and self.topk > 0 and self.topk < N:
                # 寻找每一行得分最高的 k 个节点的索引和值
                topk_vals, topk_idx = torch.topk(A_b, k=self.topk, dim=2)
                
                # 初始化一个全零的掩码屏蔽矩阵
                mask = torch.zeros_like(A_b)
                # 使用 scatter_ 将有资格入选 Top-K 的边标记为 1.0
                mask.scatter_(2, topk_idx, 1.0)
                
                # 矩阵点乘：硬生生切断微弱连接信号
                A_b = A_b * mask
                # 被斩断后概率和不再是 1，需要手动二次归一化修补
                A_b = A_b / (A_b.sum(dim=2, keepdim=True) + 1e-12)

        A_b = self.dropout(A_b)
        return A_b  

    def forward(self, x, use_ema: bool = None, return_all: bool = False):
        """
        V2 版总管线入口。
        返回参数可选配置（支持返回完整执行过程中间产物用于 Debug 或图分析）。
        """
        x_use = self._unify_input(x)  # [B, N, L]
        B, N, L_dim = x_use.shape

        gru_in = x_use.permute(1, 0, 2).contiguous() 
        H, _ = self.GRU(gru_in)  # 提取时空隐状态 H: [N, B, N]
                
        # 恢复为 [B, N, N] 供注意力交叉打分使用
        H_bnn = H.permute(1, 2, 0).contiguous()
        
        A_batch = self._attention(H_bnn)  
        # Batch 层面聚合提取共性拓扑
        A = A_batch.mean(dim=0)  

        if self.symmetrize:
            A = 0.5 * (A + A.T)

        if self.add_self_loop:
            A = A + torch.eye(N, device=A.device, dtype=A.dtype)

        # 归一化收敛值
        A_hat = self._normalize_adj(A)

        # 把本次迭代新推演出的 A_hat 融入历史长河 ema_A 中
        self._ema_update(A_hat.detach())

        # 智能判定：处于测试评估阶段时，强制使用稳定平滑的 EMA 图结构
        if use_ema is None:
            use_ema = (not self.training) or self.use_ema_during_train

        A_used = (self.ema_A if use_ema else A_hat)  

        # 结合配置，组装最终拉普拉斯算子
        if self.normalize == "sym":
            laplacian = torch.eye(N, device=A_used.device, dtype=A_used.dtype) - A_used
        elif self.normalize == "row":
            laplacian = torch.eye(N, device=A_used.device, dtype=A_used.dtype) - A_used
        else:
            d = A_used.sum(dim=1)
            D = torch.diag(d)
            laplacian = D - A_used

        # 拓展切比雪夫传播阶数
        mul_cheb = self._cheb_polynomial(laplacian, self.K)  

        # 差异化返还
        if return_all:
            # 返还字典内含批量注意力阵列，方便外部画图追踪
            return mul_cheb, A_used, A, {"A_batch": A_batch, "laplacian": laplacian}
        else:
            return mul_cheb, A_used


if __name__ == '__main__':
    pass