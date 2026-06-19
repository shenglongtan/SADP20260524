#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
# 输入的 shape：[batch_size, feature_num, series_num, window_size]
# 输出的 shape：[batch_size, feature_num, series_num, horizon_size]
#  - batch_size  (B): 训练/推理时的样本批次大小
#  - feature_num (C): 变量的特征通道数（多维异构信号特征维度）
#  - series_num  (N): 节点总数（空间拓扑图中的节点总数 Nodes）
#  - window_size (L): 观测序列长度（观测滑动窗口长度）
#  - horizon_size (p): 预测序列长度（预测滑动窗口长度）


import torch
from torch import nn
import torch.nn.functional as nn_f

# =========================================================================
# 外部依赖声明
# 请确保您的项目中存在对应的文件，或者根据实际路径进行调整
# =========================================================================
from GNN.layer import LayerNorm, linear  
from attention_matrix import GetAttMatrix  # 仅保留您需要的初代 Attention 模块

# =========================================================================
# 核心组件 1：时间卷积网络 (TC Module)
# =========================================================================
class DilatedInceptionLayer(nn.Module):
    """
    扩张 Inception 层：负责在时间维度上挖掘多尺度的长短时序列特征。
    """
    def __init__(self, c_in_num: int, c_out_num: int, dl_factor: int = 2):
        # 初始化父类
        super(DilatedInceptionLayer, self).__init__()
        # 声明一个 ModuleList，用于存放并行运行的多个卷积分支
        self.layers = nn.ModuleList()
        # 定义 Inception 模块中的 4 种时间感受野大小（卷积核长度）
        self.kernel_size = [2, 3, 6, 7]
        
        # 将期望的输出总通道数，平分给这 4 个不同的卷积分支
        c_out_num_per_branch = int(c_out_num / len(self.kernel_size))
        
        # 遍历每一种卷积核尺寸，构建并行的 2D 卷积层
        for kern in self.kernel_size:
            # 采用 2D 卷积处理 1D 时间序列：节点维度核为 1（不混合节点），时间维度核为 kern
            # dilation=(1, dl_factor) 表示时间维度上施加扩张因子，扩大感受野
            self.layers.append(
                nn.Conv2d(c_in_num, c_out_num_per_branch, (1, kern), dilation=(1, dl_factor))
            )

    def forward(self, input_x: torch.Tensor) -> torch.Tensor:
        # 定义空列表，存储 4 个卷积分支的输出张量
        x = []
        # 将输入数据分别送入 4 个并行的卷积网络中
        for i in range(len(self.kernel_size)):
            x.append(self.layers[i](input_x))
            
        # 【关键截断】：因为 4 个分支的卷积核大小不同，卷积后输出的时间序列长度也不同。
        # 最大卷积核 (7) 产生的时间序列最短。为了能在通道维度拼接，必须向最短的序列看齐。
        for i in range(len(self.kernel_size)):
            # x[-1].size(3) 是第 4 个分支（核长为 7）输出的时间序列长度
            # x[i][..., -length:] 表示在时间维度上，只截取最后（最新）的 length 个时间步
            x[i] = x[i][..., -x[-1].size(3):] 
            
        # 将对齐后的 4 个特征图在通道维度 (dim=1) 上拼接起来，恢复到总的 c_out_num
        x = torch.cat(x, dim=1)
        return x

# =========================================================================
# 核心组件 2：底层图卷积算子 (Mathematical Graph Convolution)
# =========================================================================
class GraphConvLayer(nn.Module):
    """
    【图卷积算子】：封装张量乘法的核心数学映射单元。
    负责执行多维节点特征张量与空间邻接图矩阵的高效聚合计算。
    """
    def __init__(self):
        super(GraphConvLayer, self).__init__() # 算子基类初始化

    @staticmethod
    def forward(input_x, adj_m):
        # 严谨的 4D 维度映射 (b: batch_size, c: feature_num, l: window_size)
        # 图节点信息传播流向：j 代表信息源节点(source)，i 代表目标节点(target)
        
        # input_x: 当前层的隐特征张量，形状 [batch_size, feature_num, series_num(源), window_size]
        # 对应 einsum 公式左侧的第一项: bcjl
        
        # adj_m: 图邻接矩阵，指示了源节点(j)到目标节点(i)的转移权重，形状 [series_num(目标), series_num(源)]
        # 对应 einsum 公式左侧的第二项: ij
        
        # 执行高效 Einsum 爱因斯坦求和运算：
        # 'bcjl,ij->bcil' 物理意义：将 j 维度的源节点特征，依据 adj_m(ij) 的连通权值，
        # 加权聚合至 i 维度的目标节点上，同时保持 b, c, l 三个维度的结构绝对静止。
        output_x = torch.einsum('bcjl,ij->bcil', (input_x, adj_m))
        
        # 强制执行张量内存的连续化重排，确保底层 CUDA 加速库在反向传播计算中的稳定性
        return output_x.contiguous()
    
# =========================================================================
# 核心组件 3：空间图卷积网络 (GC Module)
# =========================================================================
class MixHopLayer(nn.Module):
    """
    混合跳数传播层：解决深层 GNN 的过平滑现象，聚合 0~K 跳的空间邻居信息。
    """
    def __init__(self, c_in_num: int, c_out_num: int, layer_num: int, alpha: float):
        # 初始化父类
        super(MixHopLayer, self).__init__()
        # 实例化刚才定义的底层图卷积算子
        self.GraphConvLayer = GraphConvLayer()
        # 实例化线性层，用于将多跳拼接后的高维特征压缩回原始通道数 (K+1 跳 * 原始通道)
        self.linear = linear((layer_num + 1) * c_in_num, c_out_num)
        # layer_num 代表我们要往外传播的最大跳数 K
        self.layer_num = layer_num  
        # alpha 是超参数，控制每一步传播中保留原始自身特征的比例
        self.alpha = alpha          

    def forward(self, x, adj_m):
        # 1. 邻接矩阵加上单位阵 (torch.eye)，强制添加节点到自身的自环
        adj_m = adj_m + torch.eye(adj_m.size(0)).to(x.device)
        # 计算每个节点的度数 (即邻接矩阵每一行的权值总和)
        d = adj_m.sum(1)
        # 进行行归一化：A_norm = D^-1 * A，使得信息转移概率和为 1
        norm_adj_m = adj_m / d.view(-1, 1)
        
        # 初始特征 h (0跳特征)
        h = x
        # 将 0跳特征存入列表 out 中
        out = [h] 
        
        # 2. 循环执行 K 次信息跳跃传播
        for i in range(self.layer_num):
            # 核心公式: 当前跳特征 = alpha * 原始特征 + (1 - alpha) * 聚合后的上一跳特征
            h = self.alpha * x + (1 - self.alpha) * self.GraphConvLayer(h, norm_adj_m)
            # 将第 i 跳的结果存入列表
            out.append(h)
            
        # 3. 将 [0跳, 1跳, ..., K跳] 的所有特征张量在通道维度 (dim=1) 拼接
        ho = torch.cat(out, dim=1) 
        # 送入线性层，将拼接后的臃肿通道降维回 c_out_num
        ho = self.linear(ho)       
        # 返回聚合完成的空间图特征
        return ho

# =========================================================================
# 主网络：MTGNN (纯净版，剔除无效逻辑)
# =========================================================================
class MTGNN(nn.Module):
    """
    多变量时间序列图神经网络总成。
    """
    def __init__(
            self,
            device,
            node_num: int,            # 节点总数 N
            feature_num: int,         # 原始特征通道数 (例如水位、压力等)
            mix_hop_depth: int = 2,   # MixHop 空间卷积的层数 K
            drop_rate: float = 0.3,   # 防过拟合 Dropout 率
            dl_exp: int = 1,          # 时间卷积的扩张因数底数
            conv_channels: int = 32,  # 时空提取过程中的计算通道数
            residual_channels: int = 32, # 残差主干道的通道数
            skip_channels: int = 64,  # 汇总跳跃连接时的通道数
            end_channels: int = 128,  # 预测输出前的隐藏层通道数
            window_size: int = 12,    # 输入的历史时间步 T
            horizon_size: int = 12,   # 要预测的未来时间步
            layer_num: int = 3,       # 堆叠的时空模块 (TC+GC) 层数
            prop_alpha: float = 0.05, # MixHop 保留初始状态的比例
            layer_norm_affline: bool = True # LayerNorm 是否可学习
    ):
        # 初始化父类
        super(MTGNN, self).__init__()
        # 将基础超参数绑定为类的属性
        self.node_num = node_num
        self.drop_rate = drop_rate
        self.window_size = window_size
        self.layer_num = layer_num
        
        # 判断如果传入的跳数不为 0，则开启图卷积网络
        self.use_gcn = True if mix_hop_depth else False
        
        # 使用 ModuleList 声明网络层的容器，用于后续在 for 循环中堆叠层级
        self.filter_conv = nn.ModuleList()   # 时间卷积主分支
        self.gate_conv = nn.ModuleList()     # 时间卷积门控分支
        self.residual_conv = nn.ModuleList() # 残差 1x1 卷积分支
        self.skip_conv = nn.ModuleList()     # 跳跃输出 1x1 卷积分支
        self.gconv1 = nn.ModuleList()        # 正向 MixHop 空间卷积
        self.gconv2 = nn.ModuleList()        # 反向 (转置) MixHop 空间卷积
        self.norm = nn.ModuleList()          # 层级归一化

        # 【入口映射】：定义首个 1x1 卷积，将原始输入的多维特征映射到统一的残差主干道维度
        self.start_conv = nn.Conv2d(in_channels=feature_num, out_channels=residual_channels, kernel_size=(1, 1))

        # 【核心动态图生成】：直接锁定实例化 Attention V1 模块
        # 它期望后续接收 [Batch, Time, Node] 格式的数据
        self.latent_correlation = GetAttMatrix(node_num=node_num, window_size=window_size)

        # 【感受野计算】：根据扩张因数，计算网络理论上能看多远的历史数据
        kernel_size = 7 # 在 Inception 层中定义的最大时间核尺寸
        if dl_exp > 1:
            # 等比数列求和公式计算总感受野
            self.receptive_field = int(1 + (kernel_size - 1) * (dl_exp ** self.layer_num - 1) / (dl_exp - 1))
        else:
            # 常数序列求和计算总感受野
            self.receptive_field = self.layer_num * (kernel_size - 1) + 1

        # 初始化当前感受野追踪变量
        rf_size_i = 1 
        # 初始化时间扩张率
        new_dilation = 1
        
        # 【构建 N 层时空处理模块】
        for j in range(1, self.layer_num + 1):
            # 动态计算在经过当前第 j 层后，网络累计的感受野大小
            if dl_exp > 1:
                rf_size_j = int(rf_size_i + (kernel_size - 1) * (dl_exp ** j - 1) / (dl_exp - 1))
            else:
                rf_size_j = rf_size_i + j * (kernel_size - 1)
                
            # 1. 向列表中追加 TC (时间卷积) 的 Filter 分支和 Gate 分支
            self.filter_conv.append(DilatedInceptionLayer(residual_channels, conv_channels, dl_factor=new_dilation))
            self.gate_conv.append(DilatedInceptionLayer(residual_channels, conv_channels, dl_factor=new_dilation))
            # 追加残差通道的 1x1 恢复卷积
            self.residual_conv.append(nn.Conv2d(in_channels=conv_channels, out_channels=residual_channels, kernel_size=(1, 1)))
            
            # 2. 追加 Skip (跳跃连接) 卷积层
            # 目的：将当前层的特征跳跃输出到网络末端。为了在末端求和，这里利用卷积的 kernel_size 在时间轴上进行压缩池化
            if self.window_size > self.receptive_field:
                # 实际序列长度有富余，压缩量为 window_size - rf_size_j + 1
                self.skip_conv.append(nn.Conv2d(in_channels=conv_channels, out_channels=skip_channels, kernel_size=(1, self.window_size - rf_size_j + 1)))
            else:
                # 序列已被 pad，压缩量以网络极限 receptive_field 为基准
                self.skip_conv.append(nn.Conv2d(in_channels=conv_channels, out_channels=skip_channels, kernel_size=(1, self.receptive_field - rf_size_j + 1)))
                
            # 3. 追加 GC (图卷积) 模块
            if self.use_gcn:
                # 分别针对原图和转置图构建 MixHop，用于捕捉有向图的双向信息流
                self.gconv1.append(MixHopLayer(conv_channels, residual_channels, mix_hop_depth, prop_alpha))
                self.gconv2.append(MixHopLayer(conv_channels, residual_channels, mix_hop_depth, prop_alpha))
                
            # 4. 追加 LayerNorm 归一化层，其尺寸设置同样受序列长度条件的制约
            if self.window_size > self.receptive_field:
                self.norm.append(LayerNorm((residual_channels, node_num, self.window_size - rf_size_j + 1), elementwise_affine=layer_norm_affline))
            else:
                self.norm.append(LayerNorm((residual_channels, node_num, self.receptive_field - rf_size_j + 1), elementwise_affine=layer_norm_affline))
                
            # 每一层构建完毕，扩张率按底数倍增，为下一层扩大感受野
            new_dilation *= dl_exp 

        # 【构建输出层】：将所有跳跃连接汇总后，通过两层 MLP (用 1x1 卷积实现) 将特征投影为最终的预测步长
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels, out_channels=horizon_size, kernel_size=(1, 1), bias=True)

        # 【构建全局初始跳跃层】：直接将最原始的输入特征，经过时间轴压缩后，输送到网络末端作为基座
        if self.window_size > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=feature_num, out_channels=skip_channels, kernel_size=(1, self.window_size), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.window_size - self.receptive_field + 1), bias=True)
        else:
            self.skip0 = nn.Conv2d(in_channels=feature_num, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)

        # 预存一份节点的整数索引在显存中（LayerNorm 会用到）
        self.indices = torch.arange(self.node_num).to(device)

    def forward(self, input_m: torch.tensor):
        """
        前向传播计算管线。
        @param input_m: 数据集送入的张量，维度必须符合 [Batch, Channel, Node, Time]
        """
        # 0. 健壮性校验与长度补齐 (Padding)
        # 强制断言 DataLoader 传来的序列长度是否等于配置中的 window_size
        assert input_m.size(3) == self.window_size, '输入数据的序列长度与模型初始化的 window_size 不匹配！'
        
        # 如果配置的输入窗口长度，满足不了网络结构堆叠出的理论感受野（即输入太短，网络太深）
        if self.window_size < self.receptive_field:
            # 在时间维度的最左侧（最古老的历史数据端）填充值为 0 的序列，强制对齐长度
            input_m = nn.functional.pad(input_m, (self.receptive_field - self.window_size, 0, 0, 0))

        # 1. 动态图拓扑推断 (Attention V1 数据截取)
        # 因为 V1 是针对单变量设计的，所以对 input_m [Batch, Channel, Node, Time] 进行切片。
        # [:, 0, :, :] 物理含义：抛弃其它辅助通道，仅抽取第 0 个核心变量的数值进行构图。
        # 随后 permute(0, 2, 1) 进行轴对换，使其符合 GetAttMatrix 所需的 [Batch, Time, Node]
        attention_input = input_m[:, 0, :, :].permute(0, 2, 1).contiguous()
        
        # 调用 V1 注意力网络，获取当前 Batch 平滑推演出的邻接矩阵 adj_m [Node, Node]
        _, adj_m = self.latent_correlation(attention_input)

        # 2. 初始特征准备
        # 将完整的输入张量（包含被构图丢弃的辅助变量），通过 1x1 卷积升维/降维到残差主干道
        # x 形如: [Batch, residual_channels, Node, Time]
        x = self.start_conv(input_m)
        
        # 将完整的输入张量进行 Dropout 后，通过全局跳跃层，时间轴压缩至 1，变为 [Batch, skip_channels, Node, 1]
        skip = self.skip0(nn_f.dropout(input_m, self.drop_rate, training=self.training))

        # 3. 核心 N 层时空处理大循环
        for i in range(self.layer_num):
            # 记录当前刚进入本层的数据状态，用于最后与处理完的特征做“残差相加”
            residual = x 
            
            # --- 3.1 时间特征提取 (TC) ---
            # 通过 Tanh 和 Filter 分支提取主成分，通过 Sigmoid 和 Gate 分支控制信息流通率
            # 注意：经过 DilatedInceptionLayer 后，x 的时间维度长度变短了！
            filter_x = torch.tanh(self.filter_conv[i](x))
            gate = torch.sigmoid(self.gate_conv[i](x))
            # 门控机制：逐元素相乘
            x = filter_x * gate 
            # 施加 Dropout 防止深度学习过拟合
            x = nn_f.dropout(x, self.drop_rate, training=self.training)
            
            # --- 3.2 收集本层的跳跃特征 ---
            # 把当前层的输出结果，经过核压缩后，累加到底层的 skip 基座上
            skip += self.skip_conv[i](x)
            
            # --- 3.3 空间图特征提取 (GC) ---
            if self.use_gcn:
                # 分别沿动态图的正向 adj_m 和反向 adj_m.transpose(1, 0) 传播信息，并合并结果
                x = self.gconv1[i](x, adj_m) + self.gconv2[i](x, adj_m.transpose(1, 0))
            else:
                # 若关闭 GCN，则执行一次 1x1 的恒等通道映射
                x = self.residual_conv[i](x)
                
            # --- 3.4 残差汇接与归一化 ---
            # 核心难点：此时由于 TC 层的无 pad 卷积，x 的时间序列长度已短于最初存下的 residual。
            # 为了能够按元素相加，必须通过 [:, :, :, -x.size(3):] 将 residual 的时间轴从最右侧截断对齐！
            x = x + residual[:, :, :, -x.size(3):]
            # 对本层最终输出特征进行层归一化
            x = self.norm[i](x, self.indices)

        # 4. 预测输出网络
        # 把大循环最后一层刚结束时的输出 x，再压榨一次补充进全局 skip 张量里
        skip = self.skipE(x) + skip
        # 通过 ReLU 激活消除负数特征
        x = nn_f.relu(skip)
        
        # 经过第一层 1x1 卷积，通道数从 skip_channels 映射到 end_channels
        x = nn_f.relu(self.end_conv_1(x))
        # 经过第二层 1x1 卷积，通道数从 end_channels 映射到业务需要的预测步数 horizon_size
        x = self.end_conv_2(x) 
        
        # 最终输出前，进行转置，将格式变换为符合下游损失函数或验证器计算规范的张量。
        # 并且将本轮动态生成的图 adj_m 一并返还给主程序，用于计算图正则化或绘制拓扑图。
        return x.transpose(1, 3), adj_m
