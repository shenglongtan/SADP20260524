#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time: 2024/8/23 - 10:08
@Project: MTGNN - model
@Author: Yacan Man(曼亚灿)
@Email: manyacan@qq.com
@Website: https://www.manyacan.com
"""
from __future__ import division  # 引入精确除法，兼容Python 2的除法行为
from GNN.layer import * # 导入自定义的GNN层（如LayerNorm等，具体实现在layer.py中）
from attention_matrix import GetAttMatrix, GetAttMatrixV2  # 导入注意力矩阵生成模块
import torch.nn.functional as nn_f  # 导入PyTorch的函数式神经网络接口，重命名为nn_f
from typing import List, Tuple  # 导入类型提示模块，用于静态类型检查

class GraphConstructor(nn.Module):
    """
    邻接矩阵构造器。
    用于在没有预定义图结构时，自适应地学习节点之间的图拓扑关系。
    """
    def __init__(self, node_num, ebd_num, device, alpha: int = 3, node_feature=None, top_k: int = 0):
        """
        类对象的初始化函数。
        @param node_num: 节点（时间序列）的数量。
        @param top_k: 保留前k个最大的连接权重。top_k==0 表示返回完整的稠密邻接矩阵。
        @param ebd_num: 节点嵌入（Embedding）的维度大小。
        @param device: 使用的计算设备（CPU或GPU）。
        @param alpha: 控制激活函数饱和率的超参数。
        @param node_feature: 节点的静态特征（如地理位置等），形状为 [node_num, feature_num]。
        """
        super(GraphConstructor, self).__init__()  # 调用父类nn.Module的初始化方法
        self.node_feature = node_feature  # 保存传入的节点特征
        if self.node_feature is None:
            # 如果没有提供静态特征，则使用PyTorch的Embedding层随机初始化两个可学习的嵌入字典
            # 嵌入层: 公式(1), (2)中的E_1, E_2. 分别代表源节点嵌入和目标节点嵌入
            self.emb_0 = nn.Embedding(node_num, ebd_num)  # 源节点嵌入层
            self.emb_1 = nn.Embedding(node_num, ebd_num)  # 目标节点嵌入层
            # 对应的线性变换层，保持维度不变
            self.linear_0 = nn.Linear(ebd_num, ebd_num)
            self.linear_1 = nn.Linear(ebd_num, ebd_num)
        else:
            # 如果提供了节点特征，则将其转移到指定设备上
            self.node_feature = self.node_feature.to(device)
            # 使用线性层将节点特征的维度映射到指定的 ebd_num 维度
            self.linear_0 = nn.Linear(node_feature.shape[1], ebd_num)
            self.linear_1 = nn.Linear(node_feature.shape[1], ebd_num)
        self.top_k = top_k  # 保存 top_k 参数
        self.alpha = alpha  # 保存缩放系数 alpha
        self.device = device  # 保存设备信息
    def forward(self, indices):
        """
        前向传播函数。根据传入的节点索引生成自适应邻接矩阵。
        """
        # 公式(1): $M1 = \tanh(\alpha \cdot E1 \cdot \Theta_1)$, 公式(2): M2 = \tanh(\alpha \cdot E2 \cdot \Theta_2).
        # emb_v_0 和 emb_v_1 的形状变化: [len(indices), feature_num] → [len(indices), emb_num]
        if self.node_feature is None:
            # 如果没有静态特征，通过索引从Embedding层获取向量
            emb_v_0 = self.emb_0(indices)
            emb_v_1 = self.emb_1(indices)
        else:
            # 如果有静态特征，直接根据索引提取特征矩阵中的对应行
            emb_v_0 = self.node_feature[indices, :]
            emb_v_1 = emb_v_0  # 目标节点特征和源节点特征使用相同的初始输入
        # 经过线性层、alpha缩放以及tanh非线性激活，得到M1和M2
        emb_v_0 = torch.tanh(self.alpha * self.linear_0(emb_v_0))
        emb_v_1 = torch.tanh(self.alpha * self.linear_1(emb_v_1))
        # 公式(3): A = \text{ReLU}(\tanh(\alpha \cdot (M1 \cdot M2^T - M2 \cdot M1^T))).
        # a的形状: [len(indices), len(indices)]，即节点数 x 节点数
        # 通过 M1 * M2^T - M2 * M1^T 计算两个节点间的非对称关系得分
        a = torch.mm(emb_v_0, emb_v_1.transpose(1, 0)) - torch.mm(emb_v_1, emb_v_0.transpose(1, 0))
        # 经过alpha缩放、tanh和ReLU激活，去除负数权重，生成最终的初始邻接矩阵
        adj = nn_f.relu(torch.tanh(self.alpha * a))  # 源代码实现方式
        # adj = torch.tanh(self.alpha * a)  # 注释：如果 adj 中有负数, 在MixHopLayer类的归一化中 d = adj_m.sum(1) 可能出现0导致除零错误。
        if self.top_k > 0:  # 如果设置了 top_k，则过滤掉较小的连接权重
            # 创建一个全0的掩码矩阵（mask），形状与邻接矩阵相同
            mask = torch.zeros(indices.size(0), indices.size(0)).to(self.device)
            # mask.fill_(float('0'))
            # 为了防止数值完全相同导致topk选取不稳定，加上极小的随机噪声(0.01级别)
            # 沿着行维度(dim=1)获取前 top_k 个最大值的数值和索引
            top_k_v, top_k_i = (adj + torch.rand_like(adj) * 0.01).topk(self.top_k, 1)
            # 将mask中 top_k_i 对应的位置填充为 1
            mask.scatter_(1, top_k_i, top_k_v.fill_(1))
            # 返回过滤后的稀疏邻接矩阵（未过滤的原始矩阵也一同返回备用）
            return adj * mask, adj
        else:  # 如果 top_k 为0，直接返回完整的稠密邻接矩阵
            return adj, adj
class DilatedInceptionLayer(nn.Module):
    """
    扩张Inception层（时间卷积模块的核心组件）。
    利用不同大小的卷积核提取不同感受野的时间特征。
    """
    def __init__(self, c_in_num: int, c_out_num: int, dl_factor: int = 2):
        """
        类对象初始化函数。
        @param c_in_num: 输入通道数（通常对应残差通道 residual_channels）。
        @param c_out_num: 输出通道数（通常对应卷积通道 conv_channels）。
        @param dl_factor: 扩张率（dilation factor），用于控制时间维度的感受野跳跃。
        """
        super(DilatedInceptionLayer, self).__init__()  # 初始化父类
        self.layers = nn.ModuleList()  # 使用ModuleList保存并行的多个卷积层
        self.kernel_size = [2, 3, 6, 7]  # 定义Inception模块中包含的四种不同时间维度的卷积核大小
        # 将输出通道均分给各个并行的卷积分支
        c_out_num = int(c_out_num / len(self.kernel_size))
        # 遍历每个卷积核大小，创建对应的2D卷积层
        for kern in self.kernel_size:
            # 节点维度(dim=2)卷积核为1，时间维度(dim=3)卷积核为kern，并在时间维度上使用dl_factor进行扩张
            self.layers.append(nn.Conv2d(c_in_num, c_out_num, (1, kern), dilation=(1, dl_factor)))
    def forward(self, input_x: torch.Tensor) -> torch.Tensor:
        """
        扩张Inception层的前向传播函数。
        @param input_x: 输入张量，形状为 [batch_size, feature_num(residual_channels), node_num, window_size(receptive_field)].
        @return: 输出张量，形状为 [batch_size, conv_channels, node_num, W_out],
                 其中时间维度 W_out = (window_size - dl_factor*(kern[-1]-1)-1)+1.
        """
        x = []  # 存储不同卷积分支的输出
        for i in range(len(self.kernel_size)):
            # 逐个分支进行卷积，输出形状为 [batch_size, c_out_num, node_num, W_out_i]
            x.append(self.layers[i](input_x))
        for i in range(len(self.kernel_size)):
            # 由于卷积核大小不同，各分支输出的时间维度长度不同
            # 最大的卷积核(kern[-1]=7)产生的时间维度最短。
            # 为了能够按通道拼接，这里将所有分支的输出在时间维度上截断，使其与最短的时间序列对齐对齐（取最后的时间步）
            x[i] = x[i][..., -x[-1].size(3):]  # x[-1].size(3) 就是最小的 W_out
        # 在通道维度(dim=1)将所有分支的特征图拼接在一起
        x = torch.cat(x, dim=1)
        return x
class GraphConvLayer(nn.Module):
    """
    图卷积层。
    实现节点特征沿图拓扑结构的基本聚合操作。
    """
    def __init__(self):
        """
        类对象初始化函数。
        """
        super(GraphConvLayer, self).__init__()  # 初始化父类
    @staticmethod
    def forward(input_x, adj_m):
        """
        图卷积层的前向传播函数（静态方法，无内部可学习参数）。
        @param input_x: 节点特征张量，形状为 [batch_size, conv_channels, node_num, W_out]。
        @param adj_m: 邻接矩阵（已添加自环并归一化），形状为 [node_num, node_num]。
        @return: 聚合后的节点特征，形状为 [batch_size, conv_channels, node_num, W_out]。
        """
        # 使用爱因斯坦求和约定 (einsum) 进行批量图矩阵乘法
        # n: batch_size, c: conv_channels, w: node_num(特征输入), l: W_out
        # v: node_num(目标节点), w: node_num(源节点，对于adj_m)
        # 也就是将特征在节点维度(w)上根据邻接矩阵(vw)进行加权求和，得到新的节点维度(v)
        output_x = torch.einsum('ncwl,vw->ncvl', (input_x, adj_m))
        return output_x.contiguous()  # 确保返回的张量在内存中是连续的
class MixHopLayer(nn.Module):
    """
    混合跳数传播层（Mix-hop propagation layer）。
    用于捕捉不同跳数（hop）的图空间依赖。
    """
    def __init__(self, c_in_num: int, c_out_num: int, layer_num, alpha):
        """
        类对象初始化函数。
        @param c_in_num: 输入通道数（conv_channels）。
        @param c_out_num: 输出通道数（residual_channels）。
        @param layer_num: 混合跳数传播的层数（即最大的跳数 K）。
        @param alpha: 超参数，控制每一步传播中保留根节点（自身）原始状态的比例，用于防止过平滑。
        """
        super(MixHopLayer, self).__init__()  # 初始化父类
        self.GraphConvLayer = GraphConvLayer()  # 实例化基础的图卷积操作类
        # 线性层，用于将所有跳数 (0 到 K跳, 共 layer_num+1 个) 拼接后的特征融合降维到输出通道数
        self.linear = linear((layer_num + 1) * c_in_num, c_out_num)  # 此处的 linear 假设是在 GNN.layer 中定义的
        self.layer_num = layer_num  # 传播跳数
        self.alpha = alpha  # 保留系数
    def forward(self, x, adj_m):
        """
        混合跳数传播层的前向传播。
        @param x: 输入节点特征，形状为 [batch_size, conv_channels, node_num, W_out]。
        @param adj_m: 邻接矩阵，形状为 [node_num, node_num]。
        @return: 传播并融合后的特征张量。
        """
        # 在邻接矩阵上加上单位阵（增加自环），让节点在聚合时包含自身信息
        adj_m = adj_m + torch.eye(adj_m.size(0)).to(x.device)  
        d = adj_m.sum(1)  # 计算每个节点的度数 (按行求和)
        norm_adj_m = adj_m / d.view(-1, 1)  # 进行行归一化，即 A_norm = D^-1 * A
        h = x  # 初始状态定义为 0 跳特征
        out = [h]  # 将 0 跳特征存入列表
        # 循环执行 K 次传播
        for i in range(self.layer_num):
            # 每一步传播后，结合保留的原始信息 (alpha * x) 与新聚合的信息 ((1-alpha) * GCN(h))
            # 这是为了防止图卷积层数过多导致的节点表示过平滑（Over-smoothing）问题
            h = self.alpha * x + (1 - self.alpha) * self.GraphConvLayer(h, norm_adj_m)
            out.append(h)  # 保存每一跳的结果
        # 在通道维度(dim=1)将 0跳 到 K跳 的所有特征拼接起来
        ho = torch.cat(out, dim=1)
        # 通过线性层将多跳特征融合为指定的输出通道数
        ho = self.linear(ho)
        # 检查输出是否存在 NaN（异常值检测）
        if torch.isnan(ho.sum()).item():
            print('ho: ', out)  # 如果出现NaN，打印信息便于调试
        return ho

class MTGNN(nn.Module):
    """
    基于图神经网络的多变量时间序列预测 (MTGNN)。
    官方 GitHub: https://github.com/nnzhan/MTGNN
    ArXiv 论文: https://arxiv.org/abs/2005.11650
    """
    def __init__(
            self,
            device,
            node_num: int,
            feature_num: int,
            pd_a: torch.tensor = None,
            node_feature=None,
            mix_hop_depth: int = 2,
            drop_rate: float = 0.3,
            top_k: int = 20,
            ebd_num: int = 40,
            dl_exp: int = 1,
            conv_channels: int = 32,
            residual_channels: int = 32,
            skip_channels: int = 64,
            end_channels: int = 128,
            window_size: int = 12,
            horizon_size: int = 12,
            layer_num: int = 3,
            prop_alpha: float = 0.05,
            tanh_alpha: int = 3,
            layer_norm_affline: bool = True,
            save_adj: bool = False,
            adj_method: str = 'embedding'
    ):
        """
        类对象初始化函数。
        @param mix_hop_depth: 图卷积和混合跳数传播层的深度（对应论文图4）。如果非0，则开启图卷积(use_gcn=True)。
        @param node_num: 图中的节点（时间序列）数量。
        @param device: 运行设备（CPU或GPU）。
        @param pd_a: 预定义的邻接矩阵。如果为 None，模型将自适应构建邻接矩阵。
        @param node_feature: 节点静态特征矩阵，形状为 [node_num, feature_size]。
        @param drop_rate: Dropout 的丢弃概率。
        @param top_k: 构建图时，每个节点保留的前 k 个连接节点。
        @param ebd_num: 随机初始化节点嵌入的维度。
        @param dl_exp: 扩张卷积步长的指数增长底数。
        @param conv_channels: 时空卷积块输出的通道数。
        @param residual_channels: 残差连接的通道数。
        @param skip_channels: 跳跃连接的通道数。
        @param end_channels: 输出层最后几步使用的通道数。
        @param window_size: 特征数据集的时间窗口大小（历史序列长度）。
        @param horizon_size: 目标数据集的预测长度（未来序列长度）。
        @param feature_num: 每个节点输入的特征维度。
        @param layer_num: 包含时空卷积模块（TC+GC）的层数。
        @param prop_alpha: 混合跳数传播时保留自身状态的比例。
        @param tanh_alpha: 图构建器中tanh激活的缩放超参数。
        @param layer_norm_affline: LayerNorm 是否包含可学习的仿射参数。
        @param save_adj: 是否保存生成的图邻接矩阵。
        @param adj_method: 图邻接矩阵的生成方法 ('embedding', 'attention', 'attention_v2')。
        """
        super(MTGNN, self).__init__()  # 初始化父类
        self.node_num = node_num  # 节点数
        self.drop_rate = drop_rate  # Dropout率
        self.pd_a = pd_a  # 预定义图邻接矩阵
        self.window_size = window_size  # 输入窗口大小
        self.layer_num = layer_num  # 模块层数
        self.adj_method = adj_method  # 图生成方式
        # 根据是否传入 mix_hop_depth 决定是否应用图卷积层
        self.use_gcn = True if mix_hop_depth else False  
        # 声明多个ModuleList用于存放每一层的网络模块
        self.filter_conv = nn.ModuleList()  # 时间卷积的主分支（Filter）
        self.gate_conv = nn.ModuleList()  # 时间卷积的门控分支（Gate）
        self.residual_conv = nn.ModuleList()  # 残差卷积（用于不使用GCN的情况或者用于形状对齐）
        self.skip_conv = nn.ModuleList()  # 将每层特征映射到跳跃连接（Skip connection）维度的卷积层
        self.gconv1 = nn.ModuleList()  # 针对原邻接矩阵的图卷积模块
        self.gconv2 = nn.ModuleList()  # 针对转置邻接矩阵的图卷积模块（处理有向图的双向传播）
        self.norm = nn.ModuleList()  # 每一层后的层归一化模块
        # 初始的1x1卷积，将原始特征通道数转换为内部残差通道数
        self.start_conv = nn.Conv2d(
            in_channels=feature_num,
            out_channels=residual_channels,
            kernel_size=(1, 1)
        )
        # 实例化基于嵌入（Embedding）的图构建器（这里也可以是其他两种方法之一，根据 adj_method 参数选择）
        self.create_adj_m = GraphConstructor(node_num, ebd_num, device, tanh_alpha,
                                             node_feature, top_k)
        # 时间卷积模块内Inception结构的最大时间核大小
        kernel_size = 7  
        # 计算整个MTGNN模型最终所需的最大时间感受野（receptive_field）
        if dl_exp > 1:
            # 如果扩张因数指数递增（等比数列求和）
            self.receptive_field = int(
                1 + (kernel_size - 1) * (dl_exp ** self.layer_num - 1) / (dl_exp - 1))
        else:
            # 如果扩张因数固定（常数项求和）
            self.receptive_field = self.layer_num * (kernel_size - 1) + 1
        # 构建MTGNN主体的层级结构
        for i in range(1):  # 论文原意这里其实可能有多块（blocks），这里按代码逻辑是外层循环执行1次
            if dl_exp > 1:
                # rf_size_i 记录进入第 i 块（block）时的感受野起点（本代码块内实际只有i=0）
                rf_size_i = int(
                    1 + i * (kernel_size - 1) * (dl_exp ** self.layer_num - 1) / (
                            dl_exp - 1))
            else:
                rf_size_i = i * self.layer_num * (kernel_size - 1) + 1
            new_dilation = 1  # 初始扩张率设为1
            # 逐层构建时空模块
            for j in range(1, self.layer_num + 1):
                # 计算经过第 j 层之后的累计感受野大小
                if dl_exp > 1:
                    rf_size_j = int(
                        rf_size_i + (kernel_size - 1) * (dl_exp ** j - 1) / (dl_exp - 1))
                else:
                    rf_size_j = rf_size_i + j * (kernel_size - 1)
                # 添加时间卷积层：主分支 (Filter) 和门控分支 (Gate)
                self.filter_conv.append(
                    DilatedInceptionLayer(residual_channels, conv_channels, dl_factor=new_dilation))
                self.gate_conv.append(
                    DilatedInceptionLayer(residual_channels, conv_channels, dl_factor=new_dilation))
                # 添加1x1残差卷积，将conv_channels降维回residual_channels
                self.residual_conv.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=residual_channels,
                                                    kernel_size=(1, 1)))
                # 添加跳跃连接层。根据输入序列是否长于感受野，动态调整1D卷积核长度进行池化压缩
                if self.window_size > self.receptive_field:
                    self.skip_conv.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.window_size - rf_size_j + 1)))
                else:
                    self.skip_conv.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.receptive_field - rf_size_j + 1)))
                # 如果启用了GCN，则构建正向和反向（转置）的MixHop传播层
                if self.use_gcn:
                    self.gconv1.append(
                        MixHopLayer(conv_channels, residual_channels, mix_hop_depth, prop_alpha))
                    self.gconv2.append(
                        MixHopLayer(conv_channels, residual_channels, mix_hop_depth, prop_alpha))
                # 为当前层添加基于特征的层归一化模块
                if self.window_size > self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, node_num, self.window_size - rf_size_j + 1),
                                               elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, node_num, self.receptive_field - rf_size_j + 1),
                                               elementwise_affine=layer_norm_affline))
                new_dilation *= dl_exp  # 更新下一层的扩张率
        # 输出模块：经过所有的跳跃连接相加后，进行两次1x1卷积得到最终预测的 horizon_size 时间步
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=horizon_size,
                                    kernel_size=(1, 1),
                                    bias=True)
        # 初始的跳跃连接层：将输入数据直接跳接汇入最终的Skip部分
        if self.window_size > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=feature_num, out_channels=skip_channels,
                                   kernel_size=(1, self.window_size),
                                   bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels,
                                   kernel_size=(1, self.window_size - self.receptive_field + 1), bias=True)
        else:
            self.skip0 = nn.Conv2d(in_channels=feature_num, out_channels=skip_channels,
                                   kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1),
                                   bias=True)
        # 创建一个包含0到node_num-1的张量，用于嵌入层索引检索
        self.indices = torch.arange(self.node_num).to(device)
        # 保存是否保存邻接矩阵的标志
        self.save_adj = save_adj
        # 根据初始化参数选择动态注意力矩阵的生成方式
        if self.adj_method == 'embedding':
            ...  # 默认的生成方式，在前方 GraphConstructor 已经初始化
        elif self.adj_method == 'attention':
            # 简单的序列注意力计算类
            self.latent_correlation = GetAttMatrix(
                node_num=node_num,
                window_size=window_size
            )
        elif self.adj_method == 'attention_v2':
            # 改进版的复杂注意力机制类，加入了多头、稀疏化、指数移动平均等特性
            self.latent_correlation = GetAttMatrixV2(
                node_num=node_num,
                window_size=window_size,
                feature_num=feature_num,  # 如果输入是 [B, C, N, T]，这里传入特征数C
                K=4,  # 注意力的头数或其他降维设置
                tau=0.7,  # Softmax 的温度超参数
                sparsify="topk",  # 采用 topk 稀疏化策略
                topk=top_k,  # 复用类的 k 值
                add_self_loop=True,  # 增加自环
                symmetrize=True,  # 使邻接矩阵对称化
                normalize="sym",  # 采用对称归一化 ("sym")
                ema_momentum=0.95,  # 移动平均的动量
                use_ema_during_train=False,  # 训练期是否启用EMA
                device=device
            )
        else:
            raise ValueError('adj_method must be "attention" or "attention_v2"!!!')  # 抛出不支持的异常
    def forward(self, input_m: torch.tensor, indices=None):
        """
        主模型的前向传播函数。
        @param input_m: Dataloader传递过来的输入矩阵。
                        形状应为 [batch_size, feature_num, node_num, window_size]。
        @param indices: 可选参数，可能用于向模型传递子图信息（子节点索引）。
        @return: 输出预测结果和保留的邻接矩阵。
        """
        # ======================================  0. 检查输入尺寸  ======================================
        # 验证输入时间序列长度是否等于定义的窗口大小
        assert input_m.size(3) == self.window_size, 'The input_m.size(3) is not equal to model.window_size!'
        # 如果给定的窗口大小小于模型的理论最大感受野，则在时间维度的前方用0进行填充 (padding)
        if self.window_size < self.receptive_field:  
            # 填充格式为 (pad_left, pad_right, pad_top, pad_bottom, ...) 这里只在时间轴左侧填充
            input_m = nn.functional.pad(input_m, (self.receptive_field - self.window_size, 0, 0, 0))
        # ================================  1. 创建/获取图邻接矩阵  ==================================
        # 判断并获取网络在本次运算中将使用的邻接矩阵
        if self.adj_method == 'embedding':  # 通过GraphConstructor嵌入生成，尺寸: [node_num, node_num]
            if self.pd_a is not None:  # 如果用户提供了预先定义好的固定拓扑矩阵
                adj_m = self.pd_a  # adj_m 为无自环的邻接矩阵
                adj_m_save = adj_m  # 保留用于后续日志或分析
            else:
                # 否则，利用嵌入层生成自适应的动态拓扑结构
                adj_m, adj_m_save = self.create_adj_m(self.indices) if indices is None else self.create_adj_m(indices)
        elif self.adj_method == 'attention':
            # 利用GetAttMatrix计算，数据抽取变换，并调整维度为[batch, window, node]输入
            _, adj_m = self.latent_correlation(input_m[:, 0, :, :].permute(0, 2, 1).contiguous())
            adj_m_save = adj_m
        elif self.adj_method == 'attention_v2':
            # 利用改进版的注意力类处理完整的多特征输入
            mul_l, adj_m = self.latent_correlation(input_m)
            adj_m_save = adj_m
        else:
            raise ValueError('adj_method must be "embedding" or "attention"!!!')  # 非法方法报错
        # ==============================  2. 第一个2D卷积层（特征升维） =================================
        # nn.Conv2d(in_channels=feature_num, out_channels=residual_channels, kernel_size=(1, 1))
        # nn.Conv2d 输入形状: [N, C_in, H_in, W_in]，输出形状: [N, C_out, H_out, W_out]
        # 此处在时间和节点上做1x1卷积，单纯扩展特征维度。
        # 输入: [batch_size, feature_num, node_num, window_size(receptive_field)]
        # 输出 x 形状: [batch_size, residual_channels, node_num, window_size(receptive_field)].
        x = self.start_conv(input_m)
        # 准备全局跳跃连接 (Skip Connections)，在进入主网络层前，直接将输入端信息引至输出端前。
        # nn.Conv2d 将原始特征维度映射到 skip_channels，且在时间维度核大小为 self.receptive_field，直接将时间轴池化为1。
        # 输入形状：[batch_size, feature_num, node_num, receptive_field]
        # 输出形状: [batch_size, skip_channels, node_num, 1].
        skip = self.skip0(nn_f.dropout(input_m, self.drop_rate, training=self.training))
        # =================================  3. 时空卷积核心模块 (TC Module & GC Module)  ===================================
        for i in range(self.layer_num):
            residual = x  # 保存进入本层的初始状态用于稍后的残差相加
            # 1.1 时间卷积模块 (TC module).
            # filter_x 和 gate 输出的时间维度 W_out 会随着扩张卷积的作用而逐渐缩小
            # W_out = (window_size - dl_factor*(kern[-1]-1)-1)+1.
            filter_x = torch.tanh(self.filter_conv[i](x))  # 提取时序主成分
            gate = torch.sigmoid(self.gate_conv[i](x))  # 提取时序控制门阈值
            x = filter_x * gate  # 利用门控机制过滤信息
            x = nn_f.dropout(x, self.drop_rate, training=self.training)  # Dropout正则化防止过拟合
            # 1.2 累加跳跃连接 (Skip Connections)
            # 通过跳跃层将当前特征图压缩至时间维为1，并累加到全局的 skip 变量上
            # 这样保证了网络最后的输出可以观测到每一层的中间提取特征
            skip += self.skip_conv[i](x)
            # 1.3 图卷积模块 (GC Module)
            if self.use_gcn:
                # 开启GCN时，分别基于原邻接矩阵和转置后的邻接矩阵进行传播（考虑有向信息流动）
                # 此时输出 x 的形状: [batch_size, residual_channels, node_num, W_out].
                x = self.gconv1[i](x, adj_m) + self.gconv2[i](x, adj_m.transpose(1, 0))
            else:
                # 若不开启GCN，只通过1x1卷积进行简单的特征通道调整
                x = self.residual_conv[i](x)
            # 残差连接：将进入本层前的状态与本层输出状态相加
            # 由于经历时间卷积后序列长度变短，需要在时间维度上截取 residual 对齐当前 x
            x = x + residual[:, :, :, -x.size(3):]
            # 应用归一化层 (LayerNorm)
            if indices is None:
                x = self.norm[i](x, self.indices)
            else:
                x = self.norm[i](x, indices)
        # 将最后一层输出通过池化层对齐时间维度后，累加到全局跳跃向量中
        skip = self.skipE(x) + skip
        # 通过 ReLU 激活
        x = nn_f.relu(skip)
        # 经过输出层模块的两重 1x1 卷积，最终将时间维度转换为期望的预测长度 horizon_size
        x = nn_f.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        # 返回转置后的结果，形状变化 [batch, horizon, node, channels]，并将计算用到的邻接矩阵一起返回
        return x.transpose(1, 3), adj_m_save

if __name__ == '__main__':
    ...  # 主程序测试入口（被省略）
