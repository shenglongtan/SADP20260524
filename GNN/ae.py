#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
【核心张量维度规范说明】
本系统内的所有多维时间序列张量，必须严格遵循以下 4D 格式定义：
输入的 shape：[batch_size, feature_num, series_num, window_size]
输出的 shape：[batch_size, feature_num, series_num, window_size]

 - batch_size  (B): 训练/推理时的样本批次大小
 - feature_num (C): 变量的特征通道数（多维异构信号特征维度）
 - series_num  (N): 节点总数（空间拓扑图中的节点总数 Nodes）
 - window_size (L): 观测序列长度（观测滑动窗口长度）
=============================================================================
【模块功能摘要】
目标通道重构的简易窗口自编码器 (Window Autoencoder)。
利用 1D 卷积（在 2D 算子中通过限制空间核大小实现）对多元时间序列进行时域上的编码压缩与解码重构。
=============================================================================
"""

import torch
from torch import nn

class WindowAE(nn.Module):
    """
    【时间序列自编码器】：负责提取观测窗口内的时空潜变量，并还原目标通道信号。
    """
    def __init__(self, in_channels: int = 1, hidden_channels: int = 32, dropout_rate: float = 0.1):
        super().__init__() # 初始化 PyTorch 模型基类
        
        # 编码器部分：负责将原始低维输入映射至高维潜在特征空间 (Latent Space)
        self.encoder = nn.Sequential(
            # 第一层卷积：节点维度核尺寸为 1 (保障节点间空间独立)，时间维度核尺寸为 3 (捕获时序局部依赖)
            # 引入 padding=(0, 1) 确保卷积操作前后时间长度 window_size 不发生坍缩
            nn.Conv2d(in_channels, hidden_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.GELU(), # 使用高斯误差线性单元 (GELU) 替代普通 ReLU，提供更平滑的非线性激活流
            
            # 预留的正则化层：在空间/时间未压缩的自编码器中，加入微弱的 Dropout 可以制造信息瓶颈，
            # 防止网络退化为无意义的恒等映射（即死记硬背），提升工业异常检测的泛化能力。
            nn.Dropout(p=dropout_rate), 
            
            # 第二层卷积：在潜在空间内部进一步提取深层时序特征
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.GELU(),
        )
        
        # 解码器部分：负责将潜变量特征平滑解码，重构回原始物理观测信号
        self.decoder = nn.Sequential(
            # 解码特征转换层
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.GELU(),
            
            # 输出投影层：将隐层的高维通道重新压缩回原始的输入通道数 (如单通道目标重构)
            # 此处不使用激活函数，以允许网络输出真实物理环境下的正负连续预测值
            nn.Conv2d(hidden_channels, in_channels, kernel_size=(1, 3), padding=(0, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向重构计算管线。
        @param x: 输入时间序列张量，严格遵循规约 [batch_size, feature_num, series_num, window_size]
        @return: 重构后的时间序列张量，形状与输入完全一致
        """
        # 步骤 1：通过编码器将原始观测序列 x 映射至潜变量特征图 z
        # 输入 x 形状: [batch_size, in_channels, series_num, window_size]
        # 输出 z 形状: [batch_size, hidden_channels, series_num, window_size]
        z = self.encoder(x)
        
        # 步骤 2：通过解码器将潜变量 z 逆向重构为预测信号
        # 输出形状恢复为: [batch_size, in_channels, series_num, window_size]
        reconstructed_x = self.decoder(z)
        
        # 返回重构序列以供下游计算重构损失 (Reconstruction Loss，如 MSE)
        return reconstructed_x