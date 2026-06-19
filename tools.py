#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具模块：提供数据加载、保存、评估指标、可视化和一些公用工具函数。

说明：本文件尽量保持对外接口兼容，添加详细中文注释以便后续维护与教学使用。
@Time: 2026/1/19 - 17:49
@Project: 07_AnomalyDetection - tools
@Author: NeuroSpark(PyDeepLearning Lab)
"""

# ==========================================
# 1. 基础库导入区
# ==========================================
import os                             # 操作系统路径与文件操作
import hues                           # 控制台彩色日志输出库（项目中用于友好提示）
import json                           # 用于保存/加载 JSON 格式数据
import torch                          # PyTorch 深度学习框架核心库
import pickle                         # Python 原生二进制序列化/反序列化库
import argparse                       # 命令行参数解析库
import numpy as np                    # 科学计算与多维数组操作库
import pandas as pd                   # 数据表格处理库
import matplotlib.pyplot as plt       # 数据可视化绘图库

from datetime import datetime         # 日期与时间处理
from prettytable import PrettyTable   # 美观打印终端表格的组件（可选）
from typing import Union, List, Dict, Optional  # 类型提示注解，提升代码规范性


# ==========================================
# 2. 常量定义区：SWaT 与 WADI 数据集攻击时间表
# ==========================================
# SWaT 数据集攻击事件记录表
# 格式: (开始时间, 结束时间, ["受攻击的传感器/执行器节点列表"])
ATTACK_EVENTS_SWAT = [
    (datetime(2015, 12, 28, 10, 29, 14), datetime(2015, 12, 28, 10, 44, 53), ["MV101"]),
    (datetime(2015, 12, 28, 10, 51, 8), datetime(2015, 12, 28, 10, 58, 30), ["P102"]),
    (datetime(2015, 12, 28, 11, 22, 0), datetime(2015, 12, 28, 11, 28, 22), ["LIT101"]),
    (datetime(2015, 12, 28, 11, 47, 39), datetime(2015, 12, 28, 11, 54, 8), ["MV504"]),
    (datetime(2015, 12, 28, 11, 58, 20), datetime(2015, 12, 28, 11, 58, 20), ["No Physical Impact Attack"]),
    (datetime(2015, 12, 28, 12, 0, 55), datetime(2015, 12, 28, 12, 4, 10), ["AIT202"]),
    (datetime(2015, 12, 28, 12, 8, 25), datetime(2015, 12, 28, 12, 15, 33), ["LIT301"]),
    (datetime(2015, 12, 28, 13, 10, 10), datetime(2015, 12, 28, 13, 26, 13), ["DPIT301"]),
    (datetime(2015, 12, 28, 14, 15, 0), datetime(2015, 12, 28, 14, 15, 0), ["No Physical Impact Attack"]),
    (datetime(2015, 12, 28, 14, 16, 20), datetime(2015, 12, 28, 14, 19, 0), ["FIT401"]),
    (datetime(2015, 12, 28, 14, 19, 0), datetime(2015, 12, 28, 14, 28, 20), ["FIT401"]),
    (datetime(2015, 12, 29, 11, 10, 40), datetime(2015, 12, 29, 11, 10, 40), ["No Physical Impact Attack"]),
    (datetime(2015, 12, 29, 11, 11, 25), datetime(2015, 12, 29, 11, 15, 17), ["MV304"]),
    (datetime(2015, 12, 29, 11, 35, 40), datetime(2015, 12, 29, 11, 42, 50), ["MV303"]),
    (datetime(2015, 12, 29, 11, 52, 1), datetime(2015, 12, 29, 11, 52, 1), ["No Physical Impact Attack"]),
    (datetime(2015, 12, 29, 11, 57, 25), datetime(2015, 12, 29, 12, 2, 0), ["LIT301"]),
    (datetime(2015, 12, 29, 14, 38, 12), datetime(2015, 12, 29, 14, 50, 8), ["MV303"]),
    (datetime(2015, 12, 29, 18, 8, 55), datetime(2015, 12, 29, 18, 8, 55), ["No Physical Impact Attack"]),
    (datetime(2015, 12, 29, 18, 10, 43), datetime(2015, 12, 29, 18, 15, 1), ["AIT504"]),
    (datetime(2015, 12, 29, 18, 15, 43), datetime(2015, 12, 29, 18, 22, 17), ["AIT504"]),
    (datetime(2015, 12, 29, 18, 30, 0), datetime(2015, 12, 29, 18, 42, 0), ["MV101", "LIT101"]),
    (datetime(2015, 12, 29, 22, 55, 18), datetime(2015, 12, 29, 23, 3, 0), ["UV401", "AIT502", "P501"]),
    (datetime(2015, 12, 30, 1, 42, 34), datetime(2015, 12, 30, 1, 54, 10), ["P602", "DIT301", "MV302"]),
    (datetime(2015, 12, 30, 9, 51, 8), datetime(2015, 12, 30, 9, 56, 28), ["P203", "P205"]),
    (datetime(2015, 12, 30, 10, 1, 50), datetime(2015, 12, 30, 10, 12, 1), ["LIT401", "P401"]),
    (datetime(2015, 12, 30, 17, 4, 56), datetime(2015, 12, 30, 17, 29, 0), ["P101", "LIT301"]),
    (datetime(2015, 12, 31, 1, 17, 8), datetime(2015, 12, 31, 1, 45, 18), ["P302", "LIT401"]),
    (datetime(2015, 12, 31, 1, 45, 19), datetime(2015, 12, 31, 11, 15, 27), ["P302"]),
    (datetime(2015, 12, 31, 15, 32, 0), datetime(2015, 12, 31, 15, 34, 0), ["P201", "P203", "P205"]),
    (datetime(2015, 12, 31, 15, 47, 40), datetime(2015, 12, 31, 16, 7, 10), ["LIT101", "P101", "MV201"]),
    (datetime(2015, 12, 31, 22, 5, 34), datetime(2015, 12, 31, 22, 11, 40), ["LIT401"]),
    (datetime(2016, 1, 1, 10, 36, 0), datetime(2016, 1, 1, 10, 46, 0), ["LIT301"]),
    (datetime(2016, 1, 1, 14, 21, 12), datetime(2016, 1, 1, 14, 28, 35), ["LIT101"]),
    (datetime(2016, 1, 1, 17, 12, 40), datetime(2016, 1, 1, 17, 14, 20), ["P101"]),
    (datetime(2016, 1, 1, 17, 18, 56), datetime(2016, 1, 1, 17, 26, 56), ["P101", "P102"]),
    (datetime(2016, 1, 1, 22, 16, 1), datetime(2016, 1, 1, 22, 25, 0), ["LIT101"]),
    (datetime(2016, 1, 2, 11, 17, 2), datetime(2016, 1, 2, 11, 24, 50), ["P501", "FIT502"]),
    (datetime(2016, 1, 2, 11, 31, 38), datetime(2016, 1, 2, 11, 36, 18), ["AIT402", "AIT502"]),
    (datetime(2016, 1, 2, 11, 43, 48), datetime(2016, 1, 2, 11, 50, 28), ["FIT401", "AIT502"]),
    (datetime(2016, 1, 2, 11, 51, 42), datetime(2016, 1, 2, 11, 56, 38), ["FIT401"]),
    (datetime(2016, 1, 2, 13, 13, 2), datetime(2016, 1, 2, 13, 40, 56), ["LIT301"]),
]

# WADI 数据集攻击事件记录表
ATTACK_EVENTS_WADI = [
    # 格式: (开始时间, 结束时间, ["受攻击的节点列表"])
    (datetime(2017, 10, 9, 19, 25, 0), datetime(2017, 10, 9, 19, 50, 16), ["1_MV_001_STATUS"]),
    (datetime(2017, 10, 10, 10, 24, 10), datetime(2017, 10, 10, 10, 34, 0), ["1_FIT_001_PV"]),
    (datetime(2017, 10, 10, 10, 55, 0), datetime(2017, 10, 10, 11, 24, 0), ["2_LT_002_PV"]),
    (datetime(2017, 10, 10, 11, 7, 46), datetime(2017, 10, 10, 11, 12, 15), ["1_AIT_001_PV"]),
    (datetime(2017, 10, 10, 11, 30, 40), datetime(2017, 10, 10, 11, 44, 50),
     ["2_MCV_101_CO", "2_MCV_201_CO", "2_MCV_301_CO", "2_MCV_401_CO", "2_MCV_501_CO", "2_MCV_601_CO"]),
    (datetime(2017, 10, 10, 13, 39, 30), datetime(2017, 10, 10, 13, 50, 40),
     ["2_MCV_101_CO", "2_MCV_201_CO", "1_AIT_002_PV", "2_MV_003_STATUS"]),
    (datetime(2017, 10, 10, 14, 48, 17), datetime(2017, 10, 10, 14, 59, 55), ["1_AIT_002_PV"]),
    (datetime(2017, 10, 10, 14, 53, 44), datetime(2017, 10, 10, 15, 0, 32), ["2_MV_003_STATUS"]),
    (datetime(2017, 10, 10, 17, 40, 0), datetime(2017, 10, 10, 17, 49, 40), ["2_MCV_007_CO"]),
    (datetime(2017, 10, 11, 10, 55, 0), datetime(2017, 10, 11, 10, 56, 27), ["1_P_005_STATUS", "1_P_006_STATUS"]),
    (datetime(2017, 10, 11, 11, 17, 54), datetime(2017, 10, 11, 11, 31, 20), ["1_MV_001_STATUS"]),
    (datetime(2017, 10, 11, 11, 36, 31), datetime(2017, 10, 11, 11, 47, 0), ["2_MCV_007_CO"]),
    (datetime(2017, 10, 11, 11, 59, 0), datetime(2017, 10, 11, 12, 5, 0), ["2_MCV_007_CO"]),
    (datetime(2017, 10, 11, 12, 7, 30), datetime(2017, 10, 11, 12, 10, 52), ["2_PIC_003_SP"]),
    (datetime(2017, 10, 11, 12, 16, 0), datetime(2017, 10, 11, 12, 25, 36), ["1_P_001_STATUS", "1_P_003_STATUS"]),
    (datetime(2017, 10, 11, 15, 26, 30), datetime(2017, 10, 11, 15, 37, 0), ["2_LT_002_PV"]),
]


# ==========================================
# 3. 参数解析与环境配置工具函数
# ==========================================
def str_to_bool(input_s: str) -> bool:
    """
    将常见的字符串表示转换为 Python 的 bool (布尔) 类型。
    支持值（大小写不敏感）：
      - False: 'false', 'f', '0', 'no', 'n'
      - True:  'true', 't', '1', 'yes', 'y'
    如果传入的已经是 bool 类型，则原样返回。
    """
    if isinstance(input_s, bool):
        return input_s

    s = str(input_s).lower()
    if s in {'false', 'f', '0', 'no', 'n'}:
        return False
    if s in {'true', 't', '1', 'yes', 'y'}:
        return True

    raise ValueError(f'{input_s} is not a valid boolean value')


def get_input_paras():
    """
    从终端命令行解析输入的超参数。
    采用两阶段解析策略：
    - 第1阶段：仅解析 `--model` 字段，以确定要加载哪个图网络模型
    - 第2阶段：根据第1阶段确定的模型，添加其专属的网络架构超参数，并返回完整解析结果
    """
    # --- 第一阶段：优先解析模型名称 ---
    stage1_parser = argparse.ArgumentParser(add_help=False)
    stage1_parser.add_argument('--model', type=str, default='MTGNN', choices=['MTGNN'], help='模型名称')
    
    # stage1_parser.add_argument('--model', type=str, required=True, choices=['MTGNN'],
                            #    help='Model to use (MTGNN).')
    model_args, _ = stage1_parser.parse_known_args()

    # --- 第二阶段：构建全量参数解析器 ---
    parser = argparse.ArgumentParser(description=f'[{model_args.model}] is Selected.')

    # 基本系统与全局参数
    parser.add_argument('--model', type=str, default=model_args.model, help='使用的模型类型')
    parser.add_argument('--seed', type=int, default=None, help='随机种子，保证实验复现')

    # 数据集加载与预处理参数
    parser.add_argument('--data_path', type=str, default='D:/大学/博士论文/GNN部分/07_RA-STGNN/Data/SWat/swat_data_10s.pkl', help='输入数据存放路径')
    parser.add_argument('--repeat_exp_num', type=int, default=1, help='独立重复实验的运行次数')#日常调试（设为 1）： 像你现在这样，在调代码、看模型能不能跑通、观察 Loss 下降趋势的时候，设置为 1 是最合适的。跑通一次就能看到结果，节省时间。最终跑实验数据（设为 3、5 或 10）： 在图神经网络（GNN）的论文里，因为模型权重的随机初始化和 GPU 计算的非确定性，单次实验的结果是有偶然性的。
    parser.add_argument('--epoch_num', type=int, default=100, help='训练总 Epoch 轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小 (Batch size)')
    # parser.add_argument('--adj_data', type=str, default=None, help='外部静态邻接矩阵的路径')#我的train_attention.py里没有这个参数，tools.py里有，保持兼容性，先注释掉
    # parser.add_argument('--node_feature', type=str, default=None, help='静态节点特征的路径')#我的train_attention.py里没有这个参数，tools.py里有，保持兼容性，先注释掉

    # 时空感受野参数 (滑动窗口配置) 
    parser.add_argument('--window_size', type=int, default=36,help='历史观测窗口长度 (Look-back Window)')
    parser.add_argument('--horizon_size', type=int, default=12,help='未来预测时间步长 (Horizon Size)')
    parser.add_argument('--sliding_step', type=int, default=12,help='滑动窗口的步长(Stride)。')#'设为 1 时(默认)：逐帧连续滑动，样本量极大，相邻样本重叠度高；''设为与 window_size 相同时(如 36)：无重叠滑动，样本总量骤降至 1/36，可极大缓解 OOM 问题并提升训练速度。'
    
    # 特征与网络基础维度参数
    parser.add_argument('--ebd_num', type=int, default=40, help='随机节点嵌入维度')
    parser.add_argument('--top_k', type=int, default=0, help='自适应图保留的最大 top-k 个邻居')
    parser.add_argument('--feature_num', type=int, default=2, help='输入特征通道数量')
    parser.add_argument('--drop_rate', type=float, default=0.3, help='Dropout 随机失活率')
    parser.add_argument('--dl_exp', type=int, default=1, help='时间卷积的扩张倍数')

    # 训练控制与优化策略参数
    parser.add_argument('--cl', type=str_to_bool, default=True, help='是否开启课程学习(Curriculum Learning)')
    parser.add_argument('--shuffle_num', type=int, default=100, help='控制节点顺序随机打乱的步频')
    parser.add_argument('--split_num', type=int, default=1, help='将超大图切分为几个子图进行训练') #1表示不切分，直接全图训练
    parser.add_argument('--cl_update_num', type=int, default=2500, help='课程学习中更新预测步长的频率')
    parser.add_argument('--tanh_alpha', type=float, default=3, help='构图器中 Tanh 激活的缩放系数')
    parser.add_argument('--gd_clip', type=int, default=5, help='梯度裁剪阈值，防梯度爆炸')

    # 保存与监控参数
    parser.add_argument('--save', type=str, default='./Save/Experiments/', help='模型实验结果主保存路径')
    parser.add_argument('--save_pred_result', type=str_to_bool, default=True, help='是否保存预测输出与异常评分残差文件')
    parser.add_argument('--save_adj', type=str_to_bool, default=False, help='是否保存内部生成的动态邻接矩阵')
    parser.add_argument('--wandb', type=str_to_bool, default=False, help='是否启用 wandb 在线实验监控')

    # ===== 特定模型专属参数 (如 MTGNN) =====
    if model_args.model == 'MTGNN':
        parser.add_argument('--mix_hop_depth', type=int, default=2, help='混合跳数图卷积的传播深度')
        parser.add_argument('--prop_alpha', type=float, default=0.05, help='图传播根节点信息保留比例')
        parser.add_argument('--conv_channels', type=int, default=32, help='时空主干卷积通道数')
        parser.add_argument('--residual_channels', type=int, default=32, help='残差连接通道数')
        parser.add_argument('--skip_channels', type=int, default=64, help='跳跃连接通道数')
        parser.add_argument('--end_channels', type=int, default=128, help='末端输出层过渡通道数')
        parser.add_argument('--layer_num', type=int, default=3, help='核心时空模块堆叠的总层数')
        parser.add_argument('--opt_lr', type=float, default=0.001, help='优化器的初始学习率')
        parser.add_argument('--opt_wd', type=float, default=0.0001, help='优化器的权重衰减系数(L2正则)')
        parser.add_argument('--adj_method', type=str, default='embedding', help='生成关系矩阵的基础方法')
        parser.add_argument('--use_ae', type=str_to_bool, default=True, help='是否启用并联合训练时间窗口自编码器；默认开启')
        parser.add_argument('--ae_beta', type=float, default=0.2, help='自编码器重构损失权重')
        parser.add_argument('--ae_lambda', type=float, default=0.5, help='生成窗口重构损失权重')
        parser.add_argument('--ae_channels', type=int, default=32, help='自编码器内部隐藏层通道数')

    return parser.parse_args()

# ==============================================================================
# 4. 可视化绘制工具 (SCI 高清版与防崩溃设计)
# ==============================================================================

def set_sci_style():
    """
    [核心样式库] 全局 SCI 论文排版风格注入器。
    在每次画图前调用此函数，可以全局重置 matplotlib 的默认参数，
    确保生成的图片符合顶级学术期刊的严格排版要求。
    """
    plt.rcParams.update({
        # 1. 字体设置 (防崩溃核心)
        'font.size': 16,                         # 全局基础字号（如果觉得图里的字太小，调大这个数值，比如改为 18 或 20）
        'font.family': 'serif',                  # 强制使用衬线字体大类（学术论文标准）
        # 【字体降级策略】：系统会从左到右挨个寻找字体。
        # 首选 Times New Roman；如果 Linux 服务器上没有，就依次降级使用 SimSun(宋体) -> 系统自带的 DejaVu 等。
        # 这彻底解决了跨平台跑代码报错“找不到字体”的致命问题。
        'font.serif': ['Times New Roman', 'SimSun', 'DejaVu Serif', 'Bitstream Vera Serif', 'Computer Modern Roman'],
        
        # 2. 边框与线条设置
        'axes.linewidth': 1.5,                   # 图表最外围四条边框的粗细（加粗显得更稳重）
        'lines.linewidth': 2.0,                  # 图表内部折线的默认粗细（可被 plt.plot 里的参数覆盖）
        
        # 3. 坐标轴刻度设置 (SCI 规范要求刻度朝内)
        'xtick.direction': 'in',                 # X 轴（横轴）刻度线向图表内侧凸起
        'ytick.direction': 'in',                 # Y 轴（纵轴）刻度线向图表内侧凸起
        'xtick.major.width': 1.5,                # X 轴主刻度线的粗细，与边框粗细保持一致
        'ytick.major.width': 1.5,                # Y 轴主刻度线的粗细
        
        # 4. 高清渲染与保存设置
        'figure.dpi': 300,                       # 屏幕上显示的清晰度（DPI 越高越清晰，但渲染越慢）
        'savefig.dpi': 300,                      # 保存到硬盘的图片清晰度（SCI 期刊通常要求 300-600）
        'savefig.bbox': 'tight',                 # 保存图片时，自动裁掉外围多余的空白区域，只保留紧凑的图表本身
    })


def plt_loss(df: pd.DataFrame, df_valid: pd.DataFrame, save_path: str = None, show: bool = False):
    """
    绘制并对比训练集 (Train) 与验证集 (Validation) 的 Loss 下降曲线。
    用于判断模型是否收敛，以及是否发生了过拟合 (Overfitting)。
    """
    set_sci_style() # 注入 SCI 风格配置
    
    # 创建一个宽 8 英寸、高 6 英寸的画布 (符合 4:3 黄金图幅比例)
    plt.figure(figsize=(8, 6))
    
    # 自动生成 X 轴坐标：从 1 到 Epoch 的总轮数
    x = [i + 1 for i in range(df.shape[0])]

    # ---------------- 绘制训练集曲线 ----------------
    for i in range(df.shape[1]): # 遍历每次独立运行的数据 (Run 0, Run 1...)
        # 画实线表示训练 Loss，强制指定颜色为黑色 ('black')
        plt.plot(x, df.iloc[:, i], color='black', label=f'Train [{i}]')

    # ---------------- 绘制验证集曲线 ----------------
    for i in range(df.shape[1]):
        # ls='--' 代表虚线，强制指定颜色为红色 ('red')，alpha=0.8 稍微带点透明度防死黑
        plt.plot(x, df_valid.iloc[:, i], color='red', ls='--', alpha=0.8, label=f'Valid [{i}]')

    # 设置横纵坐标标签，fontweight='bold' 让字体加粗，增强学术感
    plt.xlabel('Epochs', fontweight='bold')
    plt.ylabel('Loss', fontweight='bold')
    
    # 添加网格线，帮助读取具体数值。linestyle=':' 是点状虚线，alpha=0.6 让网格颜色变淡
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 设置图例 (Legend) 显示
    # 调整比例为 4:3 后，图表内部空间更充裕，将图例放在右上角 (upper right)
    # 开启边框 frameon=True，避免图例文字和网格线重叠看不清
    plt.legend(loc='upper right', frameon=True)

    if save_path is not None:
        plt.savefig(save_path) # 如果提供了路径，则将图片保存到本地
    if show:
        plt.show() # 如果 show=True，则在屏幕上弹窗展示（服务器上通常不开）
        
    # 【防 OOM 神器】清空当前图表对象并释放内存空间。如果服务器批量跑循环不加这句，内存迟早会爆掉
    plt.close()


def plt_box(df: pd.DataFrame, which: str = 'MAPE', save_path: str = None, show: bool = False):
    """
    绘制指定误差指标随预测步长延伸的“箱线图 (Boxplot)”。
    箱体越窄，说明模型在多次重复实验中的稳定性越好（方差小，鲁棒性高）。
    """
    set_sci_style()
    which = which.upper() # 将输入的指标名（如 'mape'）强制转为大写，防止因为大小写写错而找不到数据
    
    # 统一 4:3 图幅比例
    plt.figure(figsize=(8, 6))
    
    # ---------------- 数据清洗与提取 ----------------
    # 1. 过滤出对应指标的行（df['Type'] == which）
    # 2. 切除前两列（Run_Id 和 Type），只保留纯数字误差列（.iloc[:, 2:]）
    # 3. 转置 (.T)，使得每一行对应一个“未来预测步长 (Horizon Step)”，每一列是一次实验结果
    plt_df = df[df['Type'] == which].iloc[:, 2:].T
    
    # X 轴坐标 (1, 2, ..., Horizon Size)
    x = [i + 1 for i in range(plt_df.shape[0])]
    # 计算每一个时间步长上，所有重复实验误差的平均值
    l_mean = plt_df.mean(axis=1)

    # ---------------- 绘制平均值趋势线 ----------------
    # 画一条贯穿全图的平均值红色连线 (zorder=3 让它显示在箱线图的上层，不被遮挡)
    plt.plot(x, l_mean, label='Mean', color='#d62728', linestyle='-', zorder=3)
    # 在折线上打上小红点
    plt.scatter(x, l_mean, color='#d62728', zorder=4)

    # ---------------- 绘制精美的学术箱线图 ----------------
    for i in range(len(plt_df)):
        plt.boxplot(
            plt_df.iloc[i, :], # 喂入当前预测步长的所有独立 Run 数据
            positions=[i + 1], # 放在 X 轴的第 i+1 个刻度上
            widths=0.5,        # 箱子的宽度（太宽会显得臃肿，0.5 刚刚好）
            patch_artist=True, # 必须开启这个，才能给箱子内部填充颜色
            
            # 箱子主体的样式：facecolor 填充色为科技蓝，透明度 0.3；边框也为蓝色
            boxprops=dict(facecolor='#1f77b4', color='#1f77b4', alpha=0.3, linewidth=1.5),
            # 箱子上下限横线的样式（胡须的帽子）
            capprops=dict(color='#1f77b4', linewidth=1.5),
            # 连接箱体和上下限的竖线样式（设为虚线）
            whiskerprops=dict(color='#1f77b4', linewidth=1.5, linestyle='--'),
            # 代表中位数的横线样式（设为亮绿色加粗，并在最高层显示 zorder=5）
            medianprops=dict(color='#2ca02c', linewidth=2, zorder=5) 
        )

    plt.xticks(ticks=x) # 强制 X 轴标出每一个预测步长，不省略
    plt.xlabel('Horizon Step', fontweight='bold')
    plt.ylabel(which, fontweight='bold') # 纵坐标自动标为对应的误差名，如 "RMSE"
    
    # axis='y' 表示只画水平方向的横向网格线，这能帮助对比不同箱子的高度，而忽略垂直杂音
    plt.grid(True, axis='y', linestyle=':', alpha=0.6) 
    
    # 设置图例位置，带有边框增加区分度
    plt.legend(loc='best', frameon=True)

    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def plt_error(df: pd.DataFrame, which: str = 'MAPE', save_path: str = None, show: bool = False):
    """
    绘制指定误差指标随预测步长延伸的“多运行轨迹折线图”。
    可以清晰看到每一单次 Run 的误差是如何随着预测越远而逐渐变大的。
    """
    set_sci_style()
    which = which.upper()
    
    # 统一 4:3 图幅比例
    plt.figure(figsize=(8, 6))

    # 数据提取方式同 plt_box
    plt_df = df[df['Type'] == which].iloc[:, 2:].T
    x = [i + 1 for i in range(plt_df.shape[0])]

    # 引入 matplotlib 自带的优质学术调色板 'tab10' (最多包含 10 种高对比度区分颜色)
    cmap = plt.get_cmap('tab10')

    # 遍历所有重复的实验 Run
    for i in range(plt_df.shape[1]):
        # 画出每个 Run 的趋势线
        # marker='o' 加上圆点标记，markersize=6 设置圆点大小
        # color=cmap(i % 10) 确保即便有多条线，系统也会依次分配完全不同的区分颜色
        plt.plot(x, plt_df.iloc[:, i], marker='o', markersize=6, alpha=0.8, 
                 color=cmap(i % 10), label=f'Run [{i}]')

    plt.xticks(ticks=x)
    plt.xlabel('Horizon Step', fontweight='bold')
    plt.ylabel(which, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    # 图例设置：
    # frameon=True 开启边框
    # loc='best' 让 matplotlib 自动寻找最不会遮挡数据线的位置摆放图例
    ncol_count = 2 if plt_df.shape[1] > 3 else 1
    plt.legend(ncol=ncol_count, loc='best', frameon=True)

    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

# ==========================================
# 5. 文件 IO 与路径处理函数 (第一部分 [首次定义])
# ==========================================
def save_path(*paths):
    """
    [首次定义] 绝对路径生成器：在硬编码的 D 盘 Export 根目录下创建带时间戳的文件夹。
    """
    cur_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root_folder = 'D:/Code/STGNN/Export/'
    path_l = [root_folder] + list(paths)
    path_l[-1] = f'[{cur_time}]_{path_l[-1]}'
    check_folder(os.path.join(*path_l[:-1]))
    return os.path.join(*path_l)


def load_pickle(pickle_file):
    """
    [首次定义] 安全兼容地读取 Pickle 二进制序列化文件，向下兼容旧版 Python2 (latin1编码)。
    """
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_adj(pkl_filename):
    """
    [首次定义] 从固定的 pickle 文件三元组中提取图的邻接矩阵特征。
    """
    _, _, adj = load_pickle(pkl_filename)
    return adj


def load_node_feature(path):
    """
    [首次定义] 从结构化 CSV/文本文件中提取并标准化节点的静态辅助特征。
    """
    if path is not None:
        with open(path) as fi:
            x = []
            for idx, li in enumerate(fi):
                if idx == 0:
                    continue  # 强行跳过首行表头
                li = li.strip().split(',')
                # 跳过前两列标识符，保留数值特征
                e = [float(t) for t in li[2:]]
                x.append(e)
            x = np.array(x)
            # 节点特征标准化 Z-score
            mean = np.mean(x, axis=0)
            std = np.std(x, axis=0)
            z = torch.tensor((x - mean) / std, dtype=torch.float)
        return z
    else:
        return None


def to_pickle(obj: Union[List, Dict], path: str):
    """
    [首次定义] 将各类 Python 对象（列表/字典）持久化压入二进制 pickle 文件。
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def get_save_path(*paths):
    """
    [首次定义] 相对路径生成器：在当前工作区 ./Export/ 目录下创建带时间戳的文件夹。
    """
    cur_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root_folder = './Export/'
    path_l = [root_folder] + list(paths)
    path_l[-1] = f'[{cur_time}]_{path_l[-1]}'
    check_folder(os.path.join(*path_l[:-1]))
    return os.path.join(*path_l)


# ==========================================
# 6. 模型掩码误差与评估管线区 (第一部分 [首次定义/被注释])
# ==========================================
# [历史废弃版本 - 被全部注释的旧版 masked_mse，原样保留]
# def masked_mse(pred_y, true_y, null_val=np.nan):
#     mask = ~torch.isnan(true_y) if np.isnan(null_val) else (true_y != null_val)
#     mask = mask.float()
#     mask /= torch.mean(mask)
#     mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
#     loss = (pred_y - true_y) ** 2
#     loss = loss * mask
#     loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
#     return torch.mean(loss)

def masked_mse(pred_y, true_y, null_val=np.nan):
    """
    [当前激活版] 掩码均方误差 (Masked MSE)。
    在计算误差时，屏蔽掉真实标签中缺失的样本点，防止污染评估指标。
    """
    null_val = float(null_val)  # 保证数值比较格式一致

    mask = ~torch.isnan(true_y) if np.isnan(null_val) else (true_y != null_val)
    mask = mask.float()

    # 如果本批次数据全部被过滤，直接返回安全数值 0
    if torch.sum(mask) == 0:
        return torch.tensor(0.0, device=pred_y.device)

    # 归一化掩码，使得最终累计损失不因掩蔽多寡而偏移
    mask_mean = torch.mean(mask)
    mask = mask / mask_mean if mask_mean > 0 else mask

    loss = (pred_y - true_y) ** 2
    loss = loss * mask

    return torch.mean(loss)


def _build_safe_mask(true_y, pred_y, null_val=np.nan):
    """
    构造掩码指标共用的安全 mask。

    返回 None 表示当前 batch 没有任何有效标签，调用方应返回安全的 0 值。
    """
    null_val = float(null_val)
    mask = ~torch.isnan(true_y) if np.isnan(null_val) else (true_y != null_val)
    mask = mask.float()
    if torch.sum(mask) == 0:
        return None

    mask_mean = torch.mean(mask)
    mask = mask / mask_mean if mask_mean > 0 else mask
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    return mask


def masked_r2(pred_y, true_y, null_val=np.nan):
    """
    [首次定义] 掩码决定系数 (Masked R²)。
    评估模型捕获未缺失数据集目标变量总变异程度的能力。
    """
    mask = _build_safe_mask(true_y, pred_y, null_val)
    if mask is None:
        return torch.tensor(0.0, device=pred_y.device, dtype=pred_y.dtype)

    masked_true_y = torch.where(mask > 0, true_y, torch.tensor(0.0, device=true_y.device))

    ss_res = torch.sum(((masked_true_y - pred_y) ** 2) * mask)
    mean_true_y = torch.sum(masked_true_y * mask) / torch.sum(mask)
    ss_tot = torch.sum(((masked_true_y - mean_true_y) ** 2) * mask)

    ss_tot_value = ss_tot.item()

    if ss_tot_value != 0:
        r2 = 1 - ss_res / ss_tot
    else:
        r2 = torch.tensor(1.0, device=pred_y.device)

    return r2


def masked_rmse(pred_y, true_y, null_val=np.nan):
    """
    [首次定义] 掩码均方根误差 (Masked RMSE)。
    """
    return torch.sqrt(masked_mse(pred_y=pred_y, true_y=true_y, null_val=null_val))


def masked_mae(pred_y, true_y, null_val=np.nan):
    """
    [首次定义] 掩码平均绝对误差 (Masked MAE)。
    """
    mask = _build_safe_mask(true_y, pred_y, null_val)
    if mask is None:
        return torch.tensor(0.0, device=pred_y.device, dtype=pred_y.dtype)

    loss = torch.abs(pred_y - true_y)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(pred_y, true_y, null_val=np.nan):
    """
    [首次定义] 带掩码的稳定版平均绝对百分比误差 (Masked MAPE)。
    利用 (|pred - true|) / (|pred| + |true| + eps) 格式防止分母过零崩溃。
    """
    mask = _build_safe_mask(true_y, pred_y, null_val)
    if mask is None:
        return torch.tensor(0.0, device=pred_y.device, dtype=pred_y.dtype)

    # 稳定分母设计，加 1e-8 偏移量
    loss = torch.abs(pred_y - true_y) / (torch.abs(pred_y) + torch.abs(true_y) + 1e-8)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def calc_error(pred, real, null_val=np.nan):
    """
    [首次定义] 统一计算并返回四大主流时间序列回归评估指标 (掩码版)。
    返回格式: (MAE, MAPE, RMSE, R²) 均安全四舍五入。
    """
    mae = masked_mae(pred, real, null_val).item()
    mape = masked_mape(pred, real, null_val).item()
    rmse = masked_rmse(pred, real, null_val).item()
    r_2 = masked_r2(pred, real, null_val).item()
    return np.round(mae, 4), np.round(mape, 4), np.round(rmse, 4), np.round(r_2, 4)


def set_random_seed(seed: int = None):
    """
    [首次定义] 全局随机种子固化函数，确保 PyTorch 与 Numpy 实验完全可复现。
    """
    if seed is not None:
        print(f'锁定系统全局随机初始化种子: {seed}.')
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)


def get_adj_m(adj_data_path: str, node_num: int, device: torch.device) -> Optional[torch.Tensor]:
    """
    [首次定义] 基于外部拓扑文件准备图卷积邻接矩阵，内部强制扣除对角自环。
    """
    if adj_data_path is not None:
        pd_a = load_adj(adj_data_path)
        pd_a = torch.tensor(pd_a) - torch.eye(node_num)
        pd_a = pd_a.to(device)
        return pd_a
    else:
        return None


def test_model(loader, true_y, scalers, model, device):
    """
    [首次定义] 模型全量测试管线评估引擎。
    返回误差四元组、真实标签阵列、预测输出阵列及动态图推演快照。
    """
    pred_y = []
    # 截取目标预测的主干通道(0号)推至显存
    true_y = torch.Tensor(true_y).transpose(1, 3)[:, 0, :, :].to(device)
    dynamic_graph_history = []

    for i, (x, y) in enumerate(loader.get_iterator()):
        input_x = torch.Tensor(x).transpose(1, 3).to(device)
        with torch.no_grad():
            output_y, _ = model(input_x)

        if output_y.dim() == 4:
            if output_y.size(1) != 1:
                raise ValueError(
                    f"模型输出应为 [B, 1, N, H]，但收到形状: {tuple(output_y.shape)}"
                )
            output_y = output_y.squeeze(1)
        elif output_y.dim() != 3:
            raise ValueError(
                f"模型输出应为 [B, 1, N, H] 或 [B, N, H]，但收到形状: {tuple(output_y.shape)}"
            )
        pred_y.append(output_y)

        if hasattr(model, 'graph_history'):
            dynamic_graph_history.append(model.graph_history)

    pred_y = torch.cat(pred_y, dim=0)[:true_y.size(0), ...]
    if pred_y.shape != true_y.shape:
        raise ValueError(
            f"预测张量与真实张量形状不一致: pred={tuple(pred_y.shape)}, true={tuple(true_y.shape)}"
        )

    # 执行关键逆归一化，反映工业界真实工程物理量纲
    if scalers is not None:
        pred_y = scalers.inverse_transform(pred_y)
        true_y = scalers.inverse_transform(true_y)

    return calc_error(pred_y, true_y), true_y, pred_y, dynamic_graph_history


def save_args_to_json(args, file_path):
    """
    [首次定义] 将 argparse 解析的超参数导出为 JSON 格式备案。
    """
    args_dict = vars(args)
    with open(file_path, 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)


def calc_error_02(pred, real):
    """
    [首次定义] 原始未脱敏（无视 NaN 和 Mask）版硬评估误差算法（常规数据集使用）。
    """
    mae = torch.mean(torch.abs(pred - real)).item()
    epsilon = 1e-10
    mape = torch.mean(torch.abs((real - pred) / (real + epsilon))).item() * 100
    rmse = torch.sqrt(torch.mean((pred - real) ** 2)).item()
    ss_res = torch.sum((real - pred) ** 2).item()
    ss_tot = torch.sum((real - torch.mean(real)) ** 2).item()
    r_2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

    return round(mae, 4), round(mape, 4), round(rmse, 4), round(r_2, 4)

class CustomScaler:
    """
    工业级数据标准化 (Z-score Normalization) 核心处理器。
    核心作用：消除不同物理量纲（如水位米、压力帕斯卡、流量升/秒）之间的数值量级差异，
    确保图神经网络等深度模型在计算特征和梯度时，不会被数值绝对值大的变量强行主导。
    """
    def __init__(self, train_data: np.ndarray, use_one: bool = False):
        # 严格拦截非法输入，确保数据格式符合 [时间步长(样本量), 传感器节点数(特征维度)] 的 2D 矩阵规范。
        if not isinstance(train_data, np.ndarray) or train_data.ndim != 2:
            raise TypeError("train_data 必须是 [样本量, 节点数] 的 2D NumPy 数组!!!")
        self.use_one = use_one
        if self.use_one:
            # 【全局映射模式】
            # 将整个系统所有传感器的所有历史读数混在一起，只计算出一个全局的均值和标准差。
            # 适用场景：所有特征本身属于同一物理量纲（例如全是同一型号的温度传感器），且希望保留它们相互之间的绝对数值落差。
            # 使用 np.nanmean/nanstd 可安全跳过系统记录的缺失值 (NaN)。
            self.mean = np.nanmean(train_data)
            self.std = np.nanstd(train_data)
        else:
            # 【独立通道映射模式】(多变量异常检测的默认且最常用模式)
            # 沿着 axis=0 (时间轴) 垂直计算，为每个传感器（列）单独计算自己的均值和标准差。
            # 适用场景：处理 SWaT 或 WADI 这类复杂系统时，水泵转速和管道压力的量纲完全不同，必须让每个传感器只和自己的历史基准线进行对比。
            self.mean = np.nanmean(train_data, axis=0)
            self.std = np.nanstd(train_data, axis=0)
        # 初始化参数计算完成后，立刻执行除零防爆检查
        self.check_zero_std()
    def check_zero_std(self):
        """
        防除零崩溃安全锁：修正“死寂变量”导致的运行时数学异常。
        物理意义：在连续的工业系统稳态运行中，某些离散设备（如常开阀门、固定设定点的恒温器）
        的读数在整个训练集周期内可能死死盯住一个值不动。这会导致该传感器的方差和标准差绝对为 0。
        如果后续直接执行 Z-score = (X - Mean) / Std，就会触发严重的除以 0 (Divide-by-Zero) 错误，导致张量被 NaN 污染。
        """
        # 找出所有标准差恰好等于 0 的列索引（即死寂传感器的 ID）
        zero_std_cols = np.where(self.std == 0)[0]
        if len(zero_std_cols) > 0:
            print(f"请注意以下列的标准差为零 (系统状态绝对静止):")
            for col in zero_std_cols:
                print(f"节点/通道 [{col}] 的标准差为零, 训练集期间所有采样值完全相同.")
            
            # 【修正逻辑】将这些列的标准差强行从 0 改为 1.0。
            # 这样在进行正向归一化计算时，(X - Mean) / 1.0 会直接等于 0。
            # 完美表达了该节点“毫无波动、彻底处于历史基准均值状态”的物理事实，同时在代码底层阻断了数学崩溃。
            self.std[zero_std_cols] = 1.0
            
    def transform(self, data):
        """前向标准化，兼容 Numpy 与 PyTorch 高维张量。"""
        if isinstance(data, np.ndarray):
            # 解除了原有的 data.ndim != 2 限制，允许对 3D 序列数据 [batch, window, node] 自动广播
            return (data - self.mean) / self.std

        elif isinstance(data, torch.Tensor):
            if data.dim() == 3:
                mean = torch.tensor(self.mean, dtype=data.dtype, device=data.device).view(1, -1, 1)
                std = torch.tensor(self.std, dtype=data.dtype, device=data.device).view(1, -1, 1)
                return (data - mean) / std
            elif data.dim() == 4:
                mean = torch.tensor(self.mean, dtype=data.dtype, device=data.device).view(1, 1, -1, 1)
                std = torch.tensor(self.std, dtype=data.dtype, device=data.device).view(1, 1, -1, 1)
                return (data - mean) / std
            else:
                raise ValueError("不支持的 PyTorch 维度深度。")
        else:
            raise TypeError("数据类型不支持, 仅支持 NumPy 数组和 PyTorch 张量!")

    def inverse_transform(self, data):
        """逆向反推去标准化，供测试泛化使用。"""
        if isinstance(data, np.ndarray):
            # 同样解除维度限制，支持多维数组的反归一化
            return data * self.std + self.mean

        elif isinstance(data, torch.Tensor):
            mean = torch.tensor(self.mean, dtype=data.dtype, device=data.device, requires_grad=False)
            std = torch.tensor(self.std, dtype=data.dtype, device=data.device, requires_grad=False)
            if data.dim() == 3:
                data = data * std.view(1, -1, 1) + mean.view(1, -1, 1)
                return data
            elif data.dim() == 4:
                # 仅对第一个核心特征通道反归一化
                data[:, 0:1, :, :] = data[:, 0:1, :, :] * std.view(1, 1, -1, 1) + mean.view(1, 1, -1, 1)
                return data
            else:
                raise ValueError("PyTorch 张量必须是3D或4D!")
        else:
            raise TypeError("仅支持 NumPy 数组或 PyTorch 张量!")

    def __len__(self):
        return 1 if self.use_one else len(self.mean)


def create_loss_df(run_num, loss_l, train_loss_df):
    """
    [首次定义] 自动延展矩阵辅助函数。
    当提前早停导致数组偏短时，填补 NaN 以对齐。
    """
    add_num = abs(len(loss_l) - train_loss_df.shape[0])
    if len(loss_l) > train_loss_df.shape[0]:
        train_loss_df = pd.concat([
            train_loss_df,
            pd.DataFrame(np.nan, index=range(add_num), columns=train_loss_df.columns)
        ], ignore_index=True)
    else:
        loss_l.extend([np.nan] * add_num)
    train_loss_df[run_num] = loss_l
    return train_loss_df


# ==========================================
# 7. 时序降采样与终端辅助打印区
# ==========================================
def downsample_block(_df: pd.DataFrame, cont_cols, status_cols, sec: int) -> pd.DataFrame:
    """
    对多变量时间序列 DataFrame 执行秒级降采样 (Downsample)。
    根据物理变量类型（连续值 -> mean，状态量 -> last）采取不同聚合策略。
    """
    _df = _df.copy()
    _df = _df.set_index("Timestamp")

    agg_dict = {}
    for c in cont_cols:
        agg_dict[c] = "mean"
    for c in status_cols:
        agg_dict[c] = "last"

    agg_dict["Train"] = "last"
    agg_dict["Attack"] = "max"

    _df = _df.resample(f"{sec}s").agg(agg_dict)
    _df = _df.dropna(how="all")

    _df["Train"] = _df["Train"].ffill().fillna(0).astype(int)
    _df["Attack"] = _df["Attack"].fillna(0).astype(int)

    _df = _df.reset_index()
    return _df


def df_to_pt(df: pd.DataFrame) -> PrettyTable:
    """
    将 Pandas DataFrame 转换为 PrettyTable 对象，用于控制台美观打印。
    """
    table = PrettyTable()
    table.field_names = df.columns.tolist()
    for index, row in df.iterrows():
        table.add_row(row.tolist())
    return table


def check_folder(save_file: str):
    """
    检查路径是否存在，不存在则递归创建。
    """
    if not os.path.exists(save_file):
        os.makedirs(save_file)
        hues.info(f"Successfully Created the folder: ['{save_file}'].")



# [以下为原程序末尾的历史废弃测试函数，原样保留]
# def get_cq_corr(dt_name: str = 'CQ'):
#     """
#     根据给定的数据集名称，加载数据并计算其皮尔逊相关系数矩阵。
#     """
#     hues.info('加载皮尔逊相关性矩阵为关系矩阵.')  
#     df = None
#     if dt_name == 'CQ':
#         df = (
#             (pd
#              .read_pickle('./Data/WDNs.pickle')
#              / 60)
#             .resample('h').sum()
#         )
#     elif dt_name == 'Industry':
#         df = pd.read_pickle('./Data/Industry.pkl').iloc[:, 1:-2]
# 
#     return df.corr().values


# def get_cq_adj(self_loops=False, undirected=False):
#     """
#     基于预定义的边（Edges），手动构建图的初始邻接矩阵。
#     """
#     hues.info('加载邻接矩阵为关系矩阵.') 
#     edges = [(0, 4), (1, 4), (1, 5), (2, 0), (2, 1), (4, 3), (5, 8), (6, 5), (7, 5), (8, 4), (9, 0)]
#     num_nodes = 10 
#     adj_m = np.zeros((num_nodes, num_nodes), dtype=int)  
# 
#     for to_node, from_node in edges:
#         adj_m[from_node][to_node] = 1
# 
#     if undirected:
#         adj_m = adj_m + adj_m.T  
#         adj_m[adj_m > 1] = 1  
# 
#     if self_loops:
#         np.fill_diagonal(adj_m, 1)  
# 
#     return adj_m

   # 以下为硬编码的候选关系矩阵生成方式（已被注释掉）
    # pd_a = torch.tensor(get_cq_adj(undirected=True)).to(device)  
    # pd_a = torch.tensor(get_cq_corr('CQ')).to(device,  
    #                                                 dtype=torch.float32)  
    
    
if __name__ == '__main__':
    ...
