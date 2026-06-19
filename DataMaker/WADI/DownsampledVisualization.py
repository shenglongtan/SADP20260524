#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DownsampledVisualization.py - 从 DownsampledVisualization.ipynb 转换而来
功能：WADI 数据集多尺度下采样的攻击事件可视化（断轴展示）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from datetime import timedelta

from tools import downsample_block, ATTACK_EVENTS_WADI

ATTACK_EVENTS = ATTACK_EVENTS_WADI

# =========================
# 1) 加载数据
# =========================
df = pd.read_pickle('../../Data/WADI/data.pkl')

hues_info_msg = f"Data shape: {df.shape}"
print(hues_info_msg)
print(df.head())

# =========================
# 2) 列类型划分：特征列 / 标签列
# =========================
label_cols = ["Timestamp", "Train", "Attack"]

# 特征列：除 Timestamp/Train/Attack 外的所有列
feature_cols = [c for c in df.columns if c not in label_cols]

# 状态列（离散/开关/阀门等）：WADI 通常以 STATUS 命名
status_cols = [c for c in feature_cols if "STATUS" in c.upper()]

# 连续列：其余都按连续量处理（PV、流量、压力等）
cont_cols = [c for c in feature_cols if c not in status_cols]

# =========================
# 3) 切分测试集
# =========================
df_test = df[df["Train"] == 0].copy()

print(f"Test raw: {df_test.shape}")

# 只保留必要列（仍保留 Timestamp/Train/Attack，便于下采样聚合）
df_test = df_test[["Timestamp", "Train", "Attack"] + feature_cols]

# =========================
# 4) 下采样尺度列表和颜色配置
# =========================
DOWNSAMPLE_LIST = [10, 30, 60, 120]

attack_colors = ['#21a675', '#4a4266', '#4b5cc4', '#f2be45', '#ed5736', '#392f41', '#ff8936', '#30dff3', '#8c4356',
                 '#b35c44', '#a98175', '#fff143', '#9b4400', '#00e500', '#f00056', '#e29c45', '#f20c00']

# =========================
# 5) 为不同下采样尺度准备数据
# =========================
df_test_dict = {}  # key: downsample_sec, value: df_test

# 原始（不下采样）
df_test_dict["orig"] = df[df["Train"] == 0].copy()

# 不同下采样尺度
for sec in DOWNSAMPLE_LIST:
    df_test_dict[sec] = downsample_block(
        df[df["Train"] == 0].copy(),
        cont_cols,
        status_cols,
        sec
    )

# =========================
# 6) Matplotlib 全局字体设置
# =========================
plt.rcParams.update({
    "font.size": 18,
    "font.family": ["Times New Roman", "SimSun"],
    # ===== 数学公式（mathtext） =====
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',  # 正体
    'mathtext.it': 'Times New Roman:italic',  # 斜体
    'mathtext.bf': 'Times New Roman:bold',  # 粗体
})

# =========================
# 7) 断轴（压缩无攻击空窗）参数
# =========================
PAD_BEFORE = timedelta(minutes=10)  # 每个攻击段前额外显示的正常区间
PAD_AFTER = timedelta(minutes=10)  # 每个攻击段后额外显示的正常区间
MERGE_GAP = timedelta(minutes=5)  # 相邻显示段之间小于该间隔则合并，避免太碎

# =========================
# 8) 生成要显示的时间段 segments（并合并重叠/相邻段）
# =========================
segments = []
for (start, end, _) in ATTACK_EVENTS:
    segments.append((start - PAD_BEFORE, end + PAD_AFTER))

segments = sorted(segments, key=lambda x: x[0])

merged = []
for s, e in segments:
    if not merged:
        merged.append([s, e])
    else:
        ps, pe = merged[-1]
        if s <= pe + MERGE_GAP:
            merged[-1][1] = max(pe, e)
        else:
            merged.append([s, e])

segments = [(s, e) for s, e in merged]

# =========================
# 9) y 轴位置映射
# =========================
y_levels = {"orig": 0}
for i, sec in enumerate(DOWNSAMPLE_LIST, 1):
    y_levels[sec] = i

yticks = list(y_levels.values())
ylabels = ["1"] + [f"{sec}" for sec in DOWNSAMPLE_LIST]

# =========================
# 10) 建图：每个 segment 一个子图（共享 y 轴）=> 实现断轴效果
# =========================
nseg = len(segments)
fig, axes = plt.subplots(
    1, nseg,
    figsize=(18, 2.5 + 0.8 * (len(DOWNSAMPLE_LIST) + 1)),
    sharey=True,
    gridspec_kw={"wspace": 0.05}
)

if nseg == 1:
    axes = [axes]

# =========================
# 11) 绘制：逐事件、逐尺度、逐 segment 绘点
# =========================
legend_handles = []
legend_labels = []
for ax, (seg_start, seg_end) in zip(axes, segments):
    for attack_id, (start, end, _) in enumerate(ATTACK_EVENTS, 1):

        # 若该攻击事件与当前 segment 无交集，跳过
        if end < seg_start or start > seg_end:
            continue

        for key, y in y_levels.items():
            _df = df_test_dict[key]

            mask = (
                    (_df["Attack"] == 1) &
                    (_df["Timestamp"] >= max(start, seg_start)) &
                    (_df["Timestamp"] <= min(end, seg_end))
            )

            times = _df.loc[mask, "Timestamp"]
            if len(times) == 0:
                continue

            ax.scatter(
                times,
                np.full(len(times), y),
                s=18 if key != "orig" else 6,
                color=attack_colors[attack_id],
                alpha=0.8,
            )

        # 添加图例句柄（只添加一次）
        if attack_id <= len(legend_labels) // len(DOWNSAMPLE_LIST):
            continue
        legend_handles.append(
            Line2D(
                [0], [0],
                marker='o',
                linestyle='None',
                markersize=7,
                markerfacecolor=attack_colors[attack_id],
                markeredgecolor=attack_colors[attack_id],
            )
        )
        legend_labels.append(f"Attack {attack_id:02d}")

    # 设置子图 x 轴范围
    ax.set_xlim(seg_start, seg_end)

    # 控制每个子图的 x 轴时间刻度
    locator = mdates.AutoDateLocator(minticks=2, maxticks=5)
    formatter = mdates.ConciseDateFormatter(locator)

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    # 获取当前刻度与标签
    ticks = ax.get_xticks()
    labels = ax.get_xticklabels()

    # 隐藏边界重复的刻度标签
    if segments.index((seg_start, seg_end)) == 0:
        # 第一个子图：隐藏最右侧刻度
        if len(labels) > 0:
            labels[-1].set_visible(False)
    elif segments.index((seg_start, seg_end)) == len(axes) - 1:
        # 最后一个子图：隐藏最左侧刻度
        if len(labels) > 0:
            labels[0].set_visible(False)
    else:
        # 中间子图：隐藏首尾刻度
        if len(labels) > 1:
            labels[0].set_visible(False)
            labels[-1].set_visible(False)

    ax.grid(axis="x", linestyle="--", alpha=0.3)

# =========================
# 12) y 轴标签只在最左侧显示
# =========================
axes[0].set_yticks(yticks)
axes[0].set_yticklabels(ylabels)
axes[0].set_ylabel('Sampling Frequency (sec)')

for ax in axes[1:]:
    ax.tick_params(axis="y", left=False, labelleft=False)

# =========================
# 13) 断轴标记：相邻子图之间画 "//"
# =========================
d = 0.012
for i in range(nseg - 1):
    ax1, ax2 = axes[i], axes[i + 1]

    # 右侧子图边界（ax1 的右边）
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, linewidth=1)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    # 左侧子图边界（ax2 的左边）
    kwargs = dict(transform=ax2.transAxes, color='k', clip_on=False, linewidth=1)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)

# =========================
# 14) 添加统一的图例
# =========================
# 重新生成图例（避免重复）
unique_handles = []
unique_labels = []
for attack_id, (start, end, _) in enumerate(ATTACK_EVENTS, 1):
    unique_handles.append(
        Line2D(
            [0], [0],
            marker='o',
            linestyle='None',
            markersize=7,
            markerfacecolor=attack_colors[attack_id],
            markeredgecolor=attack_colors[attack_id],
        )
    )
    unique_labels.append(f"Attack {attack_id:02d}")

fig.legend(
    unique_handles,
    unique_labels,
    loc="lower center",
    bbox_to_anchor=(.5, -.14),  # 越接近 0 越靠下
    ncol=8,  # 16 个事件：8 列比较舒服
    handletextpad=0.6,
    columnspacing=1.2
)

plt.tight_layout()
plt.show()
