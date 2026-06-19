#!/usr/bin/env python3                                      # 指定脚本由 Python 3 解释器执行
# -*- coding: utf-8 -*-                                     # 指定文件文本编码格式为 UTF-8，防止中文乱码
"""
Compare.py - 从 Compare.ipynb 转换而来                        # 模块说明文档：记录脚本来源
功能：多个模型/配置的性能对比可视化（箱线图）                     # 模块说明文档：记录脚本核心功能
"""

import os                                                   # 导入操作系统接口模块，用于处理文件路径和系统交互
import pandas as pd                                         # 导入 pandas 库并简写为 pd，用于处理和清洗表格数据
import seaborn as sns                                       # 导入 seaborn 库并简写为 sns，基于 matplotlib 的高级统计图表库
import matplotlib.pyplot as plt                             # 导入 matplotlib 的 pyplot 模块，用于基础图形渲染和控制
from matplotlib.gridspec import GridSpec                    # 从 matplotlib 导入 GridSpec，用于对子图进行复杂的尺寸/比例布局控制

# =========================
# 配置参数
# =========================
EXPORT_PATH = ''                                            # 定义数据文件导出的基础路径，当前为空字符串（默认当前工作目录）
DPI = 300                                                   # 定义输出图片的 DPI（分辨率），300 是标准的高清打印要求
IMG_TYPE = '.tif'                                           # 定义图片保存的默认扩展名，.tif 常用于学术论文出版
E_NAME = ['R^2', 'MAE']                                     # 定义需要从数据中提取并可视化的核心评估指标名称列表

# =========================
# 数据路径配置
# =========================
folder_name = [                                             # 定义包含不同模型实验结果的文件夹名称列表（通常为时间戳命名）
    '20260120_163808',                                      # 第一个模型的实验结果文件夹
    '20260120_185839',                                      # 第二个模型的实验结果文件夹
]
# 使用列表推导式，将基础目录与上面定义的各文件夹名称拼接，生成包含目标 .pkl 文件的绝对路径列表
abs_folder = [os.path.join(r'E:\DataForCode\2_STGNNOutput', f_name, 'Result') for f_name in folder_name]

legend_l = [                                                # 定义图例标签列表，与上面的文件夹列表一一对应，代表不同模型的配置
    '[48_12_20]',                                           # 对应第一个模型的图例名称
    '[48_12_10]',                                           # 对应第二个模型的图例名称
]

# =========================
# 加载和合并数据
# =========================
plt_df = pd.DataFrame()                                     # 初始化一个空的 Pandas DataFrame，用于将所有模型的数据拼接到一起

for i, name in enumerate(abs_folder):                       # 遍历绝对路径列表，i 为索引，name 为不带后缀的文件路径
    file_path = EXPORT_PATH + f'{name}.pkl'                 # 拼接基础路径和 .pkl 后缀，得到完整的数据文件路径
    _ = pd.read_pickle(file_path)                           # 读取该 .pkl 文件，将解析出的 DataFrame 临时存入变量 _

    print(file_path)                                        # 在控制台打印当前加载的文件路径，方便调试进度

    # 筛选指定的评估指标
    cur_df = _[_['Type'].isin(E_NAME)].copy()               # 仅保留列 'Type' 匹配 E_NAME（'R^2' 或 'MAE'）的数据行，并进行深拷贝
    cur_df['Model'] = legend_l[i]                           # 为当前提取得出的子表新增一列 'Model'，赋值为对应的图例名称
    plt_df = pd.concat([plt_df, cur_df])                    # 将当前模型处理好的数据拼接到全局汇总表 plt_df 底部

plt_df = plt_df.reset_index(drop=True)                      # 重置汇总表的行索引，并且丢弃原本混乱的旧索引
print(plt_df.head())                                        # 在控制台打印汇总表的前 5 行，检查合并是否正确

# =========================
# 数据转换为长格式（用于 seaborn 绘图）
# =========================
# pd.melt 会将数据的"宽表"形式（列名是具体预测步长）拉平为"长表"形式。
# 保留 'Run_Id', 'Model', 'Type' 不动，将其余原列名（预测步长）放在 'Horizon Step' 列，将原单元格的数值放在 'Value' 列
plt_df_long = plt_df.melt(id_vars=['Run_Id', 'Model', 'Type'], var_name='Horizon Step', value_name='Value')
print(plt_df_long.head())                                   # 打印转换后的长表前 5 行
print(plt_df_long)                                          # 打印完整的长表信息

# =========================
# 绘图配置和绘制
# =========================
plt.rcParams.update({                                       # 全局更新 matplotlib 的默认绘图参数字典
    'font.size': 25,                                        # 设置全局默认字体大小为 25
    'font.family': ['Times New Roman', 'SimSun'],           # 优先使用 Times New Roman（英文），中文后备宋体 (SimSun)
    'mathtext.fontset': 'custom',                           # 告知 matplotlib 使用自定义的数学公式字体设置
    'mathtext.rm': 'Times New Roman',                       # 数学公式中正常字体（Roman）使用 Times New Roman
    'mathtext.it': 'Times New Roman:italic',                # 数学公式中斜体部分使用 Times New Roman 的斜体
    'mathtext.bf': 'Times New Roman:bold',                  # 数学公式中粗体部分使用 Times New Roman 的粗体
})

# 子图布局
COL_NUM, ROW_NUM = 2, len(E_NAME)                           # 根据指标数量设定：行数 = 指标数(2)，列数 = 2（左边分布步长，右边汇总）
fig = plt.figure(figsize=(20, 5 * len(E_NAME)))             # 创建画布对象，固定宽度 20，高度根据指标数量动态计算（每个指标高5）
# 创建 GridSpec 对象，控制列宽
gs = GridSpec(ROW_NUM, COL_NUM, width_ratios=[0.9, 0.1])    # 控制布局网格，2列的宽度比例分别为 90% (展示所有步长) 和 10% (展示总体平均)

# 创建子图
ax_arr = []                                                 # 初始化一个列表，用于存放所有的子图句柄(Axes)
for i in range(ROW_NUM):                                    # 遍历每一行（针对每一个评估指标）
    # 第一列子图
    ax1 = fig.add_subplot(gs[i, 0])                         # 在 gs 网格的第 i 行、第 0 列位置添加第一个（宽）子图
    # 第二列子图，共享第一列的 y 轴
    ax2 = fig.add_subplot(gs[i, 1])                         # 在 gs 网格的第 i 行、第 1 列位置添加第二个（窄）子图
    ax_arr.append([ax1, ax2])                               # 将这一行的两个子图实例存入数组，方便后续调用

fig.subplots_adjust(hspace=0.1, wspace=0.045)               # 统一调整子图间距：行间距(hspace)为0.1，列间距(wspace)为0.045，非常紧凑

palette = {                                                 # 定义 Seaborn 绘图使用的调色板（不同模型对应的颜色代码）
    'None': '#f20c00',                                      # 'None' 模型使用红色 (这里注释写蓝色可能有误)
    'Week': '#ffa631',                                      # 'Week' 模型使用橙色
    'Hour': '#0eb83a',                                      # 'Hour' 模型使用绿色
    'Hour(sin-cos)': '#177cb0',                             # 'Hour(sin-cos)' 模型使用蓝色 (这里注释写红色可能有误)
    'Hour(sin-cos, Att)': '#815476'                         # 混合模型使用紫红色
}

# 左侧子图：按预测步长绘制箱线图
for i, name in enumerate(E_NAME):                           # 遍历评估指标（行）
    cur_plt = sns.boxplot(                                  # 调用 seaborn 绘制箱线图
        data=plt_df_long[plt_df_long['Type'] == name],      # 提供绘图数据：仅筛选出当前遍历到的指标(name)的行
        x='Horizon Step',                                   # X轴表示不同的预测步长
        y='Value',                                          # Y轴表示误差数值
        hue='Model',                                        # 颜色维度（hue）通过 'Model' 区分，使得同一个步长下并列多个箱子
        width=1,                                            # 箱体的相对宽度设为 1
        ax=ax_arr[i][0]                                     # 将图绑定绘制在对应的左侧子图 (第 0 列) 上
    )

    y_label = f'${name}$' if '^' in name else name          # 处理 Y 轴标签格式，如果包含上标(^)，则利用 LaTeX 语法渲染（如 $R^2$）
    cur_plt.set_ylabel(y_label)                             # 将处理好的字符串设为当前子图的 Y 轴标签

    # 删除子图的图例
    cur_plt.get_legend().remove()                           # 移除 Seaborn 自动在子图内生成的冗余图例（后续统一画在底部）

    # 获取x轴刻度位置
    xticks = cur_plt.get_xticks()                           # 获取当前子图 X 轴所有刻度点的位置列表

    # 在x轴的每个刻度之间添加竖直线
    for tick in xticks[:-1]:                                # 遍历除了最后一个以外的每个刻度
        cur_plt.axvline(x=tick + 0.5, color='#ccc', linestyle='--', linewidth=0.5) # 在每两个步长之间画一条浅灰色('--')垂直参考线，方便对齐

# 右侧子图：按模型绘制箱线图（合并所有预测步长）
for i, name in enumerate(E_NAME):                           # 再次遍历评估指标，这次画右侧汇总分布
    cur_plt = sns.boxplot(                                  # 同样使用 boxplot 画箱线图
        # 丢掉 'Horizon Step' 列，意味着不区分步长，将同一个模型所有步长的数值扔在一起看总体误差分布
        data=plt_df_long[plt_df_long['Type'] == name].drop('Horizon Step', axis=1), 
        x='Model',                                          # X轴直接代表不同模型
        y='Value',                                          # Y轴代表误差数值
        hue='Model',                                        # 按模型着色
        width=.8,                                           # 箱体宽度设为 0.8
        palette=palette,                                    # 强制指定预设好的配色方案字典
        ax=ax_arr[i][1]                                     # 绑定在右侧对应的子图上
    )

    cur_plt.set_ylabel(None)                                # 因为与左图共享Y轴含义，此处清空 Y 轴标题

    # 获取x轴刻度位置
    xticks = cur_plt.get_xticks()                           # 提取 X 轴模型对应的刻度点位置

    cur_plt.set_yticklabels([])                             # 隐藏右图 Y 轴上的刻度数值标签，保持画面干净

    # 在x轴的每个刻度之间添加竖直线
    for tick in xticks[:-1]:                                # 遍历除最后刻度以外的所有 X 刻度
        cur_plt.axvline(x=tick + 0.5, color='#ccc', linestyle='--', linewidth=0.5) # 在各个模型箱体之间也画一条分隔虚线

# 统一设置子图格式
for i in range(ROW_NUM):                                    # 遍历每一行
    for j in range(COL_NUM):                                # 遍历每一列
        ax_arr[i][j].grid(False)                            # 关闭所有子图内部默认的横竖网格线（背景变白）
        if not (i == 1 and j == 0):                         # 如果不是针对左下角的那张主图（i=1, j=0）
            ax_arr[i][j].set_xticklabels([])                # 就将其 X 轴上的文字刻度隐藏（防止上下图 X 轴重复标注）
            ax_arr[i][j].set_xlabel(None)                   # 就将其 X 轴名称（如"Horizon Step"）清空

ax_arr[1][0].set_ylabel('MAE (m)')                          # 特殊处理左下角的 Y 轴，将其显式标注为带物理单位的 'MAE (m)'

# 三个子图的图例相同，获取最后一个子图的图例
lines, labels = fig.axes[-2].get_legend_handles_labels()    # 提取整个画布倒数第二个子图（即左下角子图）的线条对象和图例标签
fig.legend(lines, labels, ncol=5, loc='lower center',       # 将提取到的图例统一部署在整个画布(Figure)层面。分 5 列横向排列，放在靠下方中心
           bbox_to_anchor=(0.5, -.1))                       # 使用 bbox_to_anchor 进行微调，将其推到图表区域底边之下 (y=-0.1)

# 显示或保存图片
plt.tight_layout()                                          # 让系统自动计算并消除所有子图/标签间的遮挡与重叠，使布局更紧密
plt.show()                                                  # 激活图像窗口进行展示（如果在 Jupyter 中则是直接输出图像）