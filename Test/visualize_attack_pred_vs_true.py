#!/usr/bin/env python
# coding: utf-8

# # Visualize Attack-Time Prediction vs Ground Truth
# 
# This notebook visualizes predicted vs true node time series around each attack event.
# 
# What it does:
# - Loads `y_true.npy`, `y_pred.npy`, and `test.npz`
# - Reconstructs time-aligned node series from window predictions
# - Selects the most affected nodes around each attack event
# - Plots attack context and `true vs pred` curves
# - Displays figures directly in Jupyter and shows a CSV summary
# 
# Edit the config cell below and re-run the notebook whenever you want to compare another run.

# In[247]:


# ==========================================
# 模块一：环境依赖与工程寻址
# 功能：导入必要的数据处理与绘图库，并动态定位工程根目录以引入自定义工具包
# ==========================================
import json  # [语法] 导入 json 模块。[功能] 用于读取 Parameters.json
import pickle  # [语法] 导入 pickle 模块。[功能] 用于加载包含特征列名的 meta.pkl 文件
import sys  # [语法] 导入 sys 模块。[功能] 用于动态修改 Python 的模块搜索路径
from pathlib import Path  # [语法] 导入 Path。[功能] 提供面向对象的跨平台路径操作

import numpy as np  # [语法] 导入 NumPy。[功能] 处理时序预测的高维张量数据
import pandas as pd  # [语法] 导入 Pandas。[功能] 负责时间序列的对齐、拼接与缺失值处理
import matplotlib.pyplot as plt  # [语法] 导入 pyplot。[功能] 提供核心的可视化绘图接口
from IPython.display import display  # [语法] 导入 display。[功能] 在 Jupyter 环境中优雅地渲染 DataFrame 表格

# Make sure project root (contains tools.py) is importable no matter where notebook starts.
def _find_project_root(start: Path) -> Path:
    # [功能] 动态根目录寻址：确保无论 Jupyter Notebook 在哪个层级启动，都能正确找到 tools.py 所在的主目录
    cur = start.resolve()  # [语法] 获取当前路径的绝对路径
    for p in [cur, *cur.parents]:  # [语法] 列表解包操作。[功能] 从当前目录开始，逐级向上遍历所有父目录
        if (p / 'tools.py').exists():  # [语法] 路径拼接与存在性检查。[功能] 判定是否到达工程根目录
            return p
    raise FileNotFoundError('Cannot find project root containing tools.py')  # [语法] 抛出异常中断执行

project_root = _find_project_root(Path.cwd())  # [功能] 获取当前运行环境的工作目录并向上寻址
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))  # [语法] 修改 sys.path。[功能] 将工程根目录强行置于环境变量首位，确保能直接 import tools

from tools import ATTACK_EVENTS_SWAT, ATTACK_EVENTS_WADI  # [功能] 显式导入不同数据集的已知攻击事件区间列表

plt.rcParams.update({
    'font.size': 14,
    'font.family': ['Times New Roman', 'SimSun']
})  # [语法] 更新 matplotlib 全局配置。[功能] 统一设定图表的基础字体及大小，兼容中英文显示

print(f'Using project_root: {project_root}')


# In[248]:


# ==========================================
# 模块二：全局超参数配置
# 功能：定义可视化分析的运行目录、目标事件、前后观察时间延展窗口及绘图色彩样式
# ==========================================
# =========================
# Config - edit these values and rerun
# =========================
RUN_DIR = Path(r'E:\DataForCode\5_AnomalyDetection\WADI_STGNN_Output\20260426_160928\0')  # [功能] 指定要分析的模型预测结果存放目录
DATA_PATH = None  # set to a dataset folder if you want to override Parameters.json  # [功能] 数据集路径，若为 None 则自动从 json 读取
EVENT_ID = 16   # None means all attack events; use 1..16 to focus on one event  # [功能] 指定要绘制的攻击事件编号
PRE_MINUTES = 10  # [功能] 绘图时间轴向攻击前延伸的观察时间（分钟）
POST_MINUTES = 10  # [功能] 绘图时间轴向攻击后延伸的观察时间（分钟）
TOPK = 3  # [功能] 在子图中，除了受攻击节点，额外展示受波及最严重的 Top K 个节点
PV_ONLY = True  # [功能] 业务过滤：是否仅分析 Process Variable (过程变量)
COLOR_LIST = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # olive
    '#17becf',  # cyan
]  # [功能] 自定义图表的离散色彩调色板，保证各变量折线区分度高
SAVE_DIR = './Export/'  # None -> RUN_DIR / 'attack_viz'  # [功能] 导出高清图表的本地磁盘路径

print('RUN_DIR =', RUN_DIR)
print('EVENT_ID =', EVENT_ID)
print('PRE/POST =', PRE_MINUTES, POST_MINUTES)
print('TOPK =', TOPK)


# In[249]:


# ==========================================
# 模块三：数据 I/O 与特征读取
# 功能：解析运行目录中的配置文件，提取并对齐真实的传感器物理标签
# ==========================================
def find_params_json(run_dir: Path) -> Path:
    # [功能] 寻找配置文件：兼容当前目录及其父级目录
    cands = [run_dir / 'Parameters.json', run_dir.parent / 'Parameters.json']
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError(f'Cannot find Parameters.json near run dir: {run_dir}')


def resolve_data_path(run_dir: Path, data_path_arg):
    # [功能] 解析数据路径：优先使用显式传入的路径，否则从 json 配置文件中提取
    if data_path_arg:
        p = Path(data_path_arg)
        if not p.exists():
            raise FileNotFoundError(f'data path not found: {p}')
        return p

    params_path = find_params_json(run_dir)
    with params_path.open('r', encoding='utf-8') as f:  # [语法] with 语句自动释放文件句柄
        params = json.load(f)

    p = Path(params['data_path'])
    if not p.exists():
        raise FileNotFoundError(f'data path from Parameters.json not found: {p}')
    return p


def select_attack_events(data_path: Path, run_dir: Path):
    # [功能] 根据数据路径或实验路径显式选择 SWaT/WADI 攻击事件表，避免模糊的 ATTACK_EVENTS 默认别名
    text = f"{data_path} {run_dir}".lower()
    if "wadi" in text:
        return ATTACK_EVENTS_WADI, "WADI"
    if "swat" in text:
        return ATTACK_EVENTS_SWAT, "SWaT"
    raise ValueError(
        "Cannot infer dataset for attack events. "
        "Please set DATA_PATH/RUN_DIR containing 'SWaT' or 'WADI', or edit select_attack_events()."
    )


def load_feature_names(data_path: Path, node_num: int) -> list[str]:
    # [功能] 从数据集元信息提取真实传感器名称，无则返回占位符
    for meta_name in ['data_meta.pkl', 'meta.pkl']:
        meta_path = data_path / meta_name
        if meta_path.exists():
            with meta_path.open('rb') as f:  # [语法] rb 表示二进制读取
                meta = pickle.load(f)
            feat_cols = meta.get('feature_cols', None)  # [语法] 字典安全取值
            if isinstance(feat_cols, list) and len(feat_cols) == node_num:
                return feat_cols
    return [f'node_{i}' for i in range(node_num)]  # [语法] 列表推导式生成兜底名称


# ==========================================
# 模块四：时序降维与连续重构
# 功能：将 STGNN 模型的 3D 滑窗预测张量 [S, N, H] 降维拼接为一维连续的 DataFrame
# ==========================================
def aggregate_windows_to_timeseries(arr: np.ndarray, y_time: np.ndarray, columns: list[str], first_step_only: bool = True) -> pd.DataFrame:
    # arr shape: [samples, nodes, horizon]  # [功能] 模型预测值 3D 张量
    # y_time shape: [samples, horizon]  # [功能] 时间戳矩阵
    samples, nodes, horizon = arr.shape
    if y_time.shape != (samples, horizon):
        raise ValueError(f'y_time shape mismatch: expected {(samples, horizon)}, got {y_time.shape}')

    out = {}

    if first_step_only:
        # Keep only the first prediction step (h=0) for each sample and stitch by its own timestamp.
        # [功能] 策略 A：仅抽取每个预测窗口的第一步（h=0）进行拼接，防止重叠导致的模糊
        ts_index = pd.to_datetime(y_time[:, 0])  # [语法] 提取第0步时间转为 DatetimeIndex
        for n, col in enumerate(columns):
            vals = arr[:, n, 0]  # [语法] 切片提取
            # [语法] 按时间戳分组求均值去重并排序
            s = pd.Series(vals, index=ts_index).groupby(level=0).mean().sort_index()
            out[col] = s
    else:
        # Fallback: merge all horizon steps and average duplicated timestamps.
        # [功能] 策略 B：展平整个视界，对所有重叠点暴力求均值
        ts_index = pd.to_datetime(y_time.reshape(-1))  # [语法] reshape(-1) 展平为 1D
        for n, col in enumerate(columns):
            vals = arr[:, n, :].reshape(-1)
            s = pd.Series(vals, index=ts_index).groupby(level=0).mean().sort_index()
            out[col] = s

    df = pd.DataFrame(out)
    df.index.name = 'Timestamp'  # [功能] 命名索引
    return df


# ==========================================
# 模块五：受波及节点效应评估
# 功能：对比攻击前与攻击期间的分布，计算偏移 Z-score 寻找受影响最深的根因特征
# ==========================================
def calc_event_effect(true_df: pd.DataFrame, st: pd.Timestamp, ed: pd.Timestamp, pre: pd.Timedelta, candidate_cols: list[str]) -> pd.Series:
    # [功能] 截取攻击前基准期与攻击发生期的数据进行对比
    baseline = true_df.loc[max(st - pre, true_df.index.min()): st, candidate_cols]  # [语法] loc 切片
    attack = true_df.loc[st:ed, candidate_cols]

    base_mean = baseline.mean(axis=0)  # [功能] 计算基准期均值
    base_std = baseline.std(axis=0).replace(0, np.nan)  # [语法] 防止标准差为 0 导致除零异常
    atk_mean = attack.mean(axis=0)  # [功能] 计算攻击期均值

    # [功能] 计算标准化绝对偏移量 (Z-score 思想)，并按受影响严重程度降序排列
    effect = ((atk_mean - base_mean).abs() / base_std).dropna().sort_values(ascending=False)
    return effect


def build_output_stem(event_idx_1based: int) -> str:
    # [功能] 生成标准化的导出文件名前缀
    event_part = f'event{event_idx_1based:02d}' if EVENT_ID is not None else 'event_all'
    pv_part = 'pvonly' if PV_ONLY else 'allnodes'
    return f'{event_part}_pre{PRE_MINUTES}m_post{POST_MINUTES}m_top{TOPK}_{pv_part}'


# ==========================================
# 模块六：可视化绘图引擎
# 功能：构建带灰度底纹的上下双联对比折线图，展示受攻击节点与衍生波及节点
# ==========================================
def plot_event(event_idx_1based: int, st: pd.Timestamp, ed: pd.Timestamp, attack_cols: list[str], true_df: pd.DataFrame, pred_df: pd.DataFrame, top_nodes: list[str], pre: pd.Timedelta, post: pd.Timedelta):
    win_true = true_df.loc[st - pre:ed + post].copy()  # [语法] 拷贝操作
    win_pred = pred_df.loc[st - pre:ed + post].copy()

    # [语法] 创建上下双联图表，共享横轴，设置高度比 1:2
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 10), sharex=True,
        gridspec_kw={'height_ratios': [1, 2]}
    )

    custom_palette = [c for c in COLOR_LIST if isinstance(c, str) and c.strip()]
    # [功能] 确定节点调色板，防止不同子图中同一个传感器的折线颜色不一致
    palette = custom_palette if len(custom_palette) > 0 else plt.rcParams['axes.prop_cycle'].by_key().get('color', list(plt.cm.tab20.colors))
    plot_nodes = [c for c in dict.fromkeys([*attack_cols, *top_nodes]) if c in win_true.columns and c in win_pred.columns]
    color_map = {node: palette[i % len(palette)] for i, node in enumerate(plot_nodes)}

    attack_cols_unique = [c for c in dict.fromkeys(attack_cols) if c in win_true.columns and c in win_pred.columns]
    for c in attack_cols_unique:
        # [功能] 子图 1 (ax1)：绘制直接遭受攻击的节点的真实/预测对比曲线
        color = color_map.get(c)
        y_true = pd.to_numeric(win_true[c], errors='coerce')  # [语法] 强转数值，错误转 NaN
        y_pred = pd.to_numeric(win_pred[c], errors='coerce')
        ax1.plot(win_true.index, y_true.values, label=f'{c} | true', color=color, linestyle='-')  # [功能] 真实值实线
        ax1.plot(win_pred.index, y_pred.values, label=f'{c} | pred', color=color, linestyle='--')  # [功能] 预测值虚线

    ax1.axvspan(st, ed, alpha=0.15, color='gray', label='Attack interval')  # [功能] 绘制半透明的灰色攻击时间底纹
    ax1.set_title('Attack point(s) context')
    ax1.set_ylabel('Status / Value')
    ax1.grid(True, alpha=0.3)

    rows = []
    for c in top_nodes:
        # [功能] 子图 2 (ax2)：绘制受波及最严重 (TopK) 的节点的对比曲线
        if c not in win_true.columns or c not in win_pred.columns:
            continue
        color = color_map.get(c)
        y_true = pd.to_numeric(win_true[c], errors='coerce')
        y_pred = pd.to_numeric(win_pred[c], errors='coerce')

        ax2.plot(win_true.index, y_true.values, label=f'{c} | true', color=color, linestyle='-')
        ax2.plot(win_pred.index, y_pred.values, label=f'{c} | pred', color=color, linestyle='--')

        atk_true = true_df.loc[st:ed, c]
        atk_pred = pred_df.loc[st:ed, c]
        err = (atk_pred - atk_true).astype(float)
        rows.append({
            'event_id': event_idx_1based,
            'node': c,
            'attack_mae': float(np.nanmean(np.abs(err.values))),  # [语法] 计算该变量误差的 MAE (忽略NaN)
            'attack_rmse': float(np.sqrt(np.nanmean(err.values ** 2))),  # [功能] 计算 RMSE
        })

    ax2.axvspan(st, ed, alpha=0.15, color='gray', label='Attack interval')  # [功能] 下方子图同样绘制灰色底纹
    ax2.set_title(f'Prediction vs Ground Truth on affected nodes (Top {len(top_nodes)})')
    ax2.set_ylabel('Value')
    ax2.set_xlabel('Time')
    ax2.grid(True, alpha=0.3)

    # [功能] 清理图例重复项并排版，统一放置在图表底部外侧
    handles, labels = [], []
    seen_labels = set()
    for ax in (ax1, ax2):
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label in seen_labels:
                continue
            seen_labels.add(label)
            handles.append(handle)
            labels.append(label)

    if handles:
        fig.legend(
            handles,
            labels,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.04),  # [语法] 图例外置参数
            ncol=min(3, max(1, len(labels))),
            frameon=True,
        )

    fig.tight_layout(rect=(0, 0.16, 1, 1))  # [语法] 压缩底部区域，为外部图例腾出空间

    save_dir = Path(SAVE_DIR) if SAVE_DIR is not None else (run_dir / 'attack_viz')
    save_dir.mkdir(parents=True, exist_ok=True)  # [语法] 保证输出目录存在
    stem = build_output_stem(event_idx_1based)
    for fmt in ['png', 'tif']:  # [功能] 循环导出 PNG 预览图和 TIF 出版级无损图
        out_path = save_dir / f'{stem}.{fmt}'
        fig.savefig(out_path, dpi=300, bbox_inches='tight')  # [语法] dpi=300, 切掉白边
    print(f'Saved figures to: {save_dir / stem}.png and .tif')

    return fig, pd.DataFrame(rows)


# In[250]:


# ==========================================
# 模块七：主控执行流水线 (Part 1 - 数据加载)
# 功能：统一读入推断张量并进行维度和文件完整性校验
# ==========================================
# =========================
# Load data
# =========================
run_dir = RUN_DIR
if not run_dir.exists():
    raise FileNotFoundError(f'run dir not found: {run_dir}')

data_path = resolve_data_path(run_dir, DATA_PATH)
ATTACK_EVENTS, DATASET_NAME = select_attack_events(data_path, run_dir)
pred_dir = run_dir / 'predictions' if (run_dir / 'predictions').exists() else run_dir
y_true_path = pred_dir / 'y_true.npy'
y_pred_path = pred_dir / 'y_pred.npy'
test_npz_path = data_path / 'test.npz'
print('dataset events:', DATASET_NAME)
print('prediction_dir:', pred_dir)

# [功能] 拦截依赖文件缺失错误
if not y_true_path.exists() or not y_pred_path.exists():
    raise FileNotFoundError(f'Missing y_true.npy/y_pred.npy in {pred_dir}')
if not test_npz_path.exists():
    raise FileNotFoundError(f'Missing test.npz in {data_path}')

y_true = np.load(y_true_path)  # [功能] 读写真值
y_pred = np.load(y_pred_path)  # [功能] 读写预测值
test_npz = np.load(test_npz_path, allow_pickle=False)

if 'y_time' not in test_npz:
    raise KeyError("test.npz has no 'y_time'. Please regenerate dataset with y_time saved.")

y_time = test_npz['y_time']  # [功能] 提取与之配套的时间戳张量

print('y_true shape:', y_true.shape)
print('y_pred shape:', y_pred.shape)
print('y_time shape:', y_time.shape)
print('data_path:', data_path)


# In[251]:


# ==========================================
# 模块七：主控执行流水线 (Part 2 - 重构降维)
# 功能：完成模型张量到时序 DataFrame 的降维转换
# ==========================================
# Reconstruct time series and feature names
if y_true.shape != y_pred.shape:
    raise ValueError(f'shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}')
if y_true.ndim != 3:
    raise ValueError(f'expected 3D arrays [samples,nodes,horizon], got {y_true.ndim}D')

_, node_num, _ = y_true.shape
feature_cols = load_feature_names(data_path, node_num)

# Use first-step stitching to avoid multi-horizon overlap artifacts in visualization.
true_df = aggregate_windows_to_timeseries(y_true, y_time, feature_cols, first_step_only=True)  # [功能] 降维
pred_df = aggregate_windows_to_timeseries(y_pred, y_time, feature_cols, first_step_only=True)  # [功能] 降维

print('stitch mode: first_step_only=True')
print('true_df shape:', true_df.shape)
print('pred_df shape:', pred_df.shape)
true_df.head()  # [语法] 展示 DateFrame 头部作验证


# In[252]:


# ==========================================
# 模块七：主控执行流水线 (Part 3 - 事件队列组装)
# 功能：根据配置文件决定绘制所有异常事件还是焦点分析单一事件
# ==========================================
# Pick events to visualize
if EVENT_ID is not None:
    if not (1 <= EVENT_ID <= len(ATTACK_EVENTS)):
        raise ValueError(f'EVENT_ID must be in [1, {len(ATTACK_EVENTS)}]')
    event_iter = [(EVENT_ID, ATTACK_EVENTS[EVENT_ID - 1])]  # [功能] 选取单一事件装入队列
else:
    event_iter = [(i + 1, ev) for i, ev in enumerate(ATTACK_EVENTS)]  # [功能] 将全部事件装入列表

print('Events to plot:', [i for i, _ in event_iter])


# In[253]:


# ==========================================
# 模块七：主控执行流水线 (Part 4 - 批量渲染与指标收集)
# 功能：遍历事件队列，定位焦点特征节点并触发引擎渲染最终对比图像
# ==========================================
plt.rcParams.update({
    'font.size': 25,
    'font.family': ['Times New Roman', 'SimSun']
})  # [功能] 动态增大绘图专用字体

# Run visualization in notebook (show figures directly, do not save images)
pre = pd.Timedelta(minutes=PRE_MINUTES)
post = pd.Timedelta(minutes=POST_MINUTES)

all_metrics = []
for event_id, (st_raw, ed_raw, attack_cols) in event_iter:
    st = pd.Timestamp(st_raw)  # [语法] 字符串转时间戳
    ed = pd.Timestamp(ed_raw)

    if PV_ONLY:
        candidate_cols = [c for c in true_df.columns if c.endswith('_PV')]  # [功能] 根据后缀筛选物理变量
    else:
        candidate_cols = list(true_df.columns)

    effect = calc_event_effect(true_df, st, ed, pre, candidate_cols)  # [功能] 提取受灾指数排行

    top_nodes = []
    # [功能] 组装图表所需显示的特征集：先加直接受击点，再加间接波及排行靠前的点
    for c in attack_cols:
        if c in true_df.columns and c not in top_nodes:
            top_nodes.append(c)
    for c in effect.index.tolist():
        if c not in top_nodes:
            top_nodes.append(c)
        if len(top_nodes) >= TOPK:
            break

    # [功能] 调度可视化引擎进行核心渲染
    fig, m_df = plot_event(
        event_idx_1based=event_id,
        st=st,
        ed=ed,
        attack_cols=attack_cols,
        true_df=true_df,
        pred_df=pred_df,
        top_nodes=top_nodes,
        pre=pre,
        post=post,
    )

    # display(fig)  # [原程序保留]
    plt.show()  # [语法] 阻塞并弹出图表窗口
    # plt.close(fig)  # [原程序保留]

    if not m_df.empty:
        all_metrics.append(m_df)  # [功能] 记录事件的 MAE 和 RMSE

    print(f'[Event {event_id:02d}] displayed figure. Top nodes: {top_nodes}')

# if len(all_metrics) > 0:  # [原程序保留]
#     metrics_df = pd.concat(all_metrics, axis=0, ignore_index=True)  # [原程序保留]
#     print('Metrics summary:')  # [原程序保留]
#     display(metrics_df.head(10))  # [原程序保留]
#     display(metrics_df)  # [原程序保留]

# print('Done.')  # [原程序保留]
