#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WADI_DataMaker_V1.py - 从 WADI_DataMaker_V1.ipynb 转换而来
功能：WADI 数据集预处理、标准化和滑动窗口样本生成
"""

import os
import hues
import pickle
import numpy as np
import pandas as pd

from tools import CustomScaler, check_folder, downsample_block

# =========================
# 0) 可调参数（先按推荐默认值）
# =========================
WINDOW_SIZE = 48  # 输入历史窗口长度
HORIZON_SIZE = 12  # 预测窗口长度

# 训练/验证划分（注意：WADI 的测试集由 Train==0 单独给出，因此这里会把 Train==1 的数据再切 train/val）
TRAIN_RATIO = 0.7

# 是否下采样（None 表示不下采样；10 表示 10 秒采样）
DOWNSAMPLE_SEC = 10  # 推荐 10；如要保持 1s：改成 None

# 保存路径（你可按自己工程目录改）
save_path = f"E:/DataForCode/5_AnomalyDetection/WADI_STGNN_[{WINDOW_SIZE}_{HORIZON_SIZE}_{DOWNSAMPLE_SEC}]/"

# =========================
# 1) 加载数据
# =========================
df = pd.read_pickle('../../Data/WADI/data.pkl')

hues.info(df.shape)
print(df.head())

# =========================
# 2) 缺失值检查
# =========================
na_count = df.isna().sum().sum()
hues.info(f"缺失值总数：{na_count}")

# =========================
# 3) 列类型划分：特征列 / 标签列
# =========================
label_cols = ["Timestamp", "Train", "Attack"]

# 特征列：除 Timestamp/Train/Attack 外的所有列
feature_cols = [c for c in df.columns if c not in label_cols]

# 状态列（离散/开关/阀门等）：WADI 通常以 STATUS 命名
status_cols = [c for c in feature_cols if "STATUS" in c.upper()]

# 连续列：其余都按连续量处理（PV、流量、压力等）
cont_cols = [c for c in feature_cols if c not in status_cols]

hues.info(f"Feature cols [{len(feature_cols)}]: {str(feature_cols)}")
hues.info(f"Status cols [{len(status_cols)}]: {str(status_cols)}")
hues.info(f"Continuous cols [{len(cont_cols)}]: {str(cont_cols)}")

# =========================
# 4) 检查状态开关列的数据分布情况
# =========================
for c in status_cols:
    vc = df[c].value_counts(dropna=False).sort_index()
    hues.log(f"Column {c}: {dict(vc)}")

# =========================
# 5) 切分 Train==1 / Train==0
# =========================
df_train_val = df[df["Train"] == 1].copy()
df_test = df[df["Train"] == 0].copy()

hues.info(f"TrainVal raw: {df_train_val.shape}")
hues.info(f"Test raw: {df_test.shape}")

# 只保留必要列（仍保留 Timestamp/Train/Attack，便于下采样聚合）
df_train_val = df_train_val[["Timestamp", "Train", "Attack"] + feature_cols]
df_test = df_test[["Timestamp", "Train", "Attack"] + feature_cols]

# =========================
# 6) 下采样（可选）
# =========================
if DOWNSAMPLE_SEC is not None:
    df_train_val = downsample_block(df_train_val, cont_cols, status_cols, DOWNSAMPLE_SEC)
    df_test = downsample_block(df_test, cont_cols, status_cols, DOWNSAMPLE_SEC)

hues.info(f"TrainVal after downsample: {df_train_val.shape}")
hues.info(f"Test after downsample: {df_test.shape}")

# =========================
# 7) 训练/验证按时间顺序切分（在 Train==1 内部再切）
#    说明：
#    - 由于测试集由 Train==0 单独给出，我们不再保留"train/val/test=0.7/0.2/0.1"的第三段
#    - 这里把 Train==1 的数据全部用完：train_ratio = 0.7/(0.7+0.2), val_ratio = 0.2/(0.7+0.2)
# =========================
tv_len = len(df_train_val)
split_idx = int(tv_len * TRAIN_RATIO)

df_train = df_train_val.iloc[:split_idx].copy()
df_val = df_train_val.iloc[split_idx:].copy()

hues.info(f"df_train: {df_train.shape}")
hues.info(f"df_val: {df_val.shape}")
hues.info(f"df_test: {df_test.shape}")

# =========================
# 8) 构造 scaler（只用 train 拟合，避免数据泄露）
#    连续列：z-score（按列）
#    状态列：不归一化（把 mean=0,std=1，相当于 transform 后原值不变）
# =========================
train_values = df_train[feature_cols].values.astype(np.float64)

scaler = CustomScaler(train_values, use_one=False)

# 让状态列保持原值：mean=0,std=1
if len(status_cols) > 0:
    status_idx = [feature_cols.index(c) for c in status_cols]
    scaler.mean[status_idx] = 0.0
    scaler.std[status_idx] = 1.0

hues.success("Scaler fitted on df_train. Status cols are kept as-is (mean=0,std=1).")

# =========================
# 9) 标准化并生成滑动窗口样本 (x,y)
#    x: [num_samples, WINDOW_SIZE, num_nodes, 1]
#    y: [num_samples, HORIZON_SIZE, num_nodes, 1]
# =========================
def build_xy_from_df(_df: pd.DataFrame):
    """
    从 DataFrame 构造 (x, y) 样本对
    
    Args:
        _df: 包含特征列的 DataFrame
        
    Returns:
        x: [num_samples, WINDOW_SIZE, num_nodes, 1]
        y: [num_samples, HORIZON_SIZE, num_nodes, 1]
    """
    # 只取特征列
    data_raw = _df[feature_cols].values.astype(np.float64)

    # 标准化
    data_std = scaler.transform(data_raw).astype(np.float32)

    # 增加最后一维 feature=1
    data_std = np.expand_dims(data_std, axis=-1)  # [T, N, 1]

    T, N, F = data_std.shape

    # 可生成样本数：从 t=WINDOW-1 开始，到 t=T-HORIZON-1 结束（含）
    num_samples = T - (WINDOW_SIZE + HORIZON_SIZE) + 1
    if num_samples <= 0:
        raise ValueError(f"Not enough timesteps: T={T}, WINDOW={WINDOW_SIZE}, HORIZON={HORIZON_SIZE}")

    x = np.zeros((num_samples, WINDOW_SIZE, N, 1), dtype=np.float32)
    y = np.zeros((num_samples, HORIZON_SIZE, N, 1), dtype=np.float32)

    # 逐样本填充（尽量保持逻辑直观，后续也好改）
    for i in range(num_samples):
        t0 = i
        x[i] = data_std[t0: t0 + WINDOW_SIZE]
        y[i] = data_std[t0 + WINDOW_SIZE: t0 + WINDOW_SIZE + HORIZON_SIZE]

    return x, y


x_train, y_train = build_xy_from_df(df_train)
x_val, y_val = build_xy_from_df(df_val)
x_test, y_test = build_xy_from_df(df_test)

hues.info(f"train x: {x_train.shape}, y: {y_train.shape}")
hues.info(f"val   x: {x_val.shape}, y: {y_val.shape}")
hues.info(f"test  x: {x_test.shape}, y: {y_test.shape}")

# =========================
# 10) 保存数据
# =========================
check_folder(save_path)

# 保存数据集（训练/验证/测试）
for cat in ["train", "val", "test"]:
    _x = locals()[f"x_{cat}"]
    _y = locals()[f"y_{cat}"]

    np.savez_compressed(
        os.path.join(save_path, f"{cat}.npz"),
        x=_x.astype(np.float32),
        y=_y.astype(np.float32),
    )
    hues.success(f"Saved {cat}.npz -> x:{_x.shape}, y:{_y.shape}")

# 保存 scaler
with open(os.path.join(save_path, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
hues.success(f"Saved scaler.pkl -> {save_path}")

# =========================
# 11) （可选）保存一些元信息，方便以后对齐列顺序/类型
# =========================
meta = {
    "window_size": WINDOW_SIZE,
    "horizon_size": HORIZON_SIZE,
    "downsample_sec": DOWNSAMPLE_SEC,
    "feature_cols": feature_cols,
    "status_cols": status_cols,
    "cont_cols": cont_cols,
    "train_ratio_eff": TRAIN_RATIO
}
with open(os.path.join(save_path, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)
hues.success("Saved meta.pkl (feature order & config).")
