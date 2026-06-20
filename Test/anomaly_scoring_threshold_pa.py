#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SADP 后处理与评估脚本。

核心约定：
1. 标准化统计量 mu/sigma 默认只从训练残差计算。
2. 路线 A 默认使用正常验证集分数分布进行无监督阈值选择，不使用验证集异常标签。
3. 测试集标签只用于最终指标计算，不参与阈值和标准化。
4. AUC-PR 使用连续异常分数，而不是二值预测标签。
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


EPS = 1e-8
SPLITS = ("train", "val", "test")

TRUE_PRED_FILES = {
    "train": ("y_train_true.npy", "y_train_pred.npy"),
    "val": ("y_val_true.npy", "y_val_pred.npy"),
    "test": ("y_true.npy", "y_pred.npy"),
}

RESIDUAL_FILES = {
    "joint": {
        "train": "train_joint_error.npy",
        "val": "val_joint_error.npy",
        "test": "test_joint_error.npy",
    },
    "mtgnn": {
        "train": "train_mtgnn_pred_error.npy",
        "val": "val_mtgnn_pred_error.npy",
        "test": "test_mtgnn_pred_error.npy",
    },
    "ae_real": {
        "train": "train_ae_rec_real_error.npy",
        "val": "val_ae_rec_real_error.npy",
        "test": "test_ae_rec_real_error.npy",
    },
    "ae_pred": {
        "train": "train_ae_rec_pred_error.npy",
        "val": "val_ae_rec_pred_error.npy",
        "test": "test_ae_rec_pred_error.npy",
    },
}

EXTRA_KEYS = ("y_time", "y_attack_window", "y_attack_point")


def str_to_bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"{v} is not a valid boolean value.")


def collect_provided_flags(argv: List[str]) -> set:
    flags = set()
    for item in argv:
        if item.startswith("--"):
            flags.add(item.split("=", 1)[0])
    return flags


def set_if_missing(args: argparse.Namespace, provided_flags: set, flag: str, attr: str, value) -> None:
    if flag not in provided_flags:
        setattr(args, attr, value)


def apply_score_preset(args: argparse.Namespace, provided_flags: set) -> None:
    if args.score_preset != "gdn":
        return

    set_if_missing(args, provided_flags, "--score-source", "score_source", "mtgnn")
    set_if_missing(args, provided_flags, "--residual-norm-method", "residual_norm_method", "robust_iqr")
    set_if_missing(args, provided_flags, "--var-reduce", "var_reduce", "max")
    set_if_missing(args, provided_flags, "--threshold-method", "threshold_method", "val_max")
    set_if_missing(args, provided_flags, "--score-smooth-window", "score_smooth_window", 4)
    set_if_missing(args, provided_flags, "--score-smooth-method", "score_smooth_method", "mean")
    set_if_missing(args, provided_flags, "--score-smooth-direction", "score_smooth_direction", "causal")
    set_if_missing(args, provided_flags, "--residual-scale-eps", "residual_scale_eps", 1e-2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SADP anomaly scoring and evaluation.")
    parser.add_argument("--score-preset", type=str, default="none", choices=["none", "gdn"],
                        help="评分预设。gdn 表示 MTGNN 预测残差 + robust_iqr + max + 因果平滑 + 验证集最大值阈值。")
    parser.add_argument("--run-dir", type=str, required=True, help="实验 run 目录，通常为 Save/Experiments/.../run_00")
    parser.add_argument("--data-path", type=str, default=None, help="可选：包含 train/val/test.npz 的数据集目录")
    parser.add_argument("--save-subdir", type=str, default="postprocess", help="输出子目录")

    parser.add_argument("--eval-granularity", type=str, default="point", choices=["point", "window"],
                        help="point 表示反投影回原始时间点评价；window 表示沿用窗口级评价")
    parser.add_argument("--score-source", type=str, default="auto",
                        choices=["auto", "joint", "mtgnn", "ae_real", "ae_pred", "ae_sum"],
                        help="异常评分残差来源：auto/joint/mtgnn/ae_real/ae_pred/ae_sum")
    parser.add_argument("--ae-sum-lambda", type=float, default=0.5,
                        help="score-source=ae_sum 时 ae_rec_pred_error 的权重，默认与训练参数 ae_lambda=0.5 对齐")
    parser.add_argument("--stats-source", type=str, default="train", choices=["train", "stats_file"],
                        help="标准化统计量来源。默认从训练残差计算；也可从外部 stats_file 读取")
    parser.add_argument("--stats-file", type=str, default=None,
                        help="外部统计量 npz 文件，需包含 mu/sigma 或 center/scale")
    parser.add_argument("--residual-norm-method", type=str, default="zscore",
                        choices=["zscore", "robust_iqr"],
                        help="残差归一化方法：zscore 使用训练残差 mean/std；robust_iqr 使用训练残差 median/IQR。")
    parser.add_argument("--residual-scale-eps", type=float, default=EPS,
                        help="残差尺度分母 epsilon。GDN 官方实现使用 1e-2；默认保持历史 zscore 的 1e-8。")

    parser.add_argument("--sigma-floor-method", type=str, default="none",
                        choices=["none", "quantile", "value"],
                        help="Robust lower bound for residual sigma: none, quantile, or value")
    parser.add_argument("--sigma-floor-quantile", type=float, default=10.0,
                        help="Quantile used when sigma-floor-method=quantile")
    parser.add_argument("--sigma-floor-value", type=float, default=0.05,
                        help="Fixed lower bound used when sigma-floor-method=value")

    parser.add_argument("--time-aggregate", type=str, default="max", choices=["max", "mean", "first", "last"],
                        help="点级反投影时，同一原始时间点被多个窗口覆盖时的聚合方式")
    parser.add_argument("--horizon-reduce", type=str, default="max", choices=["max", "mean"],
                        help="仅用于 window 评价：窗口内 horizon 维度降维方式")
    parser.add_argument("--var-reduce", type=str, default="mean", choices=["mean", "max", "topk_mean", "p95"],
                        help="传感器维度聚合方式。算法规范默认 mean")
    parser.add_argument("--var-topk", type=int, default=10, help="var-reduce=topk_mean 时使用")

    parser.add_argument("--score-smooth-window", type=int, default=1,
                        help="Temporal smoothing window for final 1D scores. 1 disables smoothing.")
    parser.add_argument("--score-smooth-method", type=str, default="mean", choices=["mean", "median"],
                        help="Temporal smoothing method for final 1D scores.")
    parser.add_argument("--score-smooth-direction", type=str, default="causal", choices=["causal", "centered"],
                        help="causal uses current/past scores only; centered is offline and uses both sides.")

    parser.add_argument("--threshold-method", type=str, default="percentile",
                        choices=["f1_val", "percentile", "mean_std", "val_max"],
                        help="路线 A 默认使用正常验证集分位数阈值；f1_val 仅作为监督阈值校准模式")
    parser.add_argument("--threshold-percentile", type=float, default=99.0)
    parser.add_argument("--threshold-k", type=float, default=3.0)
    parser.add_argument("--pa-delay", type=int, default=None,
                        help="PA 延迟半径。为空时使用 horizon_size-1")
    parser.add_argument("--save-pa", type=str_to_bool, default=True, help="是否额外输出 PA 后指标")
    args = parser.parse_args()
    apply_score_preset(args, collect_provided_flags(sys.argv[1:]))
    return args


def get_prediction_dir(run_dir: Path) -> Path:
    pred_dir = run_dir / "predictions"
    return pred_dir if pred_dir.exists() else run_dir


def load_npy_if_exists(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    return np.load(path, allow_pickle=True)


def load_predictions(run_dir: Path) -> Dict[str, np.ndarray]:
    pred_dir = get_prediction_dir(run_dir)
    data: Dict[str, np.ndarray] = {}

    for split, (true_name, pred_name) in TRUE_PRED_FILES.items():
        for name in (true_name, pred_name):
            arr = load_npy_if_exists(pred_dir / name)
            if arr is not None:
                if arr.ndim != 3:
                    raise ValueError(f"{name} 必须为 [samples, nodes, horizon]，收到 {arr.shape}")
                data[name] = arr.astype(np.float32)

    for split in ("val", "test"):
        true_name, pred_name = TRUE_PRED_FILES[split]
        if true_name not in data or pred_name not in data:
            raise FileNotFoundError(f"缺少 {split} 预测文件: {true_name}, {pred_name}")
        if data[true_name].shape != data[pred_name].shape:
            raise ValueError(f"{split} y_true/y_pred 形状不一致: {data[true_name].shape} vs {data[pred_name].shape}")

    for source_files in RESIDUAL_FILES.values():
        for name in source_files.values():
            arr = load_npy_if_exists(pred_dir / name)
            if arr is not None:
                if arr.ndim != 3:
                    raise ValueError(f"{name} 必须为 [samples, nodes, horizon]，收到 {arr.shape}")
                data[name] = arr.astype(np.float32)

    return data


def validate_split_shapes(errors: Dict[str, np.ndarray], reference: Dict[str, np.ndarray]) -> None:
    for split, err in errors.items():
        true_name, _ = TRUE_PRED_FILES[split]
        if true_name in reference and err.shape != reference[true_name].shape:
            raise ValueError(f"{split} 残差形状 {err.shape} 与 {true_name} {reference[true_name].shape} 不一致")


def select_error_tensors(
        pred_data: Dict[str, np.ndarray],
        score_source: str,
        required_splits: List[str],
        ae_sum_lambda: float,
) -> Tuple[Dict[str, np.ndarray], str]:
    def has_residual(source: str) -> bool:
        return all(RESIDUAL_FILES[source][split] in pred_data for split in required_splits)

    def require_residual(source: str) -> Dict[str, np.ndarray]:
        if not has_residual(source):
            missing = [RESIDUAL_FILES[source][split] for split in required_splits
                       if RESIDUAL_FILES[source][split] not in pred_data]
            raise FileNotFoundError(f"score-source={source} 缺少文件: {missing}")
        out = {split: pred_data[RESIDUAL_FILES[source][split]] for split in required_splits}
        validate_split_shapes(out, pred_data)
        return out

    if score_source in {"auto", "joint"} and has_residual("joint"):
        out = {split: pred_data[RESIDUAL_FILES["joint"][split]] for split in required_splits}
        validate_split_shapes(out, pred_data)
        return out, "joint_error"
    if score_source == "joint":
        return require_residual("joint"), "joint_error"

    if score_source in {"auto", "mtgnn"} and has_residual("mtgnn"):
        out = {split: pred_data[RESIDUAL_FILES["mtgnn"][split]] for split in required_splits}
        validate_split_shapes(out, pred_data)
        return out, "mtgnn_pred_error"

    if score_source == "mtgnn":
        return require_residual("mtgnn"), "mtgnn_pred_error"

    if score_source == "ae_real":
        return require_residual("ae_real"), "ae_rec_real_error"

    if score_source == "ae_pred":
        return require_residual("ae_pred"), "ae_rec_pred_error"

    if score_source == "ae_sum":
        real = require_residual("ae_real")
        pred = require_residual("ae_pred")
        out = {
            split: (real[split] + float(ae_sum_lambda) * pred[split]).astype(np.float32)
            for split in required_splits
        }
        validate_split_shapes(out, pred_data)
        return out, f"ae_sum_error(real_plus_{float(ae_sum_lambda):g}_pred)"

    out = {}
    for split in required_splits:
        true_name, pred_name = TRUE_PRED_FILES[split]
        if true_name not in pred_data or pred_name not in pred_data:
            raise FileNotFoundError(
                f"无法为 {split} 回退计算 |y_true-y_pred|，缺少 {true_name} 或 {pred_name}"
            )
        out[split] = np.abs(pred_data[true_name] - pred_data[pred_name]).astype(np.float32)
    validate_split_shapes(out, pred_data)
    return out, "computed_abs_y_true_y_pred"


def load_split_extras(run_dir: Path, data_path: Optional[str], splits: List[str]) -> Dict[str, np.ndarray]:
    pred_dir = get_prediction_dir(run_dir)
    extras: Dict[str, np.ndarray] = {}

    for split in splits:
        for key in EXTRA_KEYS:
            run_name = f"{split}_{key}.npy"
            arr = load_npy_if_exists(pred_dir / run_name)
            if arr is not None:
                extras[f"{split}_{key}"] = arr

        if data_path is None:
            continue
        npz_path = Path(data_path) / f"{split}.npz"
        if not npz_path.exists():
            continue
        with np.load(npz_path, allow_pickle=True) as npz:
            for key in EXTRA_KEYS:
                full_key = f"{split}_{key}"
                if full_key not in extras and key in npz.files:
                    extras[full_key] = npz[key]

    return extras


def require_extra(extras: Dict[str, np.ndarray], split: str, key: str) -> np.ndarray:
    full_key = f"{split}_{key}"
    if full_key not in extras:
        raise FileNotFoundError(
            f"缺少 {full_key}。点级评价需要 predictions/{full_key}.npy "
            f"或 data-path/{split}.npz 中的 {key}。"
        )
    return extras[full_key]


def unique_preserve_first(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回按首次出现顺序排列的唯一时间点，以及每个原始元素对应的新索引。

    不直接使用 np.unique 的默认排序结果，是为了避免字符串时间索引出现
    "10" 排在 "2" 前面的非时序顺序问题，从而影响点级 PA。
    """
    unique_sorted, first_idx, inverse_sorted = np.unique(values, return_index=True, return_inverse=True)
    order = np.argsort(first_idx)
    remap = np.empty_like(order)
    remap[order] = np.arange(order.size)
    return unique_sorted[order], remap[inverse_sorted]


def project_error_to_points(
        error_tensor: np.ndarray,
        y_time: np.ndarray,
        aggregate: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if error_tensor.ndim != 3:
        raise ValueError(f"error_tensor 必须为 [samples, nodes, horizon]，收到 {error_tensor.shape}")
    if y_time.ndim != 2:
        raise ValueError(f"y_time 必须为 [samples, horizon]，收到 {y_time.shape}")
    if error_tensor.shape[0] != y_time.shape[0] or error_tensor.shape[2] != y_time.shape[1]:
        raise ValueError(f"error 与 y_time 不对齐: error={error_tensor.shape}, y_time={y_time.shape}")

    samples, nodes, horizon = error_tensor.shape
    flat_time = y_time.reshape(samples * horizon)
    flat_error = np.transpose(error_tensor, (0, 2, 1)).reshape(samples * horizon, nodes)
    unique_time, inverse = unique_preserve_first(flat_time)

    if aggregate == "max":
        point_error = np.full((len(unique_time), nodes), -np.inf, dtype=np.float32)
        np.maximum.at(point_error, inverse, flat_error)
        point_error[~np.isfinite(point_error)] = 0.0
    elif aggregate == "mean":
        point_error = np.zeros((len(unique_time), nodes), dtype=np.float64)
        counts = np.zeros(len(unique_time), dtype=np.float64)
        np.add.at(point_error, inverse, flat_error)
        np.add.at(counts, inverse, 1.0)
        point_error = (point_error / np.maximum(counts[:, None], 1.0)).astype(np.float32)
    elif aggregate == "first":
        point_error = np.full((len(unique_time), nodes), np.nan, dtype=np.float32)
        filled = np.zeros(len(unique_time), dtype=bool)
        for row_idx, target_idx in enumerate(inverse):
            if not filled[target_idx]:
                point_error[target_idx] = flat_error[row_idx]
                filled[target_idx] = True
        point_error = np.nan_to_num(point_error, nan=0.0)
    elif aggregate == "last":
        point_error = np.zeros((len(unique_time), nodes), dtype=np.float32)
        for row_idx, target_idx in enumerate(inverse):
            point_error[target_idx] = flat_error[row_idx]
    else:
        raise ValueError(f"未知 time aggregate: {aggregate}")

    return unique_time, point_error.astype(np.float32)


def project_labels_to_points(y_attack_point: np.ndarray, y_time: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if y_attack_point.ndim != 2:
        raise ValueError(f"y_attack_point 必须为 [samples, horizon]，收到 {y_attack_point.shape}")
    if y_time.shape != y_attack_point.shape:
        raise ValueError(f"y_attack_point 与 y_time 不对齐: {y_attack_point.shape} vs {y_time.shape}")

    flat_time = y_time.reshape(-1)
    flat_label = y_attack_point.reshape(-1).astype(np.int8)
    unique_time, inverse = unique_preserve_first(flat_time)
    point_label = np.zeros(len(unique_time), dtype=np.int8)
    np.maximum.at(point_label, inverse, flat_label)
    return unique_time, point_label


def reduce_error_by_horizon(error_tensor: np.ndarray, horizon_reduce: str) -> np.ndarray:
    if horizon_reduce == "max":
        return error_tensor.max(axis=2)
    if horizon_reduce == "mean":
        return error_tensor.mean(axis=2)
    raise ValueError(f"未知 horizon_reduce: {horizon_reduce}")


def compute_train_stats(train_error_matrix: np.ndarray, norm_method: str) -> Tuple[np.ndarray, np.ndarray]:
    if train_error_matrix.ndim != 2:
        raise ValueError(f"训练残差统计矩阵必须为 [time, nodes]，收到 {train_error_matrix.shape}")

    if norm_method == "zscore":
        center = train_error_matrix.mean(axis=0, keepdims=True).astype(np.float32)
        scale = train_error_matrix.std(axis=0, keepdims=True).astype(np.float32)
    elif norm_method == "robust_iqr":
        center = np.median(train_error_matrix, axis=0, keepdims=True).astype(np.float32)
        q75 = np.percentile(train_error_matrix, 75, axis=0, keepdims=True)
        q25 = np.percentile(train_error_matrix, 25, axis=0, keepdims=True)
        scale = (q75 - q25).astype(np.float32)
    else:
        raise ValueError(f"未知 residual-norm-method: {norm_method}")

    scale = np.where(scale <= 0, EPS, scale).astype(np.float32)
    return center, scale


def load_stats_file(path: str, node_num: int) -> Tuple[np.ndarray, np.ndarray]:
    if path is None:
        raise ValueError("stats-source=stats_file 时必须传入 --stats-file")
    with np.load(path, allow_pickle=False) as data:
        if "mu" in data and "sigma" in data:
            mu, sigma = data["mu"], data["sigma"]
        elif "center" in data and "scale" in data:
            mu, sigma = data["center"], data["scale"]
        else:
            raise KeyError("stats-file 必须包含 mu/sigma 或 center/scale")

    mu = np.asarray(mu, dtype=np.float32).reshape(1, -1)
    sigma = np.asarray(sigma, dtype=np.float32).reshape(1, -1)
    if mu.shape[1] != node_num or sigma.shape[1] != node_num:
        raise ValueError(f"stats 节点数不匹配: mu={mu.shape}, sigma={sigma.shape}, node_num={node_num}")
    sigma = np.where(sigma <= 0, EPS, sigma).astype(np.float32)
    return mu, sigma


def sigma_summary(prefix: str, sigma: np.ndarray) -> Dict[str, float]:
    flat = sigma.reshape(-1).astype(np.float32)
    return {
        f"{prefix}_min": float(np.min(flat)),
        f"{prefix}_q10": float(np.percentile(flat, 10)),
        f"{prefix}_median": float(np.median(flat)),
        f"{prefix}_q90": float(np.percentile(flat, 90)),
        f"{prefix}_max": float(np.max(flat)),
    }


def apply_sigma_floor(
        sigma: np.ndarray,
        method: str,
        quantile: float,
        value: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    raw_sigma = np.where(sigma <= 0, EPS, sigma).astype(np.float32)

    if method == "none":
        floor = 0.0
        used_sigma = raw_sigma
    elif method == "quantile":
        if not 0.0 <= quantile <= 100.0:
            raise ValueError(f"sigma-floor-quantile must be in [0, 100], got {quantile}")
        floor = float(np.percentile(raw_sigma.reshape(-1), quantile))
        used_sigma = np.maximum(raw_sigma, floor).astype(np.float32)
    elif method == "value":
        if value < 0:
            raise ValueError(f"sigma-floor-value must be non-negative, got {value}")
        floor = float(value)
        used_sigma = np.maximum(raw_sigma, floor).astype(np.float32)
    else:
        raise ValueError(f"Unknown sigma-floor-method: {method}")

    info: Dict[str, float] = {
        "method": method,
        "quantile": float(quantile),
        "value_arg": float(value),
        "applied_value": float(floor),
        "changed_nodes": int(np.sum(used_sigma > raw_sigma)),
    }
    info.update(sigma_summary("raw_sigma", raw_sigma))
    info.update(sigma_summary("used_sigma", used_sigma))
    info.update(sigma_summary("raw_scale", raw_sigma))
    info.update(sigma_summary("used_scale", used_sigma))
    return used_sigma, info


def compute_scores_from_matrix(
        error_matrix: np.ndarray,
        center: np.ndarray,
        scale: np.ndarray,
        scale_eps: float,
        var_reduce: str,
        var_topk: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if error_matrix.ndim != 2:
        raise ValueError(f"error_matrix 必须为 [time, nodes]，收到 {error_matrix.shape}")
    if center.shape != (1, error_matrix.shape[1]) or scale.shape != (1, error_matrix.shape[1]):
        raise ValueError(
            f"center/scale 广播形状错误: error={error_matrix.shape}, "
            f"center={center.shape}, scale={scale.shape}"
        )

    if scale_eps < 0:
        raise ValueError(f"residual-scale-eps must be non-negative, got {scale_eps}")

    norm_error = (error_matrix - center) / (np.abs(scale) + float(scale_eps))
    if var_reduce == "mean":
        scores = norm_error.mean(axis=1)
    elif var_reduce == "max":
        scores = norm_error.max(axis=1)
    elif var_reduce == "topk_mean":
        k = max(1, min(int(var_topk), norm_error.shape[1]))
        part = np.partition(norm_error, kth=norm_error.shape[1] - k, axis=1)[:, -k:]
        scores = part.mean(axis=1)
    elif var_reduce == "p95":
        scores = np.percentile(norm_error, 95, axis=1)
    else:
        raise ValueError(f"未知 var_reduce: {var_reduce}")
    return scores.astype(np.float32), norm_error.astype(np.float32)


def smooth_scores(
        scores: np.ndarray,
        window: int,
        method: str,
        direction: str,
) -> Tuple[np.ndarray, Dict[str, object]]:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    window = int(window)
    info: Dict[str, object] = {
        "window": int(window),
        "method": method,
        "direction": direction,
        "enabled": bool(window > 1),
    }
    if window <= 1:
        return scores.copy(), info
    if method not in {"mean", "median"}:
        raise ValueError(f"Unknown score-smooth-method: {method}")
    if direction not in {"causal", "centered"}:
        raise ValueError(f"Unknown score-smooth-direction: {direction}")

    out = np.empty_like(scores, dtype=np.float32)
    n = scores.shape[0]
    for i in range(n):
        if direction == "causal":
            left = max(0, i - window + 1)
            right = i + 1
        else:
            before = window // 2
            after = window - before
            left = max(0, i - before)
            right = min(n, i + after)
        segment = scores[left:right]
        if method == "mean":
            out[i] = np.mean(segment, dtype=np.float64)
        else:
            out[i] = np.median(segment)
    return out.astype(np.float32), info


def best_threshold_by_val_f1(scores_val: np.ndarray, labels_val: np.ndarray) -> Tuple[float, Dict[str, float]]:
    labels_val = labels_val.astype(np.int8).reshape(-1)
    scores_val = scores_val.astype(np.float32).reshape(-1)
    if labels_val.shape[0] != scores_val.shape[0]:
        raise ValueError(f"验证集分数与标签长度不一致: {scores_val.shape[0]} vs {labels_val.shape[0]}")
    if np.unique(labels_val).size < 2:
        raise ValueError("F1 阈值搜索要求验证集同时包含正常点和异常点；当前验证集标签只有单一类别。")

    precision, recall, thresholds = precision_recall_curve(labels_val, scores_val)
    if thresholds.size == 0:
        raise ValueError("precision_recall_curve 未返回有效阈值。")
    f1 = 2.0 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + EPS)
    best_idx = int(np.nanargmax(f1))
    tau = float(thresholds[best_idx])
    pred_val = (scores_val >= tau).astype(np.int8)
    metrics = {
        "method": "f1_val",
        "threshold": tau,
        "val_precision": float(precision_score(labels_val, pred_val, zero_division=0)),
        "val_recall": float(recall_score(labels_val, pred_val, zero_division=0)),
        "val_f1": float(f1_score(labels_val, pred_val, zero_division=0)),
        "candidate_count": int(thresholds.size),
    }
    return tau, metrics


def choose_threshold(
        scores_val: np.ndarray,
        labels_val: Optional[np.ndarray],
        method: str,
        percentile: float,
        k: float,
) -> Tuple[float, Dict[str, float]]:
    if method == "f1_val":
        if labels_val is None:
            raise ValueError("threshold-method=f1_val 需要验证集标签。")
        return best_threshold_by_val_f1(scores_val, labels_val)
    if method == "percentile":
        tau = float(np.percentile(scores_val, percentile))
        return tau, {"method": "percentile", "percentile": float(percentile), "threshold": tau}
    if method == "val_max":
        tau = float(np.max(scores_val))
        return tau, {"method": "val_max", "threshold": tau}
    if method == "mean_std":
        tau = float(scores_val.mean() + k * scores_val.std())
        return tau, {"method": "mean_std", "k": float(k), "threshold": tau}
    raise ValueError(f"未知 threshold method: {method}")


def point_adjust_binary(pred: np.ndarray, delay: int) -> np.ndarray:
    pred = pred.astype(np.int8).reshape(-1)
    if delay <= 0:
        return pred.copy()
    out = pred.copy()
    idx = np.where(pred > 0)[0]
    n = len(pred)
    for i in idx:
        left = max(0, i - delay)
        right = min(n, i + delay + 1)
        out[left:right] = 1
    return out.astype(np.int8)


def safe_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(np.int8).reshape(-1)
    y_pred = y_pred.astype(np.int8).reshape(-1)
    scores = scores.astype(np.float32).reshape(-1)
    if not (len(y_true) == len(y_pred) == len(scores)):
        raise ValueError(f"指标长度不一致: y_true={len(y_true)}, y_pred={len(y_pred)}, scores={len(scores)}")

    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if np.unique(y_true).size > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
        metrics["pr_auc"] = float(average_precision_score(y_true, scores))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    return metrics


def align_by_length(*arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    min_len = min(len(a) for a in arrays)
    return tuple(a[:min_len] for a in arrays)


def build_point_matrices(
        error_tensors: Dict[str, np.ndarray],
        extras: Dict[str, np.ndarray],
        aggregate: str,
        need_train: bool,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    splits = ["val", "test"] + (["train"] if need_train else [])
    matrices: Dict[str, np.ndarray] = {}
    times: Dict[str, np.ndarray] = {}
    labels: Dict[str, np.ndarray] = {}

    for split in splits:
        y_time = require_extra(extras, split, "y_time")
        point_time, point_error = project_error_to_points(error_tensors[split], y_time, aggregate)
        matrices[split] = point_error
        times[split] = point_time

        label_key = f"{split}_y_attack_point"
        if label_key in extras:
            label_time, point_label = project_labels_to_points(extras[label_key], y_time)
            if not np.array_equal(point_time, label_time):
                raise ValueError(f"{split} 反投影后的 error time 与 label time 不一致。")
            labels[split] = point_label

    return matrices, times, labels


def build_window_matrices(
        error_tensors: Dict[str, np.ndarray],
        extras: Dict[str, np.ndarray],
        horizon_reduce: str,
        need_train: bool,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    splits = ["val", "test"] + (["train"] if need_train else [])
    matrices = {split: reduce_error_by_horizon(error_tensors[split], horizon_reduce) for split in splits}
    labels: Dict[str, np.ndarray] = {}
    for split in ("val", "test"):
        key = f"{split}_y_attack_window"
        if key in extras:
            labels[split] = extras[key].astype(np.int8)
    return matrices, labels


def save_outputs(
        out_dir: Path,
        scores_val: np.ndarray,
        scores_test: np.ndarray,
        raw_scores_val: Optional[np.ndarray],
        raw_scores_test: Optional[np.ndarray],
        norm_val: np.ndarray,
        norm_test: np.ndarray,
        center: np.ndarray,
        scale: np.ndarray,
        scale_raw: np.ndarray,
        pred_test_raw: np.ndarray,
        pred_test_pa: np.ndarray,
        labels: Dict[str, np.ndarray],
        times: Optional[Dict[str, np.ndarray]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "score_val.npy", scores_val)
    np.save(out_dir / "score_test.npy", scores_test)
    if raw_scores_val is not None:
        np.save(out_dir / "score_val_raw.npy", raw_scores_val)
    if raw_scores_test is not None:
        np.save(out_dir / "score_test_raw_score.npy", raw_scores_test)
    np.save(out_dir / "norm_err_val.npy", norm_val)
    np.save(out_dir / "norm_err_test.npy", norm_test)
    np.save(out_dir / "residual_center.npy", center.squeeze(0))
    np.save(out_dir / "residual_scale.npy", scale.squeeze(0))
    np.save(out_dir / "residual_scale_raw.npy", scale_raw.squeeze(0))
    np.save(out_dir / "residual_mu.npy", center.squeeze(0))
    np.save(out_dir / "residual_sigma.npy", scale.squeeze(0))
    np.save(out_dir / "residual_sigma_raw.npy", scale_raw.squeeze(0))
    np.save(out_dir / "pred_test_raw.npy", pred_test_raw)
    np.save(out_dir / "pred_test_pa.npy", pred_test_pa)

    for split, label in labels.items():
        np.save(out_dir / f"label_{split}.npy", label.astype(np.int8))
    if times is not None:
        for split, split_time in times.items():
            np.save(out_dir / f"time_{split}.npy", split_time)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    pred_data = load_predictions(run_dir)
    need_train_stats = args.stats_source == "train"
    required_splits = ["val", "test"] + (["train"] if need_train_stats else [])
    error_tensors, selected_source = select_error_tensors(
        pred_data,
        args.score_source,
        required_splits,
        args.ae_sum_lambda,
    )
    extras = load_split_extras(run_dir, args.data_path, required_splits)

    if args.eval_granularity == "point":
        matrices, times, labels = build_point_matrices(
            error_tensors,
            extras,
            args.time_aggregate,
            need_train=need_train_stats,
        )
        horizon_size = int(error_tensors["test"].shape[2])
    else:
        matrices, labels = build_window_matrices(
            error_tensors,
            extras,
            args.horizon_reduce,
            need_train=need_train_stats,
        )
        times = None
        horizon_size = int(error_tensors["test"].shape[2])

    node_num = int(matrices["val"].shape[1])
    if args.stats_source == "train":
        if "train" not in matrices:
            raise FileNotFoundError("stats-source=train 需要训练残差。请重新运行训练脚本导出 train_*_error.npy。")
        center, scale = compute_train_stats(matrices["train"], args.residual_norm_method)
    else:
        center, scale = load_stats_file(args.stats_file, node_num)

    scale_raw = scale.copy()
    scale, sigma_floor_info = apply_sigma_floor(
        scale,
        args.sigma_floor_method,
        args.sigma_floor_quantile,
        args.sigma_floor_value,
    )

    raw_scores_val, norm_val = compute_scores_from_matrix(
        matrices["val"], center, scale, args.residual_scale_eps, args.var_reduce, args.var_topk
    )
    raw_scores_test, norm_test = compute_scores_from_matrix(
        matrices["test"], center, scale, args.residual_scale_eps, args.var_reduce, args.var_topk
    )
    scores_val, score_smoothing_info = smooth_scores(
        raw_scores_val,
        args.score_smooth_window,
        args.score_smooth_method,
        args.score_smooth_direction,
    )
    scores_test, _ = smooth_scores(
        raw_scores_test,
        args.score_smooth_window,
        args.score_smooth_method,
        args.score_smooth_direction,
    )

    val_labels = labels.get("val")
    tau, threshold_info = choose_threshold(
        scores_val,
        val_labels,
        args.threshold_method,
        args.threshold_percentile,
        args.threshold_k,
    )

    pred_test_raw = (scores_test >= tau).astype(np.int8)
    pa_delay = int(args.pa_delay) if args.pa_delay is not None else int(max(0, horizon_size - 1))
    pred_test_pa = point_adjust_binary(pred_test_raw, pa_delay) if args.save_pa else pred_test_raw.copy()
    if args.residual_norm_method == "robust_iqr":
        standardization_desc = (
            f"robust_iqr: (e - train_median_i) / (abs(iqr_used_i) + {args.residual_scale_eps:g})"
        )
    else:
        standardization_desc = (
            f"zscore: (e - train_mean_i) / (abs(std_used_i) + {args.residual_scale_eps:g})"
        )

    summary = {
        "run_dir": str(run_dir),
        "score_preset": args.score_preset,
        "eval_granularity": args.eval_granularity,
        "score_source_arg": args.score_source,
        "selected_residual_source": selected_source,
        "ae_sum_lambda": float(args.ae_sum_lambda),
        "stats_source": args.stats_source,
        "stats_file": args.stats_file,
        "residual_norm_method": args.residual_norm_method,
        "residual_scale_eps": float(args.residual_scale_eps),
        "standardization": standardization_desc,
        "scale_floor_info": sigma_floor_info,
        "sigma_floor_info": sigma_floor_info,
        "var_reduce": args.var_reduce,
        "var_topk": int(args.var_topk),
        "score_smoothing_info": score_smoothing_info,
        "time_aggregate": args.time_aggregate if args.eval_granularity == "point" else None,
        "horizon_reduce": args.horizon_reduce if args.eval_granularity == "window" else None,
        "threshold_info": threshold_info,
        "decision_rule": "Label=1 if score >= threshold else 0",
        "threshold": float(tau),
        "route": "A_semisupervised_unsupervised_threshold" if args.threshold_method != "f1_val" else "calibrated_val_f1_threshold",
        "pa_delay": int(pa_delay),
        "node_num": int(node_num),
        "horizon_size": int(horizon_size),
        "val_score_mean": float(scores_val.mean()),
        "test_score_mean": float(scores_test.mean()),
        "test_anomaly_ratio_raw": float(pred_test_raw.mean()),
        "test_anomaly_ratio_pa": float(pred_test_pa.mean()),
        "auc_pr_uses_continuous_scores": True,
        "val_labels_used_for_threshold": bool(args.threshold_method == "f1_val"),
        "test_labels_used_for_threshold": False,
    }

    test_labels = labels.get("test")
    if test_labels is not None:
        test_labels, pred_raw_eval, pred_pa_eval, score_test_eval = align_by_length(
            test_labels, pred_test_raw, pred_test_pa, scores_test
        )
        summary["eval_raw"] = safe_binary_metrics(test_labels, pred_raw_eval, score_test_eval)
        if args.save_pa:
            summary["eval_pa"] = safe_binary_metrics(test_labels, pred_pa_eval, score_test_eval)

    out_dir = run_dir / args.save_subdir
    save_outputs(
        out_dir,
        scores_val,
        scores_test,
        raw_scores_val,
        raw_scores_test,
        norm_val,
        norm_test,
        center,
        scale,
        scale_raw,
        pred_test_raw,
        pred_test_pa,
        labels,
        times,
    )

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[DONE] anomaly postprocess outputs saved to:", out_dir)
    print("score source =", selected_source)
    print("eval granularity =", args.eval_granularity)
    print("threshold =", round(float(tau), 6), "| method =", threshold_info["method"])
    print("test anomaly ratio raw/pa =", round(float(pred_test_raw.mean()), 4), "/", round(float(pred_test_pa.mean()), 4))


if __name__ == "__main__":
    main()
