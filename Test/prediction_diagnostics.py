#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prediction diagnostics for the MTGNN forecasting branch.

This script evaluates saved physical-unit predictions:
- predictions/y_train_true.npy and y_train_pred.npy
- predictions/y_val_true.npy and y_val_pred.npy
- predictions/y_true.npy and y_pred.npy

It reports global, horizon-wise, and channel-wise forecasting metrics.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


SPLIT_FILES = {
    "train": ("y_train_true.npy", "y_train_pred.npy"),
    "val": ("y_val_true.npy", "y_val_pred.npy"),
    "test": ("y_true.npy", "y_pred.npy"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SADP MTGNN prediction diagnostics.")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--data-path", type=str, default=None,
                        help="Optional pkl/csv data file used to recover feature names.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--topk", type=int, default=15)
    return parser.parse_args()


def load_feature_names(data_path: Optional[str]) -> List[str]:
    if data_path is None:
        return []
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"data-path not found: {path}")

    import pandas as pd

    if path.suffix.lower() in {".pkl", ".pickle"}:
        df = pd.read_pickle(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path, nrows=5)
    else:
        raise ValueError(f"Unsupported data-path suffix: {path.suffix}")

    meta = {c for c in df.columns if str(c).strip().lower() in {"attack", "train"}}
    return [
        str(c) for c in df.columns
        if c not in meta and pd.api.types.is_numeric_dtype(df[c])
    ]


def load_split(pred_dir: Path, split: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    true_name, pred_name = SPLIT_FILES[split]
    true_path = pred_dir / true_name
    pred_path = pred_dir / pred_name
    if not true_path.exists() or not pred_path.exists():
        return None
    y_true = np.load(true_path, allow_pickle=True).astype(np.float32)
    y_pred = np.load(pred_path, allow_pickle=True).astype(np.float32)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"{split} shape mismatch: true={y_true.shape}, pred={y_pred.shape}")
    if y_true.ndim != 3:
        raise ValueError(f"{split} arrays must be [samples, nodes, horizon], got {y_true.shape}")
    return y_true, y_pred


def finite_flat(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def metric_block(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt, yp = finite_flat(y_true, y_pred)
    if yt.size == 0:
        return {}

    err = yp - yt
    abs_err = np.abs(err)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    smape = float(np.mean(abs_err / (np.abs(yp) + np.abs(yt) + 1e-8)))
    value_range = float(np.max(yt) - np.min(yt))
    value_std = float(np.std(yt))
    nmae_range = float(mae / (value_range + 1e-8))
    nrmse_range = float(rmse / (value_range + 1e-8))
    nrmse_std = float(rmse / (value_std + 1e-8))

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float(1.0 - ss_res / (ss_tot + 1e-8))
    corr = float(np.corrcoef(yt, yp)[0, 1]) if yt.size > 1 and np.std(yt) > 0 and np.std(yp) > 0 else float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "smape_like": smape,
        "nmae_range": nmae_range,
        "nrmse_range": nrmse_range,
        "nrmse_std": nrmse_std,
        "r2": r2,
        "corr": corr,
        "true_mean": float(np.mean(yt)),
        "true_std": value_std,
        "true_min": float(np.min(yt)),
        "true_max": float(np.max(yt)),
    }


def horizon_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> List[Dict[str, float]]:
    rows = []
    horizon = y_true.shape[2]
    for h in range(horizon):
        row = {"horizon": h + 1}
        row.update(metric_block(y_true[:, :, h], y_pred[:, :, h]))
        rows.append(row)
    return rows


def channel_metrics(y_true: np.ndarray, y_pred: np.ndarray, feature_names: Sequence[str]) -> List[Dict[str, float]]:
    rows = []
    node_num = y_true.shape[1]
    for node in range(node_num):
        row = {
            "node": node,
            "name": feature_names[node] if node < len(feature_names) else f"node_{node}",
        }
        row.update(metric_block(y_true[:, node, :], y_pred[:, node, :]))
        rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    import csv

    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def diagnose_split(
        split: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        feature_names: Sequence[str],
        topk: int,
) -> Dict[str, object]:
    overall = metric_block(y_true, y_pred)
    by_horizon = horizon_metrics(y_true, y_pred)
    by_channel = channel_metrics(y_true, y_pred, feature_names)
    worst_by_nrmse = sorted(by_channel, key=lambda r: r.get("nrmse_std", -np.inf), reverse=True)[:topk]
    worst_by_mae = sorted(by_channel, key=lambda r: r.get("mae", -np.inf), reverse=True)[:topk]
    return {
        "split": split,
        "shape": list(y_true.shape),
        "overall": overall,
        "by_horizon": by_horizon,
        "worst_channels_by_nrmse_std": worst_by_nrmse,
        "worst_channels_by_mae": worst_by_mae,
        "by_channel": by_channel,
    }


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    pred_dir = run_dir / "predictions"
    if not pred_dir.exists():
        raise FileNotFoundError(f"predictions directory not found: {pred_dir}")

    out_dir = Path(args.output_dir) if args.output_dir else run_dir / "prediction_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_names = load_feature_names(args.data_path)
    reports = {}
    compact = {}
    for split in ("train", "val", "test"):
        loaded = load_split(pred_dir, split)
        if loaded is None:
            continue
        y_true, y_pred = loaded
        report = diagnose_split(split, y_true, y_pred, feature_names, args.topk)
        reports[split] = report
        compact[split] = {
            "shape": report["shape"],
            "overall": report["overall"],
            "worst_channels_by_nrmse_std": report["worst_channels_by_nrmse_std"],
        }
        write_csv(out_dir / f"{split}_horizon_metrics.csv", report["by_horizon"])
        write_csv(out_dir / f"{split}_channel_metrics.csv", report["by_channel"])

    with (out_dir / "prediction_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)
    with (out_dir / "prediction_diagnostics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(compact, f, indent=2, ensure_ascii=False)

    print("[DONE] prediction diagnostics saved to:", out_dir)
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
