#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot channel-level diagnostics for SADP anomaly detection results.

The script reads:
- physical sensor values from predictions/y_true.npy,
- window time indices from predictions/test_y_time.npy,
- normalized residuals and labels from a postprocess directory.

It then plots, for each selected channel:
1. true sensor value,
2. normalized residual,
3. ground-truth attack label and predicted anomaly label.
"""

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_CHANNELS = ["AIT201", "AIT402", "P201", "AIT501", "FIT501"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SADP channel diagnostics.")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--postprocess-dir", type=str, required=True)
    parser.add_argument("--data-path", type=str, default=None,
                        help="Optional pkl/csv data file used to recover feature names.")
    parser.add_argument("--channels", nargs="*", default=DEFAULT_CHANNELS,
                        help="Channel names to plot. Used when --data-path is provided.")
    parser.add_argument("--nodes", nargs="*", type=int, default=None,
                        help="0-based node indices. Overrides --channels when provided.")
    parser.add_argument("--split", type=str, default="test", choices=["test"])
    parser.add_argument("--start", type=int, default=0,
                        help="Start point index after point-level projection.")
    parser.add_argument("--end", type=int, default=None,
                        help="End point index after point-level projection.")
    parser.add_argument("--max-points", type=int, default=5000,
                        help="Maximum points per figure. Use <=0 to disable truncation.")
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def load_feature_names(data_path: Optional[str]) -> List[str]:
    if data_path is None:
        return []
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"data-path not found: {path}")

    if path.suffix.lower() in {".pkl", ".pickle"}:
        import pandas as pd

        df = pd.read_pickle(path)
    elif path.suffix.lower() in {".csv"}:
        import pandas as pd

        df = pd.read_csv(path, nrows=5)
    else:
        raise ValueError(f"Unsupported data-path suffix: {path.suffix}")

    meta = {c for c in df.columns if str(c).strip().lower() in {"attack", "train"}}
    feature_cols = [
        c for c in df.columns
        if c not in meta and getattr(df[c], "dtype", None) is not None
    ]
    try:
        import pandas as pd

        feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    except Exception:
        pass
    return [str(c) for c in feature_cols]


def resolve_nodes(channels: Sequence[str], nodes: Optional[Sequence[int]], feature_names: List[str]) -> List[Tuple[int, str]]:
    if nodes:
        resolved = []
        for node in nodes:
            name = feature_names[node] if feature_names and 0 <= node < len(feature_names) else f"node_{node}"
            resolved.append((int(node), name))
        return resolved

    if not feature_names:
        raise ValueError("--channels requires --data-path to recover feature names. Otherwise use --nodes.")

    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    missing = [ch for ch in channels if ch not in name_to_idx]
    if missing:
        raise KeyError(f"Channels not found in feature names: {missing}")
    return [(name_to_idx[ch], ch) for ch in channels]


def unique_preserve_first(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    _, first_idx, inverse = np.unique(values, return_index=True, return_inverse=True)
    order = np.argsort(first_idx)
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    return values[first_idx[order]], remap[inverse]


def project_channel_to_points(values_3d: np.ndarray, y_time: np.ndarray, node: int) -> Tuple[np.ndarray, np.ndarray]:
    if values_3d.ndim != 3:
        raise ValueError(f"values must be [samples, nodes, horizon], got {values_3d.shape}")
    if y_time.shape != (values_3d.shape[0], values_3d.shape[2]):
        raise ValueError(f"y_time shape mismatch: values={values_3d.shape}, y_time={y_time.shape}")

    flat_time = y_time.reshape(-1)
    flat_values = values_3d[:, node, :].reshape(-1)
    unique_time, inverse = unique_preserve_first(flat_time)
    point_values = np.zeros(len(unique_time), dtype=np.float32)
    seen = np.zeros(len(unique_time), dtype=bool)
    for idx, group in enumerate(inverse):
        if not seen[group]:
            point_values[group] = flat_values[idx]
            seen[group] = True
    return unique_time, point_values


def as_plot_x(times: np.ndarray) -> Tuple[np.ndarray, str]:
    try:
        dt = times.astype("datetime64[ns]")
        return dt, "time"
    except Exception:
        return np.arange(len(times)), "point index"


def slice_range(length: int, start: int, end: Optional[int], max_points: int) -> slice:
    start = max(0, int(start))
    stop = length if end is None else min(length, int(end))
    if max_points and max_points > 0:
        stop = min(stop, start + max_points)
    return slice(start, stop)


def plot_one_channel(
        out_path: Path,
        channel_name: str,
        node: int,
        times: np.ndarray,
        sensor_true: np.ndarray,
        norm_err: np.ndarray,
        attack_label: Optional[np.ndarray],
        pred_label: Optional[np.ndarray],
        point_slice: slice,
) -> None:
    sl = point_slice
    x, xlabel = as_plot_x(times[sl])
    y = sensor_true[sl]
    err = norm_err[sl, node]

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2, 2, 1]})
    fig.suptitle(f"{channel_name} (node {node}) channel diagnostic", fontsize=14)

    axes[0].plot(x, y, linewidth=1.0, color="#1f77b4")
    axes[0].set_ylabel("true value")
    axes[0].grid(alpha=0.25)

    axes[1].plot(x, err, linewidth=1.0, color="#d62728")
    axes[1].axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
    axes[1].set_ylabel("norm residual")
    axes[1].grid(alpha=0.25)

    if attack_label is not None:
        axes[2].fill_between(x, 0, attack_label[sl], step="pre", alpha=0.35, color="#d62728", label="Attack")
    if pred_label is not None:
        axes[2].plot(x, pred_label[sl], linewidth=0.9, color="#2ca02c", label="Pred")
    axes[2].set_ylim(-0.05, 1.15)
    axes[2].set_ylabel("label")
    axes[2].set_xlabel(xlabel)
    axes[2].grid(alpha=0.25)
    axes[2].legend(loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    post_dir = Path(args.postprocess_dir)
    pred_dir = run_dir / "predictions"
    out_dir = Path(args.output_dir) if args.output_dir else post_dir / "channel_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_names = load_feature_names(args.data_path)
    selected = resolve_nodes(args.channels, args.nodes, feature_names)

    y_true = np.load(pred_dir / "y_true.npy", allow_pickle=True)
    y_time = np.load(pred_dir / "test_y_time.npy", allow_pickle=True)
    norm_err = np.load(post_dir / "norm_err_test.npy", allow_pickle=True)
    point_time = np.load(post_dir / "time_test.npy", allow_pickle=True)
    attack_label = np.load(post_dir / "label_test.npy", allow_pickle=True) if (post_dir / "label_test.npy").exists() else None
    pred_label = np.load(post_dir / "pred_test_raw.npy", allow_pickle=True) if (post_dir / "pred_test_raw.npy").exists() else None

    n = min(len(point_time), len(norm_err))
    if attack_label is not None:
        n = min(n, len(attack_label))
    if pred_label is not None:
        n = min(n, len(pred_label))

    point_slice = slice_range(n, args.start, args.end, args.max_points)
    for node, name in selected:
        true_time, true_values = project_channel_to_points(y_true, y_time, node)
        m = min(len(true_time), n)
        plot_one_channel(
            out_dir / f"node{node:02d}_{name}.png",
            name,
            node,
            point_time[:m],
            true_values[:m],
            norm_err[:m],
            attack_label[:m] if attack_label is not None else None,
            pred_label[:m] if pred_label is not None else None,
            point_slice,
        )

    print("[DONE] channel diagnostic plots saved to:", out_dir)
    for node, name in selected:
        print(f"node {node}: {name} -> {out_dir / f'node{node:02d}_{name}.png'}")


if __name__ == "__main__":
    main()
