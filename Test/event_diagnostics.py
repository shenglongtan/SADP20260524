#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Event-level diagnostics for SADP anomaly detection.

This script converts point-level labels into contiguous attack events and reports:
- whether each event is detected,
- detection delay,
- event-level score statistics,
- top contributing channels inside the event.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SADP event-level diagnostics.")
    parser.add_argument("--postprocess-dir", type=str, required=True)
    parser.add_argument("--data-path", type=str, default=None,
                        help="Optional pkl/csv data file used to recover feature names.")
    parser.add_argument("--topk-sensors", type=int, default=5)
    parser.add_argument("--hit-delay", type=int, default=0,
                        help="Optional tolerance in points around an event when checking hit.")
    parser.add_argument("--output-name", type=str, default="event_diagnostics")
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
    feature_cols = [
        c for c in df.columns
        if c not in meta and pd.api.types.is_numeric_dtype(df[c])
    ]
    return [str(c) for c in feature_cols]


def load_required(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return np.load(path, allow_pickle=True)


def align_by_min(*arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    n = min(len(a) for a in arrays)
    return tuple(a[:n] for a in arrays)


def contiguous_events(labels: np.ndarray) -> List[Tuple[int, int]]:
    labels = labels.astype(np.int8).reshape(-1)
    events: List[Tuple[int, int]] = []
    in_event = False
    start = 0
    for i, v in enumerate(labels):
        if v == 1 and not in_event:
            start = i
            in_event = True
        elif v == 0 and in_event:
            events.append((start, i - 1))
            in_event = False
    if in_event:
        events.append((start, len(labels) - 1))
    return events


def time_to_str(times: Optional[np.ndarray], idx: int) -> str:
    if times is None:
        return str(idx)
    value = times[idx]
    if isinstance(value, np.datetime64):
        return str(value)
    return str(value)


def top_channels_for_event(
        norm_err: np.ndarray,
        start: int,
        end: int,
        topk: int,
        feature_names: Sequence[str],
) -> List[Dict[str, object]]:
    segment = norm_err[start:end + 1]
    if segment.ndim != 2:
        return []
    node_mean = segment.mean(axis=0)
    node_max = segment.max(axis=0)
    order = np.argsort(node_mean)[::-1][:topk]
    rows: List[Dict[str, object]] = []
    for rank, node in enumerate(order, start=1):
        name = feature_names[node] if 0 <= node < len(feature_names) else f"node_{node}"
        rows.append({
            "rank": rank,
            "node": int(node),
            "name": name,
            "mean_norm_error": float(node_mean[node]),
            "max_norm_error": float(node_max[node]),
        })
    return rows


def diagnose_events(
        labels: np.ndarray,
        pred: np.ndarray,
        scores: np.ndarray,
        norm_err: np.ndarray,
        times: Optional[np.ndarray],
        feature_names: Sequence[str],
        topk: int,
        hit_delay: int,
) -> List[Dict[str, object]]:
    labels, pred, scores, norm_err = align_by_min(labels, pred, scores, norm_err)
    if times is not None:
        times = times[:len(labels)]

    events = contiguous_events(labels)
    rows: List[Dict[str, object]] = []
    n = len(labels)
    for event_id, (start, end) in enumerate(events, start=1):
        hit_left = max(0, start - hit_delay)
        hit_right = min(n - 1, end + hit_delay)
        pred_window = pred[hit_left:hit_right + 1]
        hit = bool(np.any(pred_window == 1))

        first_pred = None
        latency_points = None
        if hit:
            pred_indices = np.where(pred_window == 1)[0] + hit_left
            first_pred = int(pred_indices[0])
            latency_points = int(first_pred - start)

        event_scores = scores[start:end + 1]
        top_channels = top_channels_for_event(norm_err, start, end, topk, feature_names)
        rows.append({
            "event_id": event_id,
            "start_idx": int(start),
            "end_idx": int(end),
            "start_time": time_to_str(times, start) if times is not None else str(start),
            "end_time": time_to_str(times, end) if times is not None else str(end),
            "duration_points": int(end - start + 1),
            "hit": hit,
            "first_pred_idx": first_pred,
            "first_pred_time": time_to_str(times, first_pred) if times is not None and first_pred is not None else None,
            "latency_points": latency_points,
            "score_max": float(np.max(event_scores)),
            "score_mean": float(np.mean(event_scores)),
            "score_p95": float(np.percentile(event_scores, 95)),
            "top_channels": top_channels,
        })
    return rows


def summarize(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not rows:
        return {"event_count": 0, "hit_count": 0, "event_recall": float("nan")}
    hit_count = sum(1 for r in rows if r["hit"])
    latencies = [r["latency_points"] for r in rows if r["latency_points"] is not None]
    return {
        "event_count": len(rows),
        "hit_count": hit_count,
        "miss_count": len(rows) - hit_count,
        "event_recall": float(hit_count / len(rows)),
        "latency_mean_points": float(np.mean(latencies)) if latencies else None,
        "latency_median_points": float(np.median(latencies)) if latencies else None,
    }


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fields = [
        "event_id", "start_idx", "end_idx", "start_time", "end_time", "duration_points",
        "hit", "first_pred_idx", "first_pred_time", "latency_points",
        "score_max", "score_mean", "score_p95", "top_channels_text",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k) for k in fields if k != "top_channels_text"}
            out["top_channels_text"] = "; ".join(
                f"{c['rank']}:{c['name']}(node{c['node']},mean={c['mean_norm_error']:.3f},max={c['max_norm_error']:.3f})"
                for c in row.get("top_channels", [])
            )
            writer.writerow(out)


def main() -> None:
    args = parse_args()
    post_dir = Path(args.postprocess_dir)
    feature_names = load_feature_names(args.data_path)

    labels = load_required(post_dir / "label_test.npy").astype(np.int8)
    pred = load_required(post_dir / "pred_test_raw.npy").astype(np.int8)
    scores = load_required(post_dir / "score_test.npy").astype(np.float32)
    norm_err = load_required(post_dir / "norm_err_test.npy").astype(np.float32)
    times = np.load(post_dir / "time_test.npy", allow_pickle=True) if (post_dir / "time_test.npy").exists() else None

    rows = diagnose_events(
        labels,
        pred,
        scores,
        norm_err,
        times,
        feature_names,
        args.topk_sensors,
        args.hit_delay,
    )
    summary = summarize(rows)

    json_path = post_dir / f"{args.output_name}.json"
    csv_path = post_dir / f"{args.output_name}.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "events": rows}, f, indent=2, ensure_ascii=False)
    write_csv(csv_path, rows)

    print("[DONE] event diagnostics saved:")
    print("json =", json_path)
    print("csv  =", csv_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
