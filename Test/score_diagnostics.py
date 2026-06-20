import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional

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
QUANTILES = [0, 50, 90, 95, 99, 99.5, 99.9, 99.95, 99.99, 100]


def load_optional(path: Path) -> Optional[np.ndarray]:
    return np.load(path, allow_pickle=True) if path.exists() else None


def load_summary(path: Path) -> Dict:
    summary_path = path / "summary.json"
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def align(*arrays: np.ndarray) -> Iterable[np.ndarray]:
    min_len = min(len(a) for a in arrays)
    return tuple(a[:min_len] for a in arrays)


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    out = {"roc_auc": float("nan"), "pr_auc": float("nan")}
    if np.unique(labels).size >= 2:
        out["roc_auc"] = float(roc_auc_score(labels, scores))
        out["pr_auc"] = float(average_precision_score(labels, scores))
    return out


def current_metrics(labels: np.ndarray, pred: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    labels, pred, scores = align(labels.reshape(-1), pred.reshape(-1), scores.reshape(-1))
    out = {
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "pred_ratio": float(pred.mean()),
        "label_ratio": float(labels.mean()),
    }
    out.update(safe_auc(labels, scores))
    return out


def best_test_f1(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    labels, scores = align(labels.reshape(-1).astype(np.int8), scores.reshape(-1).astype(np.float32))
    if np.unique(labels).size < 2:
        return {}
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        return {}
    f1 = 2.0 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + EPS)
    idx = int(np.nanargmax(f1))
    tau = float(thresholds[idx])
    pred = (scores >= tau).astype(np.int8)
    return {
        "best_f1_test_only": float(f1[idx]),
        "best_precision_test_only": float(precision_score(labels, pred, zero_division=0)),
        "best_recall_test_only": float(recall_score(labels, pred, zero_division=0)),
        "best_threshold_test_only": tau,
        "best_pred_ratio_test_only": float(pred.mean()),
    }


def quantile_dict(name: str, values: np.ndarray) -> Dict[str, float]:
    values = values.reshape(-1).astype(np.float32)
    qs = np.percentile(values, QUANTILES)
    return {f"{name}_q{str(q).replace('.', '_')}": float(v) for q, v in zip(QUANTILES, qs)}


def top_sensor_table(norm_err: Optional[np.ndarray], topk: int) -> Iterable[Dict[str, float]]:
    if norm_err is None or norm_err.ndim != 2:
        return []
    mean_score = norm_err.mean(axis=0)
    p99_score = np.percentile(norm_err, 99, axis=0)
    order = np.argsort(p99_score)[::-1][:topk]
    rows = []
    for rank, node in enumerate(order, start=1):
        rows.append({
            "rank": rank,
            "node": int(node),
            "mean_norm_error": float(mean_score[node]),
            "p99_norm_error": float(p99_score[node]),
        })
    return rows


def diagnose_dir(path: Path, topk: int) -> Dict:
    summary = load_summary(path)
    score_val = load_optional(path / "score_val.npy")
    score_test = load_optional(path / "score_test.npy")
    label_test = load_optional(path / "label_test.npy")
    pred_raw = load_optional(path / "pred_test_raw.npy")
    norm_test = load_optional(path / "norm_err_test.npy")
    sigma = load_optional(path / "residual_sigma.npy")

    if score_test is None:
        raise FileNotFoundError(f"Missing score_test.npy in {path}")

    report: Dict = {
        "postprocess_dir": str(path),
        "score_source": summary.get("selected_residual_source"),
        "time_aggregate": summary.get("time_aggregate"),
        "var_reduce": summary.get("var_reduce"),
        "threshold": summary.get("threshold"),
        "threshold_info": summary.get("threshold_info"),
    }
    if score_val is not None:
        report.update(quantile_dict("val_score", score_val))
    report.update(quantile_dict("test_score", score_test))
    if sigma is not None:
        report.update(quantile_dict("sigma", sigma))

    if label_test is not None and pred_raw is not None:
        report["current_metrics"] = current_metrics(label_test, pred_raw, score_test)
        report["diagnostic_best_test_f1"] = best_test_f1(label_test, score_test)
    report["top_sensors_by_test_p99_norm_error"] = list(top_sensor_table(norm_test, topk))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--postprocess-dirs", nargs="+", required=True)
    parser.add_argument("--topk-sensors", type=int, default=10)
    args = parser.parse_args()

    reports = [diagnose_dir(Path(p), args.topk_sensors) for p in args.postprocess_dirs]
    print(json.dumps(reports, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
