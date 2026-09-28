"""Metrics for P(shot) on the 10 Hz grid (07, "Metrics").

Threshold-free: PR-AUC (average precision), ROC-AUC, Brier, calibration table.
Plain numpy so evaluation doesn't need the prediction extra.
"""

import numpy as np
import polars as pl


def _check(y, p) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=bool)
    p = np.asarray(p, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError(f"y and p must be 1-d and the same length, got {y.shape} and {p.shape}")
    if np.isnan(p).any():
        raise ValueError("p has NaN")
    return y, p


def pr_auc(y, p) -> float:
    """Average precision: sum over thresholds of precision x recall step, ties grouped.
    Same definition as sklearn's average_precision_score, not a trapezoid (which
    overstates it when positives are rare). nan without positives."""
    y, p = _check(y, p)
    n_pos = y.sum()
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-p, kind="stable")
    y, p = y[order], p[order]
    last = np.r_[np.flatnonzero(np.diff(p)), len(p) - 1]  # end of each tied group
    tp = np.cumsum(y)[last]
    precision = tp / (last + 1)
    recall_step = np.diff(np.r_[0, tp]) / n_pos
    return float((precision * recall_step).sum())


def roc_auc(y, p) -> float:
    """Mann-Whitney with average ranks for ties. nan without both classes."""
    y, p = _check(y, p)
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="stable")
    ranks = np.empty(len(p))
    sp = p[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sp)) + 1]
    ends = np.r_[starts[1:], len(sp)]
    ranks[order] = np.repeat((starts + ends + 1) / 2, ends - starts)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def brier(y, p) -> float:
    y, p = _check(y, p)
    return float(np.mean((p - y) ** 2)) if len(p) else float("nan")


def calibration(y, p, n_bins: int = 10, strategy: str = "quantile") -> pl.DataFrame:
    """Per bin: mean predicted p, observed positive rate, rows.

    Quantile bins by default: at a 2% base rate equal-width bins put almost every
    row in the first one. Empty bins (tied p) are dropped.
    """
    y, p = _check(y, p)
    if strategy == "quantile":
        edges = np.quantile(p, np.linspace(0, 1, n_bins + 1)) if len(p) else np.zeros(1)
    elif strategy == "uniform":
        edges = np.linspace(0, 1, n_bins + 1)
    else:
        raise ValueError(f"strategy must be quantile or uniform, got {strategy!r}")
    bins = np.clip(np.searchsorted(edges[1:-1], p, side="right"), 0, max(len(edges) - 2, 0))
    return (
        pl.DataFrame({"bin": bins, "p": p, "y": y})
        .group_by("bin")
        .agg(mean_p=pl.col("p").mean(), pos_rate=pl.col("y").mean(), n=pl.len())
        .sort("bin")
    )


def scored_rows(pred: pl.DataFrame, h: str) -> tuple[np.ndarray, np.ndarray]:
    """y, p over the rows every model is scored on: label_mask_{h} true, not all_estimated.

    Decided by the labels, never by p, so a model can't skip hard rows and comparisons
    stay paired (07). A null p on one of these rows is an error.
    """
    rows = pred.filter(pl.col(f"label_mask_{h}"), ~pl.col("all_estimated"))
    if rows["p"].null_count():
        raise ValueError(f"{rows['p'].null_count()} scored rows have a null p")
    return rows[f"label_shot_{h}"].to_numpy().astype(bool), rows["p"].to_numpy()


def threshold_free(pred: pl.DataFrame, h: str) -> dict:
    y, p = scored_rows(pred, h)
    return {
        "rows": len(y),
        "positives": int(y.sum()),
        "pr_auc": pr_auc(y, p),
        "roc_auc": roc_auc(y, p),
        "brier": brier(y, p),
        "base_rate": float(y.mean()) if len(y) else float("nan"),
    }
