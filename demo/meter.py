"""08 danger meter: which P(goal) a frame shows, and where it sits on the bar."""

import math

import numpy as np
import polars as pl

from demo import overlay as ov

P_MIN, P_MAX = 0.001, 0.5  # log scale: 0.1% is an empty bar, 50% a full one
TICKS = (0.01, 0.05, 0.2)
TITLE = "goal in 5 s"
METER_W = 96
STALE_S = 0.1 + 1e-6  # one grid step: an older row means the grid skipped rows here
EPS_S = 1e-6  # float slack, so a frame exactly on a grid time sees that row


def level(p: float | None) -> float | None:
    """Bar height 0-1 on the log scale, None without a value."""
    if p is None or not math.isfinite(p):
        return None
    lo, hi = math.log10(P_MIN), math.log10(P_MAX)
    return min(max((math.log10(max(p, P_MIN)) - lo) / (hi - lo), 0.0), 1.0)


def label(p: float | None) -> str:
    if p is None:
        return "--"
    return f"{100 * p:.1f}%" if p < 0.1 else f"{round(100 * p)}%"


def draw(img: np.ndarray, box: tuple[int, int, int, int], p: float | None) -> None:
    ticks = [(level(t), f"{round(100 * t)}%") for t in TICKS]
    ov.danger_meter(img, box, level(p), label(p), ticks, TITLE)


class Meter:
    """Predictions (period, t_s, col) looked up causally: a frame at period time t shows
    the latest grid row at or before t, never a later or interpolated one (08)."""

    def __init__(self, preds: pl.DataFrame, col: str):
        self.rows = {}
        for (period,), g in preds.sort("period", "t_s").partition_by("period", as_dict=True).items():
            p = g[col].cast(pl.Float64).fill_null(np.nan).to_numpy()
            self.rows[period] = (g["t_s"].to_numpy(), p)

    def value(self, period: int, t: float) -> float | None:
        if period not in self.rows:
            return None
        ts, p = self.rows[period]
        i = int(np.searchsorted(ts, t + EPS_S, side="right")) - 1
        if i < 0 or t - ts[i] > STALE_S:
            return None
        return None if np.isnan(p[i]) else float(p[i])
