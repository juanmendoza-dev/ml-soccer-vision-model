"""Metrics for P(shot) on the 10 Hz grid (07, "Metrics").

Threshold-free: PR-AUC (average precision), ROC-AUC, Brier, calibration table.
Alarms: hysteresis on P(shot), matched to shots for lead time, misses and false alarms.
Plain numpy so evaluation doesn't need the prediction extra.
"""

import numpy as np
import polars as pl

from prediction.labels import label_events

OFF_RATIO = 0.8  # an alarm ends below OFF_RATIO * tau (07)
GRACE_S = 1.0  # a shot this soon after an alarm ends still counts for it (07)
MAX_FALSE_PER_MATCH = 3.0  # tau budget (07, "Choosing tau")
MAX_NULL_HOLD_S = 2.0  # an alarm ends after this long with no prediction (07)


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


def alarms(pred: pl.DataFrame, tau: float, off_ratio: float = OFF_RATIO) -> pl.DataFrame:
    """Alarm intervals from P(shot), per (match_id, period), in t_s order (07).

    Starts when p > tau with a team in possession and the ball not dead. Ends on the
    first row with p < off_ratio * tau, a different non-null possession_team,
    ball_state == "dead", or more than MAX_NULL_HOLD_S since the last non-null p; end_s
    is that row's t_s, or the period's last t_s. A null p, a null possession_team or a
    null ball_state don't start one, and short stretches of them don't end one (05: vision
    gaps hold the meter, loose balls before a shot don't split an alarm), but a long
    cutaway can't keep an alarm alive.
    Needs match_id, period, t_s, p, possession_team, ball_state.
    """
    df = pred.select("match_id", "period", "t_s", "p", "possession_team", "ball_state").sort(
        "match_id", "period", "t_s"
    )
    match, period, t = df["match_id"].to_list(), df["period"].to_list(), df["t_s"].to_list()
    p, team, state = df["p"].to_list(), df["possession_team"].to_list(), df["ball_state"].to_list()
    off = off_ratio * tau
    out: list[tuple] = []
    active = None  # (match, period, team, start_s)
    last_p = 0.0  # t_s of the active alarm's last non-null p
    for i in range(len(t)):
        if active and (match[i], period[i]) != active[:2]:
            out.append((*active, t[i - 1], "period_end"))
            active = None
        if active:
            if state[i] == "dead":
                why = "dead"
            elif team[i] is not None and team[i] != active[2]:
                why = "possession"
            elif p[i] is not None and p[i] < off:
                why = "p"
            elif p[i] is None and t[i] - last_p > MAX_NULL_HOLD_S:
                why = "stale"
            else:
                why = None
            if why:
                out.append((*active, t[i], why))
                active = None
            elif p[i] is not None:
                last_p = t[i]
        if (
            not active
            and p[i] is not None
            and p[i] > tau
            and team[i] is not None
            and state[i] != "dead"
        ):
            active = (match[i], period[i], team[i], t[i])
            last_p = t[i]
    if active:
        out.append((*active, t[-1], "period_end"))
    return pl.DataFrame(
        out,
        schema={
            "match_id": pl.String,
            "period": pl.Int64,
            "team": pl.String,
            "start_s": pl.Float64,
            "end_s": pl.Float64,
            "ended_by": pl.String,
        },
        orient="row",
    ).with_row_index("alarm_id")


def shots_table(events: pl.DataFrame, frames: pl.DataFrame, match_id: str) -> pl.DataFrame:
    """match_id, period, t_s, team, open_play for every shot with a tracked frame.

    Open-play shots are the ones scored for misses and lead time. Set-play shots only
    stop an alarm from counting as false: the model predicts through set plays live,
    and warning before a corner header isn't a false alarm.
    """
    return (
        label_events(events, frames)
        .filter(pl.col("kind") != "goal", pl.col("period").is_not_null())
        .select(
            match_id=pl.lit(match_id),
            period=pl.col("period").cast(pl.Int64),
            t_s=pl.col("timestamp_s"),
            team="team",
            open_play=pl.col("kind") == "shot",
        )
    )


def _covering(shots: pl.DataFrame, al: pl.DataFrame, grace_s: float) -> pl.DataFrame:
    """Every (shot, alarm) pair where the alarm covers the shot: same team, started
    strictly before it, and the shot is no later than end_s + grace_s."""
    return (
        shots.join(al, on=["match_id", "period", "team"], how="inner")
        .filter(pl.col("start_s") < pl.col("t_s"), pl.col("t_s") <= pl.col("end_s") + grace_s)
        .with_columns(in_grace=pl.col("t_s") > pl.col("end_s"))
    )


def score_alarms(
    pred: pl.DataFrame, shots: pl.DataFrame, tau: float, grace_s: float = GRACE_S
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(open-play shots with alarm_id and lead_s, alarms with a `true` flag).

    A shot takes the alarm active at it, else one that ended within grace_s; lead_s is
    t_s - that alarm's start, null for a miss. One alarm can lead several shots
    (rebounds). An alarm is true if it covers any shot by its team, set play included.
    Only shots from matches in `pred` count, so one shots table can serve every fold.
    """
    al = alarms(pred, tau)
    shots = shots.filter(pl.col("match_id").is_in(pred["match_id"].unique().implode()))
    shots = shots.with_row_index("shot_id")
    pairs = _covering(shots, al, grace_s)
    best = (
        pairs.filter("open_play")
        .sort("in_grace", pl.col("start_s"), descending=[False, True])
        .group_by("shot_id", maintain_order=True)
        .first()
        .select("shot_id", "alarm_id", lead_s=pl.col("t_s") - pl.col("start_s"))
    )
    scored = shots.filter("open_play").join(best, on="shot_id", how="left").drop("shot_id")
    true_ids = pairs["alarm_id"].unique()
    return scored, al.with_columns(true=pl.col("alarm_id").is_in(true_ids.implode()))


def alarm_summary(
    pred: pl.DataFrame, shots: pl.DataFrame, tau: float, grace_s: float = GRACE_S
) -> dict:
    """Lead time, misses and false alarms at one tau. False alarms are per match in
    `pred`, including matches that never alarmed."""
    scored, al = score_alarms(pred, shots, tau, grace_s)
    n_shots, n_matches = scored.height, pred["match_id"].n_unique()
    missed = scored["lead_s"].null_count()
    false = al.filter(~pl.col("true")).height
    lead = scored["lead_s"].drop_nulls()
    return {
        "tau": tau,
        "shots": n_shots,
        "missed": missed,
        "miss_rate": missed / n_shots if n_shots else float("nan"),
        "lead_s_median": float(lead.median()) if len(lead) else float("nan"),
        "alarms": al.height,
        "false_alarms": false,
        "false_alarms_per_match": false / n_matches if n_matches else float("nan"),
    }


def tau_sweep(
    pred: pl.DataFrame, shots: pl.DataFrame, taus, grace_s: float = GRACE_S
) -> pl.DataFrame:
    """The lead time / miss / false alarm trade-off across tau (07). Picking tau is the
    caller's job, on training matches only."""
    return pl.DataFrame([alarm_summary(pred, shots, float(t), grace_s) for t in taus])


def tau_candidates(p: np.ndarray, n: int = 60) -> np.ndarray:
    """High quantiles of p, from the top 30% down to the top 0.01%: a calibrated model at
    a ~2% base rate rarely goes high, so a fixed grid would miss where its tau lives."""
    p = p[~np.isnan(p)]
    if not len(p):
        return np.array([])
    return np.unique(np.quantile(p, 1 - np.geomspace(0.3, 1e-4, n)))


def choose_tau(
    pred: pl.DataFrame,
    shots: pl.DataFrame,
    max_false_per_match: float = MAX_FALSE_PER_MATCH,
    grace_s: float = GRACE_S,
    candidates=None,
) -> dict:
    """07's rule: the lowest miss rate with at most max_false_per_match false alarms per
    match, ties to the higher tau. If nothing meets the budget, the fewest false alarms
    (ties to the higher tau) with met = False. Run it on inner validation matches only.

    Returns the chosen row of the sweep plus `met`.
    """
    if candidates is None:
        candidates = tau_candidates(pred["p"].cast(pl.Float64).to_numpy())
    if not len(candidates):
        raise ValueError("no non-null p to choose tau from")
    sweep = tau_sweep(pred, shots, candidates, grace_s)
    ok = sweep.filter(pl.col("false_alarms_per_match") <= max_false_per_match)
    if ok.height:
        best = ok.sort(["miss_rate", "tau"], descending=[False, True]).row(0, named=True)
        return {**best, "met": True}
    best = sweep.sort(["false_alarms_per_match", "tau"], descending=[False, True]).row(
        0, named=True
    )
    return {**best, "met": False}
