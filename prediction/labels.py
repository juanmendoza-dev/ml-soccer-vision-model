"""shot_within_H / goal_within_H labels and the set-play mask on the 10 Hz grid (05).

These are the only columns that look past t, so they're all named `label_*`.
"""

import numpy as np
import polars as pl

US = 1_000_000
HORIZONS = {"h5": 5, "h3": 3}  # seconds
GOAL_OUTCOMES = ("goal", "own_goal")  # own goals carry the scoring team in `team` (02)
KINDS = ("shot", "goal", "set_play")


def to_us(col: str) -> pl.Expr:
    return (pl.col(col) * US).round().cast(pl.Int64)


def label_events(events: pl.DataFrame, frames: pl.DataFrame) -> pl.DataFrame:
    """Shot and goal rows with their period, time (us) and kind: shot, goal or set_play.

    set_play = a shot or goal that isn't open play; it masks windows instead of
    labelling them. A null set_piece is unknown, so not open play.
    """
    open_play = (pl.col("set_piece") == "open_play").fill_null(False) & ~pl.col(
        "set_play_phase"
    ).fill_null(False)
    is_shot = pl.col("event_type") == "shot"
    is_goal = (pl.col("event_type") == "goal") & pl.col("outcome").is_in(GOAL_OUTCOMES)
    return (
        events.filter(is_shot | is_goal)
        .join(frames.select("frame_id", "period", "timestamp_s"), on="frame_id", how="left")
        .with_columns(
            ev_us=to_us("timestamp_s"),
            kind=pl.when(~open_play)
            .then(pl.lit("set_play"))
            .when(is_shot)
            .then(pl.lit("shot"))
            .otherwise(pl.lit("goal")),
        )
    )


def count_in_window(ev_us: np.ndarray, t_us: np.ndarray, h_us: int) -> np.ndarray:
    """Events in (t, t+h] for each t; ev_us sorted."""
    right = np.searchsorted(ev_us, t_us + h_us, side="right")
    return right - np.searchsorted(ev_us, t_us, side="right")


def add_labels(grid: pl.DataFrame, lev: pl.DataFrame) -> pl.DataFrame:
    """label_mask_*, label_shot_*, label_goal_* per horizon; labels are null where masked.

    `grid` needs grid_us, period, possession_team, eligible and set_play_phase. Only
    events by the team in possession at t count. A frame is masked if it's in a set-play
    phase itself (02, null = not) or a set-play shot/goal falls in its window.
    """
    n = grid.height
    counts = {(k, h): np.zeros(n, dtype=np.int64) for k in KINDS for h in HORIZONS}
    t_us = grid["grid_us"].to_numpy()
    period = grid["period"].to_numpy()
    team = grid["possession_team"].to_numpy()
    for (p, tm), g in lev.filter(pl.col("period").is_not_null()).group_by("period", "team"):
        rows = np.flatnonzero((period == p) & (team == tm))
        if not rows.size:
            continue
        for k in KINDS:
            ev = np.sort(g.filter(pl.col("kind") == k)["ev_us"].to_numpy())
            for h, secs in HORIZONS.items():
                counts[(k, h)][rows] = count_in_window(ev, t_us[rows], secs * US)
    open_play = grid["eligible"].to_numpy() & ~grid["set_play_phase"].fill_null(False).to_numpy()
    raw = []
    for h in HORIZONS:
        raw.append(pl.Series(f"label_mask_{h}", open_play & (counts[("set_play", h)] == 0)))
        raw += [pl.Series(f"label_{k}_{h}", counts[(k, h)] > 0) for k in ("shot", "goal")]
    return grid.with_columns(raw).with_columns(
        pl.when(pl.col(f"label_mask_{h}")).then(pl.col(f"label_{k}_{h}")).alias(f"label_{k}_{h}")
        for h in HORIZONS
        for k in ("shot", "goal")
    )


def shots_without_positive(frames10: pl.DataFrame, lev: pl.DataFrame) -> dict[str, list[dict]]:
    """Open-play shots with no positive frame before them, per horizon, with a reason."""
    shots = lev.filter(pl.col("kind") == "shot", pl.col("period").is_not_null())
    f = frames10.with_columns(t_us=to_us("t_s"))
    out = {}
    for h, secs in HORIZONS.items():
        missing = []
        for s in shots.iter_rows(named=True):
            w = f.filter(
                pl.col("period") == s["period"],
                pl.col("t_us") >= s["ev_us"] - secs * US,
                pl.col("t_us") < s["ev_us"],
            )
            if w[f"label_shot_{h}"].fill_null(False).any():
                continue
            ours = w.filter(pl.col("possession_team") == s["team"])
            if w.is_empty():
                reason = "no grid rows before the shot"
            elif ours.is_empty():
                reason = "possession never with the shooting team"
            elif not ours["eligible"].any():
                reason = "ball dead while the shooting team had it"
            else:
                reason = "masked by a set-play shot or goal"
            missing.append(
                {
                    "frame_id": s["frame_id"],
                    "period": s["period"],
                    "timestamp_s": round(s["timestamp_s"], 3),
                    "team": s["team"],
                    "outcome": s["outcome"],
                    "reason": reason,
                }
            )
        out[h] = missing
    return out
