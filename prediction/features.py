"""Hand features on the 10 Hz grid (05, "Models").

Coordinates are x_att/y_att from objects_10hz: flipped so possession_team attacks +x,
so the target goal is always at (+52.5, 0) on the 105 x 68 pitch (02).
"""

from pathlib import Path

import numpy as np
import polars as pl

GOAL_X = 52.5
POST_Y = 3.66  # half of the 7.32 m goal


def tenths(col: str = "t_s") -> pl.Expr:
    """Join key for grid rows: t_s in whole tenths, never float equality."""
    return (pl.col(col) * 10).round().cast(pl.Int64)


def goal_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.hypot(GOAL_X - x, y)


def goal_angle(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Angle the goal mouth subtends at (x, y), in radians: pi on the goal line between
    the posts, ~0.64 at the penalty spot, 0 anywhere on the goal line outside the posts.

    atan2(|cross|, dot) of the vectors to the two posts, so it stays well defined behind
    the goal line and out wide, where the usual arctan forms break or flip sign.
    """
    ax, ay = GOAL_X - x, POST_Y - y
    bx, by = GOAL_X - x, -POST_Y - y
    return np.arctan2(np.abs(ax * by - ay * bx), ax * bx + ay * by)


def ball_features(frames: pl.DataFrame, objects: pl.LazyFrame) -> pl.DataFrame:
    """frames plus ball_x, ball_y (attacking frame), ball_dist, ball_angle, ball_visible.

    One ball per grid row; null features where the grid row has no ball.
    """
    ball = (
        objects.filter(pl.col("object_type") == "ball")
        .select("period", k=tenths(), ball_x="x_att", ball_y="y_att", ball_visible="visible")
        .collect()
    )
    if ball.select("period", "k").is_duplicated().any():
        raise ValueError("more than one ball on a grid row")
    out = frames.with_columns(k=tenths()).join(ball, on=["period", "k"], how="left").drop("k")
    x = out["ball_x"].cast(pl.Float64).to_numpy()
    y = out["ball_y"].cast(pl.Float64).to_numpy()
    return out.with_columns(
        ball_dist=pl.Series(goal_distance(x, y)).fill_nan(None),
        ball_angle=pl.Series(goal_angle(x, y)).fill_nan(None),
    )


FRAME_COLS = [
    "match_id",
    "period",
    "t_s",
    "possession_team",
    "ball_state",
    "eligible",
    "all_estimated",
]


def load_match(match_id: str, processed_dir: Path, horizons) -> pl.DataFrame:
    """One match's grid rows with labels and ball features."""
    d = processed_dir / match_id
    cols = FRAME_COLS + [f"label_{x}_{h}" for h in horizons for x in ("mask", "shot")]
    frames = pl.read_parquet(d / "frames_10hz.parquet", columns=cols)
    return ball_features(frames, pl.scan_parquet(d / "objects_10hz.parquet"))
