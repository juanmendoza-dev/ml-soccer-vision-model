"""Learned possession inputs (03 stage 8, feature contract pfeat-v1).

Per match: the 10 Hz grid (05's rule, rebuilt here from 02), then 53 causal scalars per
grid row from usable native objects (VISIBLE, not interpolated, finite x/y), then the
same 53 at t - 0.5 s and t - 1 s: 159 model columns. Nothing here reads ball height,
provider velocities, player_id, events or labels, and nothing after t enters row t.

Times are integer microseconds for every comparison. A native gap longer than
1.5 / native_fps starts a new feature segment; histories never cross one. Coordinates
are rotated per period so home attacks +X.
"""

import polars as pl

CONTRACT = "pfeat-v1"
US = 1_000_000
GRID_US = 100_000  # 10 Hz
MAX_STALENESS = 1.5  # native intervals
SEG_KEY = 10**13  # key = segment * SEG_KEY + time_us keeps segments apart in one sorted array
PLAYER_TYPES = ("player", "goalkeeper")


def to_us(col: str) -> pl.Expr:
    return (pl.col(col) * US).round().cast(pl.Int64)


def tolerance_us(native_fps: float) -> int:
    if not native_fps or native_fps <= 0:
        raise ValueError(f"native_fps must be a positive declared rate, got {native_fps!r}")
    return round(MAX_STALENESS * US / native_fps)


def usable(objects: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """Balls, players and keepers that are VISIBLE, not interpolated, with finite x/y.
    Referees never count; z, vx/vy and player_id are dropped here."""
    return (
        objects.lazy()
        .filter(
            pl.col("object_type").is_in(["ball", *PLAYER_TYPES]),
            pl.col("visible").fill_null(False),
            ~pl.col("interpolated").fill_null(False),
            pl.col("x").is_finite().fill_null(False),
            pl.col("y").is_finite().fill_null(False),
        )
        .select("frame_id", "object_id", "object_type", "team", "x", "y")
        .collect()
    )


def native(frames: pl.DataFrame, native_fps: float) -> pl.DataFrame:
    """Native frames in (period, timestamp_s, frame_id) order with fi (row index), seg
    (feature segment), ts_us, key and d (+1 when home attacks +x, else -1)."""
    if frames["home_attacks_positive_x"].null_count():
        raise ValueError("home_attacks_positive_x is null on some frames")
    tol = tolerance_us(native_fps)
    return (
        frames.select("frame_id", "period", "timestamp_s", "home_attacks_positive_x")
        .with_columns(ts_us=to_us("timestamp_s"))
        .sort("period", "ts_us", "frame_id")
        .with_columns(
            d=pl.when("home_attacks_positive_x").then(1.0).otherwise(-1.0),
            fi=pl.int_range(pl.len(), dtype=pl.Int64),
            seg=(
                (pl.col("period") != pl.col("period").shift(1))
                | (pl.col("ts_us") - pl.col("ts_us").shift(1) > tol)
            )
            .fill_null(True)
            .cum_sum()
            .cast(pl.Int64),
        )
        .with_columns(key=pl.col("seg") * SEG_KEY + pl.col("ts_us"))
        .drop("home_attacks_positive_x")
    )


def grid(nat: pl.DataFrame, native_fps: float) -> pl.DataFrame:
    """05's grid: per period t = k / 10, the latest native frame at or before t, skipped
    if older than 1.5 native intervals. One row per valid (period, k) with u's fi/seg."""
    tol = tolerance_us(native_fps)
    bounds = nat.group_by("period").agg(first=pl.col("ts_us").min(), last=pl.col("ts_us").max())
    g = (
        bounds.select(
            "period",
            k=pl.int_ranges(
                (pl.col("first") + GRID_US - 1) // GRID_US, pl.col("last") // GRID_US + 1
            ),
        )
        .explode("k", empty_as_null=True)
        .drop_nulls("k")
        .with_columns(pl.col("k").cast(pl.Int64), t_us=pl.col("k").cast(pl.Int64) * GRID_US)
        .sort("period", "t_us")
    )
    g = g.join_asof(
        nat.select("period", "ts_us", "frame_id", "fi", "seg", "d"),
        left_on="t_us",
        right_on="ts_us",
        by="period",
        strategy="backward",
        check_sortedness=False,
    )
    return (
        g.filter(pl.col("t_us") - pl.col("ts_us") <= tol)
        .rename({"ts_us": "u_us"})
        .sort("period", "k")
        .select("period", "k", "t_us", "frame_id", "u_us", "fi", "seg", "d")
    )
