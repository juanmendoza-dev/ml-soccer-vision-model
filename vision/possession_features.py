"""Learned possession inputs (03 stage 8, feature contract pfeat-v1).

Per match: the 10 Hz grid (05's rule, rebuilt here from 02), then 53 causal scalars per
grid row from usable native objects (VISIBLE, not interpolated, finite x/y), then the
same 53 at t - 0.5 s and t - 1 s: 159 model columns. Nothing here reads ball height,
provider velocities, player_id, events or labels, and nothing after t enters row t.

Times are integer microseconds for every comparison. A native gap longer than
1.5 / native_fps starts a new feature segment; histories never cross one. Coordinates
are rotated per period so home attacks +X.
"""

import numpy as np
import polars as pl

from vision.state import StateConfig, infer, is_out

CONTRACT = "pfeat-v1"
US = 1_000_000
GRID_US = 100_000  # 10 Hz
MAX_STALENESS = 1.5  # native intervals
SEG_KEY = 10**13  # key = segment * SEG_KEY + time_us keeps segments apart in one sorted array
PLAYER_TYPES = ("player", "goalkeeper")
REACH_M = 1.5  # carrier_radius_m: candidate, r15 shares, last contact
TIE_M = 1e-9
BALL_HOLD_US = 1_000_000  # ball position held at most 1 s
VEL_GAP_US = 500_000  # carrier_gap_s: no longer gap between sightings inside a velocity span
CODE = {"home": 1.0, "away": -1.0}


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


def players(nat: pl.DataFrame, use: pl.DataFrame) -> pl.DataFrame:
    """Usable players and keepers per native frame, X/Y in the home frame."""
    return (
        use.filter(pl.col("object_type").is_in(PLAYER_TYPES))
        .join(nat.select("frame_id", "fi", "seg", "ts_us", "d"), on="frame_id")
        .with_columns(
            X=pl.col("d") * pl.col("x"),
            Y=pl.col("d") * pl.col("y"),
            outfield=pl.col("object_type") == "player",
        )
        .select("fi", "seg", "ts_us", "object_id", "team", "outfield", "X", "Y")
    )


def balls(nat: pl.DataFrame, use: pl.DataFrame) -> pl.DataFrame:
    """The usable ball per native frame (lowest object_id, as infer picks), in key order.
    x/y stay in schema coordinates for the rule's out check."""
    b = (
        use.filter(pl.col("object_type") == "ball")
        .sort("frame_id", "object_id")
        .unique("frame_id", keep="first", maintain_order=True)
        .select("frame_id", "x", "y")
    )
    return (
        nat.join(b, on="frame_id")
        .with_columns(bX=pl.col("d") * pl.col("x"), bY=pl.col("d") * pl.col("y"))
        .select("fi", "seg", "ts_us", "key", "x", "y", "bX", "bY")
        .sort("key")
    )


def team_code(col: str) -> pl.Expr:
    """home +1, away -1, 0 for a present object with no team."""
    return pl.col(col).replace_strict(CODE, default=0.0, return_dtype=pl.Float64)


def rule_frames(
    frames: pl.DataFrame, nat: pl.DataFrame, use: pl.DataFrame, pls: pl.DataFrame, bl: pl.DataFrame
) -> pl.DataFrame:
    """The 2D rule (03): stage 8's default rule on usable objects with every z null.
    Per native frame: rule_team, rule_carrier_age_s and rule_candidate_team (the current
    candidate within 1.5 m, by distance then object_id, none while the ball is out or
    missing, so a stale candidate never shows)."""
    flat = use.with_columns(z=pl.lit(None, pl.Float64), visible=pl.lit(True))
    state = infer(frames, flat, StateConfig())
    rule = (
        nat.select("frame_id", "fi", "period", "ts_us")
        .join(state.select("frame_id", "possession_team", "ball_carrier_id"), on="frame_id")
        .sort("fi")
        .with_columns(
            rule_team=pl.when(pl.col("possession_team").is_not_null()).then(
                team_code("possession_team")
            ),
            rule_carrier_age_s=(
                pl.col("ts_us")
                - pl.when(pl.col("ball_carrier_id").is_not_null())
                .then("ts_us")
                .forward_fill()
                .over("period")
            )
            / US,
        )
    )
    out = np.array(
        [is_out(x, y, StateConfig().out_margin_m) for x, y in zip(bl["x"], bl["y"])], dtype=bool
    )
    cand = (
        pls.join(bl.select("fi", "bX", "bY").with_columns(out=pl.Series(out)), on="fi")
        .filter(~pl.col("out"))
        .with_columns(
            dist=((pl.col("X") - pl.col("bX")) ** 2 + (pl.col("Y") - pl.col("bY")) ** 2).sqrt()
        )
        .sort("fi", "dist", "object_id")
        .unique("fi", keep="first", maintain_order=True)
        .filter(pl.col("dist") <= REACH_M)
        .select("fi", rule_candidate_team=team_code("team"))
    )
    return rule.join(cand, on="fi", how="left").select(
        "fi", "rule_team", "rule_carrier_age_s", "rule_candidate_team"
    )


def ball_features(g: pl.DataFrame, bl: pl.DataFrame) -> pl.DataFrame:
    """ball_seen, ball_age_s, ball_x/y and the 0.2 s / 1 s ball velocities at each grid
    row, from usable sightings in the row's feature segment at or before t."""
    names = ["ball_age_s", "ball_x", "ball_y", "ball_vx_02", "ball_vy_02", "ball_vx_1", "ball_vy_1"]
    if not bl.height:
        return pl.DataFrame(
            {"ball_seen": np.zeros(g.height)} | {c: np.full(g.height, np.nan) for c in names}
        )
    bkey, bts, bseg = (bl[c].to_numpy() for c in ("key", "ts_us", "seg"))
    bX, bY = bl["bX"].to_numpy(), bl["bY"].to_numpy()
    cg = np.cumsum(np.r_[False, np.diff(bts) > VEL_GAP_US])
    seg, t, u = (g[c].to_numpy() for c in ("seg", "t_us", "u_us"))
    i = np.searchsorted(bkey, seg * SEG_KEY + t, "right") - 1
    ic = np.clip(i, 0, None)
    have = (i >= 0) & (bseg[ic] == seg)
    age = np.where(have, (t - bts[ic]) / US, np.nan)
    seen = have & (bts[ic] == u)
    held = have & (t - bts[ic] <= BALL_HOLD_US)
    cols = {
        "ball_seen": seen.astype(np.float64),
        "ball_age_s": age,
        "ball_x": np.where(held, bX[ic], np.nan),
        "ball_y": np.where(held, bY[ic], np.nan),
    }
    for name, w in (("02", 200_000), ("1", 1_000_000)):
        first = np.clip(np.searchsorted(bkey, seg * SEG_KEY + t - w, "left"), 0, len(bkey) - 1)
        span = bts[ic] - bts[first]
        ok = seen & (first <= ic) & (span >= w // 2) & (cg[ic] - cg[first] == 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            cols[f"ball_vx_{name}"] = np.where(ok, (bX[ic] - bX[first]) / (span / US), np.nan)
            cols[f"ball_vy_{name}"] = np.where(ok, (bY[ic] - bY[first]) / (span / US), np.nan)
    return pl.DataFrame(cols)
