"""Learned possession inputs (03 stage 8, feature contract pfeat-v1).

Per match: the 10 Hz grid (05's rule, rebuilt here from 02), then 53 causal scalars per
grid row from usable native objects (VISIBLE, not interpolated, finite x/y), then the
same 53 at t - 0.5 s and t - 1 s: 159 model columns. Nothing here reads ball height,
provider velocities, player_id, events or labels, and nothing after t enters row t.

Times are integer microseconds for every comparison. A native gap longer than
1.5 / native_fps starts a new feature segment; histories never cross one. Coordinates
are rotated per period so home attacks +X.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

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


def dist(x: str, y: str) -> pl.Expr:
    return ((pl.col("X") - pl.col(x)) ** 2 + (pl.col("Y") - pl.col(y)) ** 2).sqrt()


def frame_features(nat: pl.DataFrame, pls: pl.DataFrame, bl: pl.DataFrame) -> pl.DataFrame:
    """Per native frame u: nearest distance, shape, pressure and view features, the
    team's nearest player to the ball (for the heading angle) and players_n (every usable
    player, unknown team included: the fallback condition, not a model input)."""
    pb = pls.join(bl.select("fi", "bX", "bY"), on="fi", how="left").with_columns(
        dist=dist("bX", "bY")
    )
    out = (
        nat.select("fi")
        .join(bl.select("fi", ball=pl.lit(True)), on="fi", how="left")
        .with_columns(pl.col("ball").fill_null(False))
    )
    for team, s in (("home", 1.0), ("away", -1.0)):
        outfield_x = pl.col("X").filter("outfield")
        by_ball = pl.col("dist"), pl.col("object_id")
        agg = (
            pb.filter(pl.col("team") == team)
            .group_by("fi")
            .agg(
                pl.col("dist").min().alias(f"near_{team}_m"),
                pl.col("X").mean().alias(f"{team}_centroid_x"),
                (pl.col("outfield") & (s * pl.col("X") > 0))
                .sum()
                .cast(pl.Float64)
                .alias(f"{team}_past_halfway"),
                outfield_x.max().alias(f"{team}_max_x"),
                outfield_x.min().alias(f"{team}_min_x"),
                pl.len().cast(pl.Float64).alias(f"{team}_visible_n"),
                (pl.col("dist") <= 5).sum().cast(pl.Float64).alias(f"{team}_within5"),
                (pl.col("dist") <= 10).sum().cast(pl.Float64).alias(f"{team}_within10"),
                pl.col("X").sort_by(*by_ball).first().alias(f"_{team}_nx"),
                pl.col("Y").sort_by(*by_ball).first().alias(f"_{team}_ny"),
            )
        )
        out = out.join(agg, on="fi", how="left").with_columns(
            pl.col(f"{team}_past_halfway", f"{team}_visible_n").fill_null(0),
            *(
                pl.when("ball").then(pl.col(c).fill_null(0))
                for c in (f"{team}_within5", f"{team}_within10")
            ),
        )
    view = (
        pl.concat([pls.select("fi", "X"), bl.select("fi", X="bX")])
        .group_by("fi")
        .agg(view_min_x=pl.col("X").min(), view_max_x=pl.col("X").max())
    )
    count = pls.group_by("fi").agg(players_n=pl.len())
    return (
        out.join(view, on="fi", how="left")
        .join(count, on="fi", how="left")
        .with_columns(pl.col("players_n").fill_null(0))
        .drop("ball")
    )


HEAD_S = (("03", 0.3), ("05", 0.5))
STILL_SPEED = 0.5  # still_speed: under it the direction is jitter


def heads(rows: pl.DataFrame, pls: pl.DataFrame) -> pl.DataFrame:
    """heads_<team>_03_m / _05_m and heads_<team>_angle per grid row. rows: fi, ball_seen,
    ball_x/y, ball_vx_02/vy_02 and the frame's _<team>_nx/_ny, in grid order."""
    r = rows.with_row_index("row")
    moving = r.filter(pl.col("ball_seen") == 1, pl.col("ball_vx_02").is_not_nan())
    out = r.select("row")
    for name, h in HEAD_S:
        pt = moving.select(
            "row",
            "fi",
            hx=pl.col("ball_x") + h * pl.col("ball_vx_02"),
            hy=pl.col("ball_y") + h * pl.col("ball_vy_02"),
        )
        d = (
            pt.join(pls.select("fi", "team", "X", "Y"), on="fi")
            .filter(pl.col("team").is_in(["home", "away"]))
            .with_columns(dist=dist("hx", "hy"))
            .group_by("row")
            .agg(
                *(
                    pl.col("dist").filter(pl.col("team") == t).min().alias(f"heads_{t}_{name}_m")
                    for t in ("home", "away")
                )
            )
        )
        out = out.join(d, on="row", how="left")
    vx, vy = pl.col("ball_vx_02"), pl.col("ball_vy_02")
    speed = (vx**2 + vy**2).sqrt()
    for t in ("home", "away"):
        dx, dy = pl.col(f"_{t}_nx") - pl.col("ball_x"), pl.col(f"_{t}_ny") - pl.col("ball_y")
        cos = (vx * dx + vy * dy) / (speed * (dx**2 + dy**2).sqrt())
        r = r.with_columns(
            pl.when(pl.col("ball_seen") == 1, vx.is_not_nan(), speed >= STILL_SPEED)
            .then(cos.clip(-1.0, 1.0).arccos())
            .alias(f"heads_{t}_angle")
        )
    return (
        out.join(r.select("row", "heads_home_angle", "heads_away_angle"), on="row")
        .sort("row")
        .drop("row")
    )


SHARE_W = (("05", 500_000), ("1", 1_000_000))


def contacts(nat: pl.DataFrame, pls: pl.DataFrame, bl: pl.DataFrame) -> pl.DataFrame:
    """Per native frame: each team's weight as nearest to the ball (any distance, and
    within 1.5 m), ties within 1e-9 m split across the tied team values (unknown's share
    goes to nobody), and the last-contact team code where the nearest is within 1.5 m
    (0 for an unknown team or a tie between teams). Frames without ball or players get
    weight 0 and no contact."""
    near = (
        pls.join(bl.select("fi", "bX", "bY"), on="fi")
        .with_columns(dist=dist("bX", "bY"), label=pl.col("team").fill_null("?"))
        .with_columns(dmin=pl.col("dist").min().over("fi"))
        .filter(pl.col("dist") <= pl.col("dmin") + TIE_M)
        .group_by("fi")
        .agg(dmin=pl.col("dmin").first(), labels=pl.col("label").unique())
        .with_columns(n=pl.col("labels").list.len())
    )
    cols = []
    for team in ("home", "away"):
        w = pl.when(pl.col("labels").list.contains(team)).then(1.0 / pl.col("n")).otherwise(0.0)
        cols += [
            w.alias(f"w_{team}_any"),
            pl.when(pl.col("dmin") <= REACH_M).then(w).otherwise(0.0).alias(f"w_{team}_r15"),
        ]
    near = near.with_columns(
        *cols,
        contact=pl.when(pl.col("dmin") <= REACH_M).then(
            pl.when(pl.col("n") == 1)
            .then(
                pl.col("labels")
                .list.first()
                .replace_strict(CODE, default=0.0, return_dtype=pl.Float64)
            )
            .otherwise(0.0)
        ),
    )
    return (
        nat.select("fi", "seg", "ts_us", "key")
        .join(near.drop("labels", "n", "dmin"), on="fi", how="left")
        .with_columns(pl.col("^w_.*$").fill_null(0.0))
        .sort("fi")
    )


def contact_features(g: pl.DataFrame, cf: pl.DataFrame, native_fps: float) -> pl.DataFrame:
    """nearest_<team>_<window>_<reach> elapsed-time shares over (max(segment start, t - W), t]
    and last_contact_team / last_contact_age_s, per grid row. Each native frame's weights
    hold until the next native frame, at most 1.5 native intervals, and never past t."""
    tol = tolerance_us(native_fps)
    key, ts, seg = (cf[c].to_numpy() for c in ("key", "ts_us", "seg"))
    nxt = np.r_[ts[1:], 0]
    same = np.r_[seg[1:] == seg[:-1], False]
    hold = np.where(same, np.minimum(nxt - ts, tol), tol)
    start = cf.group_by("seg").agg(pl.col("ts_us").min()).sort("seg")
    seg_start = dict(zip(start["seg"].to_list(), start["ts_us"].to_list()))
    gseg, t, fi = (g[c].to_numpy() for c in ("seg", "t_us", "fi"))
    s0 = np.array([seg_start[s] for s in gseg], dtype=np.int64)
    out = {}
    for team in ("home", "away"):
        for wname, w in SHARE_W:
            a = np.maximum(s0, t - w)
            ia = np.searchsorted(key, gseg * SEG_KEY + a, "right") - 1
            dur = t - a
            for reach in ("any", "r15"):
                wt = cf[f"w_{team}_{reach}"].to_numpy()
                cb = np.r_[0.0, np.cumsum(wt * hold)[:-1]]  # integral up to each frame's start
                at_t = cb[fi] + wt[fi] * np.minimum(t - ts[fi], hold[fi])
                at_a = cb[ia] + wt[ia] * np.minimum(a - ts[ia], hold[ia])
                with np.errstate(divide="ignore", invalid="ignore"):
                    share = (at_t - at_a) / dur
                out[f"nearest_{team}_{wname}_{reach}"] = np.where(dur > 0, share, np.nan)
    c = cf.filter(pl.col("contact").is_not_null())
    ckey, cts, cseg, code = (c[x].to_numpy() for x in ("key", "ts_us", "seg", "contact"))
    if not len(ckey):
        out["last_contact_team"] = out["last_contact_age_s"] = np.full(len(g), np.nan)
        return pl.DataFrame(out)
    i = np.searchsorted(ckey, key[fi], "right") - 1
    ic = np.clip(i, 0, None)
    have = (i >= 0) & (cseg[ic] == gseg)
    out["last_contact_team"] = np.where(have, code[ic], np.nan)
    out["last_contact_age_s"] = np.where(have, (t - cts[ic]) / US, np.nan)
    return pl.DataFrame(out)


TRACK_W_US = 200_000
TRACK_MIN_US = 100_000


def track_velocity(nat: pl.DataFrame, pls: pl.DataFrame) -> pl.DataFrame:
    """<team>_mean_vx_own per native frame u: the mean of -s_team * VX over the team's
    players with a velocity. A track's VX runs from its first usable sighting in
    [u - 0.2 s, u] to u, needs a span of at least 0.1 s, the same team throughout and a
    sighting on every native frame in between (no gap, same segment)."""
    ts, key, seg = (nat[c].to_numpy() for c in ("ts_us", "key", "seg"))
    start = nat.select(
        "fi", f0=pl.Series(np.searchsorted(key, seg * SEG_KEY + ts - TRACK_W_US, "left"))
    )
    by = "object_id"
    team = pl.col("team").fill_null("?")
    p = (
        pls.sort(by, "fi")
        .with_columns(
            brk=(
                (pl.col("fi") - pl.col("fi").shift(1).over(by) != 1)
                | (pl.col("seg") != pl.col("seg").shift(1).over(by))
                | (team != team.shift(1).over(by))
            ).fill_null(True)
        )
        .with_columns(run_start=pl.when("brk").then("fi").forward_fill().over(by))
        .join(start, on="fi")
        .with_columns(first=pl.max_horizontal("run_start", "f0"))
    )
    v = (
        p.join(p.select(by, first="fi", X_first="X", ts_first="ts_us"), on=[by, "first"])
        .filter(pl.col("ts_us") - pl.col("ts_first") >= TRACK_MIN_US)
        .with_columns(
            vx=(pl.col("X") - pl.col("X_first")) / (pl.col("ts_us") - pl.col("ts_first")) * US
        )
    )
    out = nat.select("fi")
    for name, sgn in (("home", 1.0), ("away", -1.0)):
        m = (
            v.filter(pl.col("team") == name)
            .group_by("fi")
            .agg((-sgn * pl.col("vx")).mean().alias(f"{name}_mean_vx_own"))
        )
        out = out.join(m, on="fi", how="left")
    return out


SHAPES = ("centroid_x", "mean_vx_own", "past_halfway", "max_x", "min_x", "visible_n")
BLOCK = [
    "rule_team", "rule_carrier_age_s", "rule_candidate_team",
    "ball_seen", "ball_age_s", "ball_x", "ball_y",
    "ball_vx_02", "ball_vy_02", "ball_vx_1", "ball_vy_1",
    "near_home_m", "near_away_m",
    *(f"nearest_{t}_{w}_{r}" for t in ("home", "away") for w, _ in SHARE_W for r in ("any", "r15")),
    "home_centroid_x", "away_centroid_x",
    "home_mean_vx_own", "away_mean_vx_own",
    "home_past_halfway", "away_past_halfway",
    "home_max_x", "home_min_x", "away_max_x", "away_min_x",
    "home_visible_n", "away_visible_n",
    *(f"diff_{s}" for s in SHAPES),
    "heads_home_03_m", "heads_home_05_m", "heads_away_03_m", "heads_away_05_m",
    "heads_home_angle", "heads_away_angle",
    "last_contact_team", "last_contact_age_s",
    "home_within5", "home_within10", "away_within5", "away_within10",
    "view_min_x", "view_max_x",
]  # fmt: skip
LAGS = (("lag05", 5), ("lag1", 10))  # grid ticks back
COLUMNS = BLOCK + [f"{c}_{name}" for name, _ in LAGS for c in BLOCK]
KEYS = ["period", "k", "t_s", "frame_id"]


def extract(
    frames: pl.DataFrame, objects: pl.DataFrame | pl.LazyFrame, native_fps: float
) -> pl.DataFrame:
    """One match's pfeat-v1 inputs: grid keys (period, k, t_s, frame_id), players_n (the
    fallback condition, not a model input) and the 159 COLUMNS as Float32, NaN for missing."""
    nat = native(frames, native_fps)
    use = usable(objects)
    g = grid(nat, native_fps)
    pls, bl = players(nat, use), balls(nat, use)
    per = (
        rule_frames(frames, nat, use, pls, bl)
        .join(frame_features(nat, pls, bl), on="fi")
        .join(track_velocity(nat, pls), on="fi")
    )
    rows = (
        g.hstack(ball_features(g, bl))
        .hstack(contact_features(g, contacts(nat, pls, bl), native_fps))
        .join(per, on="fi", how="left", maintain_order="left")
    )
    rows = rows.hstack(heads(rows, pls)).with_columns(
        (pl.col(f"home_{s}") - pl.col(f"away_{s}")).alias(f"diff_{s}") for s in SHAPES
    )
    cur = rows.with_columns(
        *(pl.col(c).cast(pl.Float64).fill_null(np.nan) for c in BLOCK),
        (pl.col("t_us") / US).alias("t_s"),
    )
    run = (
        (pl.col("k") - pl.col("k").shift(1) != 1)
        | (pl.col("seg") != pl.col("seg").shift(1))
        | (pl.col("period") != pl.col("period").shift(1))
    ).fill_null(True)
    cur = cur.with_columns(run_k=pl.when(run).then("k").forward_fill())
    lagged = []
    for name, n in LAGS:
        ok = pl.col("k") - n >= pl.col("run_k")
        lagged += [
            pl.when(ok).then(pl.col(c).shift(n)).otherwise(np.nan).alias(f"{c}_{name}")
            for c in BLOCK
        ]
    return cur.with_columns(lagged).select(
        *KEYS, "players_n", *(pl.col(c).cast(pl.Float32) for c in COLUMNS)
    )


def _block_mirror(sfx: str) -> list[pl.Expr]:
    def c(name):
        return pl.col(name + sfx)

    def neg(e):
        return 0.0 - e  # keeps 0 at +0.0

    exprs = [
        neg(c(n)).alias(n + sfx) for n in ("rule_team", "rule_candidate_team", "last_contact_team")
    ]
    exprs += [neg(c(n)).alias(n + sfx) for n in ("ball_x", "ball_vx_02", "ball_vx_1")]
    pairs = [("near_home_m", "near_away_m")]
    pairs += [
        (f"nearest_home_{w}_{r}", f"nearest_away_{w}_{r}")
        for w, _ in SHARE_W
        for r in ("any", "r15")
    ]
    pairs += [
        (f"home_{s}", f"away_{s}")
        for s in ("mean_vx_own", "past_halfway", "visible_n", "within5", "within10")
    ]
    pairs += [(f"heads_home_{s}", f"heads_away_{s}") for s in ("03_m", "05_m", "angle")]
    for h, a in pairs:
        exprs += [c(a).alias(h + sfx), c(h).alias(a + sfx)]
    exprs += [
        neg(c("away_centroid_x")).alias("home_centroid_x" + sfx),
        neg(c("home_centroid_x")).alias("away_centroid_x" + sfx),
        neg(c("away_min_x")).alias("home_max_x" + sfx),
        neg(c("away_max_x")).alias("home_min_x" + sfx),
        neg(c("home_min_x")).alias("away_max_x" + sfx),
        neg(c("home_max_x")).alias("away_min_x" + sfx),
        neg(c("view_max_x")).alias("view_min_x" + sfx),
        neg(c("view_min_x")).alias("view_max_x" + sfx),
    ]
    return exprs


def mirror(df: pl.DataFrame) -> pl.DataFrame:
    """03's transform M on every block: negate X and VX, swap home and away (team codes
    negate, 0 stays 0), then recompute the home-minus-away differences. Ages, ball_seen,
    Y and VY stay. Other columns pass through."""
    for sfx in ("", *(f"_{name}" for name, _ in LAGS)):
        df = df.with_columns(_block_mirror(sfx)).with_columns(
            (pl.col(f"home_{s}{sfx}") - pl.col(f"away_{s}{sfx}")).alias(f"diff_{s}{sfx}")
            for s in SHAPES
        )
    return df


GAMESTATE_DIR = Path("data/gamestate")
CACHE_DIR = Path("data/vision_cache")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extractor_hash(path: Path = Path(__file__)) -> str:
    """SHA256 of this module's source with line endings normalized, so a CRLF checkout
    of the same commit hashes the same."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def grid_hash(keys: pl.DataFrame) -> str:
    """SHA256 of the grid keys (period, k, frame_id) in order."""
    h = hashlib.sha256()
    for c in ("period", "k", "frame_id"):
        h.update(keys[c].cast(pl.Int64).to_numpy().tobytes())
    return h.hexdigest()


def native_inputs(match_id: str, gamestate_dir: Path) -> dict:
    d = gamestate_dir / match_id
    return {
        "frames_sha256": sha256_file(d / "frames.parquet"),
        "objects_sha256": sha256_file(d / "objects.parquet"),
        "native_fps": pl.read_parquet(d / "match.parquet")["native_fps"][0],
    }


def cache_paths(match_id: str, cache_dir: Path) -> tuple[Path, Path]:
    stem = cache_dir / match_id / f"possession_inputs_{CONTRACT}"
    return stem.with_suffix(".parquet"), stem.with_suffix(".json")


def build(
    match_id: str, gamestate_dir: Path = GAMESTATE_DIR, cache_dir: Path = CACHE_DIR
) -> pl.DataFrame:
    """Extract one match and write possession_inputs_pfeat-v1.parquet with its .json
    provenance (native input, grid and extractor hashes)."""
    d = gamestate_dir / match_id
    meta = native_inputs(match_id, gamestate_dir)
    frames = pl.read_parquet(d / "frames.parquet")
    f = extract(frames, pl.scan_parquet(d / "objects.parquet"), meta["native_fps"])
    path, side = cache_paths(match_id, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    f.write_parquet(path)
    meta |= {
        "contract": CONTRACT,
        "match_id": match_id,
        "grid_sha256": grid_hash(f),
        "extractor_sha256": extractor_hash(),
        "output_sha256": sha256_file(path),
        "rows": f.height,
        "columns": COLUMNS,
    }
    side.write_text(json.dumps(meta, indent=2))
    return f


def load(
    match_id: str, gamestate_dir: Path = GAMESTATE_DIR, cache_dir: Path = CACHE_DIR
) -> pl.DataFrame:
    """The cached inputs, refused unless every recorded hash still matches: the native
    frames/objects, the grid rebuilt from those frames, the extractor and the file itself."""
    path, side = cache_paths(match_id, cache_dir)
    if not path.exists() or not side.exists():
        raise FileNotFoundError(
            f"{path}: no cached {CONTRACT} inputs; run python -m vision.possession_features"
        )
    meta = json.loads(side.read_text())
    now = native_inputs(match_id, gamestate_dir)
    frames = pl.read_parquet(gamestate_dir / match_id / "frames.parquet")
    now["grid_sha256"] = grid_hash(grid(native(frames, now["native_fps"]), now["native_fps"]))
    now["extractor_sha256"] = extractor_hash()
    now["output_sha256"] = sha256_file(path)
    now["contract"], now["columns"] = CONTRACT, COLUMNS
    stale = [k for k, v in now.items() if meta.get(k) != v]
    if stale:
        raise ValueError(f"{path}: stale cache ({', '.join(stale)} changed); rebuild it")
    return pl.read_parquet(path)


def pff_games(gamestate_dir: Path) -> list[str]:
    return sorted(
        p.parent.name
        for p in gamestate_dir.glob("*/match.parquet")
        if pl.read_parquet(p)["source"][0] == "pff"
    )


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=f"Build {CONTRACT} possession inputs per match.")
    ap.add_argument("--games", nargs="*", help="match ids (default: every PFF game)")
    ap.add_argument("--gamestate", type=Path, default=GAMESTATE_DIR)
    ap.add_argument("--cache", type=Path, default=CACHE_DIR)
    a = ap.parse_args(argv)
    games = a.games or pff_games(a.gamestate)
    total = time.perf_counter()
    for m in games:
        t0 = time.perf_counter()
        f = build(m, a.gamestate, a.cache)
        print(f"{m}: {f.height} rows, {time.perf_counter() - t0:.1f} s", flush=True)
    print(f"{len(games)} games, {time.perf_counter() - total:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
