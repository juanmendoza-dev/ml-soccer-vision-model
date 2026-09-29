"""Hand features on the 10 Hz grid (05, "Models").

Features are in the attacking frame: rotated so possession_team attacks +x, with the
target goal at (+52.5, 0) on the 105 x 68 pitch (02). History (a held ball, velocities,
changes over the last seconds) is rotated with the anchor row's `flipped`, never the
older row's, since the flip jumps 180 degrees when possession changes (05).

Only what a broadcast would show is used (05): VISIBLE players, and with
ball_source="held" a VISIBLE ball carried forward for up to BALL_HOLD_S. PFF's
ESTIMATED positions are filled using later frames (05, Leakage), so they never reach a
feature except through ball_source="raw", kept to measure that leak.
"""

from pathlib import Path

import numpy as np
import polars as pl

from prediction import degrade as degrade_mod

GOAL_X = 52.5
POST_Y = 3.66  # half of the 7.32 m goal
BALL_SOURCES = ("held", "raw")
BALL_HOLD_S = 1.0  # same as vision's ball_max_gap_s (03): after that, no ball
VEL_ROWS = 5  # velocities difference over 0.5 s of grid rows (native 0.2 s is mostly jitter)


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


def toward_goal(x, y, vx, vy) -> np.ndarray:
    """Velocity component toward the goal center, m/s (negative = moving away)."""
    d = goal_distance(x, y)
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((GOAL_X - x) * vx - y * vy) / d


def grid_rows(frames: pl.DataFrame) -> pl.DataFrame:
    """frames sorted by (period, t_s) with r (row number), k (tenths), seg (a run of
    consecutive grid points in one period; history never crosses one) and sign (-1
    where the row is flipped)."""
    return (
        frames.with_columns(k=tenths())
        .sort("period", "k")
        .with_row_index("r")
        .with_columns(
            pl.col("r").cast(pl.Int64),
            seg=((pl.col("period").diff() != 0) | (pl.col("k").diff() != 1))
            .fill_null(True)
            .cum_sum(),
            sign=pl.when(pl.col("flipped")).then(-1.0).otherwise(1.0),
        )
    )


def lag(a: np.ndarray, n: int, seg: np.ndarray) -> np.ndarray:
    """a from n grid rows (n / 10 s) earlier in the same seg, NaN where there's none."""
    out = np.full(len(a), np.nan)
    if n < len(a):
        same = seg[n:] == seg[:-n]
        out[n:] = np.where(same, a[:-n], np.nan)
    return out


def ball_track(g: pl.DataFrame, objects: pl.LazyFrame, ball_source: str) -> dict:
    """Per grid row of g: the ball the features may use, in pitch coordinates (not
    rotated), plus age_s (seconds since that position was observed), fresh (observed on
    this row) and visible (PFF's flag; null with no ball)."""
    if ball_source not in BALL_SOURCES:
        raise ValueError(f"ball_source must be one of {BALL_SOURCES}, got {ball_source!r}")
    ball = (
        objects.filter(pl.col("object_type") == "ball")
        .select("period", "x", "y", "z", "visible", k=tenths())
        .collect()
    )
    if ball.select("period", "k").is_duplicated().any():
        raise ValueError("more than one ball on a grid row")
    b = g.select("period", "k").join(ball, on=["period", "k"], how="left", maintain_order="left")
    x, y, z = (b[c].cast(pl.Float64).fill_null(np.nan).to_numpy() for c in ("x", "y", "z"))
    visible = b["visible"]
    fresh = ~np.isnan(x)
    if ball_source == "held":
        fresh &= visible.fill_null(False).to_numpy()
    k, seg = g["k"].to_numpy(), g["seg"].to_numpy()
    last = np.maximum.accumulate(np.where(fresh, np.arange(len(x)), -1))
    src = np.maximum(last, 0)
    age = (k - k[src]) / 10
    ok = (last >= 0) & (seg[src] == seg)
    ok &= age <= (BALL_HOLD_S if ball_source == "held" else 0) + 1e-9
    pick = lambda a: np.where(ok, a[src], np.nan)
    return {
        "x": pick(x),
        "y": pick(y),
        "z": pick(z),
        "age_s": np.where(ok, age, np.nan),
        "fresh": fresh,
        "visible": visible,
    }


BOX_X = GOAL_X - 16.5
BOX_Y = 20.16
FINAL_THIRD_X = 105 / 6  # 17.5: the last third of the pitch starts here


def in_box(x, y) -> np.ndarray:
    return (x >= BOX_X) & (x <= GOAL_X) & (np.abs(y) <= BOX_Y)


def fresh_velocity(ball: dict, seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ball velocity in pitch coordinates, only between two observed positions VEL_ROWS
    apart: a held one would read as a stop."""
    obs_x = np.where(ball["fresh"], ball["x"], np.nan)
    obs_y = np.where(ball["fresh"], ball["y"], np.nan)
    dt = VEL_ROWS / 10
    return (obs_x - lag(obs_x, VEL_ROWS, seg)) / dt, (obs_y - lag(obs_y, VEL_ROWS, seg)) / dt


def ball_features(g: pl.DataFrame, ball: dict) -> dict[str, np.ndarray]:
    """Ball features per grid row of g, from ball_track's output."""
    sign, seg = g["sign"].to_numpy(), g["seg"].to_numpy()
    x, y = sign * ball["x"], sign * ball["y"]
    dist = goal_distance(x, y)
    vx, vy = fresh_velocity(ball, seg)
    vx, vy = sign * vx, sign * vy

    def dist_change(n: int) -> np.ndarray:
        # the older position rotated with this row's sign, so it's measured to today's goal
        then = goal_distance(sign * lag(ball["x"], n, seg), sign * lag(ball["y"], n, seg))
        return dist - then

    # seconds of the last 5 in which the ball was in this row's attacking final third
    third = np.where(sign > 0, ball["x"] > FINAL_THIRD_X, ball["x"] < -FINAL_THIRD_X)
    third_last_5s = rolling_count(third & ~np.isnan(ball["x"]), 50, seg) / 10

    has = ~np.isnan(x)
    return {
        "ball_x": x,
        "ball_y": y,
        "ball_z": ball["z"],
        "ball_dist": dist,
        "ball_angle": goal_angle(x, y),
        "ball_age_s": ball["age_s"],
        "ball_speed": np.hypot(vx, vy),
        "ball_vgoal": toward_goal(x, y, vx, vy),
        "ball_in_box": np.where(has, in_box(x, y), np.nan),
        "ball_dist_change_1s": dist_change(10),
        "ball_dist_change_3s": dist_change(30),
        "ball_final_third_s": third_last_5s,
    }


def rolling_count(flag: np.ndarray, n: int, seg: np.ndarray) -> np.ndarray:
    """How many of the last n rows (this one included, same seg) have flag set."""
    c = np.concatenate([[0], np.cumsum(flag)])
    r = np.arange(len(flag))
    seg_start = np.maximum.accumulate(np.where(np.diff(seg, prepend=-1) != 0, r, 0))
    start = np.maximum(r - n + 1, seg_start)
    return (c[r + 1] - c[start]).astype(float)


PLAYER_TYPES = ("player", "goalkeeper")
PRESSURE_M = 5.0


def in_lane(px, py, bx, by) -> np.ndarray:
    """(px, py) inside the triangle from the ball to the two posts (the shooting lane)."""

    def cross(ax, ay, cx, cy):
        return (cx - ax) * (py - ay) - (cy - ay) * (px - ax)

    d1 = cross(bx, by, GOAL_X, POST_Y)
    d2 = cross(GOAL_X, POST_Y, GOAL_X, -POST_Y)
    d3 = cross(GOAL_X, -POST_Y, bx, by)
    neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(neg & pos)


def player_features(g: pl.DataFrame, objects: pl.LazyFrame, bf: dict) -> dict[str, np.ndarray]:
    """Features from VISIBLE players on each grid row of g, relative to the ball in bf.
    Attackers are possession_team; with no possession nobody is either."""
    rows = g.select("period", "k", "r", "seg", "sign", "possession_team")
    p = (
        objects.filter(pl.col("object_type").is_in(PLAYER_TYPES), pl.col("visible"))
        .select("period", "object_id", "object_type", "team", "x", "y", k=tenths())
        .collect()
        .join(rows, on=["period", "k"])
    )
    # velocity between two VISIBLE sightings 0.5 s apart; rotated with the later row
    earlier = p.select("object_id", "seg", "x", "y", r=pl.col("r") + VEL_ROWS)
    p = p.join(earlier, on=["object_id", "r", "seg"], how="left", suffix="_0")
    dt = VEL_ROWS / 10
    bx = pl.Series(bf["ball_x"]).fill_nan(None)
    by = pl.Series(bf["ball_y"]).fill_nan(None)
    p = p.with_columns(
        xa=pl.col("sign") * pl.col("x"),
        ya=pl.col("sign") * pl.col("y"),
        vxa=pl.col("sign") * (pl.col("x") - pl.col("x_0")) / dt,
        vya=pl.col("sign") * (pl.col("y") - pl.col("y_0")) / dt,
        att=pl.col("team").eq(pl.col("possession_team")).fill_null(False),
        dfn=pl.col("team").ne(pl.col("possession_team")).fill_null(False),
        gk=pl.col("object_type") == "goalkeeper",
        bx=bx.gather(p["r"]),
        by=by.gather(p["r"]),
    ).with_columns(
        d=((pl.col("xa") - pl.col("bx")) ** 2 + (pl.col("ya") - pl.col("by")) ** 2).sqrt()
    )
    xa, ya = p["xa"].to_numpy(), p["ya"].to_numpy()
    b_x, b_y = p["bx"].fill_null(np.nan).to_numpy(), p["by"].fill_null(np.nan).to_numpy()
    p = p.with_columns(
        lane=pl.lit(pl.Series(in_lane(xa, ya, b_x, b_y))) & pl.col("bx").is_not_null(),
        box=pl.Series(in_box(xa, ya)),
        goal_side=pl.Series(xa > b_x),
        vgoal=pl.Series(
            toward_goal(
                xa, ya, p["vxa"].fill_null(np.nan).to_numpy(), p["vya"].fill_null(np.nan).to_numpy()
            )
        ).fill_nan(None),
        speed=(pl.col("vxa") ** 2 + pl.col("vya") ** 2).sqrt(),
        goal_dist=pl.Series(goal_distance(xa, ya)),
    )
    att, dfn = pl.col("att"), pl.col("dfn")
    agg = p.group_by("r").agg(
        n_visible_att=att.sum(),
        n_visible_def=dfn.sum(),
        att_in_box=(att & pl.col("box")).sum(),
        def_in_box=(dfn & pl.col("box")).sum(),
        att_goal_side=(att & pl.col("goal_side")).sum(),
        def_goal_side=(dfn & pl.col("goal_side")).sum(),
        lane_defenders=(dfn & pl.col("lane")).sum(),
        gk_in_lane=(dfn & pl.col("gk") & pl.col("lane")).any(),
        def_within_5m=(dfn & (pl.col("d") < PRESSURE_M)).sum(),
        press_dist=pl.col("d").filter(dfn).min(),
        gk_off_line=(GOAL_X - pl.col("xa")).filter(dfn & pl.col("gk")).first(),
        gk_ball_dist=pl.col("d").filter(dfn & pl.col("gk")).first(),
    )
    # likely carrier: PFF has none (06), so the VISIBLE attacker nearest the ball
    carrier = (
        p.filter(pl.col("att"), pl.col("d").is_not_null())
        .sort("r", "d")
        .group_by("r", maintain_order=True)
        .first()
        .select(
            "r",
            carrier_dist="d",
            carrier_speed="speed",
            carrier_vgoal="vgoal",
            carrier_goal_dist="goal_dist",
        )
    )
    out = rows.select("r").join(agg, on="r", how="left").join(carrier, on="r", how="left")
    out = out.sort("r")
    has_ball = ~np.isnan(bf["ball_x"])
    feats = {}
    for c in PLAYER_FEATURES:
        a = out[c].cast(pl.Float64).fill_null(np.nan).to_numpy()
        if c in COUNT_FEATURES:
            a = np.nan_to_num(a, nan=0.0)  # no visible players: zero of them
            if c in BALL_COUNTS:
                a = np.where(has_ball, a, np.nan)  # relative to a ball there isn't
        feats[c] = a
    return feats


COUNT_FEATURES = {
    "n_visible_att",
    "n_visible_def",
    "att_in_box",
    "def_in_box",
    "att_goal_side",
    "def_goal_side",
    "lane_defenders",
    "gk_in_lane",
    "def_within_5m",
}
BALL_COUNTS = {"att_goal_side", "def_goal_side", "lane_defenders", "gk_in_lane", "def_within_5m"}
PLAYER_FEATURES = (
    "carrier_dist",
    "carrier_speed",
    "carrier_vgoal",
    "carrier_goal_dist",
    "press_dist",
    "def_within_5m",
    "lane_defenders",
    "gk_in_lane",
    "gk_off_line",
    "gk_ball_dist",
    "att_goal_side",
    "def_goal_side",
    "att_in_box",
    "def_in_box",
    "n_visible_att",
    "n_visible_def",
)


def possession_s(g: pl.DataFrame) -> np.ndarray:
    """Seconds since possession_team last changed (in this period); NaN with no team."""
    run = (
        (
            (pl.col("period").diff() != 0)
            | pl.col("possession_team").ne_missing(pl.col("possession_team").shift())
        )
        .fill_null(True)
        .cum_sum()
    )
    s = g.select(s=pl.col("t_s") - pl.col("t_s").first().over(run))["s"].to_numpy()
    return np.where(g["possession_team"].is_null().to_numpy(), np.nan, s)


# --- lead-time features (v2, 05): an attack building over the last 2-5 s ---
# History is measured toward the anchor row's goal, with the anchor row's attackers, even
# across a turnover. So every quantity is computed in pitch coordinates for both teams
# and both ends, lagged or windowed there, and only then picked by the anchor's (team,
# end). Picking first and lagging after would measure old rows in their own frame.

TEAMS = ("home", "away")
ENDS = (1.0, -1.0)  # +1: the goal at +x, -1: the goal at -x (the row's `sign`)
NEAR_BOX_X = BOX_X - 10.0  # 26 m: the box plus 10 m in front of it
NEAR_BOX_Y = 25.0
OTHER = {"home": "away", "away": "home"}


def window(a: np.ndarray, n: int, seg: np.ndarray) -> np.ndarray:
    """(n, len(a)): a at 0 .. n-1 rows back in the same seg, NaN where there's none."""
    return np.stack([a] + [lag(a, i, seg) for i in range(1, n)])


def win(a: np.ndarray, n: int, seg: np.ndarray, how) -> np.ndarray:
    """how (np.nanmax, np.nanmean, ...) over the last n rows, NaN if none has a value."""
    w = window(a, n, seg)
    out = np.full(len(a), np.nan)
    ok = ~np.isnan(w).all(axis=0)
    out[ok] = how(w[:, ok], axis=0)
    return out


def pick(team: np.ndarray, sign: np.ndarray, per: dict) -> np.ndarray:
    """per[(team, end)] at each row's own possession_team and sign; NaN with no team."""
    out = np.full(len(sign), np.nan)
    for t in TEAMS:
        for e in ENDS:
            m = (team == t) & (sign == e)
            out[m] = per[(t, e)][m]
    return out


def pick_end(sign: np.ndarray, per: dict) -> np.ndarray:
    """per[end] at each row's sign (ball features: no team needed)."""
    return np.where(sign > 0, per[1.0], per[-1.0])


def possession_runs(g: pl.DataFrame) -> np.ndarray:
    """Run id: consecutive rows with the same possession_team in one seg."""
    return (
        g.select(
            (
                (pl.col("seg").diff() != 0)
                | pl.col("possession_team").ne_missing(pl.col("possession_team").shift())
            )
            .fill_null(True)
            .cum_sum()
        )
        .to_series()
        .to_numpy()
    )


def lead_ball_features(g: pl.DataFrame, ball: dict, bf: dict) -> dict[str, np.ndarray]:
    sign, seg = g["sign"].to_numpy(), g["seg"].to_numpy()
    vx, vy = fresh_velocity(ball, seg)
    has = ~np.isnan(ball["x"])
    dist, vgoal, box = {}, {}, {}
    for e in ENDS:
        x, y = e * ball["x"], e * ball["y"]
        dist[e] = goal_distance(x, y)
        vgoal[e] = toward_goal(x, y, e * vx, e * vy)
        box[e] = (in_box(x, y) & has).astype(float)
    now = bf["ball_dist"]
    out = {
        "ball_dist_change_5s": now - pick_end(sign, {e: lag(dist[e], 50, seg) for e in ENDS}),
        "ball_dist_max_5s": pick_end(sign, {e: win(dist[e], 50, seg, np.nanmax) for e in ENDS}),
        "ball_dist_min_5s": pick_end(sign, {e: win(dist[e], 50, seg, np.nanmin) for e in ENDS}),
        "ball_vgoal_mean_2s": pick_end(sign, {e: win(vgoal[e], 20, seg, np.nanmean) for e in ENDS}),
        "ball_vgoal_mean_5s": pick_end(sign, {e: win(vgoal[e], 50, seg, np.nanmean) for e in ENDS}),
        "ball_box_s": pick_end(sign, {e: win(box[e], 50, seg, np.nansum) for e in ENDS}) / 10,
    }

    # this possession so far: team and sign are fixed inside a run, so the anchor frame
    # is the row's own frame here
    run = possession_runs(g)
    third = np.where(has, bf["ball_x"] > FINAL_THIRD_X, False)
    runs = pl.DataFrame(
        {"run": run, "d": pl.Series(now).fill_nan(None), "third": third}
    ).with_row_index("r")
    runs = runs.with_columns(
        start=pl.when(pl.col("d").is_not_null() & (pl.col("d").is_not_null().cum_sum() == 1))
        .then(pl.col("d"))
        .forward_fill()
        .over("run"),
        closest=pl.col("d").fill_null(np.inf).cum_min().over("run"),
        third_s=pl.col("third").cum_sum().over("run") / 10,
        age=(pl.col("r") - pl.col("r").first()).over("run") / 10,
    )
    no_team = g["possession_team"].is_null().to_numpy()
    start = runs["start"].fill_null(np.nan).to_numpy()
    closest = runs["closest"].to_numpy()
    closest = np.where(np.isinf(closest), np.nan, closest)
    age = runs["age"].to_numpy()
    out |= {
        "poss_start_dist": np.where(no_team, np.nan, start),
        "poss_min_dist": np.where(no_team, np.nan, closest),
        "poss_final_third_s": np.where(no_team, np.nan, runs["third_s"].to_numpy()),
        "poss_advance_rate": np.where(no_team, np.nan, (start - now) / np.maximum(age, 1.0)),
    }
    return out


def lead_player_features(
    g: pl.DataFrame, objects: pl.LazyFrame, ball: dict
) -> dict[str, np.ndarray]:
    sign, seg = g["sign"].to_numpy(), g["seg"].to_numpy()
    team = g["possession_team"].to_numpy()
    n = g.height
    p = (
        objects.filter(pl.col("object_type").is_in(PLAYER_TYPES), pl.col("visible"))
        .select("period", "object_type", "team", "x", "y", k=tenths())
        .collect()
        .join(g.select("period", "k", "r"), on=["period", "k"])
        .filter(pl.col("team").is_in(TEAMS))
    )
    r, x, y = p["r"].to_numpy(), p["x"].to_numpy(), p["y"].to_numpy()
    tm, outfield = p["team"].to_numpy(), (p["object_type"] == "player").to_numpy()

    # carrier (v1's: the team's visible player nearest the held ball), per team
    d = np.hypot(x - ball["x"][r], y - ball["y"][r])
    carrier = {}
    for t in TEAMS:
        m = (tm == t) & ~np.isnan(d)
        order = np.lexsort((d[m], r[m]))
        rr = r[m][order]
        first = np.r_[True, rr[1:] != rr[:-1]]
        cx, cy = np.full(n, np.nan), np.full(n, np.nan)
        cx[rr[first]], cy[rr[first]] = x[m][order][first], y[m][order][first]
        for e in ENDS:
            carrier[(t, e)] = goal_distance(e * cx, e * cy)

    # per (team, end): that team's outfield players measured toward that end's goal
    near_box, deepest = {}, {}
    for t in TEAMS:
        m = (tm == t) & outfield
        for e in ENDS:
            ex = e * x[m]
            zone = (ex >= NEAR_BOX_X) & (np.abs(y[m]) <= NEAR_BOX_Y)
            near_box[(t, e)] = np.bincount(r[m], weights=zone, minlength=n)
            far = np.full(n, -np.inf)
            np.maximum.at(far, r[m], ex)
            deepest[(t, e)] = np.where(np.isinf(far), np.nan, far)

    # keyed by (attacking team, end): the other team is the defence
    keys = [(t, e) for t in TEAMS for e in ENDS]
    att_nb = {k: near_box[k] for k in keys}
    def_nb = {(t, e): near_box[(OTHER[t], e)] for t, e in keys}
    line = {(t, e): deepest[(OTHER[t], e)] for t, e in keys}
    beyond = {}
    for t, e in keys:
        m = (tm == t) & outfield
        past = e * x[m] > line[(t, e)][r[m]]  # NaN line compares False
        beyond[(t, e)] = np.where(
            np.isnan(line[(t, e)]), np.nan, np.bincount(r[m], weights=past, minlength=n)
        )

    def now(per):
        return pick(team, sign, per)

    def change(per, rows):
        return now(per) - pick(team, sign, {k: lag(v, rows, seg) for k, v in per.items()})

    line_dist = {k: GOAL_X - v for k, v in line.items()}
    gap = {(t, e): line[(t, e)] - e * ball["x"] for t, e in keys}
    return {
        "carrier_goal_dist_change_3s": change(carrier, 30),
        "carrier_goal_dist_change_5s": change(carrier, 50),
        "att_near_box": now(att_nb),
        "def_near_box": now(def_nb),
        "near_box_diff": now(att_nb) - now(def_nb),
        "att_near_box_change_3s": change(att_nb, 30),
        "def_near_box_change_3s": change(def_nb, 30),
        "def_line_dist": now(line_dist),
        "def_line_change_3s": change(line_dist, 30),
        "ball_line_gap": now(gap),
        "att_beyond_line": now(beyond),
    }


def match_features(
    frames: pl.DataFrame,
    objects: pl.LazyFrame,
    ball_source: str = "held",
    version: str = "1",
) -> pl.DataFrame:
    """frames (needs period, t_s, possession_team, flipped), sorted by (period, t_s),
    plus ball_visible and every feature in FEATURE_SETS[version]. NaN where a feature
    can't be computed from what's observed. v2 adds the lead-time features to v1's
    columns, which stay exactly as they are."""
    names = FEATURE_SETS[version]
    g = grid_rows(frames)
    ball = ball_track(g, objects, ball_source)
    bf = ball_features(g, ball)
    pf = player_features(g, objects, bf)
    feats = bf | pf | {"possession_s": possession_s(g)}
    if version == "2":
        feats |= lead_ball_features(g, ball, bf) | lead_player_features(g, objects, ball)
    return g.drop("r", "k", "seg", "sign").with_columns(
        ball_visible=ball["visible"],
        **{f: pl.Series(feats[f], dtype=pl.Float32) for f in names},
    )


FEATURES = (
    "ball_x",
    "ball_y",
    "ball_z",
    "ball_dist",
    "ball_angle",
    "ball_age_s",
    "ball_speed",
    "ball_vgoal",
    "ball_in_box",
    "ball_dist_change_1s",
    "ball_dist_change_3s",
    "ball_final_third_s",
    "possession_s",
    *PLAYER_FEATURES,
)
LEAD_FEATURES = (
    "ball_dist_change_5s",
    "ball_dist_max_5s",
    "ball_dist_min_5s",
    "ball_vgoal_mean_2s",
    "ball_vgoal_mean_5s",
    "ball_box_s",
    "poss_start_dist",
    "poss_min_dist",
    "poss_final_third_s",
    "poss_advance_rate",
    "carrier_goal_dist_change_3s",
    "carrier_goal_dist_change_5s",
    "att_near_box",
    "def_near_box",
    "near_box_diff",
    "att_near_box_change_3s",
    "def_near_box_change_3s",
    "def_line_dist",
    "def_line_change_3s",
    "ball_line_gap",
    "att_beyond_line",
)
# cached features are keyed by version: add a new one when any definition changes
FEATURE_SETS = {"1": FEATURES, "2": FEATURES + LEAD_FEATURES}
FEATURES_VERSION = "1"  # the default; v1 is the baseline every earlier run used


FRAME_COLS = [
    "match_id",
    "period",
    "t_s",
    "possession_team",
    "flipped",
    "ball_state",
    "eligible",
    "all_estimated",
]


def load_match(
    match_id: str,
    processed_dir: Path,
    horizons,
    ball_source: str = "held",
    cache: bool = True,
    degrade=(),
    degrade_seed: int = degrade_mod.DEFAULT_SEED,
    degrade_stats: dict | None = None,
    features_version: str = FEATURES_VERSION,
) -> pl.DataFrame:
    """One match's grid rows with labels and features. Features are cached next to the
    resampled tables, keyed by features_version and ball_source.

    With `degrade` (05, "Vision sensitivity test") the objects are degraded in memory
    first, never cached, and the clean cache is neither read nor written. Frames and
    labels are untouched, so scored rows stay the same. Realized severity (on the rows
    horizons[0] scores) is added into degrade_stats."""
    d = processed_dir / match_id
    cols = FRAME_COLS + [f"label_{x}_{h}" for h in horizons for x in ("mask", "shot")]
    frames = pl.read_parquet(d / "frames_10hz.parquet", columns=cols).sort("period", "t_s")
    path = d / f"features_v{features_version}_{ball_source}.parquet"
    keep = ["period", "t_s", "ball_visible", *FEATURE_SETS[features_version]]
    objects = pl.scan_parquet(d / "objects_10hz.parquet")
    if degrade:
        if ball_source != "held":
            raise ValueError("degradations need ball_source='held' (05)")
        cache = False
        scored = frames.filter(pl.col(f"label_mask_{horizons[0]}"), ~pl.col("all_estimated"))
        degraded, stats = degrade_mod.apply(
            objects.collect(), match_id, degrade, degrade_seed, scored
        )
        objects = degraded.lazy()
        if degrade_stats is not None:
            for arm, (num, den) in stats.items():
                acc = degrade_stats.setdefault(arm, [0.0, 0.0])
                acc[0] += num
                acc[1] += den
    if cache and path.exists():
        feats = pl.read_parquet(path)
    else:
        feats = match_features(
            frames.select("period", "t_s", "possession_team", "flipped"),
            objects,
            ball_source,
            features_version,
        ).select(keep)
        if cache:
            feats.write_parquet(path)
    if feats.height != frames.height:
        raise ValueError(f"{match_id}: {feats.height} feature rows for {frames.height} grid rows")
    return frames.hstack(feats.drop("period", "t_s"))
