"""What the overlay shows, worked out from game state (02) in meters: possession and thirds
tallies, score, clock, ticker lines and the shooting-lane count. Nothing here draws.

Every value at frame t uses frames <= t only (08 "Element definitions").
"""

import numpy as np
import polars as pl

from prediction.features import GOAL_X, POST_Y, in_lane

THIRD_X = 17.5  # m from halfway: a third of the 105 m pitch
TICKER_S = 4.0
PLAYER_TYPES = ["player", "goalkeeper"]
THIRDS = ("def", "mid", "att")


def attack_sign(team: str | None, home_attacks_positive_x: bool) -> int:
    """+1 if `team` attacks +x on this frame, -1 if it attacks -x, 0 with no team."""
    if team is None:
        return 0
    return 1 if (team == "home") == bool(home_attacks_positive_x) else -1


def possession_tally(frames: pl.DataFrame, ball: pl.DataFrame) -> pl.DataFrame:
    """Per frame_id, cumulative from the first frame: `<team>_n` counted possession frames
    (ball alive, possession_team set) and `<team>_<third>` those of them with a ball row,
    by the third the ball is in from that team's attacking direction. `ball` is frame_id,
    x (one row per frame at most)."""
    f = (
        frames.select(
            "frame_id",
            "period",
            "timestamp_s",
            "ball_state",
            "possession_team",
            "home_attacks_positive_x",
        )
        .sort("period", "timestamp_s")
        .join(ball.select("frame_id", bx="x"), on="frame_id", how="left")
    )
    counted = (pl.col("ball_state") == "alive") & pl.col("possession_team").is_not_null()
    sign = (
        pl.when((pl.col("possession_team") == "home") == pl.col("home_attacks_positive_x"))
        .then(1.0)
        .otherwise(-1.0)
    )
    xa = sign * pl.col("bx")
    third = (
        pl.when(xa < -THIRD_X)
        .then(pl.lit("def"))
        .when(xa > THIRD_X)
        .then(pl.lit("att"))
        .when(xa.is_not_null())
        .then(pl.lit("mid"))
    )
    f = f.with_columns(counted=counted, third=third)
    cols = []
    for team in ("home", "away"):
        mine = pl.col("counted") & (pl.col("possession_team") == team)
        cols.append(mine.cast(pl.Int64).cum_sum().alias(f"{team}_n"))
        for t in THIRDS:
            cols.append(
                (mine & (pl.col("third") == t)).cast(pl.Int64).cum_sum().alias(f"{team}_{t}")
            )
    return f.select("frame_id", *cols)


def shares(row: dict) -> dict:
    """Possession % per team and each team's thirds shares (0-1), from one tally row.
    None where nothing has been counted yet."""
    total = row["home_n"] + row["away_n"]
    out = {
        "home_pct": row["home_n"] / total if total else None,
        "away_pct": row["away_n"] / total if total else None,
    }
    for team in ("home", "away"):
        n = sum(row[f"{team}_{t}"] for t in THIRDS)
        for t in THIRDS:
            out[f"{team}_{t}"] = row[f"{team}_{t}"] / n if n else None
    return out


def score(events: pl.DataFrame, frame_id: int) -> tuple[int, int]:
    goals = events.filter(pl.col("event_type") == "goal", pl.col("frame_id") <= frame_id)
    return (
        goals.filter(pl.col("team") == "home").height,
        goals.filter(pl.col("team") == "away").height,
    )


def clock(period: int, timestamp_s: float) -> str:
    """Match clock as mm:ss: 45 min per earlier regular period, 15 per extra-time one."""
    base = {1: 0, 2: 45, 3: 90, 4: 105, 5: 120}.get(period, 0) * 60
    m, s = divmod(int(base + timestamp_s), 60)
    return f"{m:02d}:{s:02d}"


def event_text(e: dict, names: dict[str, str]) -> str:
    team = names.get(e["team"], e["team"] or "?")
    if e["event_type"] == "goal":
        return f"GOAL  {team}"
    if e["event_type"] == "shot":
        bits = [f"Shot  {team}"]
        if e.get("set_piece") and e["set_piece"] != "open_play":
            bits.append(e["set_piece"].replace("_", " "))
        if e.get("outcome"):
            bits.append(e["outcome"].replace("_", " "))
        return "  -  ".join(bits)
    return f"{e['event_type'].replace('_', ' ')}  {team}"


def ticker(events: pl.DataFrame, frame_id: int, fps: float) -> list[dict]:
    """Events from their frame_id for TICKER_S, newest first."""
    span = round(TICKER_S * fps)
    live = events.filter(pl.col("frame_id") <= frame_id, pl.col("frame_id") > frame_id - span)
    return live.sort("frame_id", descending=True).to_dicts()


def lane(
    players: pl.DataFrame,
    ball_x: float,
    ball_y: float,
    possession_team: str | None,
    home_attacks_positive_x: bool,
) -> tuple[list[tuple[float, float]], int] | None:
    """The shooting lane at one frame: the triangle (ball, near post, far post) in pitch
    meters and the number of VISIBLE players of the other team inside it (05's
    lane_defenders). None with no possession team. `players` is one frame's objects."""
    sign = attack_sign(possession_team, home_attacks_positive_x)
    if sign == 0:
        return None
    d = players.filter(
        pl.col("object_type").is_in(PLAYER_TYPES),
        pl.col("visible"),
        pl.col("team").is_not_null(),
        pl.col("team") != possession_team,
    )
    inside = in_lane(
        sign * d["x"].to_numpy(), sign * d["y"].to_numpy(), sign * ball_x, sign * ball_y
    )
    tri = [(ball_x, ball_y), (sign * GOAL_X, sign * POST_Y), (sign * GOAL_X, -sign * POST_Y)]
    return tri, int(np.sum(inside))
