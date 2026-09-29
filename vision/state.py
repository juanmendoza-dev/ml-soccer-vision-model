"""Stage 8 (03): ball carrier, possession team and ball state from pitch coordinates.

One causal state machine, stepped frame by frame: a frame's output depends on that
frame and earlier ones only, so the same code runs live in the pipeline and offline on
dataset tracking (vision.state_check compares it with provider values). Thresholds are
seconds and meters, never frames, so it works at any frame rate.

It reads only what vision would see: the VISIBLE ball and VISIBLE players and
goalkeepers. On PFF that also keeps its ESTIMATED positions out, which use later frames
(05, Leakage).
"""

import math
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np
import polars as pl

HALF_X, HALF_Y = 52.5, 34.0
PENALTY_SPOT_X = HALF_X - 11.0
PENALTY_AREA_X, PENALTY_AREA_Y = HALF_X - 16.5, 20.16
GOAL_AREA_X, GOAL_AREA_Y = HALF_X - 5.5, 9.16
SPOT_R = 2.0  # "at" a corner, the center spot or a penalty spot
PLAYER_TYPES = ("player", "goalkeeper")


@dataclass(frozen=True)
class StateConfig:
    carrier_radius_m: float = 1.5
    carrier_max_z_m: float = 1.0
    carrier_min_s: float = 0.3
    carrier_gap_s: float = 0.5
    ball_lost_s: float = 2.0
    out_margin_m: float = 0.0
    still_speed: float = 0.5
    still_spot_s: float = 1.0
    still_any_s: float = 1.0
    kick_speed: float = 3.0
    speed_window_s: float = 0.2

    def to_dict(self) -> dict:
        return asdict(self)


def is_out(x: float, y: float, margin: float) -> bool:
    return abs(x) > HALF_X + margin or abs(y) > HALF_Y + margin


def at_restart_spot(x: float, y: float) -> bool:
    """Corner, center spot, penalty spot, or inside a goal area."""
    ax, ay = abs(x), abs(y)
    return (
        math.hypot(ax - HALF_X, ay - HALF_Y) <= SPOT_R
        or math.hypot(x, y) <= SPOT_R
        or math.hypot(ax - PENALTY_SPOT_X, y) <= SPOT_R
        or (ax >= GOAL_AREA_X and ay <= GOAL_AREA_Y)
    )


def in_penalty_area(x: float, y: float) -> bool:
    return abs(x) >= PENALTY_AREA_X and abs(y) <= PENALTY_AREA_Y


class StateMachine:
    """Feed frames in time order with step(). Each call returns (ball_state,
    possession_team, ball_carrier_id) for that frame, with None where 02 wants null."""

    def __init__(self, config: StateConfig | None = None):
        self.cfg = config or StateConfig()
        self.period = None

    def _reset(self, period):
        self.period = period
        self.possession = None
        self.carrier = None
        self.cand, self.cand_since, self.cand_team = None, None, None
        self.last_ball_t = None
        self.sightings = deque()  # (t, x, y) of the visible ball, for speed
        self.dead = False
        self.still_since = None

    def _speed(self, t: float) -> float | None:
        w = self.cfg.speed_window_s
        while self.sightings and self.sightings[0][0] < t - w - 1e-9:
            self.sightings.popleft()
        if len(self.sightings) < 2:
            return None
        t0, x0, y0 = self.sightings[0]
        t1, x1, y1 = self.sightings[-1]
        if t1 - t0 < w / 2:  # too short to tell jitter from motion
            return None
        return math.hypot(x1 - x0, y1 - y0) / (t1 - t0)

    def step(
        self,
        period: int,
        t: float,
        ball: tuple[float, float, float | None] | None,
        candidate: str | None = None,
        candidate_team: str | None = None,
    ) -> tuple[str | None, str | None, str | None]:
        """ball: the visible ball's (x, y, z or None), None when not seen. candidate: the
        nearest player within reach of the ball (see candidate()), with its team."""
        c = self.cfg
        if period != self.period:
            self._reset(period)

        out = ball is not None and is_out(ball[0], ball[1], c.out_margin_m)
        if out:  # nobody carries a ball that's out
            candidate = candidate_team = None

        # carrier: the same candidate for carrier_min_s, held through short ball gaps
        if ball is None:
            gap = self.last_ball_t is None or t - self.last_ball_t > c.carrier_gap_s
            if gap:
                self.carrier = None
                self.cand = self.cand_since = None
        else:
            if candidate != self.cand:
                self.cand, self.cand_since, self.cand_team = candidate, t, candidate_team
                self.carrier = None
            if self.cand is not None and t - self.cand_since >= c.carrier_min_s - 1e-9:
                self.carrier = self.cand
                if self.cand_team is not None:
                    self.possession = self.cand_team

        # ball state
        if ball is not None:
            x, y, _ = ball
            self.last_ball_t = t
            self.sightings.append((t, x, y))
            speed = self._speed(t)
            if out:
                self.dead = True
                self.still_since = None
            elif speed is not None:
                if speed < c.still_speed:
                    if self.still_since is None:
                        self.still_since = t
                    still = t - self.still_since
                    if (at_restart_spot(x, y) and still >= c.still_spot_s - 1e-9) or (
                        not in_penalty_area(x, y) and still >= c.still_any_s - 1e-9
                    ):
                        self.dead = True
                else:
                    self.still_since = None
                    if speed >= c.kick_speed:
                        self.dead = False
        if self.last_ball_t is None or t - self.last_ball_t > c.ball_lost_s:
            state = None
        else:
            state = "dead" if self.dead else "alive"
        return state, self.possession, self.carrier


def candidate(
    ball: tuple[float, float, float | None] | None,
    xy: np.ndarray,
    ids: list[str],
    teams: list[str | None],
    config: StateConfig | None = None,
) -> tuple[str | None, str | None]:
    """The live pipeline's candidate: nearest of the visible players (xy (n, 2) meters)
    to the visible ball, if close enough and the ball isn't in the air."""
    c = config or StateConfig()
    if ball is None or not len(ids):
        return None, None
    x, y, z = ball
    if z is not None and z >= c.carrier_max_z_m:
        return None, None
    d = np.hypot(xy[:, 0] - x, xy[:, 1] - y)
    d = np.where(np.isfinite(d), d, np.inf)
    # nearest; ties by object_id so the result doesn't depend on detection order
    i = min(range(len(ids)), key=lambda j: (d[j], ids[j]))
    if d[i] > c.carrier_radius_m:
        return None, None
    return ids[i], teams[i]


def infer(
    frames: pl.DataFrame, objects: pl.LazyFrame | pl.DataFrame, config: StateConfig | None = None
) -> pl.DataFrame:
    """Stage 8 over a whole game state (02 frames and objects): frame_id plus inferred
    ball_state, possession_team and ball_carrier_id. Candidates are found in bulk, then
    the state machine steps through the frames in (period, timestamp_s) order."""
    c = config or StateConfig()
    obj = objects.lazy().filter(pl.col("visible").fill_null(False), pl.col("x").is_not_null())
    ball = (
        obj.filter(pl.col("object_type") == "ball")
        .sort("frame_id", "object_id")
        .unique("frame_id", keep="first", maintain_order=True)
        .select("frame_id", bx="x", by="y", bz="z")
    )
    cand = (
        obj.filter(pl.col("object_type").is_in(PLAYER_TYPES))
        .select("frame_id", "object_id", "team", "x", "y")
        .join(ball, on="frame_id")
        .with_columns(
            d=((pl.col("x") - pl.col("bx")) ** 2 + (pl.col("y") - pl.col("by")) ** 2).sqrt()
        )
        .sort("frame_id", "d", "object_id")
        .unique("frame_id", keep="first", maintain_order=True)
        .filter(
            pl.col("d") <= c.carrier_radius_m,
            pl.col("bz").is_null() | (pl.col("bz") < c.carrier_max_z_m),
        )
        .select("frame_id", cand="object_id", cand_team="team")
    )
    rows = (
        frames.lazy()
        .select("frame_id", "period", "timestamp_s")
        .join(ball, on="frame_id", how="left")
        .join(cand, on="frame_id", how="left")
        .sort("period", "timestamp_s", "frame_id")
        .collect()
    )
    sm = StateMachine(c)
    out_state, out_poss, out_carrier = [], [], []
    cols = ("period", "timestamp_s", "bx", "by", "bz", "cand", "cand_team")
    for period, t, x, y, z, who, team in zip(*(rows[k].to_list() for k in cols), strict=True):
        s, p, car = sm.step(period, t, None if x is None else (x, y, z), who, team)
        out_state.append(s)
        out_poss.append(p)
        out_carrier.append(car)
    return pl.DataFrame(
        {
            "frame_id": rows["frame_id"],
            "ball_state": pl.Series(out_state, dtype=pl.String),
            "possession_team": pl.Series(out_poss, dtype=pl.String),
            "ball_carrier_id": pl.Series(out_carrier, dtype=pl.String),
        }
    )
