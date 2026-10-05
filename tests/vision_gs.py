"""A small game state in vision's format (vision.writer): one period, the ball carried
toward +x by home player h1, an away player nearby, three more players. For stage 8 and
inference tests."""

from pathlib import Path

import polars as pl

from converters.common import causal_velocities
from gamestate.schema import SCHEMA_VERSION

FRAME_SCHEMA = {
    "match_id": pl.String,
    "frame_id": pl.Int64,
    "period": pl.Int64,
    "timestamp_s": pl.Float64,
    "home_attacks_positive_x": pl.Boolean,
    "ball_state": pl.String,
    "possession_team": pl.String,
    "ball_carrier_id": pl.String,
    "view_polygon": pl.List(pl.Float64),
    "set_play_phase": pl.Boolean,
}
OBJECT_SCHEMA = {
    "match_id": pl.String,
    "frame_id": pl.Int64,
    "object_id": pl.String,
    "object_type": pl.String,
    "team": pl.String,
    "player_id": pl.String,
    "x": pl.Float64,
    "y": pl.Float64,
    "z": pl.Float64,
    "visible": pl.Boolean,
    "interpolated": pl.Boolean,
    "confidence": pl.Float64,
}


def ball_x(t: float) -> float:
    return 5.0 + 4.0 * t  # 4 m/s toward the +x goal


def write_match(d: Path, n=60, fps=10.0, teams=True, possession=None, ball_state=None) -> Path:
    """possession / ball_state: None leaves them null (as vision.run writes them), a str
    fills every frame, a list gives one value per frame."""
    d.mkdir(parents=True, exist_ok=True)
    per = lambda v: v if isinstance(v, list) else [v] * n  # noqa: E731
    frames = pl.DataFrame(
        {
            "match_id": ["m"] * n,
            "frame_id": list(range(n)),
            "period": [1] * n,
            "timestamp_s": [f / fps for f in range(n)],
            "home_attacks_positive_x": [True] * n,
            "ball_state": per(ball_state),
            "possession_team": per(possession),
            "ball_carrier_id": [None] * n,
            "view_polygon": [None] * n,
            "set_play_phase": [None] * n,
        },
        schema=FRAME_SCHEMA,
    )
    rows = []
    for f in range(n):
        t = f / fps
        bx = ball_x(t)
        people = [
            ("h1", "home", bx - 0.5, 0.3),
            ("h2", "home", bx - 12.0, 10.0),
            ("a1", "away", bx + 6.0, -2.0),
            ("a2", "away", bx + 15.0, 4.0),
            ("k1", "away", 50.0, 0.0),
        ]
        for oid, team, x, y in people:
            rows.append(
                ("m", f, oid, "goalkeeper" if oid == "k1" else "player", team if teams else None,
                 None, x, y, None, True, False, 0.9)
            )
        rows.append(("m", f, "ball", "ball", None, None, bx, 0.0, None, True, False, 0.8))
    objects = pl.DataFrame(rows, schema=OBJECT_SCHEMA, orient="row")
    objects = causal_velocities(objects, frames, fps=fps)
    match = pl.DataFrame(
        {
            "match_id": ["m"],
            "schema_version": [SCHEMA_VERSION],
            "source": ["vision"],
            "competition": [None],
            "season": [None],
            "date": [None],
            "home_team": ["Home"],
            "away_team": ["Away"],
            "native_fps": [fps],
        },
        schema_overrides={"competition": pl.String, "season": pl.String, "date": pl.Date},
    )
    events = pl.DataFrame(
        schema={
            "match_id": pl.String,
            "frame_id": pl.Int64,
            "event_type": pl.String,
            "team": pl.String,
            "player_id": pl.String,
            "x": pl.Float64,
            "y": pl.Float64,
            "outcome": pl.String,
            "set_piece": pl.String,
            "set_play_phase": pl.Boolean,
        }
    )
    players = pl.DataFrame(
        schema={
            "match_id": pl.String,
            "player_id": pl.String,
            "team": pl.String,
            "jersey_number": pl.Int64,
            "position": pl.String,
            "name": pl.String,
        }
    )
    for name, df in (("match", match), ("frames", frames), ("objects", objects), ("events", events), ("players", players)):
        df.write_parquet(d / f"{name}.parquet")
    return d
