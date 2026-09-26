import datetime as dt

import polars as pl
import pytest

from gamestate.schema import SCHEMA_VERSION
from gamestate.validate import main, validate_match

MATCH_ID = "m1"


def tiny_match() -> dict[str, pl.DataFrame]:
    """Three frames, two players, one ball, a referee and one shot. Valid under 0.3."""
    n = 3
    frames = pl.DataFrame(
        {
            "match_id": [MATCH_ID] * n,
            "frame_id": [0, 1, 2],
            "period": [1, 1, 1],
            "timestamp_s": [0.0, 0.1, 0.2],
            "home_attacks_positive_x": [True] * n,
            "ball_state": ["alive", "alive", None],
            "possession_team": ["home", "home", None],
            "ball_carrier_id": ["p1", "p1", None],
            "view_polygon": [[-10.0, -30, 10, -30, 20, 30, -20, 30], None, None],
        },
        schema_overrides={"view_polygon": pl.List(pl.Float64)},
    )
    rows = []
    for fid in range(n):
        rows += [
            ("p1", "player", "home", "h9", 30.0 + fid, 1.0, None),
            ("p2", "goalkeeper", "away", "a1", 50.0, 0.0, None),
            ("r", "referee", None, None, 20.0, 5.0, None),
            ("ball", "ball", None, None, 30.5 + fid, 1.0, 0.1),
        ]
    objects = pl.DataFrame(
        {
            "match_id": MATCH_ID,
            "frame_id": [fid for fid in range(n) for _ in range(4)],
            "object_id": [r[0] for r in rows],
            "object_type": [r[1] for r in rows],
            "team": [r[2] for r in rows],
            "player_id": [r[3] for r in rows],
            "x": [r[4] for r in rows],
            "y": [r[5] for r in rows],
            "z": [r[6] for r in rows],
            "vx": 0.0,
            "vy": 0.0,
            "visible": True,
            "interpolated": False,
            "confidence": 1.0,
        },
        schema_overrides={"z": pl.Float64},
    )
    events = pl.DataFrame(
        {
            "match_id": [MATCH_ID],
            "frame_id": [2],
            "event_type": ["shot"],
            "team": ["home"],
            "player_id": ["h9"],
            "x": [32.0],
            "y": [1.0],
            "outcome": ["saved"],
            "set_piece": ["open_play"],
            "set_play_phase": [False],
        }
    )
    players = pl.DataFrame(
        {
            "match_id": MATCH_ID,
            "player_id": ["h9", "a1"],
            "team": ["home", "away"],
            "jersey_number": [9, 1],
            "position": ["ST", "GK"],
            "name": [None, "Keeper"],
        },
        schema_overrides={"name": pl.String},
    )
    match = pl.DataFrame(
        {
            "match_id": [MATCH_ID],
            "schema_version": [SCHEMA_VERSION],
            "source": ["pff"],
            "competition": ["FIFA World Cup"],
            "season": ["2022"],
            "date": [dt.date(2022, 11, 20)],
            "home_team": ["A"],
            "away_team": ["B"],
            "native_fps": [29.97],
        }
    )
    return {
        "match": match,
        "objects": objects,
        "frames": frames,
        "events": events,
        "players": players,
    }


def write(tmp_path, tables, match_id=MATCH_ID):
    d = tmp_path / match_id
    d.mkdir()
    for name, df in tables.items():
        df.write_parquet(d / f"{name}.parquet")
    return d


def errors_after(tmp_path, table, fn):
    tables = tiny_match()
    tables[table] = fn(tables[table])
    return validate_match(write(tmp_path, tables))


def test_valid_match_passes(tmp_path):
    assert validate_match(write(tmp_path, tiny_match())) == []


def test_empty_events_allowed(tmp_path):
    assert errors_after(tmp_path, "events", lambda df: df.clear()) == []


def test_missing_file(tmp_path):
    d = write(tmp_path, tiny_match())
    (d / "players.parquet").unlink()
    assert validate_match(d) == ["players.parquet: missing"]


def test_cli_exit_codes(tmp_path, capsys):
    good = write(tmp_path, tiny_match())
    assert main([str(good)]) == 0
    assert main([str(tmp_path / "nope")]) == 1


@pytest.mark.parametrize(
    "table, fn, expected",
    [
        ("objects", lambda df: df.drop("z"), "objects.z: missing column"),
        ("frames", lambda df: df.with_columns(period=pl.lit(1.0)), "expected int"),
        ("match", lambda df: df.with_columns(schema_version=pl.lit("0.2")), "not allowed"),
        ("match", lambda df: df.with_columns(source=pl.lit("statsbomb")), "not allowed"),
        ("objects", lambda df: df.with_columns(x=pl.lit(None, pl.Float64)), "non-null"),
        ("objects", lambda df: df.with_columns(vx=pl.lit(float("nan"))), "non-null"),
        ("events", lambda df: df.with_columns(set_piece=pl.lit("penalty_kick")), "not allowed"),
        ("frames", lambda df: df.with_columns(ball_state=pl.lit("paused")), "not allowed"),
        ("players", lambda df: df.clear(), "players: no rows"),
    ],
)
def test_column_errors(tmp_path, table, fn, expected):
    errors = errors_after(tmp_path, table, fn)
    assert any(expected in e for e in errors), errors


@pytest.mark.parametrize(
    "table, fn, expected",
    [
        # match
        ("match", lambda df: df.with_columns(match_id=pl.lit("m2")), "doesn't match directory"),
        ("match", lambda df: df.with_columns(native_fps=pl.lit(0.0)), "native_fps"),
        ("players", lambda df: df.with_columns(match_id=pl.lit("m2")), "players.match_id"),
        # frames
        ("frames", lambda df: df.with_columns(frame_id=pl.lit(0)), "duplicate frame_ids"),
        ("frames", lambda df: df.with_columns(period=pl.Series([2, 1, 1])), "backwards"),
        (
            "frames",
            lambda df: df.with_columns(timestamp_s=pl.Series([0.2, 0.1, 0.0])),
            "decreases within a period",
        ),
        (
            "frames",
            lambda df: df.with_columns(view_polygon=pl.Series([[1.0, 2.0], None, None])),
            "exactly 8",
        ),
        (
            "frames",
            lambda df: df.with_columns(period=pl.Series([1, 1, 5])),
            "not marked dead",
        ),
        (
            "frames",
            lambda df: df.with_columns(possession_team=pl.lit(None, pl.String)),
            "no possession_team",
        ),
        (
            "frames",
            lambda df: df.with_columns(ball_carrier_id=pl.Series(["r", None, None])),
            "carrier isn't a player",
        ),
        # objects
        ("objects", lambda df: df.with_columns(x=pl.col("x") * 20), "off the pitch"),
        ("objects", lambda df: pl.concat([df, df.head(1)]), "duplicate (frame_id, object_id)"),
        (
            "objects",
            lambda df: df.with_columns(frame_id=pl.col("frame_id") + 10),
            "frame_id not in frames",
        ),
        (
            "objects",
            lambda df: pl.concat(
                [
                    df,
                    df.filter(pl.col("object_type") == "ball").with_columns(object_id=pl.lit("b2")),
                ]
            ),
            "more than one ball",
        ),
        (
            "objects",
            lambda df: df.with_columns(team=pl.lit("home")),
            "ball/referee rows with a team",
        ),
        ("objects", lambda df: df.with_columns(z=pl.lit(0.0)), "non-ball rows with z"),
        ("objects", lambda df: df.with_columns(confidence=pl.lit(1.5)), "outside [0, 1]"),
        (
            "objects",
            lambda df: df.with_columns(player_id=pl.col("player_id").replace("h9", "ghost")),
            "ids not in players",
        ),
        # events
        ("events", lambda df: df.with_columns(frame_id=pl.lit(99)), "frame_id not in frames"),
        (
            "events",
            lambda df: df.with_columns(event_type=pl.lit("goal"), outcome=pl.lit("disallowed")),
            "goal events marked disallowed",
        ),
        # players
        ("players", lambda df: df.with_columns(player_id=pl.lit("h9")), "duplicate player_ids"),
    ],
)
def test_semantic_errors(tmp_path, table, fn, expected):
    errors = errors_after(tmp_path, table, fn)
    assert any(expected in e for e in errors), errors


def test_shootout_shot_rejected(tmp_path):
    tables = tiny_match()
    tables["frames"] = tables["frames"].with_columns(
        period=pl.Series([1, 1, 5]), ball_state=pl.Series(["alive", "alive", "dead"])
    )
    errors = validate_match(write(tmp_path, tables))
    assert any("shot events in the shootout" in e for e in errors), errors
