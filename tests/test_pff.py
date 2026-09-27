import bz2
import json

import polars as pl
import pytest

from converters.common import ConversionReport
from converters.pff import RAW_DIR, convert_game, link_events, move_shootout, read_tracking
from gamestate.schema import PFF_CONFIDENCE

ROSTER = pl.DataFrame(
    {
        "player_id": ["100", "200"],
        "team": ["home", "away"],
        "jersey_number": [1, 9],
        "position": ["GK", "CF"],
        "name": ["Keeper", "Striker"],
        "started": [True, True],
    }
)


def frame(num, period=1, event=None, balls=None, away_jersey="9"):
    return {
        "frameNum": num,
        "period": period,
        "periodElapsedTime": num / 29.97,
        "videoTimeMs": num / 29.97 * 1000,
        "homePlayers": [
            {"jerseyNum": "1", "confidence": "HIGH", "visibility": "VISIBLE", "x": -40.0, "y": 1.0}
        ],
        "awayPlayers": [
            {
                "jerseyNum": away_jersey,
                "confidence": "LOW",
                "visibility": "ESTIMATED",
                "x": 5.0,
                "y": -2.0,
            }
        ],
        "homePlayersSmoothed": [],
        "balls": [{"visibility": "VISIBLE", "x": 0.5, "y": 0.5, "z": 1.5}]
        if balls is None
        else balls,
        "game_event": event,
    }


@pytest.fixture
def tracking(tmp_path):
    lines = [
        frame(10, event={"game_event_type": "FIRSTKICKOFF", "home_ball": True}),
        frame(10, event=None),  # duplicate frameNum
        frame(11, balls=[]),
        frame(12, event={"game_event_type": "OUT", "home_ball": None}),
        frame(13, event={"game_event_type": "OTB", "home_ball": False}, away_jersey="77"),
    ]
    path = tmp_path / "t.jsonl.bz2"
    with bz2.open(path, "wt") as f:
        f.writelines(json.dumps(x) + "\n" for x in lines)
    report = ConversionReport(match_id="t", source="pff")
    frames, objects = read_tracking(path, ROSTER, report)
    return frames, objects, report


def test_duplicate_frames_dropped_and_logged(tracking):
    frames, _, report = tracking
    assert frames["frame_id"].to_list() == [10, 11, 12, 13]
    assert any(d["count"] == 1 and "duplicate" in d["reason"] for d in report.dropped)


def test_ball_state_and_possession_follow_game_events(tracking):
    frames, _, _ = tracking
    assert frames["ball_state"].to_list() == ["alive", "alive", "dead", "alive"]
    assert frames["possession_team"].to_list() == ["home", "home", "home", "away"]


def test_visibility_confidence_and_z(tracking):
    _, objects, _ = tracking
    keeper = objects.filter(pl.col("object_id") == "home_1").row(0, named=True)
    assert keeper["object_type"] == "goalkeeper"
    assert keeper["player_id"] == "100"
    assert keeper["visible"] is True
    assert keeper["confidence"] == PFF_CONFIDENCE["HIGH"]
    assert keeper["z"] is None
    striker = objects.filter(pl.col("object_id") == "away_9").row(0, named=True)
    assert striker["visible"] is False
    assert striker["confidence"] == PFF_CONFIDENCE["LOW"]
    ball = objects.filter(pl.col("object_type") == "ball")
    assert ball["z"].to_list() == [1.5, 1.5, 1.5]  # frame 11 had no ball
    assert ball["confidence"].unique().to_list() == [1.0]


def test_unmatched_jersey_is_unresolved_not_dropped(tracking):
    _, objects, report = tracking
    unknown = objects.filter(pl.col("object_id") == "away_77")
    assert unknown.height == 1 and unknown["player_id"][0] is None
    assert report.unresolved == [{"what": "jersey", "detail": ["away_77"]}]


def test_shootout_frames_move_to_period_5():
    frames = pl.DataFrame(
        {
            "frame_id": [1, 2, 3, 4],
            "period": [3, 4, 4, 4],
            "timestamp_s": [900.0, 900.0, 901.0, 902.0],
            "video_time_s": [7000.0, 8000.0, 8001.0, 8002.0],
            "ball_state": ["alive"] * 4,
            "possession_team": ["home"] * 4,
        }
    )
    report = ConversionReport(match_id="t", source="pff")
    out = move_shootout(frames, 8001.0, report)
    # The END frame itself (8001.0) stays in period 4.
    assert out["period"].to_list() == [3, 4, 4, 5]
    assert out["timestamp_s"].to_list() == [900.0, 900.0, 901.0, 1.0]
    assert out["ball_state"].to_list() == ["alive", "alive", "alive", "dead"]
    assert out["possession_team"].to_list() == ["home", "home", "home", None]
    assert report.changed[0]["count"] == 1
    assert move_shootout(frames, None, report).equals(frames)


def test_events_snap_only_within_half_a_second():
    frames = pl.DataFrame({"frame_id": [100, 101, 103, 104]})
    events = pl.DataFrame(
        {"frame_id": [101, 102, 110, 500], "event_type": ["shot", "shot", "shot", "goal"]}
    )
    report = ConversionReport(match_id="t", source="pff")
    out = link_events(events, frames, report)
    # 102 moves to a neighbour, 110 is 6 frames out (still < 0.5 s), 500 is past tracking.
    assert out["frame_id"].to_list()[:2] == [101, 101] or out["frame_id"].to_list()[:2] == [
        101,
        103,
    ]
    assert out.height == 3 and 500 not in out["frame_id"].to_list()
    assert {d["reason"].split()[0]: d["count"] for d in report.dropped} == {"goal": 1}
    assert report.changed[0]["count"] == 2


DEV_GAMES = ["10502", "10504", "10505"]
SCORES = {"10502": (3, 1), "10504": (3, 1), "10505": (3, 0)}

needs_data = pytest.mark.skipif(
    not all((RAW_DIR / "Tracking Data" / f"{g}.jsonl.bz2").exists() for g in DEV_GAMES),
    reason="PFF tracking not downloaded",
)


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    out = tmp_path_factory.mktemp("gamestate")
    errors = {g: convert_game(g, RAW_DIR, out) for g in DEV_GAMES}
    return out, errors


def load(out, game, table):
    return pl.read_parquet(out / game / f"{table}.parquet")


def report(out, game):
    return json.loads((out / game / "conversion_report.json").read_text())


@needs_data
@pytest.mark.parametrize("game", DEV_GAMES)
def test_output_validates(converted, game):
    _, errors = converted
    assert errors[game] == []


@needs_data
@pytest.mark.parametrize("game", DEV_GAMES)
def test_score_from_goal_events(converted, game):
    out, _ = converted
    h, a = SCORES[game]
    assert report(out, game)["checks"]["score"] == {"home": h, "away": a}


@needs_data
@pytest.mark.parametrize("game", DEV_GAMES)
def test_home_keeper_side_per_period(converted, game):
    out, _ = converted
    gk_x = report(out, game)["checks"]["home_gk_mean_x_by_period"]
    assert gk_x["1"] < -30 and gk_x["2"] > 30
    frames = load(out, game, "frames")
    by_period = dict(
        frames.group_by("period").agg(pl.col("home_attacks_positive_x").first()).iter_rows()
    )
    assert by_period == {1: True, 2: False}


@needs_data
@pytest.mark.parametrize("game", DEV_GAMES)
def test_plus_y_is_the_attacking_teams_left(converted, game):
    out, _ = converted
    sides = report(out, game)["checks"]["plus_y_left_minus_right_m"]
    assert set(sides) == {"home_p1", "home_p2", "away_p1", "away_p2"}
    assert all(v > 5 for v in sides.values())


@needs_data
@pytest.mark.parametrize("game", DEV_GAMES)
def test_shots_line_up_with_the_tracked_ball(converted, game):
    out, _ = converted
    checks = report(out, game)["checks"]
    assert checks["shot_to_ball_m"]["n"] == checks["shots"] - checks["shots_without_ball"]
    assert checks["shot_to_ball_m"]["n"] >= 0.8 * checks["shots"]
    assert checks["shot_to_ball_m"]["median"] < 1.0
    # Independent of the event's own ball position: the coded player vs the tracked ball.
    align = checks["event_player_to_ball_m_by_period"]
    assert set(align) == {"1", "2"} and all(v["median_m"] < 2 for v in align.values())


@needs_data
@pytest.mark.parametrize("game", DEV_GAMES)
def test_speed_and_visibility_checks_reported(converted, game):
    out, _ = converted
    checks = report(out, game)["checks"]
    # Reported, not clamped. Raw (unsmoothed) positions jitter, so some rows are fast.
    assert 6 < checks["player_speed_p99_mps"] < 12
    assert checks["player_rows_over_12_mps"] > 0
    assert 0.2 < checks["frames_all_players_estimated"] < 0.45
    objects = load(out, game, "objects")
    assert objects.filter(~pl.col("visible") & ~pl.col("interpolated")).height == 0
    assert objects.filter((pl.col("object_type") != "ball") & pl.col("z").is_not_null()).height == 0
    assert set(objects["confidence"].unique()) <= {0.33, 0.67, 1.0}


@needs_data
@pytest.mark.parametrize("game", DEV_GAMES)
def test_every_player_has_a_roster_id(converted, game):
    out, _ = converted
    assert report(out, game)["unresolved"] == []
    players = load(out, game, "objects").filter(pl.col("object_type") != "ball")
    assert players["player_id"].null_count() == 0


@needs_data
@pytest.mark.parametrize("game", DEV_GAMES)
def test_duplicates_logged_and_nulls_where_02_says(converted, game):
    out, _ = converted
    r = report(out, game)
    dup = [d for d in r["dropped"] if "duplicate frameNum" in d["reason"]]
    assert dup and dup[0]["count"] > 0
    frames = load(out, game, "frames")
    assert frames["ball_carrier_id"].null_count() == frames.height
    assert frames["view_polygon"].null_count() == frames.height
    assert 0.5 < r["checks"]["frames_ball_alive"] < 0.8
    assert load(out, game, "match")["native_fps"][0] == 29.97
