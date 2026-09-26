import json

import polars as pl
import pytest

from converters.metrica import RAW_DIR, convert_game, player_ref, shot_outcome


@pytest.mark.parametrize(
    "subtype, expected",
    [
        ("ON TARGET-GOAL", "goal"),
        ("HEAD-ON TARGET-GOAL", "goal"),
        ("ON TARGET-SAVED", "saved"),
        ("BLOCKED", "blocked"),
        ("HEAD-WOODWORK-OUT", "woodwork"),
        ("OFF TARGET-HEAD-OUT", "off_target"),
        ("OFF TARGET", "off_target"),
        (None, "unknown"),
    ],
)
def test_shot_outcome(subtype, expected):
    assert shot_outcome(subtype) == expected


def test_player_ref_handles_the_header_typo():
    assert player_ref("Away", "Player 26") == "away_26"
    assert player_ref("Home", "Player9") == "home_9"
    assert player_ref("Home", None) is None


needs_data = pytest.mark.skipif(
    not (RAW_DIR / "Sample_Game_1").exists(), reason="Metrica raw data not downloaded"
)


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    out = tmp_path_factory.mktemp("gamestate")
    errors = {g: convert_game(g, RAW_DIR, out) for g in (1, 2)}
    return out, errors


def load(out, game, table):
    return pl.read_parquet(out / f"metrica-{game}" / f"{table}.parquet")


def report(out, game):
    return json.loads((out / f"metrica-{game}" / "conversion_report.json").read_text())


@needs_data
@pytest.mark.parametrize("game", [1, 2])
def test_output_validates(converted, game):
    _, errors = converted
    assert errors[game] == []


@needs_data
@pytest.mark.parametrize("game, score", [(1, {"home": 3, "away": 1}), (2, {"home": 3, "away": 2})])
def test_scores(converted, game, score):
    out, _ = converted
    goals = load(out, game, "events").filter(pl.col("event_type") == "goal")
    assert {t: goals.filter(pl.col("team") == t).height for t in score} == score


@needs_data
def test_own_goal_goes_to_the_other_team(converted):
    out, _ = converted
    og = load(out, 1, "events").filter(pl.col("outcome") == "own_goal")
    assert og.height == 1
    assert og["team"][0] == "away"
    assert og["player_id"][0] is None


@needs_data
@pytest.mark.parametrize("game", [1, 2])
def test_home_attacks_positive_x_in_period_1(converted, game):
    out, _ = converted
    frames = load(out, game, "frames")
    by_period = dict(
        frames.group_by("period").agg(pl.col("home_attacks_positive_x").first()).iter_rows()
    )
    assert by_period == {1: True, 2: False}
    gk_x = report(out, game)["checks"]["home_gk_mean_x_by_period"]
    assert gk_x["1"] < -30 and gk_x["2"] > 30


@needs_data
@pytest.mark.parametrize("game", [1, 2])
def test_shots_line_up_with_the_tracked_ball(converted, game):
    out, _ = converted
    check = report(out, game)["checks"]["shot_to_ball_m"]
    assert check["n"] == 24
    assert check["median"] < 1.0


@needs_data
def test_penalty_shot_is_a_set_piece(converted):
    out, _ = converted
    shots = load(out, 2, "events").filter(pl.col("event_type") == "shot")
    assert shots.filter(pl.col("set_piece") == "penalty").height == 1
