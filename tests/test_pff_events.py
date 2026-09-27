import polars as pl
import pytest

from converters.common import ConversionReport
from converters.pff_events import (
    RAW_DIR,
    classify_restarts,
    frame_of,
    game_ids,
    load_metadata,
    parse_events,
    restart_of,
)

# 06: tracking files still missing (second Drive part). Hardcoded so the tracked
# counts don't change when they're downloaded.
NO_TRACKING = {
    "10503", "10506", "10507", "10508", "10510", "10511", "10517",
    "3812", "3813", "3819", "3827", "3834", "3848",
}  # fmt: skip

# Final scores after extra time, shootouts excluded, as (PFF home, PFF away).
SCORES = {
    "3812": ("Senegal", 0, "Netherlands", 2),
    "3813": ("England", 6, "Iran", 2),
    "3814": ("Qatar", 0, "Ecuador", 2),
    "3815": ("United States", 1, "Wales", 1),
    "3816": ("Argentina", 1, "Saudi Arabia", 2),
    "3817": ("Denmark", 0, "Tunisia", 0),
    "3818": ("Mexico", 0, "Poland", 0),
    "3819": ("France", 4, "Australia", 1),
    "3820": ("Morocco", 0, "Croatia", 0),
    "3821": ("Germany", 1, "Japan", 2),
    "3822": ("Spain", 7, "Costa Rica", 0),
    "3823": ("Belgium", 1, "Canada", 0),
    "3824": ("Switzerland", 1, "Cameroon", 0),
    "3825": ("Uruguay", 0, "South Korea", 0),
    "3826": ("Portugal", 3, "Ghana", 2),
    "3827": ("Brazil", 2, "Serbia", 0),
    "3828": ("Wales", 0, "Iran", 2),
    "3829": ("Qatar", 1, "Senegal", 3),
    "3830": ("Netherlands", 1, "Ecuador", 1),
    "3831": ("England", 0, "United States", 0),
    "3832": ("Tunisia", 0, "Australia", 1),
    "3833": ("Poland", 2, "Saudi Arabia", 0),
    "3834": ("France", 2, "Denmark", 1),
    "3835": ("Argentina", 2, "Mexico", 0),
    "3836": ("Japan", 0, "Costa Rica", 1),
    "3837": ("Belgium", 0, "Morocco", 2),
    "3838": ("Croatia", 4, "Canada", 1),
    "3839": ("Spain", 1, "Germany", 1),
    "3840": ("Cameroon", 3, "Serbia", 3),
    "3841": ("South Korea", 2, "Ghana", 3),
    "3842": ("Brazil", 1, "Switzerland", 0),
    "3843": ("Portugal", 2, "Uruguay", 0),
    "3844": ("Ecuador", 1, "Senegal", 2),
    "3845": ("Netherlands", 2, "Qatar", 0),
    "3846": ("Wales", 0, "England", 3),
    "3847": ("Iran", 0, "United States", 1),
    "3848": ("Australia", 1, "Denmark", 0),
    "3849": ("Tunisia", 1, "France", 0),
    "3850": ("Poland", 0, "Argentina", 2),
    "3851": ("Saudi Arabia", 1, "Mexico", 2),
    "3852": ("Croatia", 0, "Belgium", 0),
    "3853": ("Canada", 1, "Morocco", 2),
    "3854": ("Japan", 2, "Spain", 1),
    "3855": ("Costa Rica", 2, "Germany", 4),
    "3856": ("Ghana", 0, "Uruguay", 2),
    "3857": ("South Korea", 2, "Portugal", 1),
    "3858": ("Serbia", 2, "Switzerland", 3),
    "3859": ("Cameroon", 1, "Brazil", 0),
    "10502": ("Netherlands", 3, "United States", 1),
    "10503": ("Argentina", 2, "Australia", 1),
    "10504": ("France", 3, "Poland", 1),
    "10505": ("England", 3, "Senegal", 0),
    "10506": ("Japan", 1, "Croatia", 1),
    "10507": ("Brazil", 4, "South Korea", 1),
    "10508": ("Morocco", 0, "Spain", 0),
    "10509": ("Portugal", 6, "Switzerland", 1),
    "10510": ("Croatia", 1, "Brazil", 1),
    "10511": ("Netherlands", 2, "Argentina", 2),
    "10512": ("Morocco", 1, "Portugal", 0),
    "10513": ("England", 1, "France", 2),
    "10514": ("Argentina", 3, "Croatia", 0),
    "10515": ("France", 2, "Morocco", 0),
    "10516": ("Croatia", 2, "Morocco", 1),
    "10517": ("Argentina", 3, "France", 3),
}


def test_frame_of_matches_tracking_frame_numbers():
    # 10502 kickoff: eventTime 179.046 is tracking frameNum 5366.
    assert frame_of(179.046) == 5366


@pytest.mark.parametrize(
    "event_type, setpiece, expected",
    [
        ("SECONDKICKOFF", "K", "K"),
        ("OTB", "K", "K"),
        ("END", None, "END"),
        ("OTB", "F", "F"),
        ("OTB", "C", "C"),
        ("OTB", "O", None),
        ("OUT", None, None),
    ],
)
def test_restart_of(event_type, setpiece, expected):
    row = {"gameEvents": {"gameEventType": event_type, "setpieceType": setpiece}}
    assert restart_of(row) == expected


def test_score_table_adds_up_to_the_tournament_total():
    assert len(SCORES) == 64
    assert sum(h + a for _, h, _, a in SCORES.values()) == 172


needs_data = pytest.mark.skipif(
    not (RAW_DIR / "Event Data").exists(), reason="PFF raw data not downloaded"
)


def restart_row(kind, home, period, x, player_x=None):
    ball = [] if x is None else [{"x": x, "y": 0.0}]
    players = [] if player_x is None else [{"playerId": 7, "x": player_x, "y": 0.0}]
    return {
        "gameEvents": {"period": period, "homeTeam": home, "setpieceType": kind, "playerId": 7},
        "startTime": 100.0,
        "ball": ball,
        "homePlayers": players,
        "awayPlayers": [],
    }


def test_only_corners_and_final_third_free_kicks_start_a_set_play():
    # raw coordinates, home starts left: home attacks +x in periods 1 and 3, -x in 2 and 4
    meta = {"homeTeamStartLeft": True, "homeTeamStartLeftExtraTime": True}
    rows = [
        restart_row("C", True, 1, -50.0),  # corner: always
        restart_row("F", True, 1, 20.0),  # home, final third
        restart_row("F", True, 1, 10.0),  # home, attacking half short of the final third
        restart_row("F", True, 2, 20.0),  # home in period 2 attacks -x: own half
        restart_row("F", False, 1, -30.0),  # away in period 1 attacks -x: final third
        restart_row("F", True, 1, None, player_x=25.0),  # no ball: taker's position
        restart_row("F", True, 1, None),  # no position at all: counts
        restart_row("F", True, 4, -20.0),  # extra time, period 4 like period 2
        restart_row("T", True, 1, 50.0),  # throw-ins never
    ]
    out = classify_restarts(rows, meta)
    assert [c["set_play"] for c in out] == [True, True, False, False, True, True, True, True]
    assert out[3]["x_att"] == -20.0
    # home starting right: the same situations have mirrored raw positions
    mirrored = [
        r
        | {"ball": [{"x": -b["x"], "y": 0.0} for b in r["ball"]]}
        | {"homePlayers": [q | {"x": -q["x"]} for q in r["homePlayers"]]}
        for r in rows
    ]
    flipped = classify_restarts(
        mirrored, {"homeTeamStartLeft": False, "homeTeamStartLeftExtraTime": False}
    )
    assert [c["set_play"] for c in flipped] == [c["set_play"] for c in out]
    assert [c["x_att"] for c in flipped] == [c["x_att"] for c in out]


@pytest.fixture(scope="module")
def parsed():
    reports, events = {}, []
    for g in game_ids():
        reports[g] = ConversionReport(match_id=g, source="pff")
        events.append(parse_events(g, RAW_DIR, reports[g]))
    return pl.concat(events), reports


def counts(events: pl.DataFrame) -> dict:
    shots = events.filter(pl.col("event_type") == "shot")
    open_play = shots.filter(pl.col("set_piece") == "open_play")
    per_match = shots.group_by("match_id").len()["len"]
    return {
        "shots": shots.height,
        "open_play_strict": open_play.height,
        "open_play_proxy": open_play.filter(~pl.col("set_play_phase")).height,
        "goals": events.filter(pl.col("event_type") == "goal").height,
        "open_play_shot_goals": open_play.filter(pl.col("outcome") == "goal").height,
        "shots_per_match": (per_match.min(), per_match.median(), per_match.max()),
    }


@needs_data
def test_all_64_games_parse(parsed):
    events, _ = parsed
    assert events["match_id"].n_unique() == 64


@needs_data
def test_counts_match_06_all_games(parsed):
    events, _ = parsed
    assert counts(events) == {
        "shots": 1518,
        "open_play_strict": 1450,
        "open_play_proxy": 1180,  # 1,153 before own-half/midfield free kicks were dropped (06)
        "goals": 172,
        "open_play_shot_goals": 149,
        "shots_per_match": (9, 23, 41),
    }


@needs_data
def test_counts_match_06_tracked_games(parsed):
    events, _ = parsed
    tracked = events.filter(~pl.col("match_id").is_in(list(NO_TRACKING)))
    assert tracked["match_id"].n_unique() == 51
    assert counts(tracked) == {
        "shots": 1179,
        "open_play_strict": 1125,
        "open_play_proxy": 912,  # was 890 with every free kick
        "goals": 129,
        "open_play_shot_goals": 113,
        "shots_per_match": (9, 22, 41),
    }


@needs_data
def test_shots_by_set_piece(parsed):
    events, _ = parsed
    shots = events.filter(pl.col("event_type") == "shot")
    assert dict(shots.group_by("set_piece").len().iter_rows()) == {
        "open_play": 1450,
        "free_kick": 44,
        "penalty": 24,
    }
    goals = shots.filter(pl.col("outcome") == "goal")
    assert dict(goals.group_by("set_piece").len().iter_rows()) == {
        "open_play": 149,
        "free_kick": 2,
        "penalty": 17,
    }


@needs_data
def test_disallowed_goals(parsed):
    events, reports = parsed
    disallowed = events.filter(pl.col("outcome") == "disallowed")
    assert disallowed.height == 22
    assert set(disallowed["event_type"]) == {"shot"}
    assert disallowed.filter(pl.col("match_id") == "3816").height == 3  # Argentina-Saudi Arabia
    # Ziyech's free kick vs Belgium, coded as a cross: no event, but logged.
    assert reports["3837"].checks["disallowed_non_shot_goals"] == 1


@needs_data
def test_shootouts(parsed):
    _, reports = parsed
    kicks = {
        g: r.checks["shootout"]["kicks"]
        for g, r in reports.items()
        if r.checks["shootout"]["kicks"]
    }
    assert set(kicks) == {"10506", "10508", "10510", "10511", "10517"}
    assert sum(kicks.values()) == 41
    assert sum(r.checks["shootout"]["goals"] for r in reports.values()) == 26


@needs_data
def test_non_shot_goals(parsed):
    events, reports = parsed
    goals = events.filter(pl.col("event_type") == "goal")
    shot_goals = events.filter((pl.col("event_type") == "shot") & (pl.col("outcome") == "goal"))
    assert goals.height - shot_goals.height == 4
    # Sabiri: CR + SH, counted once.
    dup = [d for d in reports["3837"].dropped if "coded twice" in d["reason"]]
    assert dup and dup[0]["count"] == 1
    # Own goals go to the other team with no player.
    own = goals.filter(pl.col("outcome") == "own_goal")
    assert own["player_id"].null_count() == own.height
    assert dict(zip(own["match_id"], own["team"])) == {
        "10503": "away",  # Enzo Fernandez for Australia
        "3853": "home",  # Aguerd for Canada
        "3855": "home",  # Costa Rica's 2nd, coded on Neuer's touch
    }


@needs_data
def test_placeholder_event_dropped(parsed):
    _, reports = parsed
    dropped = [d for d in reports["3833"].dropped if "not in 1-4" in d["reason"]]
    assert dropped == [{"table": "events", "count": 1, "reason": dropped[0]["reason"]}]
    assert not any(
        "not in 1-4" in d["reason"] for g, r in reports.items() if g != "3833" for d in r.dropped
    )


@needs_data
def test_nothing_unresolved(parsed):
    _, reports = parsed
    assert {g: r.unresolved for g, r in reports.items() if r.unresolved} == {}


@needs_data
@pytest.mark.parametrize("game", sorted(SCORES))
def test_final_score(parsed, game):
    events, _ = parsed
    home, h, away, a = SCORES[game]
    meta = load_metadata(game)
    assert (meta["homeTeam"]["name"], meta["awayTeam"]["name"]) == (home, away)
    goals = events.filter((pl.col("match_id") == game) & (pl.col("event_type") == "goal"))
    assert (
        goals.filter(pl.col("team") == "home").height,
        goals.filter(pl.col("team") == "away").height,
    ) == (h, a)
