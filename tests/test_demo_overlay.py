"""08 pitch view: the tallies, bands, lane count and ticker, and a render on a tiny match."""

import numpy as np
import polars as pl
import pytest

cv2 = pytest.importorskip("cv2")

from demo import overlay as ov
from demo import render, tally
from demo.pitch import Pitch
from prediction.features import in_lane

FPS = 10.0


def frames_df(n=40, period=1, start=0, poss="home", state="alive", home_pos=True):
    return pl.DataFrame(
        {
            "match_id": "m",
            "frame_id": list(range(start, start + n)),
            "period": period,
            "timestamp_s": [(start + i) / FPS for i in range(n)],
            "home_attacks_positive_x": home_pos,
            "ball_state": state,
            "possession_team": poss,
            "ball_carrier_id": None,
            "view_polygon": None,
            "set_play_phase": None,
        },
        schema_overrides={
            "ball_carrier_id": pl.String,
            "view_polygon": pl.List(pl.Float64),
            "set_play_phase": pl.Boolean,
            "possession_team": pl.String,
        },
    )


def ball_df(frame_ids, x):
    return pl.DataFrame(
        {"frame_id": list(frame_ids), "x": x if isinstance(x, list) else [x] * len(frame_ids)}
    )


# speed bands and tint


@pytest.mark.parametrize(
    ("vx", "vy", "band"),
    [
        (5.49, 0, None),
        (5.5, 0, "high"),
        (0, -6.99, "high"),
        (7.0, 0, "sprint"),
        (-5, -5, "sprint"),
        (None, 3, None),
        (3, None, None),
        (float("nan"), 9, None),
    ],
)
def test_speed_bands(vx, vy, band):
    assert ov.speed_band(vx, vy) == band


def test_tint_guessed_beats_low_confidence():
    assert ov.tint(True, 1.0) == "guessed"
    assert ov.tint(True, 0.33) == "guessed"
    assert ov.tint(False, 0.33) == "low"
    assert ov.tint(False, 0.5) == "solid"
    assert ov.tint(False, None) == "solid"


# possession tally


def test_tally_counts_only_alive_frames_with_a_team():
    f = pl.concat(
        [
            frames_df(10, poss="home"),
            frames_df(5, start=10, poss="away"),
            frames_df(5, start=15, poss="away", state="dead"),
            frames_df(5, start=20, poss=None),
        ]
    )
    t = tally.possession_tally(f, ball_df(range(25), 0.0))
    last = t.row(-1, named=True)
    assert (last["home_n"], last["away_n"]) == (10, 5)
    sh = tally.shares(last)
    assert sh["home_pct"] == pytest.approx(2 / 3)


def test_tally_never_changes_when_later_frames_are_added():
    early = frames_df(20, poss="home")
    later = pl.concat([early, frames_df(20, start=20, poss="away")])
    ball = ball_df(range(40), [float(i) for i in range(40)])
    a = tally.possession_tally(early, ball)
    b = tally.possession_tally(later, ball).filter(pl.col("frame_id") < 20)
    assert a.equals(b)


def test_thirds_follow_the_attacking_direction():
    xs = [-30.0, 0.0, 30.0, 30.0]
    for home_pos, want in ((True, (1, 1, 2)), (False, (2, 1, 1))):
        t = tally.possession_tally(frames_df(4, home_pos=home_pos), ball_df(range(4), xs))
        last = t.row(-1, named=True)
        assert (last["home_def"], last["home_mid"], last["home_att"]) == want
    # the same frames for away: mirrored
    t = tally.possession_tally(frames_df(4, poss="away"), ball_df(range(4), xs))
    last = t.row(-1, named=True)
    assert (last["away_def"], last["away_mid"], last["away_att"]) == (2, 1, 1)


def test_thirds_skip_frames_with_no_ball_and_cut_at_17_5():
    t = tally.possession_tally(frames_df(4), ball_df([0, 1, 2], [-17.5, 17.5, 17.51]))
    last = t.row(-1, named=True)
    assert last["home_n"] == 4
    assert (last["home_def"], last["home_mid"], last["home_att"]) == (0, 2, 1)


def test_shares_are_none_before_anything_counts():
    t = tally.possession_tally(frames_df(3, poss=None), ball_df(range(3), 0.0))
    sh = tally.shares(t.row(-1, named=True))
    assert sh["home_pct"] is None and sh["away_att"] is None


# lane


def players_df(rows):
    return pl.DataFrame(rows, schema=["object_type", "team", "visible", "x", "y"], orient="row")


@pytest.mark.parametrize("home_pos", [True, False])
@pytest.mark.parametrize("poss", ["home", "away"])
def test_lane_count_is_in_lane_over_visible_defenders(home_pos, poss):
    rng = np.random.default_rng(0)
    other = "away" if poss == "home" else "home"
    rows = [
        (
            "goalkeeper" if i == 0 else "player",
            other if i % 3 else poss,
            bool(i % 4),
            float(x),
            float(y),
        )
        for i, (x, y) in enumerate(zip(rng.uniform(-52, 52, 60), rng.uniform(-30, 30, 60)))
    ]
    p = players_df(rows)
    bx, by = 20.0, 5.0
    tri, n = tally.lane(p, bx, by, poss, home_pos)
    sign = tally.attack_sign(poss, home_pos)
    d = p.filter(pl.col("visible"), pl.col("team") == other)
    want = int(
        in_lane(sign * d["x"].to_numpy(), sign * d["y"].to_numpy(), sign * bx, sign * by).sum()
    )
    assert n == want
    assert {round(abs(x), 2) for x, _ in tri[1:]} == {52.5}
    assert tri[1][0] == sign * 52.5


def test_lane_is_none_without_a_possession_team():
    p = players_df([("player", "away", True, 40.0, 0.0)])
    assert tally.lane(p, 30.0, 0.0, None, True) is None


def test_lane_ignores_invisible_own_team_and_unknown_players():
    p = players_df(
        [
            ("player", "away", True, 45.0, 0.0),  # counts
            ("player", "away", False, 46.0, 0.0),  # guessed position
            ("player", "home", True, 47.0, 0.0),  # own team
            ("player", None, True, 48.0, 0.0),  # no team
            ("referee", "away", True, 49.0, 0.0),  # not a player
        ]
    )
    assert tally.lane(p, 30.0, 0.0, "home", True)[1] == 1


# ticker, score, clock


def events_df():
    return pl.DataFrame(
        {
            "match_id": "m",
            "frame_id": [10, 10, 30],
            "event_type": ["shot", "goal", "shot"],
            "team": ["home", "home", "away"],
            "player_id": None,
            "x": [40.0, 40.0, -30.0],
            "y": [0.0, 0.0, 5.0],
            "outcome": ["goal", "goal", "saved"],
            "set_piece": ["open_play", "open_play", "corner"],
            "set_play_phase": [False, False, True],
        },
        schema_overrides={"player_id": pl.String},
    )


def test_ticker_shows_an_event_from_its_frame_for_four_seconds():
    e = events_df()
    span = round(tally.TICKER_S * FPS)
    assert tally.ticker(e, 9, FPS) == []
    assert [r["event_type"] for r in tally.ticker(e, 10, FPS)] == ["goal"]
    assert [r["event_type"] for r in tally.ticker(e, 10 + span - 1, FPS)] == ["shot", "goal"]
    assert [r["frame_id"] for r in tally.ticker(e, 10 + span, FPS)] == [30]


def test_ticker_keeps_a_goal_shot_with_no_goal_event_on_its_frame():
    e = events_df().filter(pl.col("event_type") == "shot")
    assert [r["outcome"] for r in tally.ticker(e, 12, FPS)] == ["goal"]


def test_event_text():
    names = {"home": "Home FC", "away": "Away United"}
    rows = events_df().to_dicts()
    assert tally.event_text(rows[1], names) == "GOAL  Home FC"
    assert tally.event_text(rows[2], names) == "Shot  Away United  -  corner  -  saved"
    assert tally.event_text(rows[0], names) == "Shot  Home FC  -  goal"


def test_score_counts_goals_up_to_t():
    e = events_df()
    assert tally.score(e, 9) == (0, 0)
    assert tally.score(e, 10) == (1, 0)


def test_clock():
    assert tally.clock(1, 0.0) == "00:00"
    assert tally.clock(1, 61.9) == "01:01"
    assert tally.clock(2, 30.0) == "45:30"
    assert tally.clock(3, 0.0) == "90:00"
    assert tally.clock(4, 60.0) == "106:00"


# drawing


def test_pitch_maps_plus_y_up_and_the_arrow_tip_follows_velocity():
    p = Pitch(scale=10, pad=30)
    cx, cy = p.px(0, 0)
    ux, uy = p.px(0, 0 + ov.ARROW_S * 4.0)  # vy = +4 m/s
    assert ux == cx and uy < cy
    rx, _ = p.px(0 + ov.ARROW_S * 4.0, 0)
    assert rx > cx


def test_drawing_off_the_image_edge_doesnt_raise():
    img = np.zeros((50, 50, 3), np.uint8)
    ov.ball_marker(img, (-5, 49), 6, "guessed", (100, 100))
    ov.ball_trail(img, [(-10, -10), (2, 2), None, (60, 60)])
    ov.lane_cone(img, [(-20, 10), (70, 0), (70, 40)], 3)
    ov.player_marker(img, (49, 49), 9, (1, 2, 3), "sprint", "low", "10")


# render


def write_match(tmp_path, n=60):
    d = tmp_path / "m"
    d.mkdir()
    f = pl.concat([frames_df(n), frames_df(n, period=2, start=1000, home_pos=False)])
    f.write_parquet(d / "frames.parquet")
    pl.DataFrame(
        {"match_id": "m", "home_team": "Home FC", "away_team": "Away United", "native_fps": FPS}
    ).write_parquet(d / "match.parquet")
    e = events_df().with_columns(pl.col("frame_id") + 1000)
    e.write_parquet(d / "events.parquet")
    rows = []
    for fid in f["frame_id"]:
        rows.append(
            (fid, "ball", "ball", None, 10.0 + (fid % 100) * 0.3, 2.0, 3.0, 0.0, True, False, 1.0)
        )
        for k, team in enumerate(["home", "away"] * 3):
            rows.append(
                (
                    fid,
                    f"{team}_{k}",
                    "player",
                    team,
                    -20.0 + 8 * k,
                    -5.0 + k,
                    6.0,
                    0.0,
                    k != 1,
                    k == 1,
                    0.33 if k == 2 else 1.0,
                )
            )
    pl.DataFrame(rows, schema=render.OBJECT_COLS, orient="row").with_columns(
        match_id=pl.lit("m"), player_id=pl.lit(None, pl.String), z=pl.lit(None, pl.Float64)
    ).write_parquet(d / "objects.parquet")
    return d


def test_render_writes_every_frame_in_range(tmp_path):
    d = write_match(tmp_path)
    out = tmp_path / "clip.mp4"
    render.main(
        ["--match", "m", "--gamestate", str(tmp_path), "--frames", "1005-1034", "--out", str(out)]
    )
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 30
    ok, img = cap.read()
    assert ok and img.shape[1::-1] == render.Scene(d, 1005, 1034).size


def test_render_step_and_goal_window_stays_in_its_period(tmp_path):
    d = write_match(tmp_path)
    first, last = render.goal_window(d, 1, before=15.0, after=5.0)
    assert (first, last) == (1000, 1059)  # 15 s back crosses into period 1, 5 s on past its end
    out = tmp_path / "goal.mp4"
    render.main(
        [
            "--match",
            "m",
            "--gamestate",
            str(tmp_path),
            "--goal",
            "1",
            "--step",
            "2",
            "--out",
            str(out),
        ]
    )
    assert int(cv2.VideoCapture(str(out)).get(cv2.CAP_PROP_FRAME_COUNT)) == 30
    with pytest.raises(SystemExit):
        render.goal_window(d, 2, 15.0, 5.0)


def test_a_frame_before_the_goal_shows_no_goal(tmp_path):
    d = write_match(tmp_path)
    s = render.Scene(d, 1000, 1059)
    before, at = s.draw(1009), s.draw(1010)
    footer = slice(s.size[1] - render.FOOTER, s.size[1])
    assert (before[footer] == ov.PANEL_BG).all()  # empty ticker
    assert not np.array_equal(before[footer], at[footer])
