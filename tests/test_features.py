import math

import numpy as np
import polars as pl
import pytest

from prediction.features import (
    BALL_HOLD_S,
    FEATURES,
    goal_angle,
    goal_distance,
    in_lane,
    load_match,
    match_features,
)


def angle(x, y):
    return float(goal_angle(np.array([x]), np.array([y]))[0])


def test_goal_angle_landmarks():
    assert angle(52.5, 0.0) == pytest.approx(math.pi)  # on the line, between the posts
    assert angle(41.5, 0.0) == pytest.approx(2 * math.atan(3.66 / 11))  # penalty spot, ~0.64
    assert angle(52.5, 34.0) == pytest.approx(0.0, abs=1e-9)  # corner flag
    assert angle(0.0, 0.0) == pytest.approx(2 * math.atan(3.66 / 52.5))  # centre spot


def test_goal_angle_shrinks_out_wide_and_is_symmetric():
    xs = np.full(5, 36.0)
    ys = np.array([0.0, 5.0, 10.0, 20.0, 30.0])
    a = goal_angle(xs, ys)
    assert np.all(np.diff(a) < 0)
    assert goal_angle(xs, -ys) == pytest.approx(a)


def test_goal_angle_behind_the_line_stays_sane():
    # past the goal line (PFF balls reach x = 67): small, positive, no sign flip
    for x, y in [(60.0, 10.0), (67.0, 0.5), (55.0, 20.0)]:
        assert 0 <= angle(x, y) < math.pi


def test_goal_distance():
    assert goal_distance(np.array([41.5]), np.array([0.0]))[0] == pytest.approx(11.0)


def frames_of(n, poss="home", flipped=False, period=1, start=0.0):
    t = np.round(start + np.arange(n) * 0.1, 1)
    return pl.DataFrame(
        {
            "period": [period] * n,
            "t_s": t,
            "possession_team": [poss] * n,
            "flipped": [flipped] * n,
        }
    )


def obj(period, t_s, object_id, x, y, visible=True, team=None, object_type=None, z=0.0):
    if object_type is None:
        object_type = "ball" if object_id == "ball" else "player"
    return {
        "period": period,
        "t_s": t_s,
        "object_id": object_id,
        "object_type": object_type,
        "team": team,
        "x": x,
        "y": y,
        "z": z,
        "visible": visible,
    }


def objects_of(rows):
    return pl.DataFrame(rows).lazy()


def feats(frames, rows, ball_source="held"):
    return match_features(frames, objects_of(rows), ball_source)


def col(df, c):
    return df[c].cast(pl.Float64).fill_nan(None).to_list()


def test_ball_joins_by_tenths():
    frames = frames_of(3)
    rows = [obj(1, 0.1 + 1e-9, "ball", 41.5, 0.0)]  # float noise on the ball still joins
    got = feats(frames, rows, "raw")
    assert col(got, "ball_dist")[1] == pytest.approx(11.0)
    assert col(got, "ball_dist")[0] is None
    assert got["ball_visible"].to_list() == [None, True, None]


def test_two_balls_on_a_row_is_an_error():
    rows = [obj(1, 0.0, "ball", 0.0, 0.0), obj(1, 0.0, "ball", 1.0, 0.0)]
    with pytest.raises(ValueError, match="more than one ball"):
        feats(frames_of(1), rows)


def test_held_ball_carries_the_last_visible_one_for_a_second():
    n = 20
    rows = [obj(1, 0.0, "ball", 30.0, 5.0)] + [
        obj(1, round(i * 0.1, 1), "ball", 99.0 + i, 0.0, visible=False) for i in range(1, n)
    ]
    held = feats(frames_of(n), rows)
    age = col(held, "ball_age_s")
    hold_rows = round(BALL_HOLD_S * 10)
    assert age[: hold_rows + 1] == pytest.approx([i / 10 for i in range(hold_rows + 1)])
    assert all(a is None for a in age[hold_rows + 1 :])
    assert col(held, "ball_x")[: hold_rows + 1] == pytest.approx([30.0] * (hold_rows + 1))
    assert held["ball_visible"].to_list()[1] is False
    # raw takes the ESTIMATED ball as it is
    raw = feats(frames_of(n), rows, "raw")
    assert col(raw, "ball_x")[5] == pytest.approx(104.0)
    assert col(raw, "ball_age_s")[5] == 0.0


def test_held_ball_never_crosses_a_period_or_a_grid_gap():
    frames = pl.concat(
        [frames_of(3), frames_of(3, start=1.0), frames_of(3, period=2)]  # 0.3-0.9 missing
    )
    rows = [obj(1, 0.2, "ball", 10.0, 0.0), obj(2, 0.0, "ball", 10.0, 0.0, visible=False)]
    got = col(feats(frames, rows), "ball_x")
    assert got == [None, None, 10.0, None, None, None, None, None, None]


def test_history_is_rotated_with_the_anchor_rows_flip():
    # the ball moves 30 -> 35 in pitch x over 1 s; possession (and the flip) changes
    # at the last row. Measured to the new team's goal the ball moved 5 m away from it
    n = 11
    frames = frames_of(n).with_columns(
        flipped=pl.Series([False] * (n - 1) + [True]),
        possession_team=pl.Series(["home"] * (n - 1) + ["away"]),
    )
    rows = [obj(1, round(i * 0.1, 1), "ball", 30.0 + i / 2, 0.0) for i in range(n)]
    got = feats(frames, rows)
    last = got.row(n - 1, named=True)
    assert last["ball_x"] == pytest.approx(-35.0)
    assert last["ball_dist_change_1s"] == pytest.approx(5.0)
    assert last["ball_vgoal"] == pytest.approx(-5.0)  # 5 m/s away from the new goal
    assert col(got, "ball_vgoal")[n - 2] == pytest.approx(5.0)  # old team: toward goal
    assert last["possession_s"] == 0.0
    assert col(got, "possession_s")[n - 2] == pytest.approx(0.9)


def test_velocity_only_between_visible_sightings():
    n = 13
    rows = [obj(1, round(i * 0.1, 1), "ball", 10.0 + i, 0.0, visible=(i != 3)) for i in range(n)]
    rows += [
        obj(1, round(i * 0.1, 1), "h9", 20.0 + i, 0.0, team="home", visible=(i != 7))
        for i in range(n)
    ]
    got = feats(frames_of(n), rows)
    speed = col(got, "ball_speed")
    assert speed[:5] == [None] * 5
    assert speed[5] == pytest.approx(10.0)
    assert speed[8] is None  # 0.5 s back is the ESTIMATED row 3
    assert speed[9] == pytest.approx(10.0)
    carrier = col(got, "carrier_speed")
    assert carrier[6] == pytest.approx(10.0)
    assert carrier[7] is None  # not visible now: the only attacker isn't there
    assert carrier[11] == pytest.approx(10.0)
    assert carrier[12] is None  # 0.5 s back is the ESTIMATED row 7


def test_in_lane():
    xs = np.array([45.0, 45.0, 51.0, 30.0])
    ys = np.array([0.0, 5.0, 3.0, 0.0])
    assert in_lane(xs, ys, np.full(4, 40.0), np.zeros(4)).tolist() == [True, False, True, False]


def test_player_features_by_hand():
    # home attacks +x, ball at (40, 0)
    rows = [
        obj(1, 0.0, "ball", 40.0, 0.0),
        obj(1, 0.0, "h1", 39.0, 0.0, team="home"),  # carrier, 1 m
        obj(1, 0.0, "h2", 45.0, 10.0, team="home"),  # in the box, goal side
        obj(1, 0.0, "h3", 0.0, 0.0, team="home", visible=False),  # ESTIMATED: ignored
        obj(1, 0.0, "a1", 43.0, 0.0, team="away"),  # in the lane, 3 m: pressure
        obj(1, 0.0, "a2", 44.0, 6.0, team="away"),  # outside the lane, in the box
        obj(1, 0.0, "ak", 51.0, 0.5, team="away", object_type="goalkeeper"),
        obj(1, 0.0, "a3", 20.0, 0.0, team="away"),  # behind the ball
    ]
    r = feats(frames_of(1), rows).row(0, named=True)
    assert r["carrier_dist"] == pytest.approx(1.0)
    assert r["carrier_goal_dist"] == pytest.approx(13.5)
    assert r["press_dist"] == pytest.approx(3.0)
    assert r["def_within_5m"] == 1
    assert r["lane_defenders"] == 2  # a1 and the keeper
    assert r["gk_in_lane"] == 1
    assert r["gk_off_line"] == pytest.approx(1.5)
    assert r["gk_ball_dist"] == pytest.approx(math.hypot(11.0, 0.5))
    assert r["att_goal_side"] == 1
    assert r["def_goal_side"] == 3
    assert r["att_in_box"] == 2  # the box starts at x = 36
    assert r["def_in_box"] == 3
    assert r["n_visible_att"] == 2
    assert r["n_visible_def"] == 4
    assert r["ball_in_box"] == 1
    assert r["carrier_speed"] is None or math.isnan(r["carrier_speed"])  # one frame only


def test_features_in_the_attacking_frame():
    # same scene with away in possession and the pitch flipped: same features
    home = [
        obj(1, 0.0, "ball", 40.0, 2.0),
        obj(1, 0.0, "h1", 39.0, 2.0, team="home"),
        obj(1, 0.0, "a1", 43.0, 0.0, team="away"),
    ]
    away = [
        {**o, "x": -o["x"], "y": -o["y"], "team": {"home": "away", "away": "home"}.get(o["team"])}
        for o in home
    ]
    a = feats(frames_of(1), home)
    b = feats(frames_of(1, poss="away", flipped=True), away)
    assert a.select(FEATURES).row(0) == pytest.approx(b.select(FEATURES).row(0), nan_ok=True)


def synth(seed, n=120):
    """A random match stretch: 22 players and a ball, possession flipping every 3 s,
    about a third of everything ESTIMATED."""
    rng = np.random.default_rng(seed)
    t = np.round(np.arange(n) * 0.1, 1)
    poss = np.where((t // 3) % 2 == 0, "home", "away")
    frames = pl.DataFrame(
        {"period": [1] * n, "t_s": t, "possession_team": poss, "flipped": poss == "away"}
    )
    rows = []
    ids = [("ball", None, "ball")] + [
        (f"{team}_{i}", team, "goalkeeper" if i == 1 else "player")
        for team in ("home", "away")
        for i in range(1, 12)
    ]
    for oid, team, kind in ids:
        x = np.cumsum(rng.normal(0, 0.5, n)) + rng.uniform(-50, 50)
        y = np.cumsum(rng.normal(0, 0.5, n)) + rng.uniform(-30, 30)
        vis = rng.random(n) > 0.35
        for i in range(n):
            rows.append(
                obj(
                    1,
                    t[i],
                    oid,
                    float(x[i]),
                    float(y[i]),
                    bool(vis[i]),
                    team,
                    kind,
                    z=float(abs(rng.normal())),
                )
            )
    return frames, pl.DataFrame(rows, infer_schema_length=None)


def same(a: pl.DataFrame, b: pl.DataFrame):
    for f in FEATURES:
        x, y = a[f].to_numpy(), b[f].to_numpy()
        assert np.allclose(x, y, equal_nan=True), f


@pytest.mark.parametrize("ball_source", ["held", "raw"])
def test_features_only_use_the_past(ball_source):
    frames, objects = synth(1)
    cut = 6.0
    base = match_features(frames, objects.lazy(), ball_source)
    later = pl.col("t_s") > cut
    frames_b = frames.with_columns(
        possession_team=pl.when(later).then(pl.lit("home")).otherwise("possession_team"),
        flipped=pl.when(later).then(True).otherwise("flipped"),
    )
    objects_b = objects.with_columns(
        x=pl.when(later).then(pl.col("x") * -3 + 7).otherwise("x"),
        y=pl.when(later).then(0.0).otherwise("y"),
        visible=pl.when(later).then(~pl.col("visible")).otherwise("visible"),
    )
    moved = match_features(frames_b, objects_b.lazy(), ball_source)
    upto = pl.col("t_s") <= cut
    same(base.filter(upto), moved.filter(upto))


def test_estimated_positions_never_reach_a_held_feature():
    """PFF fills ESTIMATED positions from later frames (05): scramble every one of them
    and nothing may move."""
    frames, objects = synth(2)
    est = ~pl.col("visible")
    scrambled = objects.with_columns(
        x=pl.when(est).then(pl.col("x") * 7 - 20).otherwise("x"),
        y=pl.when(est).then(-pl.col("y") + 3).otherwise("y"),
        z=pl.when(est).then(9.0).otherwise("z"),
    )
    same(match_features(frames, objects.lazy()), match_features(frames, scrambled.lazy()))
    # the check has teeth: raw does move
    raw_a = match_features(frames, objects.lazy(), "raw")
    raw_b = match_features(frames, scrambled.lazy(), "raw")
    assert not np.allclose(raw_a["ball_x"].to_numpy(), raw_b["ball_x"].to_numpy(), equal_nan=True)


def test_load_match_cache_round_trip(tmp_path):
    frames, objects = synth(3, n=40)
    d = tmp_path / "m1"
    d.mkdir()
    lab = {
        "match_id": ["m1"] * frames.height,
        "ball_state": ["alive"] * frames.height,
        "eligible": [True] * frames.height,
        "all_estimated": [False] * frames.height,
        "label_mask_h5": [True] * frames.height,
        "label_shot_h5": [False] * frames.height,
    }
    frames.with_columns(**{k: pl.Series(v) for k, v in lab.items()}).reverse().write_parquet(
        d / "frames_10hz.parquet"
    )
    objects.write_parquet(d / "objects_10hz.parquet")
    first = load_match("m1", tmp_path, ["h5"])
    assert list(d.glob("features_v*_held.parquet"))
    again = load_match("m1", tmp_path, ["h5"])
    assert first["t_s"].to_list() == sorted(first["t_s"].to_list())
    same(first, again)
    same(first, load_match("m1", tmp_path, ["h5"], cache=False))
