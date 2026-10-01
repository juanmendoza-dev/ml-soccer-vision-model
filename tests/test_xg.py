"""xG v1 (05 "xG v1 build"): StatsBomb coordinates, the population filter, and parity of the
training adapter with the shot model's own columns."""

import math

import numpy as np
import polars as pl
import pytest

from prediction import xg
from prediction.features import POST_Y, match_features


def test_coordinates_keep_the_goal_and_the_penalty_spot():
    gx, gy = xg.to_m([120, 120], [36, 44])
    assert gx.tolist() == [52.5, 52.5]
    assert gy[0] - gy[1] == pytest.approx(2 * POST_Y, abs=0.01)  # 8 yd = 7.32 m
    sx, sy = xg.to_m(108, 40)
    assert 52.5 - sx == pytest.approx(11.0, abs=0.05) and sy == 0
    _, left = xg.to_m(110, 0)
    assert left > 0  # StatsBomb's y = 0 is the shooter's left, 02's +y


def shot(x, y, defenders):
    return {"x": x, "y": y, "defenders": defenders}


SCENE = shot(40.0, 0.0, [(43.0, 0.0, False), (44.0, 6.0, False), (51.0, 0.5, True), (20.0, 0.0, False)])


def test_columns_match_the_shot_models_features_on_the_same_scene():
    # the by-hand scene of test_features.test_player_features_by_hand, with its attackers
    frames = pl.DataFrame({"period": [1], "t_s": [0.0], "possession_team": ["home"], "flipped": [False]})
    rows = [
        ("ball", "ball", None, 40.0, 0.0),
        ("h1", "player", "home", 39.0, 0.0),
        ("h2", "player", "home", 45.0, 10.0),
        ("a1", "player", "away", 43.0, 0.0),
        ("a2", "player", "away", 44.0, 6.0),
        ("ak", "goalkeeper", "away", 51.0, 0.5),
        ("a3", "player", "away", 20.0, 0.0),
    ]
    objects = pl.DataFrame(
        [{"period": 1, "t_s": 0.0, "object_id": o, "object_type": t, "team": team, "x": x, "y": y,
          "z": None, "visible": True} for o, t, team, x, y in rows],
        schema_overrides={"z": pl.Float64},
    ).lazy()
    want = match_features(frames, objects).select(xg.XG_FEATURES).cast(pl.Float64).row(0)
    got = xg.columns([SCENE]).row(0)
    assert got == pytest.approx(want, nan_ok=True)
    r = dict(zip(xg.XG_FEATURES, got, strict=True))
    assert r["lane_defenders"] == 2 and r["gk_in_lane"] == 1 and r["def_within_5m"] == 1
    assert r["press_dist"] == pytest.approx(3.0) and r["gk_off_line"] == pytest.approx(1.5)
    assert r["ball_dist"] == pytest.approx(12.5)


def test_shots_never_see_each_other():
    other = shot(30.0, -10.0, [(31.0, -10.0, False)])
    both = xg.columns([SCENE, other])
    assert both.row(0) == pytest.approx(xg.columns([SCENE]).row(0), nan_ok=True)
    assert both.row(1) == pytest.approx(xg.columns([other]).row(0), nan_ok=True)


def test_a_keeper_off_camera_is_missing_not_zero():
    r = dict(zip(xg.XG_FEATURES, xg.columns([shot(40.0, 0.0, [(43.0, 0.0, False)])]).row(0), strict=True))
    assert math.isnan(r["gk_off_line"]) and math.isnan(r["gk_ball_dist"])
    assert r["gk_in_lane"] == 0 and r["lane_defenders"] == 1


def event(kind, team, period, ts, location=None, **extra):
    e = {"id": extra.pop("id", f"{kind}-{team}-{period}-{ts}"), "type": {"name": kind},
         "team": {"id": team}, "period": period, "timestamp": ts, "location": location}
    return e | extra


def pass_(kind, team, period, ts, x):
    return event("Pass", team, period, ts, [x, 40], **{"pass": {"type": {"name": kind}}})


def shot_event(sid, team, period, ts, kind="Open Play", outcome="Goal"):
    return event("Shot", team, period, ts, [108.0, 40.0], id=sid,
                 shot={"type": {"name": kind}, "outcome": {"name": outcome}, "statsbomb_xg": 0.3})


def test_set_play_phase_is_02s_proxy():
    events = [
        pass_("Corner", 1, 1, "00:10:00.000", 120),
        shot_event("in5", 1, 1, "00:10:05.000"),
        shot_event("in12", 1, 1, "00:10:12.000"),
        pass_("Free Kick", 1, 1, "00:20:00.000", 70),  # short of the final third
        shot_event("fk_mid", 1, 1, "00:20:04.000"),
        pass_("Free Kick", 1, 1, "00:30:00.000", 85),
        shot_event("fk_final", 1, 1, "00:30:04.000"),
        pass_("Corner", 2, 1, "00:40:00.000", 120),  # the other team's corner
        shot_event("other_team", 1, 1, "00:40:03.000"),
        shot_event("other_period", 1, 2, "00:10:03.000"),
        shot_event("pen", 1, 2, "00:50:00.000", kind="Penalty"),
    ]
    out = {s["shot_id"]: s for s in xg.match_shots(7, events, [])}
    phase = {k: v["set_play_phase"] for k, v in out.items()}
    assert phase == {"in5": True, "in12": False, "fk_mid": False, "fk_final": True,
                     "other_team": False, "other_period": False, "pen": False}
    assert not out["pen"]["open_play"] and out["in5"]["open_play"]
    assert not any(s["has_360"] for s in out.values())


def test_360_frames_give_defenders_in_meters_and_the_keeper_flag():
    events = [shot_event("s", 1, 1, "00:01:00.000")]
    frame = {"event_uuid": "s", "visible_area": [], "freeze_frame": [
        {"teammate": True, "actor": True, "keeper": False, "location": [108.0, 40.0]},
        {"teammate": False, "actor": False, "keeper": True, "location": [119.0, 40.0]},
        {"teammate": False, "actor": False, "keeper": False, "location": [110.0, 38.0]},
    ]}
    (s,) = xg.match_shots(7, events, [frame])
    assert s["has_360"] and s["goal"] and s["match_id"] == "7"
    assert len(s["defenders"]) == 2
    (kx, ky, keeper), (dx, dy, other) = s["defenders"]
    assert keeper and not other
    assert kx == pytest.approx(52.5 - 0.9144) and ky == 0
    assert dy == pytest.approx(2 * 0.9144)


def test_folds_are_by_match_and_fixed():
    ids = [str(i) for i in range(23)]
    a, b = xg.folds(ids * 3), xg.folds(list(reversed(ids)))
    assert a == b and set(a.values()) == set(range(5))
    assert sorted(np.bincount(list(a.values()))) == [4, 4, 5, 5, 5]


def test_the_choice_needs_both_pooled_and_four_folds():
    y = np.array([0, 1, 0, 1, 0], float)
    good, bad = np.array([0.1, 0.9, 0.1, 0.9, 0.1]), np.array([0.5] * 5)

    def cv(lgbm_folds):
        return {"oof": {"lgbm": good, "baseline": bad},
                "per_fold": pl.DataFrame({"lgbm_log_loss": lgbm_folds, "baseline_log_loss": [0.5] * 5})}

    assert xg.choose(cv([0.1, 0.1, 0.1, 0.1, 0.9]), y)[0] == "lgbm"
    assert xg.choose(cv([0.1, 0.1, 0.1, 0.9, 0.9]), y)[0] == "baseline"
    assert xg.choose({**cv([0.1] * 5), "oof": {"lgbm": bad, "baseline": good}}, y)[0] == "baseline"
