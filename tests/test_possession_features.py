from pathlib import Path

import numpy as np
import polars as pl
import pytest

from prediction.resample import build_grid
from vision import possession_features as pf

FPS = 10.0
GS = Path("data/gamestate")


def missing(v):
    """NaN or null: the extractor fills nulls with NaN only when it assembles the matrix."""
    return v is None or np.isnan(v)


def make_frames(times, period=1, home_pos=True):
    return pl.DataFrame(
        {
            "frame_id": list(range(len(times))),
            "period": [period] * len(times),
            "timestamp_s": [float(t) for t in times],
            "home_attacks_positive_x": [home_pos] * len(times),
        }
    )


def test_usable_keeps_visible_real_sightings_only():
    objects = pl.DataFrame(
        {
            "frame_id": [0, 0, 0, 0, 0, 0],
            "object_id": ["ball", "h1", "h2", "h3", "ref", "a1"],
            "object_type": ["ball", "player", "player", "player", "referee", "goalkeeper"],
            "team": [None, "home", "home", "home", None, "away"],
            "x": [0.0, 1.0, 2.0, float("nan"), 3.0, 4.0],
            "y": [0.0] * 6,
            "visible": [True, True, False, True, True, True],
            "interpolated": [False, True, False, False, False, False],
        }
    )
    assert pf.usable(objects)["object_id"].to_list() == ["ball", "a1"]


def test_segments_break_on_gaps_and_periods():
    frames = pl.concat(
        [
            make_frames([0.0, 0.1, 0.2, 0.4, 0.5]),
            make_frames([0.6, 0.7], period=2).with_columns(frame_id=pl.col("frame_id") + 10),
        ]
    )
    nat = pf.native(frames, FPS)
    # 0.2 -> 0.4 is two intervals, over 1.5; the period change breaks too
    assert nat["seg"].to_list() == [1, 1, 1, 2, 2, 3, 3]


def test_grid_matches_the_resampler():
    times = [0.03, 0.13, 0.2, 0.23, 0.6, 0.71, 0.83, 0.93]
    frames = make_frames(times).with_columns(frame_id=pl.col("frame_id") * 3)
    ours = pf.grid(pf.native(frames, FPS), FPS)
    theirs, skipped = build_grid(frames, FPS)
    assert skipped > 0
    assert ours["frame_id"].to_list() == theirs["frame_id"].to_list()
    assert ours["t_us"].to_list() == theirs["grid_us"].to_list()


@pytest.mark.skipif(not (GS / "10502").exists(), reason="needs PFF game state")
def test_grid_matches_the_resampler_on_pff():
    frames = pl.read_parquet(GS / "10502" / "frames.parquet")
    fps = pl.read_parquet(GS / "10502" / "match.parquet")["native_fps"][0]
    ours = pf.grid(pf.native(frames, fps), fps)
    theirs, _ = build_grid(frames, fps)
    assert ours.select("period", "t_us", "frame_id").equals(
        theirs.select("period", t_us="grid_us", frame_id="frame_id")
    )


def objs(rows):
    """rows: (frame_id, object_id, object_type, team, x, y)."""
    return pl.DataFrame(
        rows,
        schema=["frame_id", "object_id", "object_type", "team", "x", "y"],
        orient="row",
    ).with_columns(visible=pl.lit(True), interpolated=pl.lit(False))


def ten_hz(seconds, home_pos=True):
    return make_frames([round(i / 10, 6) for i in range(int(seconds * 10) + 1)], home_pos=home_pos)


def ball_at(frames, rows):
    nat = pf.native(frames, FPS)
    g = pf.grid(nat, FPS)
    return g, pf.ball_features(g, pf.balls(nat, pf.usable(objs(rows))))


def test_ball_age_and_the_one_second_hold():
    frames = ten_hz(2)
    rows = [(f, "ball", "ball", None, 3.0, 4.0) for f in range(5)]  # seen 0.0 .. 0.4 s
    g, b = ball_at(frames, rows)
    at = dict(zip(g["k"].to_list(), b.iter_rows(named=True)))
    assert at[4]["ball_seen"] == 1 and at[4]["ball_age_s"] == 0
    assert at[10]["ball_seen"] == 0 and at[10]["ball_age_s"] == pytest.approx(0.6)
    assert at[14]["ball_x"] == 3.0 and at[14]["ball_y"] == 4.0  # exactly 1 s: still held
    assert missing(at[15]["ball_x"]) and at[15]["ball_age_s"] == pytest.approx(1.1)


def test_ball_velocity_spans_and_direction():
    rows = [(f, "ball", "ball", None, 2.0 * f / 10, 1.0) for f in range(21)]  # +2 m/s in x
    g, b = ball_at(ten_hz(2), rows)
    at = dict(zip(g["k"].to_list(), b.iter_rows(named=True)))
    assert missing(at[0]["ball_vx_02"])  # no span yet
    assert at[1]["ball_vx_02"] == pytest.approx(2.0)  # 0.1 s span is W/2
    assert at[5]["ball_vx_02"] == pytest.approx(2.0) and at[5]["ball_vy_02"] == 0
    assert missing(at[4]["ball_vx_1"]) and at[5]["ball_vx_1"] == pytest.approx(2.0)
    # away attacks +x: home frame is rotated, so the same ball moves -x
    g, b = ball_at(ten_hz(2, home_pos=False), rows)
    assert b["ball_vx_02"][10] == pytest.approx(-2.0) and b["ball_x"][10] == pytest.approx(-2.0)


def test_ball_velocity_needs_no_long_gap_between_sightings():
    frames_seen = [0, 1, *range(7, 11)]  # 0.1 -> 0.7 s is a 0.6 s hole
    rows = [(f, "ball", "ball", None, float(f), 0.0) for f in frames_seen]
    _, b = ball_at(ten_hz(1), rows)
    assert missing(b["ball_vx_1"][10])
    assert b["ball_vx_02"][10] == pytest.approx(10.0)


def test_ball_history_resets_at_a_segment():
    frames = make_frames([0.0, 0.1, 0.2, 0.5, 0.6, 0.7])  # 0.2 -> 0.5 breaks the segment
    rows = [(f, "ball", "ball", None, 1.0, 1.0) for f in (0, 1, 2)] + [
        (4, "ball", "ball", None, 1.0, 1.0)
    ]
    g, b = ball_at(frames, rows)
    at = dict(zip(g["k"].to_list(), b.iter_rows(named=True)))
    assert at[5]["ball_seen"] == 0 and missing(
        at[5]["ball_age_s"]
    )  # 0.2 s sighting is another segment
    assert at[7]["ball_age_s"] == pytest.approx(0.1)


def rule_at(frames, rows):
    nat = pf.native(frames, FPS)
    use = pf.usable(objs(rows))
    r = pf.rule_frames(frames, nat, use, pf.players(nat, use), pf.balls(nat, use))
    return r.sort("fi")


def test_rule_carrier_age_and_candidate():
    rows = []
    for f in range(21):
        rows.append((f, "ball", "ball", None, 0.0, 0.0))
        rows.append((f, "a9", "player", "away", 0.0 if f < 10 else 5.0, 1.0))
        rows.append((f, "h1", "player", "home", 3.0, 0.0))
    r = rule_at(ten_hz(2), rows)
    assert missing(r["rule_team"][0]) and missing(r["rule_carrier_age_s"][0])
    assert r["rule_candidate_team"][0] == -1.0
    assert r["rule_team"][3] == -1.0 and r["rule_carrier_age_s"][9] == 0.0  # carrier from 0.3 s
    assert r["rule_carrier_age_s"][15] == pytest.approx(0.6)  # last carried at 0.9 s
    assert missing(r["rule_candidate_team"][15])  # nobody within 1.5 m
    assert r["rule_team"][20] == -1.0


def test_rule_candidate_is_absent_without_a_ball_or_out():
    rows = [(f, "h1", "player", None, 0.0, 0.0) for f in range(5)]
    rows += [(f, "ball", "ball", None, 0.5, 0.0) for f in (0, 1)]
    rows += [(3, "ball", "ball", None, 53.0, 0.0), (3, "h2", "player", "home", 53.0, 0.0)]
    r = rule_at(ten_hz(0.4), rows)
    assert r["rule_candidate_team"][0] == 0.0  # present, team unknown
    assert missing(r["rule_candidate_team"][2])  # no ball: no stale candidate
    assert missing(r["rule_candidate_team"][3])  # ball out


def frame_at(rows, frames=None, home_pos=True):
    frames = frames if frames is not None else make_frames([0.0], home_pos=home_pos)
    nat = pf.native(frames, FPS)
    use = pf.usable(objs(rows))
    return pf.frame_features(nat, pf.players(nat, use), pf.balls(nat, use)).sort("fi")


SCENE = [
    (0, "ball", "ball", None, 0.0, 0.0),
    (0, "h1", "player", "home", 3.0, 4.0),  # 5 m
    (0, "h2", "player", "home", 0.0, 0.0),  # exactly on halfway, on the ball
    (0, "hk", "goalkeeper", "home", -50.0, 0.0),
    (0, "a1", "player", "away", -8.0, 0.0),
    (0, "u1", "player", None, 60.0, 0.0),
]


def test_frame_features_shape_pressure_and_view():
    f = frame_at(SCENE).row(0, named=True)
    assert f["near_home_m"] == 0.0 and f["near_away_m"] == 8.0
    assert f["home_centroid_x"] == pytest.approx(-47 / 3)  # keepers count for the centroid
    assert f["home_past_halfway"] == 1  # h1 only: h2 is on halfway, the keeper isn't outfield
    assert f["away_past_halfway"] == 1  # away attacks -X
    assert (f["home_max_x"], f["home_min_x"]) == (3.0, 0.0)  # keeper excluded
    assert f["home_visible_n"] == 3 and f["away_visible_n"] == 1 and f["players_n"] == 5
    assert (f["home_within5"], f["home_within10"], f["away_within5"], f["away_within10"]) == (
        2,
        2,
        0,
        1,
    )
    assert (f["view_min_x"], f["view_max_x"]) == (-50.0, 60.0)  # unknown team counts for the view


def test_frame_features_rotate_and_handle_missing_ball():
    f = frame_at(SCENE, home_pos=False).row(0, named=True)
    assert (f["home_max_x"], f["home_min_x"]) == (0.0, -3.0)
    assert f["away_past_halfway"] == 0 and f["home_past_halfway"] == 0
    f = frame_at(SCENE[1:]).row(0, named=True)
    assert f["near_home_m"] is None and f["home_within5"] is None
    assert f["home_visible_n"] == 3 and f["home_centroid_x"] == pytest.approx(-47 / 3)
    f = frame_at([(0, "ball", "ball", None, 0.0, 0.0)]).row(0, named=True)
    assert f["home_visible_n"] == 0 and f["home_within5"] == 0 and f["home_centroid_x"] is None


def test_heads_extrapolate_from_past_sightings_only():
    rows = []
    for f in range(4):
        rows.append((f, "ball", "ball", None, 1.0 * f, 0.0))  # 10 m/s in x
        rows.append((f, "h1", "player", "home", 8.0, 0.0))
        rows.append((f, "a1", "player", "away", 3.0, 5.0))
    frames = make_frames([0.0, 0.1, 0.2, 0.3])
    nat = pf.native(frames, FPS)
    g = pf.grid(nat, FPS)
    use = pf.usable(objs(rows))
    pls, bl = pf.players(nat, use), pf.balls(nat, use)
    rows_g = g.hstack(pf.ball_features(g, bl)).join(pf.frame_features(nat, pls, bl), on="fi")
    h = pf.heads(rows_g, pls).row(3, named=True)  # ball at 3 m
    assert h["heads_home_03_m"] == pytest.approx(2.0)  # point (6, 0)
    assert h["heads_home_05_m"] == pytest.approx(0.0)  # point (8, 0)
    assert h["heads_away_03_m"] == pytest.approx(np.hypot(3, 5))
    assert h["heads_home_angle"] == pytest.approx(0.0)
    assert h["heads_away_angle"] == pytest.approx(np.pi / 2)
    assert missing(pf.heads(rows_g, pls).row(0, named=True)["heads_home_03_m"])  # no velocity yet


def test_heads_angle_needs_a_moving_ball():
    rows = []
    for f in range(3):
        rows.append((f, "ball", "ball", None, 0.01 * f, 0.0))  # 0.1 m/s
        rows.append((f, "h1", "player", "home", 8.0, 0.0))
    frames = make_frames([0.0, 0.1, 0.2])
    nat = pf.native(frames, FPS)
    g = pf.grid(nat, FPS)
    use = pf.usable(objs(rows))
    pls, bl = pf.players(nat, use), pf.balls(nat, use)
    rows_g = g.hstack(pf.ball_features(g, bl)).join(pf.frame_features(nat, pls, bl), on="fi")
    h = pf.heads(rows_g, pls).row(2, named=True)
    assert missing(h["heads_home_angle"]) and h["heads_home_03_m"] == pytest.approx(
        8.0 - 0.02 - 0.03
    )
    assert missing(h["heads_away_03_m"])  # nobody on away


def contact_at(frames, rows):
    nat = pf.native(frames, FPS)
    g = pf.grid(nat, FPS)
    use = pf.usable(objs(rows))
    cf = pf.contacts(nat, pf.players(nat, use), pf.balls(nat, use))
    return dict(zip(g["k"].to_list(), pf.contact_features(g, cf, FPS).iter_rows(named=True)))


def nearest(f, team, x=0.5, other=5.0):
    """Ball at 0, `team`'s player x m away, the other team's `other` m away."""
    rival = "away" if team == "home" else "home"
    return [
        (f, "ball", "ball", None, 0.0, 0.0),
        (f, "p", "player", team, x, 0.0),
        (f, "q", "player", rival, other, 0.0),
    ]


def test_shares_are_elapsed_time_in_the_window():
    rows = [r for f in range(11) for r in nearest(f, "home" if f < 5 else "away")]
    at = contact_at(ten_hz(1), rows)
    assert at[5]["nearest_home_1_any"] == pytest.approx(1.0)  # window clipped to the segment start
    assert at[10]["nearest_home_05_any"] == pytest.approx(0.0)
    assert at[10]["nearest_away_05_r15"] == pytest.approx(1.0)
    assert at[10]["nearest_home_1_any"] == pytest.approx(0.5)
    assert at[10]["nearest_home_1_any"] + at[10]["nearest_away_1_any"] == pytest.approx(1.0)
    assert missing(at[0]["nearest_home_05_any"])  # zero-length window


def test_shares_count_the_partial_first_interval():
    frames = make_frames([0.05 + i / 10 for i in range(11)])  # 0.05 .. 1.05
    rows = [r for f in range(11) for r in nearest(f, "home" if f == 4 else "away")]
    at = contact_at(frames, rows)
    # t = 1.0, window (0.5, 1.0]: frame 0.45 holds until 0.55, so 0.05 s of home
    assert at[10]["nearest_home_05_any"] == pytest.approx(0.1)


def test_shares_ties_reach_and_unseen_time():
    rows = [r for f in range(11) for r in nearest(f, "home", x=2.0, other=2.0)]
    at = contact_at(ten_hz(1), rows)
    assert at[10]["nearest_home_1_any"] == pytest.approx(0.5)  # tie split
    assert at[10]["nearest_home_1_r15"] == 0.0  # 2 m is out of reach
    rows = [r for f in range(5) for r in nearest(f, "home")] + [
        (f, "p", "player", "home", 0.5, 0.0) for f in range(5, 11)
    ]
    at = contact_at(ten_hz(1), rows)
    assert at[10]["nearest_home_1_any"] == pytest.approx(
        0.5
    )  # no ball: weight 0, time still counts
    unk = [(f, "ball", "ball", None, 0.0, 0.0) for f in range(11)]
    unk += [(f, "h", "player", "home", 1.0, 0.0) for f in range(11)]
    unk += [(f, "u", "player", None, -1.0, 0.0) for f in range(11)]
    at = contact_at(ten_hz(1), unk)
    assert at[10]["nearest_home_1_any"] == pytest.approx(0.5)  # unknown's half goes to nobody
    assert at[10]["nearest_away_1_any"] == 0.0


def test_last_contact_team_and_age():
    rows = [r for f in range(4) for r in nearest(f, "away")]
    rows += [
        r for f in range(4, 11) for r in nearest(f, "home", x=3.0, other=6.0)
    ]  # nobody in reach
    at = contact_at(ten_hz(1), rows)
    assert at[10]["last_contact_team"] == -1.0
    assert at[10]["last_contact_age_s"] == pytest.approx(0.7)
    tie = [r for f in range(2) for r in nearest(f, "home", x=1.0, other=1.0)]
    at = contact_at(ten_hz(0.1), tie)
    assert at[1]["last_contact_team"] == 0.0 and at[1]["last_contact_age_s"] == 0.0
    frames = make_frames([0.0, 0.1, 0.5, 0.6])  # a segment break after 0.1 s
    at = contact_at(frames, [r for f in (0, 1) for r in nearest(f, "home")])
    assert at[1]["last_contact_team"] == 1.0 and missing(at[6]["last_contact_team"])


def velocity_at(frames, rows):
    nat = pf.native(frames, FPS)
    use = pf.usable(objs(rows))
    return pf.track_velocity(nat, pf.players(nat, use)).sort("fi")


def test_track_velocity_toward_own_goal():
    rows = [
        (f, "h1", "player", "home", -1.0 * f / 10, 0.0) for f in range(6)
    ]  # home retreats 1 m/s
    rows += [
        (f, "a1", "player", "away", -2.0 * f / 10, 0.0) for f in range(6)
    ]  # away advances 2 m/s
    v = velocity_at(ten_hz(0.5), rows)
    assert missing(v["home_mean_vx_own"][0])  # one sighting
    assert v["home_mean_vx_own"][1] == pytest.approx(1.0)  # 0.1 s span is enough
    assert v["home_mean_vx_own"][5] == pytest.approx(1.0)
    assert v["away_mean_vx_own"][5] == pytest.approx(-2.0)


def test_track_velocity_needs_an_unbroken_track():
    rows = [(f, "h1", "player", "home", 1.0 * f, 0.0) for f in (0, 1, 3, 4)]  # missing on frame 2
    v = velocity_at(ten_hz(0.4), rows)
    assert missing(v["home_mean_vx_own"][3])  # run restarts at frame 3
    assert v["home_mean_vx_own"][4] == pytest.approx(-10.0)  # 3 -> 4 is 0.1 s
    rows = [(f, "h1", "player", "home" if f < 2 else "away", 1.0 * f, 0.0) for f in range(4)]
    v = velocity_at(ten_hz(0.3), rows)
    assert missing(v["away_mean_vx_own"][2]) and v["away_mean_vx_own"][3] == pytest.approx(10.0)


def random_scene(seed=0, fps=25.0, seconds=6.0, drop=(2.0, 2.3)):
    """A messy synthetic match: random walks, dropouts, an unknown-team player, a missing
    ball now and then, and a native gap (a feature segment break) in `drop`."""
    rng = np.random.default_rng(seed)
    n = int(seconds * fps)
    times = [round(i / fps, 6) for i in range(n) if not drop[0] <= i / fps < drop[1]]
    frames = make_frames(times)
    ids = [(f"h{i}", "player", "home") for i in range(4)] + [
        (f"a{i}", "player", "away") for i in range(4)
    ]
    ids += [("hk", "goalkeeper", "home"), ("ak", "goalkeeper", "away"), ("u0", "player", None)]
    pos = {o: rng.uniform([-40, -25], [40, 25]) for o, _, _ in ids}
    ball = np.array([0.0, 0.0])
    rows, vis = [], []
    for fid in range(len(times)):
        ball = ball + rng.normal(0, 0.4, 2)
        if rng.random() > 0.1:
            rows.append((fid, "ball", "ball", None, *ball))
            vis.append(rng.random() > 0.05)
        for o, kind, team in ids:
            pos[o] = pos[o] + rng.normal(0, 0.15, 2)
            if o != "h0" or rng.random() > 0.2:
                rows.append((fid, o, kind, team, *pos[o]))
                vis.append(rng.random() > 0.1)
    objects = objs(rows).with_columns(visible=pl.Series(vis))
    return frames, objects, fps


def swap_teams(objects):
    return objects.with_columns(
        x=-pl.col("x"),
        team=pl.col("team").replace_strict({"home": "away", "away": "home"}, default=None),
    )


def assert_same(a, b, cols=pf.COLUMNS):
    for c in cols:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-5, equal_nan=True, err_msg=c)


def test_extract_shape_and_order():
    frames, objects, fps = random_scene()
    f = pf.extract(frames, objects, fps)
    assert f.columns == [*pf.KEYS, "players_n", *pf.COLUMNS] and len(pf.COLUMNS) == 159
    assert all(f[c].dtype == pl.Float32 for c in pf.COLUMNS)
    assert f["k"].is_sorted() and f.height > 0


def test_mirror_is_its_own_inverse():
    frames, objects, fps = random_scene()
    f = pf.extract(frames, objects, fps)
    assert_same(pf.mirror(pf.mirror(f)), f)


def test_mirrored_scene_gives_mirrored_features():
    frames, objects, fps = random_scene(seed=3)
    assert_same(
        pf.extract(frames, swap_teams(objects), fps), pf.mirror(pf.extract(frames, objects, fps))
    )


def test_lag_blocks_copy_the_past_row_and_break_at_gaps():
    frames, objects, fps = random_scene(seed=1)
    f = pf.extract(frames, objects, fps).with_row_index("i")
    by_k = {k: i for i, k in zip(f["i"], f["k"])}
    for k in (35, 45, 59):  # frames resume at 2.32 s, so the run starts again at k = 24
        row, back = f.row(by_k[k], named=True), f.row(by_k[k - 10], named=True)
        assert row["ball_age_s_lag1"] == back["ball_age_s"] or np.isnan(back["ball_age_s"])
        assert row["home_centroid_x_lag1"] == back["home_centroid_x"]
    row = f.row(by_k[25], named=True)
    assert all(np.isnan(row[f"{c}_lag05"]) for c in pf.BLOCK)  # 20 is before the break
    assert np.isnan(f.row(by_k[28], named=True)["home_centroid_x_lag05"])  # 23 is skipped
    assert not np.isnan(f.row(by_k[29], named=True)["home_centroid_x_lag05"])


def test_nothing_after_t_changes_row_t():
    frames, objects, fps = random_scene(seed=2)
    cut = 3.0
    later = frames.filter(pl.col("timestamp_s") > cut)["frame_id"]
    changed = objects.with_columns(
        x=pl.when(pl.col("frame_id").is_in(later.implode())).then(-pl.col("x") * 3).otherwise("x"),
        visible=pl.when(pl.col("frame_id").is_in(later.implode()))
        .then(~pl.col("visible"))
        .otherwise("visible"),
    )
    a, b = (pf.extract(frames, o, fps).filter(pl.col("t_s") <= cut) for o in (objects, changed))
    assert a.height == b.height
    assert_same(a, b)
    shorter = frames.filter(pl.col("timestamp_s") <= cut + 0.001)
    c = pf.extract(shorter, objects, fps).filter(pl.col("t_s") <= cut)
    assert_same(a, c)


def test_invisible_objects_change_nothing():
    frames, objects, fps = random_scene(seed=4)
    hidden = ~pl.col("visible")
    scrambled = objects.with_columns(
        x=pl.when(hidden).then(pl.col("x") + 17.0).otherwise("x"),
        team=pl.when(hidden).then(pl.lit("away")).otherwise("team"),
        object_id=pl.when(hidden).then(pl.lit("zz")).otherwise("object_id"),
        z=pl.lit(5.0),
        vx=pl.lit(99.0),
        player_id=pl.lit("p"),
    )
    assert_same(pf.extract(frames, objects, fps), pf.extract(frames, scrambled, fps))
