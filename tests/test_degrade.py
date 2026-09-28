import numpy as np
import polars as pl
import pytest

from prediction import degrade
from prediction.features import FEATURES, load_match, match_features

ALL = [a if lvl is None else f"{a}:{lvl}" for a, (_, lvl) in degrade.ARMS.items()]
CASES = ALL + ["target"]


def synth(seed=0, n=600, periods=(1, 2), estimated=0.3):
    """Two periods of a ball and 10 players (one keeper a side) wandering around, with a
    share of every object ESTIMATED."""
    rng = np.random.default_rng(seed)
    parts = []
    ids = [("ball", None, "ball")] + [
        (f"{team}_{i}", team, "goalkeeper" if i == 1 else "player")
        for team in ("home", "away")
        for i in range(1, 6)
    ]
    for period in periods:
        t = np.round(np.arange(n) * 0.1, 1)
        for oid, team, kind in ids:
            parts.append(
                pl.DataFrame(
                    {
                        "match_id": "m1",
                        "frame_id": np.arange(n) + 10_000 * period,
                        "object_id": oid,
                        "object_type": kind,
                        "team": pl.Series([team] * n, dtype=pl.String),
                        "x": np.cumsum(rng.normal(0, 0.4, n)) + rng.uniform(-30, 30),
                        "y": np.cumsum(rng.normal(0, 0.4, n)) + rng.uniform(-20, 20),
                        "z": np.abs(rng.normal(0, 1, n)) if kind == "ball" else np.full(n, None),
                        "visible": rng.random(n) > estimated,
                        "interpolated": False,
                        "confidence": 1.0,
                        "period": period,
                        "t_s": t,
                        "vx": 0.0,
                        "x_att": 0.0,
                    },
                    schema_overrides={"z": pl.Float64},
                )
            )
    # 02: visible=False implies interpolated=True
    return (
        pl.concat(parts)
        .with_columns(interpolated=~pl.col("visible"))
        .sort("period", "t_s", "object_id")
    )


def run(objects, specs, seed=degrade.DEFAULT_SEED):
    return degrade.apply(objects, "m1", specs, seed)[0]


def equal(a: pl.DataFrame, b: pl.DataFrame):
    assert a.columns == b.columns
    for c in a.columns:
        x, y = a[c], b[c]
        if x.dtype.is_float():
            assert np.allclose(
                x.fill_null(np.nan).to_numpy(), y.fill_null(np.nan).to_numpy(), equal_nan=True
            ), c
        else:
            assert x.to_list() == y.to_list(), c


def test_parse_orders_arms_and_expands_presets():
    assert degrade.canonical(["team_flip:0.1", "ball_miss:0.25"]) == [
        "ball_miss:0.25",
        "team_flip:0.1",
    ]
    assert degrade.canonical(["no_ball_z"]) == ["no_ball_z"]
    target = degrade.canonical(["target"])
    assert len(target) == len(degrade.ARMS)
    assert "camera_drift:0.66" in target and "player_noise:0.66" in target
    for bad in (
        ["nope:1"],
        ["ball_miss:1"],
        ["ball_miss:-0.1"],
        ["ball_miss:x"],
        ["ball_miss:0.1", "ball_miss:0.2"],
        ["id_fragment:0"],
        ["no_ball_z:1"],
    ):
        with pytest.raises(ValueError):
            degrade.parse(bad)


def test_uniform_is_a_pure_function_of_its_keys():
    a = degrade.uniform(1, 2, np.arange(1000))
    assert np.array_equal(a, degrade.uniform(1, 2, np.arange(1000)))
    assert np.array_equal(a[500:], degrade.uniform(1, 2, np.arange(500, 1000)))
    assert not np.array_equal(a, degrade.uniform(1, 3, np.arange(1000)))
    assert 0 <= a.min() and a.max() < 1 and abs(a.mean() - 0.5) < 0.03


@pytest.mark.parametrize("spec", CASES)
def test_deterministic_given_the_seed(spec):
    objects = synth()
    a = run(objects, [spec])
    equal(a, run(objects, [spec]))
    if spec != "no_ball_z":
        b = run(objects, [spec], seed=1)
        assert not all(a[c].equals(b[c]) for c in a.columns), "the seed changed nothing"


@pytest.mark.parametrize("spec", CASES)
def test_degraded_rows_only_depend_on_the_past(spec):
    """Truncate or perturb everything after t: nothing at or before t may change."""
    objects = synth(1)
    cut = 33.3
    base = run(objects, [spec])
    upto = pl.col("t_s") <= cut
    later = pl.col("t_s") > cut
    # the second period is untouched by both, so it must match too
    keep = upto | (pl.col("period") == 2)
    truncated = run(objects.filter(keep), [spec])
    equal(base.filter(keep), truncated)
    moved = objects.with_columns(
        x=pl.when(later).then(pl.col("x") * -3 + 7).otherwise("x"),
        visible=pl.when(later).then(~pl.col("visible")).otherwise("visible"),
        team=pl.when(later).then(pl.lit("home")).otherwise("team"),
    )
    equal(base.filter(upto), run(moved, [spec]).filter(upto))


@pytest.mark.parametrize("spec", CASES)
def test_estimated_positions_never_steer_a_degradation(spec):
    """ESTIMATED positions use later frames (05): scramble them and every row that was
    VISIBLE must degrade the same way."""
    objects = synth(2)
    est = ~pl.col("visible")
    scrambled = objects.with_columns(
        x=pl.when(est).then(pl.col("x") * 7 - 20).otherwise("x"),
        y=pl.when(est).then(-pl.col("y") + 3).otherwise("y"),
    )
    vis = objects["visible"]
    equal(run(objects, [spec]).filter(vis), run(scrambled, [spec]).filter(vis))


@pytest.mark.parametrize(
    "spec",
    [
        "camera_drift:0",
        "player_noise:0",
        "ball_noise:0",
        "ball_false:0",
        "ball_miss:0",
        "geom_loss:0",
        "player_miss:0",
        "team_flip:0",
        "team_unknown:0",
        "id_fragment:inf",
    ],
)
def test_zero_severity_changes_nothing(spec):
    objects = synth(3)
    got = run(objects, [spec])
    equal(got, objects.drop(*degrade.DROPPED, strict=False))


def test_realized_severity_is_close_to_asked():
    objects = synth(4, n=3000, estimated=0.0)
    _, stats = degrade.apply(objects, "m1", ["ball_miss:0.3", "player_miss:0.2"])
    got = degrade.summarize(stats)
    assert got["ball_miss"]["share"] == pytest.approx(0.3, abs=0.06)
    assert got["player_miss"]["share"] == pytest.approx(0.2, abs=0.03)
    _, stats = degrade.apply(objects, "m1", ["player_noise:2", "id_fragment:2"])
    got = degrade.summarize(stats)
    assert got["player_noise"]["sigma_m"] == pytest.approx(2, rel=0.1)
    assert got["id_fragment"]["track_life_s"] == pytest.approx(2, rel=0.1)


def test_ball_gaps_are_contiguous():
    objects = synth(5, n=3000, estimated=0.0)
    got = run(objects, ["ball_miss:0.3"]).filter(pl.col("object_type") == "ball")
    hidden = ~got.filter(pl.col("period") == 1)["visible"].to_numpy()
    starts = np.flatnonzero(hidden[1:] & ~hidden[:-1])
    assert hidden.sum() / max(len(starts), 1) > 10  # mean gap 2 s = 20 rows, not 1


def test_geom_loss_hides_whole_frames():
    objects = synth(6, estimated=0.0)
    got = run(objects, ["geom_loss:0.3"])
    per_frame = got.group_by("period", "t_s").agg(pl.col("visible").n_unique())
    assert per_frame["visible"].max() == 1
    assert 0 < (~got["visible"]).mean() < 1


def test_camera_drift_moves_a_frame_together():
    """Offset plus a scale around the ball: every object's offset from the ball scales
    by the same factor within a frame."""
    objects = synth(7, estimated=0.0)
    got = run(objects, ["camera_drift:3"])
    f = pl.col("t_s") == 12.3
    a, b = objects.filter(f, pl.col("period") == 1), got.filter(f, pl.col("period") == 1)
    ball_a = a.filter(pl.col("object_type") == "ball").select("x", "y").row(0)
    ball_b = b.filter(pl.col("object_type") == "ball").select("x", "y").row(0)
    ra = np.hypot(a["x"] - ball_a[0], a["y"] - ball_a[1]).to_numpy()
    rb = np.hypot(b["x"] - ball_b[0], b["y"] - ball_b[1]).to_numpy()
    ratio = rb[ra > 1] / ra[ra > 1]
    assert np.allclose(ratio, ratio[0]) and ratio[0] != 1


def test_ball_false_moves_only_the_visible_ball_by_5_to_30_m():
    objects = synth(8, n=3000)
    got = run(objects, ["ball_false:0.2"])
    ball = pl.col("object_type") == "ball"
    d = np.hypot(
        got.filter(ball)["x"] - objects.filter(ball)["x"],
        got.filter(ball)["y"] - objects.filter(ball)["y"],
    ).to_numpy()
    vis = objects.filter(ball)["visible"].to_numpy()
    moved = d > 1e-9
    assert moved.any() and not (moved & ~vis).any()
    assert d[moved].min() >= 5 - 1e-9 and d[moved].max() <= 30 + 1e-9
    assert got.filter(~ball)["x"].equals(objects.filter(~ball)["x"])


def test_team_errors_touch_only_visible_players_and_favor_the_ball():
    objects = synth(9, n=3000, estimated=0.2)
    got = run(objects, ["team_flip:0.2"])
    changed = (got["team"] != objects["team"]).fill_null(False).to_numpy()
    player = objects["object_type"].is_in(degrade.PLAYER_TYPES).to_numpy()
    assert changed.any() and not (changed & ~(objects["visible"].to_numpy() & player)).any()
    assert set(got["team"].drop_nulls()) == {"home", "away"}
    unknown = run(objects, ["team_unknown:0.2"])
    assert unknown["team"].null_count() > objects["team"].null_count()


def test_no_ball_z():
    got = run(synth(10), ["no_ball_z"])
    assert got["z"].null_count() == got.height


def frames_for(objects):
    g = objects.select("period", "t_s").unique().sort("period", "t_s")
    return g.with_columns(possession_team=pl.lit("home"), flipped=pl.lit(False))


def test_id_fragmentation_kills_player_velocities():
    objects = synth(11, estimated=0.0)
    frames = frames_for(objects)
    clean = match_features(frames, objects.lazy())
    broken = match_features(frames, run(objects, ["id_fragment:0.05"]).lazy())
    assert clean["carrier_speed"].fill_nan(None).drop_nulls().len() > 100
    assert broken["carrier_speed"].fill_nan(None).drop_nulls().len() < 0.2 * clean.height
    # positions are untouched, so position features stay the same
    assert np.allclose(
        clean["carrier_goal_dist"].to_numpy(),
        broken["carrier_goal_dist"].to_numpy(),
        equal_nan=True,
    )


def write_match(tmp_path, objects):
    d = tmp_path / "m1"
    d.mkdir()
    frames = frames_for(objects)
    n = frames.height
    frames.with_columns(
        match_id=pl.lit("m1"),
        ball_state=pl.lit("alive"),
        eligible=pl.lit(True),
        all_estimated=pl.lit(False),
        label_mask_h5=pl.lit(True),
        label_shot_h5=pl.Series([False] * n),
    ).write_parquet(d / "frames_10hz.parquet")
    objects.write_parquet(d / "objects_10hz.parquet")
    return d


def test_degraded_load_never_touches_the_clean_cache(tmp_path):
    objects = synth(12)
    d = write_match(tmp_path, objects)
    clean = load_match("m1", tmp_path, ["h5"])
    cache = d / "features_v1_held.parquet"
    assert cache.exists()
    # poison the clean cache: a degraded load must not read it
    pl.read_parquet(cache).with_columns(ball_x=pl.lit(999.0, pl.Float32)).write_parquet(cache)
    before = sorted(p.name for p in d.iterdir())
    stats = {}
    got = load_match("m1", tmp_path, ["h5"], degrade=["ball_noise:0"], degrade_stats=stats)
    assert sorted(p.name for p in d.iterdir()) == before
    assert pl.read_parquet(cache)["ball_x"].max() == 999.0
    assert got["ball_x"].max() < 999
    for f in FEATURES:  # severity 0 reproduces the clean features exactly
        assert np.allclose(got[f].to_numpy(), clean[f].to_numpy(), equal_nan=True), f
    assert stats["ball_noise"][1] > 0
    with pytest.raises(ValueError, match="held"):
        load_match("m1", tmp_path, ["h5"], ball_source="raw", degrade=["ball_miss:0.1"])


def test_degraded_features_keep_the_scored_rows(tmp_path):
    objects = synth(13)
    write_match(tmp_path, objects)
    clean = load_match("m1", tmp_path, ["h5"], cache=False)
    got = load_match("m1", tmp_path, ["h5"], degrade=["target"])
    cols = ["period", "t_s", "eligible", "all_estimated", "label_mask_h5", "label_shot_h5"]
    assert got.select(cols).equals(clean.select(cols))
    assert not np.allclose(got["ball_x"].to_numpy(), clean["ball_x"].to_numpy(), equal_nan=True)
