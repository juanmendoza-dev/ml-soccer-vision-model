"""Inferred possession arm (05, "Inferred possession"; 07 #6): stage 8's possession feeds
the model's inputs, while labels and scored rows stay the provider's."""

import numpy as np
import polars as pl
import pytest

from prediction import cv, possession
from prediction.features import FEATURES, FRAME_COLS, load_match, match_features
from prediction.resample import process_game
from vision.state import StateConfig

FPS = 25.0
# period, seconds, home_attacks_positive_x
PERIODS = ((1, 20.0, True), (2, 10.0, False))


def native(period: int, t: float) -> dict:
    """Where everything is at native time t. Period 1: home_1 carries the ball for 8 s,
    away_1 takes it until 14 s, then the ball is in the air with nobody on it. Period 2:
    away_1 carries it throughout. home_3 sits on the ball the whole time but is ESTIMATED,
    so stage 8 never sees it."""
    if period == 1:
        if t < 8:
            ball, on = (10 + t, 0.0, 0.0), "home_1"
        elif t < 14:
            ball, on = (18 - 2 * (t - 8), 5.0, 0.0), "away_1"
        else:
            ball, on = (6 + 5 * (t - 14), 10.0, 2.0), None
    else:
        ball, on = (t, 0.0, 0.0), "away_1"
    bx, by, bz = ball
    pos = {
        "ball": (bx, by, bz, True),
        "home_1": (bx + 0.5, by, None, True) if on == "home_1" else (-5.0, 20.0, None, True),
        "away_1": (bx - 0.5, by, None, True) if on == "away_1" else (25.0, -20.0, None, True),
        "home_2": (-20.0, -10.0, None, True),
        "away_2": (30.0, -15.0, None, True),
        "away_gk": (50.0, 0.0, None, True),
        "home_3": (bx, by + 0.2, None, False),
    }
    return pos


KINDS = {
    "ball": ("ball", None),
    "home_1": ("player", "home"),
    "home_2": ("player", "home"),
    "home_3": ("player", "home"),
    "away_1": ("player", "away"),
    "away_2": ("player", "away"),
    "away_gk": ("goalkeeper", "away"),
}


def provider(period: int, t: float) -> str | None:
    """PFF's possession: null for the first 0.5 s, home all of period 1, and in period 2
    home until 5 s, then away."""
    if period == 1:
        return None if t < 0.5 else "home"
    return "home" if t < 5 else "away"


def game(shift_after: float | None = None):
    """02 tables for one synthetic match at 25 fps. shift_after moves every object and
    turns away_1 home on native frames after period 1's shift_after (and all of period 2),
    to check nothing earlier changes."""
    frames, objects, fid = [], [], 0
    for period, seconds, home_pos in PERIODS:
        for i in range(round(seconds * FPS) + 1):
            t = i / FPS
            frames.append(
                {
                    "match_id": "syn",
                    "frame_id": fid,
                    "period": period,
                    "timestamp_s": t,
                    "home_attacks_positive_x": home_pos,
                    "ball_state": "alive",
                    "possession_team": provider(period, t),
                    "ball_carrier_id": None,
                    "view_polygon": None,
                    "set_play_phase": False,
                }
            )
            later = shift_after is not None and (period == 2 or t > shift_after + 1e-9)
            for oid, (x, y, z, vis) in native(period, t).items():
                kind, team = KINDS[oid]
                if later:
                    x += 20.0
                    team = "home" if oid == "away_1" else team
                objects.append(
                    {
                        "match_id": "syn",
                        "frame_id": fid,
                        "object_id": oid,
                        "object_type": kind,
                        "team": team,
                        "player_id": None,
                        "x": x,
                        "y": y,
                        "z": z,
                        "vx": 0.0,
                        "vy": 0.0,
                        "visible": vis,
                        "interpolated": not vis,
                        "confidence": 1.0,
                    }
                )
            fid += 1
    frames = pl.DataFrame(
        frames,
        schema_overrides={
            "possession_team": pl.String,
            "ball_carrier_id": pl.String,
            "view_polygon": pl.List(pl.Float64),
        },
    )
    objects = pl.DataFrame(
        objects, schema_overrides={"team": pl.String, "player_id": pl.String, "z": pl.Float64}
    )
    # a home open-play shot at 12 s in period 1, so there are positives
    shot = frames.filter(pl.col("period") == 1, (pl.col("timestamp_s") - 12.0).abs() < 1e-9)
    events = pl.DataFrame(
        {
            "match_id": ["syn"],
            "frame_id": shot["frame_id"],
            "event_type": ["shot"],
            "team": ["home"],
            "player_id": pl.Series([None], dtype=pl.String),
            "x": [40.0],
            "y": [0.0],
            "outcome": ["saved"],
            "set_piece": ["open_play"],
            "set_play_phase": [False],
        }
    )
    match = pl.DataFrame(
        {"match_id": ["syn"], "source": ["pff"], "native_fps": [FPS], "schema_version": ["0.6"]}
    )
    return frames, objects, events, match


def build(root, shift_after=None):
    """Write the game state and resample it; returns (gamestate_dir, processed_dir)."""
    gs, proc = root / "gamestate", root / "processed"
    d = gs / "syn"
    d.mkdir(parents=True)
    frames, objects, events, match = game(shift_after)
    frames.write_parquet(d / "frames.parquet")
    objects.write_parquet(d / "objects.parquet")
    events.write_parquet(d / "events.parquet")
    match.write_parquet(d / "match.parquet")
    process_game("syn", gs, proc)
    return gs, proc


def load(gs, proc, which, **kw):
    return load_match("syn", proc, ["h5", "h3"], possession=which, gamestate_dir=gs, **kw)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    gs, proc = build(tmp_path_factory.mktemp("clean"))
    return (
        gs,
        proc,
        load(gs, proc, "provider", cache=False),
        load(gs, proc, "inferred", cache=False),
    )


def at(df, period, lo, hi):
    return df.filter(pl.col("period") == period, pl.col("t_s") >= lo, pl.col("t_s") <= hi)


def test_provider_run_is_unchanged(built):
    """With the flag off, load_match is what it was: features from frames_10hz's own
    possession, and the table the cache round-trips."""
    _, proc, prov, _ = built
    frames = pl.read_parquet(proc / "syn" / "frames_10hz.parquet").sort("period", "t_s")
    objects = pl.scan_parquet(proc / "syn" / "objects_10hz.parquet")
    feats = match_features(frames.select("period", "t_s", "possession_team", "flipped"), objects)
    assert prov.select(FEATURES).equals(feats.select(FEATURES))
    cached = load_match("syn", proc, ["h5", "h3"])  # writes the provider cache
    assert cached.equals(prov)
    assert load_match("syn", proc, ["h5", "h3"]).equals(prov)  # and reads it back


def test_labels_and_scored_rows_stay_the_providers(built):
    _, _, prov, inf = built
    cols = FRAME_COLS + [f"label_{x}_{h}" for h in ("h5", "h3") for x in ("mask", "shot")]
    assert inf.columns == prov.columns
    assert inf.select(cols).equals(prov.select(cols))
    assert prov["label_shot_h5"].any()
    # the features do differ, or the test above says nothing
    assert not inf.select(FEATURES).equals(prov.select(FEATURES))


def test_inferred_possession_follows_stage_8(built):
    gs, proc, _, _ = built
    state = possession.inferred_state("syn", gs, proc, cache=False)
    f10 = pl.read_parquet(proc / "syn" / "frames_10hz.parquet")
    swapped = possession.swap_possession(f10, state)
    team = lambda p, lo, hi: at(swapped, p, lo, hi)["possession_team"].unique().to_list()
    assert team(1, 0.0, 0.2) == [None]  # nobody has carried it yet
    assert team(1, 1.0, 8.0) == ["home"]
    # away_1 took it, and it's kept while in the air. home_3 is nearer the ball than
    # away_1 but ESTIMATED: if stage 8 saw it, home would keep the ball throughout
    assert team(1, 9.0, 20.0) == ["away"]
    assert team(2, 1.0, 10.0) == ["away"]


def test_flipped_uses_the_resamplers_rule():
    """Swapping in the provider's own possession reproduces frames_10hz's flipped."""
    frames, _, _, _ = game()
    f10 = pl.DataFrame(
        {
            "frame_id": frames["frame_id"],
            "home_attacks_positive_x": frames["home_attacks_positive_x"],
            "possession_team": pl.Series([None] * frames.height, dtype=pl.String),
            "flipped": pl.Series([None] * frames.height, dtype=pl.Boolean),
        }
    )
    got = possession.swap_possession(f10, frames.select("frame_id", "possession_team"))
    want = frames.select(
        flipped=pl.when(pl.col("possession_team").is_not_null()).then(
            (pl.col("possession_team") == "home") != pl.col("home_attacks_positive_x")
        )
    )
    assert got["flipped"].equals(want["flipped"])


def test_rotation_and_roles_follow_the_inferred_team(built):
    """Period 1 after 9 s: PFF says home (attacks +x, not flipped), stage 8 says away, so
    the inferred arm sees the pitch rotated and the teams swapped."""
    _, _, prov, inf = built
    p, i = at(prov, 1, 9.0, 13.5), at(inf, 1, 9.0, 13.5)
    assert not p["flipped"].any() and not i["flipped"].any()  # the table stays PFF's
    assert np.allclose(i["ball_x"].to_numpy(), -p["ball_x"].to_numpy())
    assert np.allclose(i["ball_y"].to_numpy(), -p["ball_y"].to_numpy())
    # home_1 and home_2 visible, away_1, away_2 and the keeper visible
    assert (p["n_visible_att"] == 2).all() and (p["n_visible_def"] == 3).all()
    assert (i["n_visible_att"] == 3).all() and (i["n_visible_def"] == 2).all()
    # stage 8's change at ~8.3 s restarts possession_s; PFF's never changed
    assert i["possession_s"].max() < 6 < p["possession_s"].min()
    # where they agree (period 1, 1-8 s), the features are the same, except possession_s:
    # stage 8's run started at 0.3 s, PFF's at 0.5 s
    rest = [f for f in FEATURES if f != "possession_s"]
    assert at(prov, 1, 1.0, 7.9).select(rest).equals(at(inf, 1, 1.0, 7.9).select(rest))


def test_an_inferred_null_on_a_scored_row_rotates_nothing(built):
    """Period 2's first 0.2 s: PFF has home (scored, and flipped since home attacks -x in
    period 2), stage 8 has nobody until away_1 has carried for 0.3 s."""
    _, _, prov, inf = built
    p, i = at(prov, 2, 0.0, 0.2), at(inf, 2, 0.0, 0.2)
    assert p.height == 3 and p["label_mask_h5"].all() and p["flipped"].all()
    assert (i["n_visible_att"] == 0).all() and (i["n_visible_def"] == 0).all()
    assert i["possession_s"].is_nan().all() and i["carrier_dist"].is_nan().all()
    assert np.allclose(i["ball_x"].to_numpy(), -p["ball_x"].to_numpy())  # not rotated


def test_inferred_only_uses_the_past(tmp_path):
    """Objects after period 1's 10 s change (moved 20 m, away_1 turns home); nothing on a
    row at or before 10 s may change, stage 8's state included."""
    a_gs, a_proc = build(tmp_path / "a")
    b_gs, b_proc = build(tmp_path / "b", shift_after=10.0)
    a = load(a_gs, a_proc, "inferred", cache=False)
    b = load(b_gs, b_proc, "inferred", cache=False)
    early = (pl.col("period") == 1) & (pl.col("t_s") <= 10.0 + 1e-9)
    assert a.filter(early).equals(b.filter(early))
    assert not a.filter(~early).equals(b.filter(~early))
    frames = pl.read_parquet(a_gs / "syn" / "frames.parquet")
    past = frames.filter(pl.col("period") == 1, pl.col("timestamp_s") <= 10.0)["frame_id"]
    sa = possession.inferred_state("syn", a_gs, a_proc, cache=False)
    sb = possession.inferred_state("syn", b_gs, b_proc, cache=False)
    keep = pl.col("frame_id").is_in(past.implode())
    assert sa.filter(keep).equals(sb.filter(keep))
    assert not sa.equals(sb)


def test_caches_are_separate_and_keyed_by_the_config(tmp_path):
    gs, proc = build(tmp_path)
    d = proc / "syn"
    prov = load(gs, proc, "provider")
    clean = d / "features_v1_held.parquet"
    before = clean.read_bytes()
    stats = {}
    inf = load(gs, proc, "inferred", possession_stats=stats)
    key = possession.config_key(StateConfig())
    assert {p.name for p in d.glob("*.parquet")} == {
        "frames_10hz.parquet",
        "objects_10hz.parquet",
        "features_v1_held.parquet",
        f"features_v1_held_pinf_{key}.parquet",
        f"state_inferred_age_{key}.parquet",
    }
    assert clean.read_bytes() == before
    assert load(gs, proc, "provider").equals(prov)
    assert load(gs, proc, "inferred").equals(inf)  # from its own cache
    assert stats["stage8_s"] > 0 and stats["features_s"] > 0
    assert 0 < stats["possession_differs"] < stats["scored_rows"]
    assert stats["flipped_differs"] > 0 and stats["inferred_null"] > 0
    # another stage 8 config gets its own caches
    other = StateConfig(carrier_radius_m=2.5)
    assert possession.config_key(other) != key
    load(gs, proc, "inferred", state_config=other)
    assert len(list(d.glob("state_inferred_*.parquet"))) == 2
    # a resample drops them, like the feature caches
    process_game("syn", gs, proc)
    assert not list(d.glob("state_inferred_*")) and not list(d.glob("features_v*"))


def test_refused_combinations(tmp_path, capsys):
    gs, proc = build(tmp_path)
    with pytest.raises(ValueError, match="degradations"):
        load(gs, proc, "inferred", degrade=["ball_miss:0.1"])
    with pytest.raises(ValueError, match="gamestate_dir"):
        load_match("syn", proc, ["h5"], possession="inferred")
    with pytest.raises(ValueError, match="possession must be"):
        load(gs, proc, "pff")
    for argv in (
        ["--model", "gnn", "--possession", "inferred"],
        ["--model", "tgnn", "--possession", "inferred"],
        ["--model", "lgbm", "--possession", "inferred", "--degrade", "ball_miss:0.1"],
    ):
        with pytest.raises(SystemExit):
            cv.main(argv)
    assert "lgbm/floor only" in capsys.readouterr().err


def test_run_config_records_the_possession_source():
    """The possession keys go into run.json, not into the model's constructor."""
    from test_cv import world  # pytest puts tests/ on the path

    folds, data, shots = world()
    extra = {
        "possession": "inferred",
        "state_config": StateConfig().to_dict(),
        "possession_scored": {"scored_rows": 1},
    }
    _, meta = cv.run_cv("floor", ["h5"], folds, data, shots, log=lambda m: None, data_config=extra)
    assert {k: meta["config"][k] for k in extra} == extra


# --- stale possession (05, "Stale possession"): carrier age, v3, arm U ---


def test_carrier_age_by_hand():
    """0 on a carrier frame, seconds since the last one while nobody carries, null before
    the period's first carrier, and a new period starts over. Row order is frames'."""
    frames = pl.DataFrame(
        {
            "frame_id": [5, 0, 1, 2, 3, 4, 6, 7],
            "period": [2, 1, 1, 1, 1, 1, 2, 2],
            "timestamp_s": [0.0, 0.0, 0.5, 1.0, 1.5, 3.0, 0.5, 1.0],
        }
    )
    state = pl.DataFrame(
        {
            "frame_id": [0, 1, 2, 3, 4, 5, 6, 7],
            "ball_carrier_id": [None, "h1", "h1", None, None, None, None, "a1"],
        },
        schema_overrides={"ball_carrier_id": pl.String},
    )
    got = possession.carrier_age(frames, state).to_list()
    # frame 5 (period 2, before any carrier) is null even though period 1 had one
    assert got == [None, None, 0.0, 0.0, 0.5, 2.0, None, 0.0]


def test_carrier_age_on_the_synthetic_match(built):
    """home_1 carries from 0.3 s, away_1 from 8.3 s, then the ball goes up at 14 s and
    nobody is a candidate: age grows from the last carrier frame (13.96 s). Period 2
    starts over: null until away_1 has carried for 0.3 s."""
    gs, proc, _, _ = built
    state = possession.inferred_state("syn", gs, proc, cache=False)
    frames = pl.read_parquet(gs / "syn" / "frames.parquet")
    s = frames.select("frame_id", "period", "timestamp_s").join(state, on="frame_id")
    p1 = s.filter(pl.col("period") == 1)
    age = lambda lo, hi: p1.filter(pl.col("timestamp_s").is_between(lo, hi))["carrier_age_s"]
    assert age(0.0, 0.25).is_null().all()
    assert (age(0.3, 7.96) == 0).all() and (age(8.3, 13.96) == 0).all()
    late = p1.filter(pl.col("timestamp_s") >= 14.0)
    assert np.allclose(late["carrier_age_s"], late["timestamp_s"] - 13.96)
    assert late["possession_team"].unique().to_list() == ["away"]  # still carried forward
    p2 = s.filter(pl.col("period") == 2)
    assert p2.filter(pl.col("timestamp_s") < 0.28)["carrier_age_s"].is_null().all()
    assert (p2.filter(pl.col("timestamp_s") >= 0.3)["carrier_age_s"] == 0).all()


def test_v1_inferred_is_match_features_on_stage_8s_possession(built):
    """The new code leaves the v1 inferred arm as it was: features from the swapped
    possession, nothing else."""
    gs, proc, _, inf = built
    frames = pl.read_parquet(proc / "syn" / "frames_10hz.parquet").sort("period", "t_s")
    state = possession.inferred_state("syn", gs, proc, cache=False)
    inputs = possession.swap_possession(frames, state).select(
        "period", "t_s", "possession_team", "flipped"
    )
    feats = match_features(inputs, pl.scan_parquet(proc / "syn" / "objects_10hz.parquet"))
    assert inf.select(FEATURES).equals(feats.select(FEATURES))


@pytest.mark.parametrize("which", ["provider", "inferred"])
def test_v3_is_v1_plus_the_carrier_age(built, which):
    gs, proc, prov, inf = built
    v1 = prov if which == "provider" else inf
    v3 = load(gs, proc, which, cache=False, features_version="3")
    assert v3.columns == [*v1.columns, "poss_carrier_age_s"]
    assert v3.drop("poss_carrier_age_s").equals(v1)
    # the row's native frame's age, NaN before the first carrier
    f10 = pl.read_parquet(proc / "syn" / "frames_10hz.parquet").sort("period", "t_s")
    state = possession.inferred_state("syn", gs, proc, cache=False)
    want = f10.join(state, on="frame_id", how="left", maintain_order="left")["carrier_age_s"]
    got = v3["poss_carrier_age_s"].cast(pl.Float64).fill_nan(None)
    assert np.allclose(got.fill_null(-1), want.fill_null(-1), atol=1e-5)
    assert got.is_null().any() and (got > 3).any()


def test_unknown_nulls_possession_exactly_where_stale(built):
    """Arm U with S = 2 s: past 15.96 s of period 1 stage 8's away is dropped. There
    nobody is an attacker or defender and possession_s is NaN; everywhere else the
    features are the plain inferred arm's. Labels and scored rows stay PFF's."""
    gs, proc, prov, inf = built
    stats = {}
    unk = load(gs, proc, "inferred", cache=False, stale_after=2.0, possession_stats=stats)
    f10 = pl.read_parquet(proc / "syn" / "frames_10hz.parquet").sort("period", "t_s")
    state = possession.inferred_state("syn", gs, proc, cache=False)
    age = f10.join(state, on="frame_id", how="left", maintain_order="left")["carrier_age_s"]
    stale = (age > 2.0).fill_null(False).to_numpy()
    assert stale.any() and not stale.all()
    s = unk.filter(pl.Series(stale))
    assert (s["n_visible_att"] == 0).all() and (s["n_visible_def"] == 0).all()
    assert s["possession_s"].is_nan().all()
    # the fresh rows match the plain inferred arm
    assert unk.filter(pl.Series(~stale)).equals(inf.filter(pl.Series(~stale)))
    cols = FRAME_COLS + [f"label_{x}_{h}" for h in ("h5", "h3") for x in ("mask", "shot")]
    assert unk.select(cols).equals(prov.select(cols))
    scored = (prov["label_mask_h5"] & ~prov["all_estimated"]).to_numpy()
    assert stats["stale_unknown"] == int((stale & scored).sum()) > 0
    summary = possession.summarize(stats)
    assert summary["stale_unknown_share"] > 0
    assert summary["inferred_null_share"] >= summary["stale_unknown_share"]


def test_stale_features_only_use_the_past(tmp_path):
    """v3 and arm U: objects after period 1's 10 s change nothing at rows <= 10 s."""
    a_gs, a_proc = build(tmp_path / "a")
    b_gs, b_proc = build(tmp_path / "b", shift_after=10.0)
    early = (pl.col("period") == 1) & (pl.col("t_s") <= 10.0 + 1e-9)
    for kw in (
        {"features_version": "3"},
        {"stale_after": 2.0},
        {"stale_after": 2.0, "features_version": "3"},
    ):
        for which in ("provider", "inferred") if "stale_after" not in kw else ("inferred",):
            a = load(a_gs, a_proc, which, cache=False, **kw)
            b = load(b_gs, b_proc, which, cache=False, **kw)
            assert a.filter(early).equals(b.filter(early)), (which, kw)
            assert not a.filter(~early).equals(b.filter(~early))


def test_stale_caches_are_separate(tmp_path):
    gs, proc = build(tmp_path)
    d = proc / "syn"
    key = possession.config_key(StateConfig())
    old = {
        "features_v1_held.parquet": load(gs, proc, "provider"),
        f"features_v1_held_pinf_{key}.parquet": load(gs, proc, "inferred"),
    }
    before = {n: (d / n).read_bytes() for n in old}
    new = {
        f"features_v3_held_s8_{key}.parquet": load(gs, proc, "provider", features_version="3"),
        f"features_v3_held_pinf_{key}.parquet": load(gs, proc, "inferred", features_version="3"),
        f"features_v1_held_pinf_{key}_unk2.parquet": load(gs, proc, "inferred", stale_after=2.0),
        f"features_v1_held_pinf_{key}_unk5.parquet": load(gs, proc, "inferred", stale_after=5.0),
    }
    assert {p.name for p in d.glob("features_*")} == set(old) | set(new)
    assert {p.name for p in d.glob("state_*")} == {f"state_inferred_age_{key}.parquet"}
    assert {n: (d / n).read_bytes() for n in old} == before
    for n, df in {**old, **new}.items():  # each reads back from its own cache
        kw = {"features_version": "3"} if "_v3_" in n else {}
        if "_unk" in n:
            kw["stale_after"] = float(n.split("_unk")[1].split(".")[0])
        which = "inferred" if "pinf" in n else "provider"
        assert load(gs, proc, which, **kw).equals(df), n
    assert not new[f"features_v1_held_pinf_{key}_unk2.parquet"].equals(
        new[f"features_v1_held_pinf_{key}_unk5.parquet"]
    )
    process_game("syn", gs, proc)
    assert not list(d.glob("state_*")) and not list(d.glob("features_v*"))


def test_stale_refused_combinations(tmp_path, capsys):
    gs, proc = build(tmp_path)
    for kw in ({"stale_after": 2.0}, {"stale_after": 0.0, "which": "inferred"}):
        which = kw.pop("which", "provider")
        with pytest.raises(ValueError, match="stale_after"):
            load(gs, proc, which, **kw)
    with pytest.raises(ValueError, match="degradations"):
        load(gs, proc, "provider", features_version="3", degrade=["ball_miss:0.1"])
    with pytest.raises(ValueError, match="gamestate_dir"):
        load_match("syn", proc, ["h5"], features_version="3")
    for argv, msg in (
        (["--stale-possession", "unknown", "--stale-after", "10"], "needs --possession inferred"),
        (["--possession", "inferred", "--stale-possession", "unknown"], "positive --stale-after"),
        (["--possession", "inferred", "--stale-after", "10"], "needs --stale-possession"),
        (["--model", "gnn", "--features-version", "3"], "lgbm/floor only"),
        (["--model", "lgbm", "--features-version", "3", "--degrade", "ball_miss:0.1"], "--degrade"),
    ):
        with pytest.raises(SystemExit):
            cv.main(["--model", "lgbm", *argv] if "--model" not in argv else argv)
        assert msg in capsys.readouterr().err


def test_run_config_records_the_stale_arm():
    from test_cv import world

    folds, data, shots = world()
    extra = {"stale_possession": {"arm": "unknown", "after_s": 10.0}}
    _, meta = cv.run_cv("floor", ["h5"], folds, data, shots, log=lambda m: None, data_config=extra)
    assert meta["config"]["stale_possession"] == extra["stale_possession"]
