import ast
import json
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from prediction.lgbm import PARAMS as BASELINE
from prediction.resample import add_frame_flags, build_grid
from vision import possession_features as pf
from vision import possession_model as pm
from vision import possession_train as pt

GS = Path("data/gamestate")
PROCESSED = Path("data/processed")
SMALL = {"min_data_in_leaf": 20}  # synthetic matches are far too small for 500-row leaves


def game(seed, seconds=40.0, fps=10.0):
    """A synthetic PFF-like match whose label is the team nearest the ball, with dead
    spells, null possession, set plays and frames where every player is ESTIMATED."""
    rng = np.random.default_rng(seed)
    n = int(seconds * fps)
    ids = [(f"h{i}", "player", "home") for i in range(4)] + [
        (f"a{i}", "player", "away") for i in range(4)
    ]
    ids += [("hk", "goalkeeper", "home"), ("ak", "goalkeeper", "away")]
    pos = {o: rng.uniform([-40, -25], [40, 25]) for o, _, _ in ids}
    ball = np.zeros(2)
    rows, vis, poss, state = [], [], [], []
    for f in range(n):
        near = min(ids, key=lambda o: np.hypot(*(pos[o[0]] - ball)))
        ball = ball + 0.3 * (pos[near[0]] - ball) + rng.normal(0, 0.5, 2)
        hidden = 100 <= f < 110  # every player ESTIMATED
        rows.append((f, "ball", "ball", None, *ball))
        vis.append(True)
        for o, kind, team in ids:
            pos[o] = pos[o] + rng.normal(0, 0.3, 2)
            rows.append((f, o, kind, team, *pos[o]))
            vis.append(not hidden and rng.random() > 0.05)
        poss.append(None if 50 <= f < 60 else near[2])
        state.append("dead" if 200 <= f < 230 else "alive")
    frames = pl.DataFrame(
        {
            "frame_id": list(range(n)),
            "period": [1] * n,
            "timestamp_s": [round(i / fps, 6) for i in range(n)],
            "home_attacks_positive_x": [seed % 2 == 0] * n,
            "ball_state": state,
            "possession_team": poss,
            "set_play_phase": [300 <= f < 320 for f in range(n)],
        }
    )
    objects = pl.DataFrame(
        rows, schema=["frame_id", "object_id", "object_type", "team", "x", "y"], orient="row"
    ).with_columns(
        visible=pl.Series(vis),
        interpolated=pl.lit(False),
        z=pl.lit(None, pl.Float64),
        vx=pl.lit(0.0),
        vy=pl.lit(0.0),
        player_id=pl.col("object_id"),
    )
    return frames, objects, fps


def write_game(root, match_id, frames, objects, fps):
    d = root / match_id
    d.mkdir(parents=True)
    frames.write_parquet(d / "frames.parquet")
    objects.write_parquet(d / "objects.parquet")
    pl.DataFrame({"match_id": [match_id], "source": ["pff"], "native_fps": [fps]}).write_parquet(
        d / "match.parquet"
    )


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    """Eight cached synthetic matches: (gamestate dir, cache dir, {id: match_rows})."""
    root = tmp_path_factory.mktemp("pm")
    gs, cache = root / "gs", root / "cache"
    tables = {}
    for i in range(8):
        m = f"m{i}"
        write_game(gs, m, *game(i))
        pf.build(m, gs, cache)
        tables[m] = pm.match_rows(m, gs, cache)
    return gs, cache, tables


def test_params_are_the_baselines_plus_seed_and_threads():
    assert pm.PARAMS == {**BASELINE, "seed": 20260927, "num_threads": 6}


def test_canonical_makes_the_mirror_an_exact_involution(synthetic):
    _, _, tables = synthetic
    x = pm.canonical(tables["m0"].select(pf.COLUMNS))
    a, b = x.to_numpy(), pf.mirror(pf.mirror(x)).to_numpy()
    assert ((a == b) | (np.isnan(a) & np.isnan(b))).all()


def test_training_rows_drop_dead_unlabelled_and_all_estimated_but_keep_set_plays():
    frames, objects, fps = game(1)
    lab = pm.labels(frames, objects, fps).join(
        frames.select("frame_id", "ball_state", "possession_team", "set_play_phase"),
        on="frame_id",
    )
    by = dict(zip(lab["frame_id"], lab.iter_rows(named=True)))
    assert not by[210]["train"] and by[210]["ball_state"] == "dead"
    assert not by[55]["train"] and by[55]["label"] is None
    assert not by[105]["train"] and by[105]["all_estimated"]
    assert by[310]["train"] and by[310]["set_play_phase"]
    assert lab.filter("train")["label"].is_in([0.0, 1.0]).all()
    home = lab.filter(pl.col("possession_team") == "home")
    assert (home["label"] == 1.0).all()


def test_all_estimated_matches_the_resampler_on_a_synthetic_match():
    frames, objects, fps = game(2)
    ours = pm.labels(frames, objects, fps)
    grid, _ = build_grid(frames, fps)
    theirs = add_frame_flags(grid, objects)
    assert ours["frame_id"].to_list() == theirs["frame_id"].to_list()
    assert ours["all_estimated"].to_list() == theirs["all_estimated"].to_list()
    assert ours["all_estimated"].any()


def pff_games():
    return pf.pff_games(GS) if GS.exists() else []


@pytest.mark.skipif(not pff_games(), reason="needs PFF game state and resampled frames")
def test_labels_match_the_resampler_on_every_pff_match():
    games = pff_games()
    assert len(games) == 64
    for m in games:
        fps = pl.read_parquet(GS / m / "match.parquet")["native_fps"][0]
        frames = pl.read_parquet(GS / m / "frames.parquet")
        ours = pm.labels(frames, pl.scan_parquet(GS / m / "objects.parquet"), fps)
        theirs = pl.read_parquet(PROCESSED / m / "frames_10hz.parquet")
        assert ours["frame_id"].equals(theirs["frame_id"]), m
        assert ours["all_estimated"].to_list() == theirs["all_estimated"].to_list(), m
        alive = ours.join(
            frames.select("frame_id", "ball_state"), on="frame_id", maintain_order="left"
        )["ball_state"]
        eligible = (alive == "alive").fill_null(False) & ours["label"].is_not_null()
        assert eligible.to_list() == theirs["eligible"].to_list(), m
        train = theirs["eligible"] & ~theirs["all_estimated"]
        assert ours["train"].to_list() == train.to_list(), m


def test_es_split_is_the_baselines_draw():
    ids = [f"{10500 + i}" for i in range(51)]
    rng = np.random.default_rng(20260927)
    expected = sorted(rng.choice(np.array(sorted(ids)), 8, replace=False).tolist())
    assert pm.es_split(reversed(ids)) == expected  # round(0.15 * 51) = 8, from sorted IDs
    with pytest.raises(ValueError):
        pm.es_split(["a", "b"])


def test_design_strides_by_grid_tick_and_mirrors_with_flipped_labels(synthetic):
    _, _, tables = synthetic
    t = tables["m3"]
    x, y, n = pm.design([t], stride=2)
    kept = t.filter(pl.col("train"), pl.col("k") % 2 == 0)
    assert n == kept.height and x.shape == (2 * n, 159) and x.dtype == np.float32
    f = pm.canonical(kept.select(pf.COLUMNS))
    np.testing.assert_array_equal(x[:n], f.to_numpy())
    np.testing.assert_array_equal(x[n:], pf.mirror(f).to_numpy())
    np.testing.assert_array_equal(y[n:], 1.0 - y[:n])
    np.testing.assert_array_equal(y[:n], kept["label"].to_numpy())


def test_fit_is_deterministic_and_keeps_es_matches_whole(synthetic, tmp_path):
    _, _, tables = synthetic
    ids = sorted(tables)[:7]
    a = pm.fit(tables, ids, tmp_path / "a", params=SMALL)
    b = pm.fit(tables, list(reversed(ids)), tmp_path / "b", params=SMALL)
    assert a["model_sha256"] == b["model_sha256"]
    assert (tmp_path / "a" / "model.txt").read_bytes() == (
        tmp_path / "b" / "model.txt"
    ).read_bytes()
    assert a["es_ids"] == pm.es_split(ids) and not set(a["es_ids"]) & set(a["probe_ids"])
    assert sorted(a["probe_ids"] + a["es_ids"]) == ids == a["refit_ids"]
    assert a["rows"]["refit"]["after_stride"] == sum(tables[m]["train"].sum() for m in ids)
    assert a["best_iteration"] >= 1 and a["training_sha256"] == pm.ids_sha256(ids)


def test_fit_refuses_a_set_without_eligible_rows(synthetic, tmp_path):
    _, _, tables = synthetic
    empty = {m: t.with_columns(train=pl.lit(False)) for m, t in tables.items()}
    with pytest.raises(ValueError, match="no eligible rows"):
        pm.fit(empty, sorted(empty), tmp_path, params=SMALL)


def test_symmetrized_probability_sums_to_one_under_the_mirror(synthetic, tmp_path):
    import lightgbm as lgb

    _, _, tables = synthetic
    ids = sorted(tables)[:7]
    pm.fit(tables, ids, tmp_path, params=SMALL)
    booster = lgb.Booster(model_file=str(tmp_path / "model.txt"))
    x = tables["m7"]
    # ordinary rows, all-NaN rows and a row whose home and away values tie
    blank = x.head(3).with_columns(pl.lit(np.nan, pl.Float32).alias(c) for c in pf.COLUMNS)
    tied = x.head(1).with_columns(
        pl.col(f"home_{s}").alias(f"away_{s}") for s in ("centroid_x", "visible_n", "within5")
    )
    rows = pl.concat([x, blank, tied])
    p = pm.p_home(booster, rows)
    pmir = pm.p_home(booster, pf.mirror(pm.canonical(rows)))
    assert p.dtype == np.float64 and np.isfinite(p).all()
    assert np.abs(p + pmir - 1.0).max() <= 1e-12
    assert p.std() > 0  # the synthetic label is learnable, so p isn't constant


FOLDS = {f"g{i:02d}": i % 5 for i in range(20)}


def hand_manifest(folds=FOLDS, dedup=True):
    """A sealed manifest for `folds` with placeholder model hashes (structure only)."""
    fits = pm.plan(folds, dedup)
    trs = {
        name: {
            "dir": f"fits/{name}",
            "training_ids": ids,
            "training_sha256": pm.ids_sha256(ids),
            "model_sha256": "-",
        }
        for name, ids in pm.trainings(fits).items()
    }
    logical = {
        lid: {k: f[k] for k in ("training", "role", "outer", "fold", "predicts")}
        | {"training_sha256": trs[f["training"]]["training_sha256"]}
        for lid, f in fits.items()
    }
    return {
        "sealed": True,
        "dedup_pairs": dedup,
        "folds": {"assignments": folds},
        "trainings": trs,
        "logical": logical,
    }


def test_plan_is_25_logical_fits_over_15_trainings():
    fits = pm.plan(FOLDS)
    assert len(fits) == 25 and len(pm.trainings(fits)) == 15
    assert len(pm.trainings(pm.plan(FOLDS, dedup=False))) == 25
    users = {}
    for lid, f in fits.items():
        users.setdefault(f["training"], []).append(lid)
        outer = {m for m, j in FOLDS.items() if j == f["outer"]}
        assert not set(f["training_ids"]) & (outer | set(f["predicts"])), lid
        assert set(f["predicts"]) == {m for m, j in FOLDS.items() if j == f["fold"]}
    for name, lids in users.items():
        if name.startswith("pair_"):
            a, b = map(int, name.split("_")[1:])
            assert sorted(lids) == [f"outer_{a}_inner_{b}", f"outer_{b}_inner_{a}"]
            assert fits[lids[0]]["training_ids"] == fits[lids[1]]["training_ids"]
    # final aliases outer_j's test output: every match is predicted once, out of fold
    tests = [m for k in range(5) for m in fits[f"outer_{k}"]["predicts"]]
    assert sorted(tests) == sorted(FOLDS)


def test_manifest_check_accepts_the_plan_and_fails_closed():
    pm.check_manifest(hand_manifest())
    pm.check_manifest(hand_manifest(dedup=False))

    def broken(edit):
        man = hand_manifest()
        edit(man)
        with pytest.raises(ValueError, match="manifest"):
            pm.check_manifest(man)

    broken(lambda m: m.update(sealed=False))
    broken(lambda m: m["logical"].pop("outer_3_inner_1"))
    broken(lambda m: m["trainings"].pop("pair_1_3"))
    broken(lambda m: m["logical"]["outer_0"].update(predicts=m["logical"]["outer_1"]["predicts"]))
    broken(lambda m: m["logical"]["outer_2_inner_4"].update(training="outer_2"))
    broken(lambda m: m["logical"]["outer_1_inner_0"].update(training_sha256="-"))

    def leak(m):  # outer 0's test match sneaks into a training set, hash updated to match
        tr = m["trainings"]["pair_1_2"]
        tr["training_ids"] = sorted([*tr["training_ids"], "g00"])
        tr["training_sha256"] = pm.ids_sha256(tr["training_ids"])

    broken(leak)


def test_rule_grid_matches_the_cached_rule_columns(synthetic):
    gs, _, tables = synthetic
    for m in ("m0", "m1"):
        frames = pl.read_parquet(gs / m / "frames.parquet")
        rule = pm.rule_grid(frames, pl.read_parquet(gs / m / "objects.parquet"), 10.0)
        t = tables[m]
        assert rule.select("period", "k", "frame_id").equals(t.select("period", "k", "frame_id"))
        code = rule["rule_possession_team"].replace_strict(
            {"home": 1.0, "away": -1.0}, default=None, return_dtype=pl.Float64
        )
        np.testing.assert_array_equal(code.fill_null(np.nan), t["rule_team"].cast(pl.Float64))
        np.testing.assert_allclose(
            rule["carrier_age_s"].fill_null(np.nan),
            t["rule_carrier_age_s"].cast(pl.Float64),
            rtol=1e-6,
            equal_nan=True,
        )
        assert rule["ball_carrier_id"].is_not_null().any()
        assert rule["ball_state"].is_in(["alive", "dead"]).any()


def scramble_hidden(frames, objects, seed=0):
    """Scramble what a VISIBLE-only consumer must never read: invisible objects'
    position, team and identity, every z, provider vx/vy and player_id, and PFF's
    possession and ball state."""
    rng = np.random.default_rng(seed)
    n = objects.height
    hidden = ~objects["visible"]
    objects = objects.with_columns(
        x=pl.when(hidden).then(pl.Series(rng.uniform(-50, 50, n))).otherwise("x"),
        y=pl.when(hidden).then(pl.Series(rng.uniform(-30, 30, n))).otherwise("y"),
        team=pl.when(hidden).then(pl.lit("away")).otherwise("team"),
        object_id=pl.when(hidden)
        .then(pl.lit("ghost") + pl.int_range(n).cast(pl.String))
        .otherwise("object_id"),
        z=pl.Series(rng.uniform(0, 3, n)),
        vx=pl.Series(rng.normal(0, 5, n)),
        vy=pl.Series(rng.normal(0, 5, n)),
        player_id=pl.lit("p") + pl.Series(rng.integers(0, 99, n)).cast(pl.String),
    )
    frames = frames.with_columns(
        possession_team=pl.lit("away"), ball_state=pl.lit("dead"), set_play_phase=pl.lit(True)
    )
    return frames, objects


def test_the_2d_rule_is_mirror_symmetric_with_the_same_object_ids():
    """Fallback rows keep the 2D rule's possession, so it must flip with the scene: teams
    swapped and x negated give the other team, the same carrier and the same ball state."""
    frames, objects, fps = game(1)  # both teams carry in this one
    a = pm.rule_grid(frames, objects, fps)
    swapped = objects.with_columns(
        x=-pl.col("x"),
        team=pl.col("team").replace_strict({"home": "away", "away": "home"}, default=None),
    )
    b = pm.rule_grid(frames, swapped, fps)
    flip = {"home": "away", "away": "home"}
    assert b["rule_possession_team"].to_list() == [
        flip.get(t) for t in a["rule_possession_team"].to_list()
    ]
    assert a["rule_possession_team"].drop_nulls().n_unique() == 2
    assert b.select("ball_carrier_id", "ball_state", "carrier_age_s").equals(
        a.select("ball_carrier_id", "ball_state", "carrier_age_s")
    )


def test_rule_grid_is_visible_only_and_ignores_height_and_pff_state():
    frames, objects, fps = game(4)
    a = pm.rule_grid(frames, objects, fps)
    b = pm.rule_grid(*scramble_hidden(frames, objects), fps)
    assert a.equals(b)


@pytest.mark.parametrize("module", [pm, pt])
def test_vision_model_modules_never_import_prediction(module):
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    names += [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not [m for m in names if m.split(".")[0] in ("prediction", "evaluation")]


MODEL_ID = "lgbm-v1-nested5x4-s1"


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Ten cached synthetic matches, two per fold, trained and sealed by the driver."""
    root = tmp_path_factory.mktemp("pt")
    gs, cache, models = root / "gs", root / "cache", root / "models"
    folds = {f"g{i:02d}": i % 5 for i in range(10)}
    for i, m in enumerate(folds):
        write_game(gs, m, *game(100 + i))
        pf.build(m, gs, cache)
    spec = {
        "n_folds": 5,
        "seed": 0,
        "frozen": {"pff": "test"},
        "matches": [{"match_id": m, "source": "pff", "fold": f} for m, f in folds.items()],
    }
    folds_path = root / "folds.json"
    folds_path.write_text(json.dumps(spec))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pt, "FOLD_SIZES", [2] * 5)
        man = pt.run(MODEL_ID, None, gs, cache, models, folds_path, log=lambda _: None)
        yield {
            "gs": gs,
            "cache": cache,
            "models": models,
            "folds_path": folds_path,
            "folds": folds,
            "dir": models / MODEL_ID,
            "manifest": man,
        }


def test_driver_seals_25_logical_fits_over_15_trainings(trained):
    man = json.loads((trained["dir"] / pm.MANIFEST).read_text())
    assert man == json.loads(json.dumps(trained["manifest"]))
    pm.check_manifest(man)
    assert man["sealed"] and man["dedup_pairs"] and man["determinism"]["same"]
    assert len(man["logical"]) == 25 and len(man["trainings"]) == 15
    assert sorted(man["inputs"]) == sorted(trained["folds"])
    for name, tr in man["trainings"].items():
        fit_dir = trained["dir"] / tr["dir"]
        assert pf.sha256_file(fit_dir / "model.txt") == tr["model_sha256"], name
        assert pf.sha256_file(fit_dir / "fit.json") == tr["fit_sha256"], name
        assert set(tr["es_ids"]) <= set(tr["training_ids"])


def test_a_sealed_model_id_is_never_retrained(trained):
    t = trained
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pt, "FOLD_SIZES", [2] * 5)
        with pytest.raises(ValueError, match="sealed"):
            pt.run(MODEL_ID, None, t["gs"], t["cache"], t["models"], t["folds_path"])


def files_under(root):
    return sorted(p for p in root.rglob("*") if p.is_file())


def test_predict_match_uses_the_right_fit_per_context(trained):
    d, gs, cache = trained["dir"], trained["gs"], trained["cache"]
    before = files_under(gs.parent)
    out = pm.predict_match(d, "outer_0", "g00", gs, cache)
    assert files_under(gs.parent) == before  # writes nothing
    assert out.columns == pm.STATE_COLUMNS
    assert (out["fit_id"] == "outer_0").all()
    fb = out.filter("fallback")
    assert fb.height and fb["p_home"].is_null().all()
    assert fb["possession_team"].to_list() == fb["rule_possession_team"].to_list()
    live = out.filter(~pl.col("fallback"))
    assert live["p_home"].is_not_null().all()
    assert live["possession_team"].to_list() == pm.team(live["p_home"].to_numpy()).tolist()
    # one match, two outer contexts: test under outer_0, inner-oof under outer_1
    inner = pm.predict_match(d, "outer_1_inner_0", "g00", gs, cache)
    assert not inner["p_home"].equals(out["p_home"])
    man = trained["manifest"]  # both inner IDs of a pair are one booster
    assert man["logical"]["outer_1_inner_0"]["training"] == "pair_0_1"
    assert man["logical"]["outer_0_inner_1"]["training"] == "pair_0_1"


def test_predict_match_fails_closed(trained, tmp_path):
    d, gs, cache = trained["dir"], trained["gs"], trained["cache"]
    with pytest.raises(ValueError, match="doesn't predict"):
        pm.predict_match(d, "outer_0", "g01", gs, cache)
    with pytest.raises(ValueError, match="no logical fit"):
        pm.predict_match(d, "live_all64", "g00", gs, cache)

    def tampered(name, edit):
        copy = tmp_path / name / MODEL_ID
        shutil.copytree(d, copy)
        man = json.loads((copy / pm.MANIFEST).read_text())
        edit(copy, man)
        (copy / pm.MANIFEST).write_text(json.dumps(man))
        return copy

    def append_line(copy, man):
        with open(copy / "fits/outer_0/model.txt", "a") as fh:
            fh.write("\n")

    with pytest.raises(ValueError, match="model hash"):
        pm.predict_match(tampered("model", append_line), "outer_0", "g00", gs, cache)
    stale = tampered("inputs", lambda c, m: m["inputs"]["g00"].update(grid_sha256="-"))
    with pytest.raises(ValueError, match="grid_sha256"):
        pm.predict_match(stale, "outer_0", "g00", gs, cache)
    unsealed = tampered("unsealed", lambda c, m: m.update(sealed=False))
    with pytest.raises(ValueError, match="not sealed"):
        pm.predict_match(unsealed, "outer_0", "g00", gs, cache)


def test_outer_test_labels_cannot_change_that_contexts_weights(trained, tmp_path):
    gs, cache, folds = trained["gs"], trained["cache"], trained["folds"]
    tables, _ = pt.load_tables(sorted(folds), gs, cache)
    flipped = {
        m: t.with_columns(label=1.0 - pl.col("label")) if folds[m] == 0 else t
        for m, t in tables.items()
    }
    ids = trained["manifest"]["trainings"]["outer_0"]["training_ids"]
    rec = pm.fit(flipped, ids, tmp_path)
    assert rec["model_sha256"] == trained["manifest"]["trainings"]["outer_0"]["model_sha256"]


def test_tie_goes_home_and_hard_labels_swap_away_from_it():
    p = np.array([0.5, 0.2, 0.8, 0.5 - 1e-12])
    assert pm.team(p).tolist() == ["home", "away", "home", "away"]
    assert pm.team(1.0 - p[1:3]).tolist() == ["home", "away"]


def test_learned_output_is_causal(trained):
    """Fixed weights; frames after the cut are scrambled or dropped. Every grid row up to
    the cut keeps its features, rule state and probability."""
    import lightgbm as lgb

    frames, objects, fps = game(7)
    cut = 250  # grid tick k == native frame id here (10 fps from t = 0)
    later = objects["frame_id"] > cut
    rng = np.random.default_rng(1)
    noisy = objects.with_columns(
        x=pl.when(later).then(pl.Series(rng.uniform(-50, 50, objects.height))).otherwise("x")
    )
    short = (
        frames.filter(pl.col("frame_id") <= cut + 40),
        objects.filter(pl.col("frame_id") <= cut + 40),
    )
    booster = lgb.Booster(model_file=str(trained["dir"] / "fits/outer_0/model.txt"))
    base = pf.extract(frames, objects, fps)
    p0 = pm.p_home(booster, base)
    r0 = pm.rule_grid(frames, objects, fps)
    for f, o in ((frames, noisy), short):
        x = pf.extract(f, o, fps)
        n = int((x["k"] <= cut).sum())
        assert x.head(n).equals(base.head(n))
        np.testing.assert_array_equal(pm.p_home(booster, x)[:n], p0[:n])
        assert pm.rule_grid(f, o, fps).head(n).equals(r0.head(n))


def test_learned_inputs_ignore_everything_invisible_and_pff_state():
    frames, objects, fps = game(5)
    a = pf.extract(frames, objects, fps)
    f2, o2 = scramble_hidden(frames, objects, seed=3)
    assert pf.extract(f2, o2, fps).equals(a)
