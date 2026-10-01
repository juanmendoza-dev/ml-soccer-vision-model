"""Learned possession plumbing (03 stage 8, "Prediction plumbing"; 05 "Learned
possession"): per-context learned state and features, hash checks on every read, the
fold data provider in prediction.cv and the refused combinations. Ten copies of
test_possession's synthetic match, two per fold, with a model trained and sealed by
vision.possession_train."""

import argparse
import json
import shutil

import polars as pl
import pytest
from test_possession import game

from prediction import cv, possession
from prediction.features import FEATURES, FRAME_COLS, load_match, match_features
from prediction.resample import process_game
from vision import possession_features as vpf
from vision import possession_train as pt
from vision.state import StateConfig

IDS = [f"s{i}" for i in range(10)]
MODEL = "lgbm-v1-nested5x4-s1"
CONFIG = StateConfig(possession_model=MODEL)
KEY = possession.config_key(CONFIG)
H = ["h5", "h3"]
LABELS = [f"label_{x}_{h}" for h in H for x in ("mask", "shot")]


def fold_of(m):
    return IDS.index(m) % 5


def make_world(root):
    gs, proc, vc, models = (root / n for n in ("gamestate", "processed", "vision", "models"))
    tables = dict(zip(("frames", "objects", "events", "match"), game(), strict=True))
    for m in IDS:
        d = gs / m
        d.mkdir(parents=True)
        for name, t in tables.items():
            t.with_columns(match_id=pl.lit(m)).write_parquet(d / f"{name}.parquet")
        process_game(m, gs, proc, vc)
        vpf.build(m, gs, vc)
    folds = {
        "n_folds": 5,
        "seed": 7,
        "frozen": {"pff": "test"},
        "matches": [
            {"match_id": m, "source": "pff", "fold": fold_of(m), "open_play_shots": 1} for m in IDS
        ],
    }
    folds_path = root / "folds.json"
    folds_path.write_text(json.dumps(folds))
    return {"root": root, "gs": gs, "proc": proc, "vc": vc, "models": models, "folds": folds}


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    w = make_world(tmp_path_factory.mktemp("learned"))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pt, "FOLD_SIZES", [2] * 5)
        pt.run(MODEL, None, w["gs"], w["vc"], w["models"], w["root"] / "folds.json", lambda _: 0)
    return w


@pytest.fixture
def w(trained, monkeypatch):
    monkeypatch.setattr(possession, "MODELS_DIR", trained["models"])
    monkeypatch.setattr(possession, "VISION_CACHE_DIR", trained["vc"])
    return trained


@pytest.fixture
def copy(trained, tmp_path, monkeypatch):
    """A private copy of the data (not the models) to tamper with."""
    out = dict(trained)
    for n in ("gs", "proc", "vc"):
        out[n] = tmp_path / n
        shutil.copytree(trained[n], out[n])
    monkeypatch.setattr(possession, "MODELS_DIR", trained["models"])
    monkeypatch.setattr(possession, "VISION_CACHE_DIR", out["vc"])
    return out


def learned(w, m, ctx, **kw):
    return load_match(
        m,
        w["proc"],
        H,
        possession="inferred",
        gamestate_dir=w["gs"],
        state_config=CONFIG,
        context=ctx,
        **kw,
    )


def state(w, m, ctx):
    return possession.learned_state(m, CONFIG, ctx, w["gs"], w["proc"])


def test_contexts_name_their_fit_and_cache():
    c = possession.context_for
    assert (c(2, 2).role, c(2, 2).fit_id, c(2, 2).suffix) == ("test", "outer_2", "outer2_test")
    inner = c(2, 4)
    assert (inner.role, inner.fit_id, inner.suffix) == (
        "inner-oof",
        "outer_2_inner_4",
        "outer2_inner-oof4",
    )
    final = c(None, 3)
    assert (final.role, final.fit_id, final.suffix) == ("final", "outer_3", "outer3_test")
    for bad in (("test", 1, 2), ("inner-oof", 1, 1), ("final", 1, 1), ("oof", 1, 2)):
        with pytest.raises(ValueError, match="invalid context"):
            possession.Context(*bad)


def test_learned_load_keeps_provider_labels_and_uses_the_contexts_possession(w):
    ctx = possession.context_for(0, 0)
    stats = {}
    got = learned(w, "s0", ctx, possession_stats=stats)
    prov = load_match("s0", w["proc"], H)
    assert got.select(FRAME_COLS + LABELS).equals(prov.select(FRAME_COLS + LABELS))
    st = state(w, "s0", ctx)
    assert (st["fit_id"] == "outer_0").all()
    frames = pl.read_parquet(w["proc"] / "s0" / "frames_10hz.parquet").sort("period", "t_s")
    inputs = possession.swap_learned(frames, st).select(
        "period", "t_s", "possession_team", "flipped"
    )
    want = match_features(inputs, pl.scan_parquet(w["proc"] / "s0" / "objects_10hz.parquet"))
    assert got.select(FEATURES).equals(want.select(FEATURES))
    assert stats["scored_rows"] > 0 and stats["grid_rows"] == st.height
    assert stats["fallback_rows"] == int(st["fallback"].sum())
    d = w["proc"] / "s0"
    for stem in (
        f"features_v1_held_pinf_{KEY}_outer0_test",
        f"state_inferred_age_{KEY}_outer0_test",
    ):
        assert (d / f"{stem}.parquet").exists() and (d / f"{stem}.json").exists()
    assert not list(d.glob(f"*_{KEY}.parquet"))  # never an unsuffixed learned cache
    assert learned(w, "s0", ctx).equals(got)  # from its own cache, hashes rechecked


def test_one_match_in_two_outer_contexts_reads_two_fits(w):
    test = state(w, "s0", possession.context_for(0, 0))
    inner = state(w, "s0", possession.context_for(1, 0))
    assert test["fit_id"][0] == "outer_0" and inner["fit_id"][0] == "outer_1_inner_0"
    assert (w["proc"] / "s0" / f"state_inferred_age_{KEY}_outer1_inner-oof0.parquet").exists()
    # final is outer_0's test output, byte for byte the same file
    assert state(w, "s0", possession.context_for(None, 0)).equals(test)
    # only test-role loads count
    stats = {}
    learned(w, "s0", possession.context_for(1, 0), possession_stats=stats)
    learned(w, "s0", possession.context_for(None, 0), possession_stats=stats)
    assert "scored_rows" not in stats and "fallback_rows" not in stats


def test_learned_loads_refuse_bad_requests(w):
    with pytest.raises(ValueError, match="fold"):
        state(w, "s0", possession.context_for(1, 1))  # s0 is in fold 0
    with pytest.raises(ValueError, match="context"):
        learned(w, "s0", None)
    with pytest.raises(ValueError, match="context"):
        load_match("s0", w["proc"], H, "held", possession="inferred", gamestate_dir=w["gs"],
                   context=possession.context_for(0, 0))  # fmt: skip
    with pytest.raises(ValueError, match="possession='inferred'"):
        load_match("s0", w["proc"], H, state_config=CONFIG, context=possession.context_for(0, 0))
    with pytest.raises(ValueError, match="v1, held"):
        learned(w, "s0", possession.context_for(0, 0), features_version="2")
    with pytest.raises(FileNotFoundError, match="possession_train"):
        possession.learned_manifest(StateConfig(possession_model="lgbm-v1-nested5x4-s2"))


def test_learned_caches_are_refused_once_anything_they_came_from_changes(copy):
    ctx = possession.context_for(0, 0)
    d = copy["proc"] / "s0"
    learned(copy, "s0", ctx)
    feats = d / f"features_v1_held_pinf_{KEY}_outer0_test.parquet"
    st = d / f"state_inferred_age_{KEY}_outer0_test.parquet"
    good_feats, good_state = feats.read_bytes(), st.read_bytes()

    pl.read_parquet(feats).head(5).write_parquet(feats)
    with pytest.raises(ValueError, match="output_sha256"):
        learned(copy, "s0", ctx)
    feats.write_bytes(good_feats)

    pl.read_parquet(st).with_columns(possession_team=pl.lit("home")).write_parquet(st)
    with pytest.raises(ValueError, match="output_sha256"):
        state(copy, "s0", ctx)
    st.write_bytes(good_state)

    side = st.with_suffix(".json")
    good_side = side.read_bytes()
    side.unlink()
    with pytest.raises(ValueError, match="no provenance"):
        state(copy, "s0", ctx)
    side.write_bytes(good_side)
    assert learned(copy, "s0", ctx).height  # restored: accepted again

    # a reconvert without a resample: the native objects change under the caches
    objects = copy["gs"] / "s0" / "objects.parquet"
    pl.read_parquet(objects).with_columns(x=pl.col("x") + 1).write_parquet(objects)
    with pytest.raises(ValueError, match="objects_sha256"):
        learned(copy, "s0", ctx)  # the learned state cache refuses
    for f in d.glob(f"state_inferred_age_{KEY}_outer1_*"):
        f.unlink()
    with pytest.raises(ValueError, match="objects_sha256"):
        state(copy, "s0", possession.context_for(1, 0))  # a rebuild: the input cache refuses


def test_resample_removes_every_learned_and_vision_cache(copy):
    learned(copy, "s0", possession.context_for(0, 0))
    d, vc = copy["proc"] / "s0", copy["vc"] / "s0"
    assert list(d.glob("state_inferred_*.json")) and list(vc.glob("possession_inputs_*"))
    process_game("s0", copy["gs"], copy["proc"], copy["vc"])
    assert not list(d.glob("state_inferred_*")) and not list(d.glob("features_v*"))
    assert not list(vc.glob("possession_inputs_*"))
    assert (copy["models"] / MODEL / "manifest.json").exists()  # models are kept


def run_args(w, tmp_path, **kw):
    return argparse.Namespace(
        **{
            "model": "lgbm",
            "features_version": "1",
            "processed": w["proc"],
            "gamestate": w["gs"],
            "horizons": H,
            "ball_source": "held",
            "degrade_seed": 0,
            "degrade_arm": "both",
            "possession": "inferred",
            "stale_possession": "none",
            "stale_after": None,
            "one_fold": None,
            "run_id": "learned",
            "runs": tmp_path,
            "no_report": True,
        }
        | kw
    )


def test_cv_runs_each_context_once_for_both_horizons(copy, tmp_path, monkeypatch):
    calls = []
    real = cv.load_data

    def spy(*a, **kw):
        calls.append(kw.get("contexts"))
        return real(*a, **kw)

    monkeypatch.setattr(cv, "load_data", spy)
    cv.run(run_args(copy, tmp_path), [], copy["folds"], CONFIG)
    contexts = [c for c in calls if c]
    assert len(contexts) == 6  # outer 0-4 and final, not 12
    assert {m: c.suffix for m, c in contexts[0].items()}["s5"] == "outer0_test"  # fold 0
    assert {m: c.suffix for m, c in contexts[0].items()}["s1"] == "outer0_inner-oof1"
    assert all(c.role == "final" for c in contexts[-1].values())
    meta = json.loads((tmp_path / "learned" / "run.json").read_text())
    cfg = meta["config"]
    assert cfg["state_config"] == CONFIG.to_dict() and cfg["possession"] == "inferred"
    block = cfg["possession_model"]
    assert block["model_id"] == MODEL and len(block["trainings"]) == 15
    assert sorted(block["state_sha256"]) == IDS
    # every match counted once, from its outer-test role
    prov = cv.load_data(IDS, copy["proc"], copy["gs"], H)[0]
    scored = prov.filter(pl.col("label_mask_h5"), ~pl.col("all_estimated")).height
    assert cfg["possession_scored"]["scored_rows"] == scored
    assert block["grid_rows_test"] == prov.height
    for h in H:
        for k in range(5):
            assert meta["folds"][h][str(k)]["context"]["test"] == f"outer{k}_test"
        assert "final" in meta["tau"][h]
    preds = pl.read_parquet(tmp_path / "learned" / "predictions.parquet")
    assert preds.height == prov.height


def test_a_context_with_the_wrong_possession_is_refused():
    from test_cv import world

    folds, data, shots = world(10)
    fold_of_ = {m["match_id"]: m["fold"] for m in folds["matches"]}

    def provider(outer):
        def tag(m):
            j = fold_of_[m]
            return f"outer{j}_test"  # wrong: training matches must be inner-oof

        return data.with_columns(
            possession_context=pl.col("match_id").map_elements(tag, return_dtype=pl.String)
        )

    with pytest.raises(ValueError, match="want outer0_inner-oof"):
        cv.run_cv("floor", ["h5"], folds, data, shots, log=lambda m: None, provider=provider)


@pytest.mark.parametrize(
    "extra,why",
    [
        (["--possession", "provider"], "needs --possession inferred"),
        (["--model", "floor"], "lgbm only"),
        (["--features-version", "2"], "features v1 only"),
        (["--ball-source", "raw"], "held ball only"),
        (["--stale-possession", "unknown", "--stale-after", "10"], "no stale-possession arm"),
    ],
)
def test_cli_refuses_learned_combinations(extra, why, capsys):
    argv = ["--model", "lgbm", "--possession", "inferred", "--possession-model", MODEL]
    with pytest.raises(SystemExit):
        cv.main(argv + extra)
    assert why in capsys.readouterr().err


def test_fill_builds_every_outer_context(copy):
    for f in copy["proc"].glob("*/state_inferred_age_*"):
        f.unlink()
    n = possession.fill(CONFIG, copy["gs"], copy["proc"], log=lambda _: 0)
    assert n == 5 * len(IDS)
    for m in IDS:
        names = {f.name for f in (copy["proc"] / m).glob(f"state_inferred_age_{KEY}_*.parquet")}
        assert (
            len(names) == 5 and f"state_inferred_age_{KEY}_outer{fold_of(m)}_test.parquet" in names
        )


def test_cv_refuses_a_model_trained_on_other_folds(w, tmp_path):
    folds = json.loads(json.dumps(w["folds"]))
    folds["matches"][0]["fold"], folds["matches"][1]["fold"] = 1, 0
    with pytest.raises(ValueError, match="folds"):
        cv.run(run_args(w, tmp_path), [], folds, CONFIG)
