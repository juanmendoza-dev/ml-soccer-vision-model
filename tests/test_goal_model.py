"""05 "Offline demo model": the fold map, and GoalModel's P(shot) x xG -> map."""

import json

import numpy as np
import polars as pl
import pytest

lgb = pytest.importorskip("lightgbm")

from converters.common import sha256
from prediction import goal_model as gm
from prediction.features import FEATURES
from prediction.pgoal import apply_map, fit_map
from prediction.xg import XG_FEATURES


def pgoal_rows(seed=0, n=4000):
    rng = np.random.default_rng(seed)
    p = rng.uniform(1e-4, 0.2, n)
    return pl.DataFrame(
        {
            "fold": rng.integers(0, 5, n),
            "label_mask_h5": np.ones(n, bool),
            "all_estimated": np.zeros(n, bool),
            "p_goal_h5": p,
            "label_goal_h5": rng.random(n) < p * 1.8,
        }
    )


def test_fold_map_is_the_other_folds_fit():
    rows = pgoal_rows()
    other = rows.filter(pl.col("fold") != 0)
    want = fit_map(other["p_goal_h5"].to_numpy(), other["label_goal_h5"].to_numpy().astype(float))
    assert gm.fold_map(rows, 0) == pytest.approx(want)


def test_fold_map_ignores_the_folds_own_labels():
    rows = pgoal_rows()
    flipped = rows.with_columns(
        label_goal_h5=pl.when(pl.col("fold") == 0).then(~pl.col("label_goal_h5")).otherwise("label_goal_h5")
    )
    assert gm.fold_map(rows, 0) == gm.fold_map(flipped, 0)


def test_train_ids_leave_the_fold_out_sorted():
    folds = {"matches": [{"match_id": i, "fold": f} for i, f in (("3812", 1), ("10517", 0), ("10502", 2))]}
    assert gm.train_ids(folds, 0) == ["10502", "3812"]


def write_model(tmp_path, ab=(0.1, 0.8)):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(400, len(FEATURES)))
    y = (x[:, 3] + rng.normal(size=400) > 1).astype(float)
    booster = lgb.train(
        {"objective": "binary", "verbosity": -1, "min_data_in_leaf": 20},
        lgb.Dataset(x, y, feature_name=list(FEATURES)),
        num_boost_round=5,
    )
    xg_dir = tmp_path / "xg"
    xg_dir.mkdir()
    (xg_dir / "manifest.json").write_text(json.dumps({"model": "logistic", "features": list(XG_FEATURES)}))
    k = len(XG_FEATURES)
    (xg_dir / "model.json").write_text(json.dumps({"w": [-1.0] + [0.0] * k, "mean": [0.0] * k, "std": [1.0] * k}))
    d = tmp_path / "goal"
    d.mkdir()
    booster.save_model(str(d / "model.txt"))
    man = {
        "model_id": "goal-test",
        "features": list(FEATURES),
        "map": {"a": ab[0], "b": ab[1]},
        "xg": {"dir": str(xg_dir), "file": "model.json", "sha256": sha256(xg_dir / "model.json")},
    }
    (d / "manifest.json").write_text(json.dumps(man))
    return d, booster


def feats(n=50, seed=2):
    rng = np.random.default_rng(seed)
    df = pl.DataFrame({f: rng.normal(size=n).astype(np.float32) for f in FEATURES})
    no_ball = np.arange(n) % 7 == 0
    return df.with_columns(ball_dist=pl.Series(np.where(no_ball, np.nan, np.abs(df["ball_dist"].to_numpy())), dtype=pl.Float32)), no_ball


def test_predict_is_pshot_times_xg_through_the_map(tmp_path):
    d, booster = write_model(tmp_path)
    df, no_ball = feats()
    out = gm.GoalModel(d).predict(df)
    p_shot = booster.predict(df.select(pl.col(f).cast(pl.Float32) for f in FEATURES).to_numpy())
    xg = 1 / (1 + np.exp(1.0))  # the logistic xG above: intercept -1, zero weights
    want = apply_map(p_shot * xg, (0.1, 0.8))
    got = out["p_goal_h5"].to_numpy()
    assert np.allclose(got[~no_ball], want[~no_ball])
    assert out["p_goal_h5"].is_null().to_numpy()[no_ball].all()
    assert out["xg"].is_null().to_numpy()[no_ball].all()
    assert np.allclose(out["p_shot_h5"].to_numpy(), p_shot)  # p_shot needs no ball


def test_a_changed_xg_file_is_refused(tmp_path):
    d, _ = write_model(tmp_path)
    (tmp_path / "xg" / "model.json").write_text(json.dumps({"w": [0.0] * 9, "mean": [0.0] * 8, "std": [1.0] * 8}))
    with pytest.raises(ValueError, match="xG"):
        gm.GoalModel(d)
