import numpy as np
import polars as pl
import pytest

from prediction import cv
from prediction.features import FEATURES
from prediction.floor import LogisticFloor

N = 300  # 30 s per match


def match(mid, seed):
    """Ball walks toward goal; a home shot at 20 s, positives in the 5 s before it."""
    rng = np.random.default_rng(seed)
    t = np.round(np.arange(N) * 0.1, 1)
    dist = np.clip(60 - 2.5 * t + rng.normal(0, 3, N), 1, None)
    dist[t > 20] = 60
    return pl.DataFrame(
        {
            "match_id": [mid] * N,
            "period": [1] * N,
            "t_s": t,
            "possession_team": ["home"] * N,
            "ball_state": ["alive"] * N,
            "eligible": [True] * N,
            "all_estimated": [False] * N,
            "label_mask_h5": [True] * N,
            "label_shot_h5": (t >= 15) & (t < 20),
            "ball_visible": [True] * N,
            "ball_dist": dist,
            "ball_angle": 1 / dist,
        }
    )


def world(n=15):
    ids = [f"m{i:02d}" for i in range(n)]
    folds = {
        "n_folds": 5,
        "seed": 7,
        "frozen": {},
        "matches": [
            {"match_id": i, "source": "pff", "fold": k % 5, "open_play_shots": 1}
            for k, i in enumerate(ids)
        ],
    }
    data = pl.concat([match(i, k) for k, i in enumerate(ids)]).with_row_index("row")
    shots = pl.DataFrame(
        {
            "match_id": ids,
            "period": [1] * n,
            "t_s": [20.0] * n,
            "team": ["home"] * n,
            "open_play": [True] * n,
        }
    )
    return folds, data, shots


def test_every_prediction_is_out_of_fold(monkeypatch):
    folds, data, shots = world()
    fold_of = {m["match_id"]: m["fold"] for m in folds["matches"]}
    fits = []

    class Spy(LogisticFloor):
        def fit(self, df, h):
            self.trained_on = set(df["match_id"].unique())
            fits.append(self.trained_on)
            return super().fit(df, h)

        def predict(self, df):
            assert not self.trained_on & set(df["match_id"].unique())
            return super().predict(df)

    monkeypatch.setitem(cv.MODELS, "spy", (Spy, {"features": [], "l2": 1e-6}))
    preds, meta = cv.run_cv("spy", ["h5"], folds, data, shots, log=lambda *_: None)
    # 5 folds x (inner fit + outer fit) + final inner fit
    assert len(fits) == 11
    for fold in range(5):
        outer = [f for f in fits if all(fold_of[i] != fold for i in f)]
        assert outer, f"no fit left fold {fold} out"
    assert preds["p_h5"].null_count() == 0
    assert set(meta["tau"]["h5"]) == {"0", "1", "2", "3", "4", "final"}
    assert set(meta["tau_met"]["h5"]) == set(meta["tau"]["h5"])


def test_floor_learns_the_toy_and_meets_the_budget():
    folds, data, shots = world()
    preds, meta = cv.run_cv("floor", ["h5"], folds, data, shots, log=lambda *_: None)
    assert all(meta["tau_met"]["h5"].values())
    y = data["label_shot_h5"].to_numpy()
    p = preds["p_h5"].to_numpy()
    assert p[y].mean() > p[~y].mean()
    assert meta["folds"]["h5"]["0"]["coef"]["ball_dist"] < 0


def test_lgbm_runs_through_cv_and_records_its_fits():
    pytest.importorskip("lightgbm")
    folds, data, shots = world()
    data = data.with_columns(
        pl.lit(float("nan")).alias(f) for f in FEATURES if f not in data.columns
    )
    preds, meta = cv.run_cv("lgbm", ["h5"], folds, data, shots, log=lambda *_: None)
    y = data["label_shot_h5"].to_numpy()
    p = preds["p_h5"].to_numpy()
    assert p[y].mean() > p[~y].mean()
    f0 = meta["folds"]["h5"]["0"]
    assert f0["best_iter"] >= 1 and f0["es_matches"]
    assert max(f0["coef"], key=f0["coef"].get) in {"ball_dist", "ball_angle"}  # angle = 1/dist
    assert meta["config"]["ball_source"] == "held"
    # early stopping only ever holds out training matches of that fold
    fold_of = {m["match_id"]: m["fold"] for m in folds["matches"]}
    for k, info in meta["folds"]["h5"].items():
        assert all(fold_of[i] != int(k) for i in info["es_matches"])


def test_test_arm_fits_on_data_and_predicts_test_data(monkeypatch):
    """--degrade-arm test: every fit and tau sees the clean data, only the held-out
    predictions read the degraded copy."""
    folds, data, shots = world()
    degraded = data.with_columns(ball_dist=pl.lit(1.0), ball_angle=pl.lit(1.0))
    seen = []

    class Spy(LogisticFloor):
        def fit(self, df, h):
            seen.append(("fit", df["ball_dist"].max()))
            return super().fit(df, h)

    monkeypatch.setitem(cv.MODELS, "spy", (Spy, {"features": [], "l2": 1e-6}))
    preds, meta = cv.run_cv(
        "spy",
        ["h5"],
        folds,
        data,
        shots,
        log=lambda *_: None,
        test_data=degraded,
        data_config={"degrade": ["ball_noise:1"], "degrade_arm": "test"},
    )
    assert all(v > 1 for _, v in seen)
    # every held-out row saw the same constant features, so one p per fold
    per_fold = (
        preds.join(
            pl.DataFrame(
                [{"match_id": m["match_id"], "fold": m["fold"]} for m in folds["matches"]]
            ),
            on="match_id",
        )
        .group_by("fold")
        .agg(pl.col("p_h5").n_unique())
    )
    assert per_fold["p_h5"].max() == 1
    assert meta["config"]["degrade"] == ["ball_noise:1"]
    with pytest.raises(ValueError, match="same rows"):
        cv.run_cv(
            "spy", ["h5"], folds, data, shots, log=lambda *_: None, test_data=degraded.head(10)
        )


def test_lgbm_takes_its_features_from_the_versioned_config():
    pytest.importorskip("lightgbm")
    folds, data, shots = world()
    config = {"features": ["ball_dist"], "features_version": "test"}
    preds, meta = cv.run_cv(
        "lgbm", ["h5"], folds, data, shots, log=lambda *_: None, data_config=config
    )
    assert meta["config"]["features"] == ["ball_dist"]
    assert meta["config"]["features_version"] == "test"
    assert list(meta["folds"]["h5"]["0"]["coef"]) == ["ball_dist"]
