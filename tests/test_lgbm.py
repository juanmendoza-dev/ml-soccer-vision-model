import numpy as np
import polars as pl
import pytest

pytest.importorskip("lightgbm")

from evaluation.metrics import pr_auc
from prediction.features import FEATURES
from prediction.lgbm import LGBMModel

FAST = {"max_rounds": 200, "early_stopping": 20}


def data(n_matches=10, n=2000, seed=0):
    """Shots likelier close to goal with few defenders in the lane; everything else noise,
    and a third of the ball features missing."""
    rng = np.random.default_rng(seed)
    N = n_matches * n
    cols = {f: rng.normal(size=N) for f in FEATURES}
    dist = rng.uniform(5, 60, N)
    lane = rng.integers(0, 4, N).astype(float)
    true_p = 1 / (1 + np.exp(-(-0.5 - 0.1 * dist - 0.8 * lane)))
    cols["ball_dist"] = np.where(rng.random(N) < 0.33, np.nan, dist)
    cols["lane_defenders"] = lane
    return pl.DataFrame(
        {
            **cols,
            "match_id": np.repeat([f"m{i}" for i in range(n_matches)], n),
            "label_mask_h5": np.ones(N, dtype=bool),
            "label_shot_h5": rng.random(N) < true_p,
            "eligible": np.ones(N, dtype=bool),
            "all_estimated": np.zeros(N, dtype=bool),
        }
    )


def test_learns_the_signal_and_uses_the_right_features():
    train, test = data(seed=0), data(seed=1)
    m = LGBMModel(**FAST).fit(train, "h5")
    p = m.predict(test).to_numpy()
    y = test["label_shot_h5"].to_numpy()
    assert pr_auc(y, p) > 2 * y.mean()
    top2 = sorted(m.coef, key=m.coef.get)[-2:]
    assert set(top2) == {"ball_dist", "lane_defenders"}
    assert sum(m.coef.values()) == pytest.approx(1.0)
    # calibrated-ish without class weights: mean p near the base rate
    assert p.mean() == pytest.approx(y.mean(), abs=0.02)


def test_early_stopping_holds_out_whole_training_matches():
    m = LGBMModel(**FAST).fit(data(), "h5")
    assert len(m.es_matches) == 2  # 15% of 10
    assert set(m.es_matches) <= {f"m{i}" for i in range(10)}
    assert 1 <= m.best_iter <= FAST["max_rounds"]
    assert m.fit_info["best_iter"] == m.best_iter


def test_deterministic():
    df = data()
    a = LGBMModel(**FAST).fit(df, "h5").predict(df).to_numpy()
    b = LGBMModel(**FAST).fit(df, "h5").predict(df).to_numpy()
    assert np.array_equal(a, b)


def test_ignores_masked_and_all_estimated_rows():
    df = data(n_matches=6)
    poison = data(n_matches=6, seed=3).with_columns(
        label_shot_h5=pl.col("ball_dist").fill_nan(0) > 30, label_mask_h5=pl.lit(False)
    )
    poison2 = poison.with_columns(label_mask_h5=pl.lit(True), all_estimated=pl.lit(True))
    a = LGBMModel(**FAST).fit(df, "h5")
    b = LGBMModel(**FAST).fit(pl.concat([df, poison, poison2]), "h5")
    assert a.n_train == b.n_train
    assert np.allclose(a.predict(df).to_numpy(), b.predict(df).to_numpy())


def test_predict_nulls_where_05_says_so():
    m = LGBMModel(**FAST).fit(data(n_matches=4), "h5")
    rows = data(n_matches=1, n=3).with_columns(
        eligible=pl.Series([True, False, True]), all_estimated=pl.Series([False, False, True])
    )
    p = m.predict(rows).to_list()
    assert 0 < p[0] < 1 and p[1] is None and p[2] is None


def test_reads_only_its_own_feature_list():
    df = data().with_columns(extra=pl.lit(1.0))
    m = LGBMModel(**FAST, features=["ball_dist", "extra"]).fit(df, "h5")
    assert list(m.coef) == ["ball_dist", "extra"]
    # a feature it wasn't given can't reach it: dropping one changes nothing
    a = m.predict(df).to_numpy()
    assert np.array_equal(a, m.predict(df.drop("lane_defenders")).to_numpy(), equal_nan=True)
