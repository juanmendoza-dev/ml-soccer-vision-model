import numpy as np
import polars as pl
import pytest

from prediction.features import goal_angle, goal_distance
from prediction.floor import LogisticFloor


def data(n=20_000, seed=0):
    """Shots more likely close to goal and at a wide angle, like the real thing."""
    rng = np.random.default_rng(seed)
    dist = rng.uniform(5, 60, n)
    ang = rng.uniform(0.05, 1.5, n)
    true_p = 1 / (1 + np.exp(-(-1.0 - 0.08 * dist + 1.5 * ang)))
    return pl.DataFrame(
        {
            "ball_dist": dist,
            "ball_angle": ang,
            "label_mask_h5": np.ones(n, dtype=bool),
            "label_shot_h5": rng.random(n) < true_p,
            "eligible": np.ones(n, dtype=bool),
            "all_estimated": np.zeros(n, dtype=bool),
        }
    )


def test_fit_recovers_the_signs_and_beats_the_base_rate():
    df = data()
    m = LogisticFloor().fit(df, "h5")
    assert m.coef["ball_dist"] < 0 < m.coef["ball_angle"]
    p = m.predict(df).to_numpy()
    y = df["label_shot_h5"].to_numpy()
    assert p[y].mean() > p[~y].mean()
    # unstandardized slope close to the true -0.08 per metre
    assert m.coef["ball_dist"] / m.std[0] == pytest.approx(-0.08, abs=0.02)


def test_fit_ignores_masked_and_all_estimated_rows():
    df = data()
    # poison: masked / all-estimated rows where every far ball is a shot
    poison = data(seed=1).with_columns(
        label_shot_h5=pl.col("ball_dist") > 30,
        label_mask_h5=pl.lit(False),
    )
    poison2 = poison.with_columns(label_mask_h5=pl.lit(True), all_estimated=pl.lit(True))
    a = LogisticFloor().fit(df, "h5").coef
    b = LogisticFloor().fit(pl.concat([df, poison, poison2]), "h5").coef
    assert a == pytest.approx(b)


def test_predict_nulls_and_missing_ball_fallback():
    m = LogisticFloor().fit(data(), "h5")
    rows = pl.DataFrame(
        {
            "ball_dist": [10.0, None, 10.0, 10.0],
            "ball_angle": [0.8, None, 0.8, 0.8],
            "eligible": [True, True, False, True],
            "all_estimated": [False, False, False, True],
        }
    )
    p = m.predict(rows).to_list()
    assert 0 < p[0] < 1
    assert p[1] == pytest.approx(m.base_rate)  # eligible, no ball: training base rate
    assert p[2] is None and p[3] is None  # ineligible, all-ESTIMATED


def test_base_rate_counts_rows_without_a_ball():
    df = data(n=1000).with_columns(
        ball_dist=pl.when(pl.int_range(pl.len()) < 500).then(None).otherwise("ball_dist")
    )
    m = LogisticFloor().fit(df, "h5")
    assert m.base_rate == pytest.approx(df["label_shot_h5"].mean())


def test_heavy_angle_tail_doesnt_diverge():
    # the real fold 0 fit: shot rate peaks at mid angles and drops again on the goal line,
    # with a thin tail up to pi. Plain Newton steps blew up by the 4th iteration there,
    # and do here too (checked by removing the step halving)
    rng = np.random.default_rng(1)
    n = 50_000
    x, y = rng.uniform(-50, 52.5, n), rng.uniform(-34, 34, n)
    dist, ang = goal_distance(x, y), goal_angle(x, y)
    rate = np.select(
        [ang <= 0.2, ang <= 0.5, ang <= 1, ang <= 2, ang <= 3],
        [0.012, 0.17, 0.22, 0.17, 0.12],
        0.09,
    )
    df = pl.DataFrame(
        {
            "ball_dist": dist,
            "ball_angle": ang,
            "label_mask_h5": np.ones(n, dtype=bool),
            "label_shot_h5": rng.random(n) < rate,
            "eligible": np.ones(n, dtype=bool),
            "all_estimated": np.zeros(n, dtype=bool),
        }
    )
    m = LogisticFloor().fit(df, "h5")
    assert np.abs(m.w).max() < 20
    assert m.coef["ball_dist"] < 0
