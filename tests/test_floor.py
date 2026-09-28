import numpy as np
import polars as pl
import pytest

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
    # real data: angle is ~0.13 +- 0.09 with a few rows at pi on the goal line, which
    # sent plain Newton steps off to overflow
    df = data(n=50_000).with_columns(ball_angle=pl.col("ball_angle") * 0.1)
    tail = data(n=40, seed=3).with_columns(
        ball_angle=pl.lit(np.pi), ball_dist=pl.lit(0.5), label_shot_h5=pl.lit(False)
    )
    m = LogisticFloor().fit(pl.concat([df, tail]), "h5")
    assert np.all(np.isfinite(m.w))
    assert m.coef["ball_dist"] < 0
