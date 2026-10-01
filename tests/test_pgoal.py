"""P(goal) recalibration (05 "Recalibration"): the two-number map and its cross-fitting."""

import numpy as np
import polars as pl
import pytest

from prediction import pgoal


def sample(n, a, b, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.0005, 0.05, n)
    y = (rng.uniform(size=n) < pgoal.apply_map(p, (a, b))).astype(float)
    return p, y


def test_fit_map_recovers_a_known_map():
    p, y = sample(200_000, 0.8, 1.1)
    a, b = pgoal.fit_map(p, y)
    assert a == pytest.approx(0.8, abs=0.1) and b == pytest.approx(1.1, abs=0.05)


def test_the_map_keeps_the_order_and_refuses_a_reversal():
    p = np.array([1e-4, 1e-3, 1e-2, 0.2])
    assert np.all(np.diff(pgoal.apply_map(p, (0.5, 1.2))) > 0)
    q, y = sample(50_000, -6.0, -1.0)  # goals get rarer as p rises
    with pytest.raises(ValueError, match="reverse"):
        pgoal.fit_map(q, y)


def rows_of(seed=0, matches=10):
    rng = np.random.default_rng(seed)
    n = 4000
    p, y = sample(n * matches, 0.9, 1.0, seed)
    return pl.DataFrame({
        "match_id": np.repeat([f"m{i}" for i in range(matches)], n),
        "label_mask_h5": True, "label_mask_h3": True, "all_estimated": False,
        "p_goal_h5": p, "p_goal_h3": p, "label_goal_h5": y.astype(bool), "label_goal_h3": y.astype(bool),
    }).with_columns(pl.when(pl.Series(rng.uniform(size=n * matches) < 0.01)).then(None)
                    .otherwise(pl.col("p_goal_h5")).alias("p_goal_h5"))


FOLDS = {f"m{i}": i % 5 for i in range(10)}


def test_a_folds_own_labels_never_reach_its_map():
    rows = rows_of()
    base, _ = pgoal.recalibrate(rows, FOLDS)
    flipped = rows.with_columns(
        pl.when(pl.col("match_id").is_in(["m0", "m5"])).then(~pl.col("label_goal_h5"))
        .otherwise(pl.col("label_goal_h5")).alias("label_goal_h5")
    )
    other, maps = pgoal.recalibrate(flipped, FOLDS)
    fold0 = pl.col("match_id").is_in(["m0", "m5"])
    assert base.filter(fold0)["p_goal_cal_h5"].equals(other.filter(fold0)["p_goal_cal_h5"])
    assert not base.filter(~fold0)["p_goal_cal_h5"].equals(other.filter(~fold0)["p_goal_cal_h5"])
    assert set(maps["h5"]["folds"]) == {0, 1, 2, 3, 4}


def test_recalibration_fixes_the_scale_and_leaves_missing_rows_missing():
    rows = rows_of()
    out, maps = pgoal.recalibrate(rows, FOLDS)
    ok = out.filter(pl.col("p_goal_h5").is_not_null())
    goals = ok["label_goal_h5"].sum()
    assert ok["p_goal_cal_h5"].sum() == pytest.approx(goals, rel=0.05)
    assert ok["p_goal_h5"].sum() < 0.5 * goals  # the raw p runs about 2.4x low here
    assert out.filter(pl.col("p_goal_h5").is_null())["p_goal_cal_h5"].null_count() == out["p_goal_h5"].null_count()
    assert maps["h5"]["rows"] == ok.height and maps["h5"]["goal_rows"] == goals
