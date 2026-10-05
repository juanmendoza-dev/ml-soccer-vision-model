"""demo.check: the demo regression table (ball fix plan, Phase 0)."""

import polars as pl
import pytest

from demo import check


def rows(ps, start_k=100):
    return pl.DataFrame(
        {"k": list(range(start_k, start_k + len(ps))), "vis_p": ps, "pff_p": [0.01] * len(ps)}
    ).with_columns(t=pl.col("k") / 10)


def test_meter_reads_the_row_at_each_time_before_the_goal():
    j = rows([0.001 * i for i in range(60)])  # t 10.0 .. 15.9
    out = check.meter_at(j, goal_t=15.0, befores=(5, 1, 0.5))
    assert out[0] == (5, pytest.approx(0.0), 0.01)  # k 100
    assert out[1][1] == pytest.approx(0.040)  # k 140
    assert out[2][1] == pytest.approx(0.045)  # k 145
    assert check.meter_at(j, goal_t=30.0, befores=(1,))[0][1:] == (None, None)


def test_false_peaks_skip_rows_near_a_shot():
    ps = [0.0] * 60
    ps[5] = 0.05  # t 10.5, 4.5 s before the shot at 15: a false peak
    ps[30] = 0.08  # t 13.0, 2 s before: near the shot, not false
    ps[59] = 0.03  # t 15.9, 0.9 s after: near too
    peaks = check.false_peaks(rows(ps), shot_ts=[15.0])
    assert peaks["t"].to_list() == [pytest.approx(10.5)]


def test_ball_distance_splits_by_pff_visibility():
    vis = pl.DataFrame({"k": [1, 2, 3], "x": [0.0, 10.0, 0.0], "y": [0.0, 0.0, 0.0]})
    pff = pl.DataFrame(
        {"k": [1, 2, 3], "x": [1.0, 0.0, 0.0], "y": [0.0, 0.0, 3.0], "visible": [True, True, False]}
    )
    d = check.ball_distance(vis, pff)
    assert d["visible"]["n"] == 2 and d["visible"]["within_2m"] == pytest.approx(0.5)
    assert d["estimated"]["median_m"] == pytest.approx(3.0)


def test_join_rows_keys_vision_time_onto_the_pff_grid():
    vis = pl.DataFrame(
        {
            "t_s": [0.0, 0.1],
            "frame_id": [0, 3],
            "predicted": [True, True],
            "possession_team": ["home", None],
            "p_goal_h5": [0.01, 0.02],
        }
    )
    pff = pl.DataFrame(
        {"k": [1000, 1001], "possession_team": ["home", "away"], "p_goal_cal_h5": [0.1, 0.2]}
    )
    j = check.join_rows(vis, pff, shift=100.0)
    assert j["k"].to_list() == [1000, 1001] and j["pff_p"].to_list() == [0.1, 0.2]
    assert j["t"].to_list() == [pytest.approx(100.0), pytest.approx(100.1)]
