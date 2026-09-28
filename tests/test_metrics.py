import math

import numpy as np
import polars as pl
import pytest

from evaluation.metrics import (
    alarm_summary,
    alarms,
    brier,
    calibration,
    pr_auc,
    roc_auc,
    score_alarms,
    scored_rows,
    shots_table,
    tau_sweep,
)

TAU = 0.5


def test_pr_auc_by_hand():
    # ranked 1 0 1 0: precision 1 at recall 1/2, 2/3 at recall 1
    assert pr_auc([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]) == pytest.approx(1 / 2 + 2 / 3 / 2)


def test_pr_auc_ties_are_one_threshold():
    # all tied: a single step at precision = base rate, no credit for lucky order
    assert pr_auc([1, 0, 0, 0], [0.5] * 4) == pytest.approx(0.25)
    assert pr_auc([0, 0, 0, 1], [0.5] * 4) == pytest.approx(0.25)


def test_roc_auc_by_hand_and_ties():
    assert roc_auc([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]) == pytest.approx(0.75)
    assert roc_auc([1, 0], [0.5, 0.5]) == pytest.approx(0.5)


def test_one_class_is_nan():
    assert math.isnan(pr_auc([0, 0], [0.1, 0.2]))
    assert math.isnan(roc_auc([1, 1], [0.1, 0.2]))


def test_nan_p_is_an_error():
    with pytest.raises(ValueError):
        pr_auc([1, 0], [0.5, float("nan")])


def test_brier():
    assert brier([1, 0], [0.75, 0.25]) == pytest.approx(0.0625)


def test_matches_sklearn_on_random_data():
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y = rng.random(2000) < 0.03
    p = np.round(rng.random(2000) * 0.6 + y * 0.3, 2)  # rounded so there are ties
    assert pr_auc(y, p) == pytest.approx(metrics.average_precision_score(y, p))
    assert roc_auc(y, p) == pytest.approx(metrics.roc_auc_score(y, p))
    assert brier(y, p) == pytest.approx(metrics.brier_score_loss(y, p))


def test_calibration_quantile_bins_are_even():
    rng = np.random.default_rng(1)
    p = rng.random(1000)
    y = rng.random(1000) < p
    cal = calibration(y, p, n_bins=4)
    assert cal["n"].to_list() == [250] * 4
    assert cal["mean_p"].is_sorted()
    assert (cal["pos_rate"] - cal["mean_p"]).abs().max() < 0.1


def test_calibration_uniform():
    cal = calibration([0, 1, 1], [0.05, 0.95, 0.99], n_bins=10, strategy="uniform")
    assert cal["bin"].to_list() == [0, 9]
    assert cal["n"].to_list() == [1, 2]


def frames(p, team="home", state="alive", match="m", period=1, label=None, estimated=False):
    """A 10 Hz stretch starting at t = 0. Scalars are broadcast."""
    n = len(p)

    def col(v):
        return v if isinstance(v, list) else [v] * n

    return pl.DataFrame(
        {
            "match_id": col(match),
            "period": col(period),
            "t_s": [round(0.1 * i, 1) for i in range(n)],
            "p": p,
            "possession_team": col(team),
            "ball_state": col(state),
            "label_mask_h5": [True] * n,
            "label_shot_h5": col(False if label is None else label),
            "all_estimated": col(estimated),
        },
        schema_overrides={"p": pl.Float64, "possession_team": pl.String, "ball_state": pl.String},
    )


def spans(pred):
    return [
        (r["start_s"], r["end_s"], r["ended_by"]) for r in alarms(pred, TAU).iter_rows(named=True)
    ]


def test_dip_above_off_holds_drop_below_ends():
    # 0.45 is between 0.8 * tau and tau: holds. 0.3 is below 0.8 * tau: ends
    assert spans(frames([0.1, 0.6, 0.45, 0.7, 0.3, 0.1])) == [(0.1, 0.4, "p")]


def test_possession_change_and_dead_ball_end_it():
    got = spans(frames([0.6, 0.6, 0.6], team=["home", "home", "away"]))
    # the away row ends home's alarm and starts away's on the same row
    assert got == [(0.0, 0.2, "possession"), (0.2, 0.2, "period_end")]
    assert spans(frames([0.6, 0.6, 0.1], state=["alive", "dead", "dead"])) == [(0.0, 0.1, "dead")]


def test_nulls_hold():
    # null p (cutaway), null possession (loose ball) and null ball_state (vision gap)
    pred = frames(
        [0.6, None, 0.6, 0.6, 0.1],
        team=["home", "home", None, "home", "home"],
        state=["alive", "alive", "alive", None, "alive"],
    )
    assert spans(pred) == [(0.0, 0.4, "p")]


def test_no_start_without_possession_or_on_a_dead_ball():
    assert spans(frames([0.9, 0.9], team=None)) == []
    assert spans(frames([0.9, 0.9], state="dead")) == []


def test_alarm_never_crosses_a_period():
    pred = pl.concat([frames([0.6, 0.6], period=1), frames([0.6, 0.1], period=2)])
    assert alarms(pred, TAU).select("period", "start_s", "end_s", "ended_by").rows() == [
        (1, 0.0, 0.1, "period_end"),
        (2, 0.0, 0.1, "p"),
    ]


def shots(*rows):
    """(t_s, team, open_play) in match m, period 1."""
    return pl.DataFrame(
        {
            "match_id": ["m"] * len(rows),
            "period": [1] * len(rows),
            "t_s": [r[0] for r in rows],
            "team": [r[1] for r in rows],
            "open_play": [r[2] for r in rows],
        },
        schema={
            "match_id": pl.String,
            "period": pl.Int64,
            "t_s": pl.Float64,
            "team": pl.String,
            "open_play": pl.Boolean,
        },
    )


# alarm from 0.2 to 1.0 (ends on the 0.1 at t = 1.0), then nothing to 4.9 s
QUIET = [0.1, 0.1] + [0.9] * 8 + [0.1] * 40


def test_rebound_shares_the_alarm_start():
    scored, al = score_alarms(frames(QUIET), shots((0.6, "home", True), (0.9, "home", True)), TAU)
    assert scored["lead_s"].to_list() == pytest.approx([0.4, 0.7])
    assert al["true"].to_list() == [True]


def test_grace_after_the_end():
    # ended at 1.0: 1.5 is inside the 1 s grace, 2.5 is a miss and the alarm is false
    s = alarm_summary(frames(QUIET), shots((1.5, "home", True)), TAU)
    assert (s["missed"], s["false_alarms"], s["lead_s_median"]) == (0, 0, pytest.approx(1.3))
    s = alarm_summary(frames(QUIET), shots((2.5, "home", True)), TAU)
    assert (s["missed"], s["false_alarms"]) == (1, 1)


def test_active_alarm_beats_one_in_grace():
    # alarm A 0.0-0.3, alarm B from 0.5; a shot at 0.8 is in A's grace but B is active
    pred = frames([0.9, 0.9, 0.9, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9])
    scored, al = score_alarms(pred, shots((0.8, "home", True)), TAU)
    assert scored["lead_s"].to_list() == pytest.approx([0.3])
    assert al["true"].to_list() == [True, True]  # A still covered it within grace


def test_alarm_must_start_before_the_shot():
    pred = frames([0.1, 0.1, 0.9, 0.9, 0.1])
    s = alarm_summary(pred, shots((0.2, "home", True)), TAU)
    assert (s["missed"], s["false_alarms"]) == (1, 1)


def test_other_teams_shot_doesnt_count():
    s = alarm_summary(frames(QUIET), shots((0.6, "away", True)), TAU)
    assert (s["missed"], s["false_alarms"]) == (1, 1)


def test_set_play_shot_isnt_scored_but_makes_the_alarm_true():
    s = alarm_summary(frames(QUIET), shots((0.6, "home", False)), TAU)
    assert (s["shots"], s["missed"], s["false_alarms"]) == (0, 0, 0)


def test_false_alarms_are_per_match_including_quiet_ones():
    pred = pl.concat([frames(QUIET, match="m"), frames([0.1] * 10, match="quiet")])
    s = alarm_summary(pred, shots(), TAU)
    assert (s["false_alarms"], s["false_alarms_per_match"]) == (1, 0.5)


def test_tau_sweep_trades_misses_for_false_alarms():
    pred = frames([0.1, 0.6, 0.6, 0.1, 0.1, 0.9, 0.9, 0.1, 0.1, 0.1])
    sweep = tau_sweep(pred, shots((0.8, "home", True)), [0.5, 0.8])
    assert sweep["alarms"].to_list() == [2, 1]
    assert sweep["missed"].to_list() == [0, 0]


def test_scored_rows_are_picked_by_labels():
    pred = frames([0.2, 0.3, None], label=[False, True, False], estimated=[False, False, True])
    y, p = scored_rows(pred, "h5")
    assert y.tolist() == [False, True] and p.tolist() == [0.2, 0.3]
    with pytest.raises(ValueError, match="null p"):
        scored_rows(frames([0.2, None]), "h5")


def test_shots_table_keeps_set_plays_and_drops_goal_rows():
    frames_ = pl.DataFrame(
        {"frame_id": [1, 2, 3], "period": [1, 1, 1], "timestamp_s": [1.0, 2.0, 3.0]}
    )
    events = pl.DataFrame(
        {
            "frame_id": [1, 2, 2, 3],
            "event_type": ["shot", "shot", "goal", "shot"],
            "team": ["home", "away", "away", "home"],
            "outcome": ["saved", "goal", "goal", "off_target"],
            "set_piece": ["open_play", "open_play", "open_play", "corner"],
            "set_play_phase": [False, False, False, True],
        }
    )
    got = shots_table(events, frames_, "m")
    assert got.select("t_s", "team", "open_play").rows() == [
        (1.0, "home", True),
        (2.0, "away", True),
        (3.0, "home", False),
    ]


def test_shots_from_other_matches_are_ignored():
    both = pl.concat(
        [shots((0.6, "home", True)), shots((0.6, "home", True)).with_columns(match_id=pl.lit("x"))]
    )
    s = alarm_summary(frames(QUIET), both, TAU)
    assert (s["shots"], s["missed"]) == (1, 0)
