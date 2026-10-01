"""The learned-possession gate (03 stage 8, "Checks and fixed bars"): the churn count, the
bucket edges, the bars' boundaries, the baseline stop and the grid join, then one end-to-end
run on the synthetic fold world of test_possession_learned."""

import json
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest
from test_possession_learned import CONFIG, MODEL, make_world, trained, w  # noqa: F401

from prediction import possession
from prediction import possession_gate as gate

ROOT = Path(__file__).resolve().parent.parent


def grid(period, teams):
    return pl.DataFrame({"period": period, "k": list(range(len(teams))), "team": teams})


def test_churn_carries_the_last_team_over_nulls():
    assert gate.churn(grid(1, ["home", None, "home"])) == 0
    assert gate.churn(grid(1, ["home", None, "away"])) == 1
    assert gate.churn(grid(1, [None, None, "away", "home", "home", None, "away"])) == 2
    assert gate.churn(grid(1, [None, None, None])) == 0


def test_churn_never_counts_across_a_period_boundary():
    both = pl.concat([grid(1, ["home", "home"]), grid(2, ["away", "away"])])
    assert gate.churn(both) == 0
    both = pl.concat([grid(1, ["home", "away"]), grid(2, ["away", "home"])])
    assert gate.churn(both) == 2
    # the last team of period 1 isn't carried into period 2's leading nulls
    both = pl.concat([grid(1, ["home"]), grid(2, [None, "away"])])
    assert gate.churn(both) == 0


def test_churn_reads_rows_in_grid_order_whatever_their_storage_order():
    g = grid(1, ["home", "away", "home"]).reverse()
    assert gate.churn(g) == 2


def test_bucket_edges():
    since = [None, 0.0, 0.999, 1.0, 2.999, 3.0, 9.999, 10.0, 300.0]
    a = pl.DataFrame(
        {
            "since": since,
            "seen": [True] * len(since),
            "gap": [0.0] * len(since),
            "p_state": [None] * len(since),
        }
    )
    got = gate.label_rows(a)["since_chg"].to_list()
    assert got == [
        "e none",
        "a <1s",
        "a <1s",
        "b 1-3s",
        "b 1-3s",
        "c 3-10s",
        "c 3-10s",
        "d >10s",
        "d >10s",
    ]
    # first 3 s is [0, 3): 3.0 is outside it; the late bucket includes exactly 10 s
    rows = gate.label_rows(a).with_columns(pos=pl.lit(False), d=pl.lit(True))
    t = gate.tally(rows, "d")
    assert (t["first3_rows"], t["late_rows"]) == (4, 2)


def test_gap_buckets_for_a_ball_not_seen():
    a = pl.DataFrame(
        {
            "since": [1.0] * 6,
            "seen": [True, False, False, False, False, False],
            "gap": [0.0, 0.2, 1.0, 5.0, 11.0, None],
            "p_state": [None] * 6,
        }
    )
    assert gate.label_rows(a)["ball"].to_list() == [
        "0 visible",
        "1 gap<0.5s",
        "2 gap0.5-2s",
        "3 gap2-10s",
        "4 gap>10s",
        "5 never",
    ]


def counts(**kw):
    base = {
        "rows": 1_977_379,
        "positives": 49_812,
        "first3_rows": 334_949,
        "late_rows": 1_162_041,
        "overall": 200_000,
        "first3": 100_000,
        "positive_dis": 3_000,
        "late_dis": 52_000,
    }
    return base | kw


DEFAULT = counts(overall=281_606, first3=154_806, positive_dis=3_862, late_dis=52_455)


def verdict(learned, churn=10_000):
    return {b.name: b.passed for b in gate.evaluate(learned, churn, DEFAULT)}


def test_every_bar_passes_exactly_at_its_limit_and_fails_one_over():
    names = ["overall", "first-3", "positive", "late", "churn"]
    ok = verdict(counts(overall=253_445, first3=131_585, positive_dis=3_862), 18_182)
    assert all(ok.values()), ok

    def failed(learned, churn=10_000):
        bad = [k for k, v in verdict(learned, churn).items() if not v]
        return [n for n in names if any(n in k for k in bad)]

    assert failed(counts(overall=253_446)) == ["overall"]
    assert failed(counts(first3=131_586)) == ["first-3"]
    assert failed(counts(positive_dis=3_863)) == ["positive"]
    assert failed(counts(), churn=18_183) == ["churn"]


def test_late_bar_is_the_defaults_unrounded_rate_plus_one_point():
    limit = 52_455 / 1_162_041 + 0.010
    just_in = int(limit * 1_162_041)  # floor: under the limit
    assert verdict(counts(late_dis=just_in))["late-change rate (>= 10 s)"]
    assert not verdict(counts(late_dis=just_in + 1))["late-change rate (>= 10 s)"]
    # 5.514%, not the rounded 4.5% + 1 point: the spec says the unrounded default rate
    assert limit == pytest.approx(0.055140, abs=1e-6)


def test_bars_refuse_different_denominators():
    with pytest.raises(ValueError, match="denominators differ on rows"):
        gate.evaluate(counts(rows=1_977_378), 0, DEFAULT)


def test_a_baseline_mismatch_raises_naming_the_count():
    churns = {"churn_pff": 14_546, "churn_default": 10_074}
    gate.check_baseline(
        DEFAULT | {"first3_rows": 1} | {"positives": 49_812},
        churns,
        gate.BASELINE | {"rows": 1_977_379},
    )
    with pytest.raises(ValueError, match="baseline mismatch.*overall 281607 != 281606"):
        gate.check_baseline(DEFAULT | {"overall": 281_607}, churns)
    with pytest.raises(ValueError, match="churn_pff 14547 != 14546"):
        gate.check_baseline(DEFAULT, {"churn_pff": 14_547, "churn_default": 10_074})


def test_constants_are_the_specs():
    b = gate.BARS
    assert b["overall"] == 253_445 and b["first3"] == 131_585 and b["positive_dis"] == 3_862
    assert b["churn"] == 18_182 and b["late_margin"] == 0.010
    assert gate.BASELINE["rows"] == 1_977_379 and gate.BASELINE["positives"] == 49_812
    assert gate.BASELINE["overall"] == 281_606 and gate.BASELINE["first3"] == 154_806


def test_tick_is_the_learned_states_key():
    t = pl.DataFrame({"t_s": [0.0, 0.1, 0.3, 12.5]})
    assert t.select(gate.tick())["t_s"].to_list() == [0, 1, 3, 125]


def scored_rows(frame_ids):
    n = len(frame_ids)
    return pl.DataFrame(
        {"period": [1] * n, "k": list(range(n)), "frame_id": frame_ids, "pos": [False] * n}
    )


def state_rows(frame_ids, teams, fallback=None):
    n = len(frame_ids)
    return pl.DataFrame(
        {
            "period": [1] * n,
            "k": list(range(n)),
            "frame_id": frame_ids,
            "possession_team": teams,
            "fallback": fallback or [False] * n,
        }
    )


def native(frame_ids, poss):
    n = len(frame_ids)
    return pl.DataFrame(
        {
            "frame_id": frame_ids,
            "p_poss": poss,
            "p_state": [None] * n,
            "seen": [True] * n,
            "gap": [0.0] * n,
            "since": [0.5] * n,
            "dis": [False] * n,
        }
    )


def test_a_native_frame_reused_at_two_ticks_is_counted_twice():
    rows = gate.match_rows(
        "m",
        0,
        scored_rows([7, 7, 8]),
        state_rows([7, 7, 8], ["home", "away", "home"]),
        native([7, 8], ["home", "home"]),
    )
    assert rows.height == 3
    assert rows["dis_learned"].to_list() == [False, True, False]


def test_a_null_learned_possession_counts_as_disagreement():
    rows = gate.match_rows(
        "m",
        0,
        scored_rows([1, 2]),
        state_rows([1, 2], ["home", None]),
        native([1, 2], ["home", "home"]),
    )
    assert rows["dis_learned"].to_list() == [False, True]


def test_rows_without_a_pff_team_drop_out_like_the_old_scripts_join():
    rows = gate.match_rows(
        "m",
        0,
        scored_rows([1, 2, 3]),
        state_rows([1, 2, 3], ["home"] * 3),
        native([1, 3], ["home", "away"]),
    )
    assert rows["frame_id"].to_list() == [1, 3]


def test_join_errors_on_a_missing_tick_or_a_different_frame():
    with pytest.raises(ValueError, match="doesn't cover"):
        gate.join_learned(scored_rows([1, 2, 3]), state_rows([1, 2], ["home", "home"]))
    with pytest.raises(ValueError, match="frame_id differs"):
        gate.join_learned(scored_rows([1, 2]), state_rows([1, 9], ["home", "home"]))
    with pytest.raises(Exception, match="(?i)join|duplicate|1:1"):
        dup = pl.concat(
            [state_rows([1, 2], ["home", "home"]), state_rows([1, 2], ["away", "away"])]
        )
        gate.join_learned(scored_rows([1, 2]), dup)


def test_scored_grid_keeps_masked_rows_that_arent_all_estimated():
    g = pl.DataFrame(
        {
            "period": [1, 1, 1],
            "t_s": [0.0, 0.1, 0.2],
            "frame_id": [5, 6, 6],
            "label_mask_h5": [True, False, True],
            "all_estimated": [False, False, True],
            "label_shot_h5": [True, False, False],
        }
    )
    out = gate.scored_grid(g)
    assert out.to_dicts() == [{"period": 1, "k": 0, "frame_id": 5, "pos": True}]


# --- end to end on the synthetic fold world ---


def run_gate(world, **kw):
    return gate.run(
        CONFIG, world["gs"], world["proc"], world["root"] / "folds.json", log=lambda _: None, **kw
    )


def default_counts(world, monkeypatch):
    """What the default rule scores on the synthetic world, captured at the baseline check."""
    seen, real = {}, gate.check_baseline
    monkeypatch.setattr(gate, "check_baseline", lambda d, c, b=None: seen.update(d | c))
    run_gate(world)
    monkeypatch.setattr(gate, "check_baseline", real)
    return seen


def test_run_reads_only_test_contexts_and_scores_every_match(w, monkeypatch):  # noqa: F811
    asked = []
    real = possession.learned_state

    def spy(m, config, context, *a, **kw):
        asked.append((m, context))
        return real(m, config, context, *a, **kw)

    monkeypatch.setattr(possession, "learned_state", spy)
    seen = {}
    monkeypatch.setattr(gate, "check_baseline", lambda d, c, b=None: seen.update(d | c))
    res = run_gate(w)
    assert [m for m, _ in asked] == sorted(w["folds"]["matches"][i]["match_id"] for i in range(10))
    assert all(c.role == "test" and c.outer == c.fold for _, c in asked)
    assert res["rows"]["match"].n_unique() == 10
    assert set(res["rows"]["fold"].unique()) == {0, 1, 2, 3, 4}
    assert len(res["bars"]) == 5
    assert res["learned"]["rows"] == res["default"]["rows"] == res["rows"].height
    assert len(res["per_match"]) == 10


def test_a_baseline_mismatch_stops_the_gate_before_any_bar(w, monkeypatch):  # noqa: F811
    def no_bars(*a, **k):
        raise AssertionError("bars were evaluated")

    monkeypatch.setattr(gate, "evaluate", no_bars)
    with pytest.raises(ValueError, match="baseline mismatch"):
        run_gate(w)


def test_run_with_the_matching_baseline_returns_a_decision_and_saves_it(w, monkeypatch, tmp_path):  # noqa: F811
    want = default_counts(w, monkeypatch)
    res = run_gate(w, baseline=want)
    assert res["passed"] == all(b.passed for b in res["bars"])
    text = gate.report(res, MODEL)
    for needle in (
        "Per fold",
        "Positives",
        "Ball visibility",
        "Fallbacks and churn per match",
        "S10",
    ):
        assert needle in text
    out = gate.save(res, MODEL, tmp_path / "runs")
    saved = json.loads((out / "gate_h5.json").read_text())
    assert saved["model_id"] == MODEL and len(saved["bars"]) == 5 and len(saved["per_match"]) == 10
    assert (out / "gate_h5.md").read_text() == text


def test_the_gate_refuses_folds_that_differ_from_the_frozen_ones(w, tmp_path):  # noqa: F811
    folds = json.loads((w["root"] / "folds.json").read_text())
    folds["matches"][0]["fold"] = (folds["matches"][0]["fold"] + 1) % 5
    other = tmp_path / "folds.json"
    other.write_text(json.dumps(folds))
    with pytest.raises(ValueError, match="folds differ"):
        gate.run(CONFIG, w["gs"], w["proc"], other, log=lambda _: None)


# --- CLI ---


@pytest.mark.parametrize(
    "args",
    [
        ["--possession-model", "x"],
        ["--possession-model", "x", "--scored", "h3"],
        ["--possession-model", "x", "--scored", "h5", "--state", "team_near_s=0.1"],
        ["--possession-model", "x", "--scored", "h5", "--residual"],
        ["--possession-model", "x", "--scored", "h5", "10502"],
    ],
)
def test_cli_refuses_anything_but_the_gates_own_arguments(args):
    p = subprocess.run(
        [sys.executable, "scripts/possession_split.py", *args],
        check=False,
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT), "PATH": ""},
        capture_output=True,
        text=True,
    )
    assert p.returncode == 2 and "--scored h5" in p.stderr
