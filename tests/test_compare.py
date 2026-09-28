import polars as pl
import pytest
from test_report import build_world, oracle, run

from evaluation.compare import compare, main, wins
from evaluation.report import ReportError


@pytest.fixture
def world(tmp_path):
    return build_world(tmp_path)


def flat(mid, p=0.3):
    return oracle(mid).with_columns(p_h5=pl.lit(p))


def cmp(world, a, b):
    return compare(a, b, world / "gamestate", world / "processed", world / "folds.json")


def test_wins_counts_by_direction():
    assert wins([0.2, 0.3, 0.1], [0.1, 0.4, 0.05], higher=True) == 2
    assert wins([0.2, 0.3, 0.1], [0.1, 0.4, 0.05], higher=False) == 1


def test_oracle_beats_a_flat_run_on_every_fold(world):
    both = ("a", "b")
    a = run(world, pl.concat([oracle(m) for m in both]), tau={"0": 0.5, "1": 0.5}, name="A")
    b = run(world, pl.concat([flat(m) for m in both]), tau={"0": 0.5, "1": 0.5}, name="B")
    text = cmp(world, a, b)
    assert "A = A, B = B" in text
    assert "PR-AUC (higher is better): A better on **2/2** folds" in text
    assert "A wins" in text
    # alarms at each run's tau: the oracle catches both shots, the flat run neither
    assert "miss rate (lower is better): A better on **2/2** folds" in text
    assert "| pooled |" in text


def test_alarm_metrics_need_a_tau_in_both_runs(world):
    both = ("a", "b")
    a = run(world, pl.concat([oracle(m) for m in both]), tau={"0": 0.5, "1": 0.5}, name="A")
    b = run(world, pl.concat([flat(m) for m in both]), name="B")
    text = cmp(world, a, b)
    assert "miss rate" not in text
    assert "no per-fold τ" in text


def test_runs_must_cover_the_same_matches(world):
    a = run(world, pl.concat([oracle("a"), oracle("b")]), name="A")
    b = run(world, oracle("a"), name="B")
    with pytest.raises(ReportError):
        cmp(world, a, b)


def test_cli(world, capsys):
    both = ("a", "b")
    a = run(world, pl.concat([oracle(m) for m in both]), name="A")
    b = run(world, pl.concat([flat(m) for m in both]), name="B")
    args = [str(a), str(b), "--gamestate", str(world / "gamestate")]
    args += ["--processed", str(world / "processed"), "--folds", str(world / "folds.json")]
    assert main(args + ["--out", str(world / "c.md")]) == 0
    assert (world / "c.md").read_text().startswith("# Compare")
    b_short = run(world, flat("a"), name="short")
    assert main([str(a), str(b_short)] + args[2:]) == 1  # different matches
