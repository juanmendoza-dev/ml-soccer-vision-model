import json

import polars as pl
import pytest

from evaluation.report import ReportError, main, render
from evaluation.runs import load_run, save_run

N = 50  # 5 s of 10 Hz grid per match
SHOT_T = 3.0  # one open-play home shot per match; rows 1.0-2.9 s are positive


def make_match(root, mid, source, shot=True):
    gs = root / "gamestate" / mid
    gs.mkdir(parents=True)
    pl.DataFrame({"match_id": [mid], "source": [source]}).write_parquet(gs / "match.parquet")
    pl.DataFrame(
        {"frame_id": [0, 1], "period": [1, 1], "timestamp_s": [0.0, SHOT_T]}
    ).write_parquet(gs / "frames.parquet")
    pl.DataFrame(
        {
            "frame_id": [1] if shot else [],
            "event_type": ["shot"] if shot else [],
            "team": ["home"] if shot else [],
            "outcome": ["saved"] if shot else [],
            "set_piece": ["open_play"] if shot else [],
            "set_play_phase": [False] if shot else [],
        },
        schema={
            "frame_id": pl.Int64,
            "event_type": pl.String,
            "team": pl.String,
            "outcome": pl.String,
            "set_piece": pl.String,
            "set_play_phase": pl.Boolean,
        },
    ).write_parquet(gs / "events.parquet")
    t = [round(0.1 * i, 1) for i in range(N)]
    pr = root / "processed" / mid
    pr.mkdir(parents=True)
    pl.DataFrame(
        {
            "match_id": [mid] * N,
            "period": [1] * N,
            "t_s": t,
            "possession_team": ["home"] * N,
            "ball_state": ["alive"] * N,
            "all_estimated": [False] * N,
            "label_mask_h5": [True] * N,
            "label_shot_h5": [shot and x < SHOT_T and x >= SHOT_T - 2 for x in t],
        }
    ).write_parquet(pr / "frames_10hz.parquet")


def oracle(mid):
    """0.9 on the positive rows, 0.1 elsewhere: one alarm from 1.0 s, lead 2.0 s."""
    t = [round(0.1 * i, 1) for i in range(N)]
    return pl.DataFrame(
        {
            "match_id": [mid] * N,
            "period": [1] * N,
            "t_s": t,
            "p_h5": [0.9 if 1.0 <= x < SHOT_T else 0.1 for x in t],
        }
    )


@pytest.fixture
def world(tmp_path):
    for mid in ("a", "b"):
        make_match(tmp_path, mid, "pff")
    make_match(tmp_path, "met", "metrica")
    make_match(tmp_path, "ids", "idsse")
    folds = {
        "n_folds": 2,
        "seed": 1,
        "frozen": {"pff": "2026-09-26"},
        "matches": [
            {"match_id": "a", "source": "pff", "fold": 0, "open_play_shots": 1},
            {"match_id": "b", "source": "pff", "fold": 1, "open_play_shots": 1},
        ],
    }
    (tmp_path / "folds.json").write_text(json.dumps(folds))
    return tmp_path


def run(world, preds, tau=None, name="r"):
    meta = {"model": "test", "horizons": ["h5"], "config": {"x": 1}}
    if tau is not None:
        meta["tau"] = {"h5": tau}
    return save_run(world / "runs" / name, preds, meta)


def report(world, run_dir, **kw):
    return render(run_dir, world / "gamestate", world / "processed", world / "folds.json", **kw)


def test_save_run_round_trip_and_stamps(world):
    d = run(world, oracle("a"))
    meta, preds = load_run(d)
    assert meta["run_id"] == "r" and meta["model"] == "test"
    assert {"git_commit", "git_dirty", "created"} <= set(meta)
    assert preds.columns == ["match_id", "period", "t_s", "p_h5"]


def test_save_run_rejects_bad_p(world):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        run(world, oracle("a").with_columns(p_h5=pl.lit(1.5)))
    with pytest.raises(ValueError, match="missing columns"):
        run(world, oracle("a").drop("p_h5"))


def test_report_pooled_and_alarms(world):
    d = run(world, pl.concat([oracle("a"), oracle("b")]), tau={"0": 0.5, "1": 0.5})
    text = report(world, d)
    assert "2 matches, folds [0, 1]" in text
    # 2 shots, none missed, lead 2.0 s, no false alarms
    assert "| 2 | 0 | 0.000 | 2.000 | 2.00–2.00 | 2 | 0 | 0.000 |" in text
    assert "| 2–3 s | 2 |" in text
    assert "PR-AUC" in text and "mean ± std" in text and "τ sweep" in text


def test_each_fold_uses_its_own_tau(world):
    # fold 1's tau is above every p, so its shot is missed and fold 0's isn't
    d = run(world, pl.concat([oracle("a"), oracle("b")]), tau={"0": 0.5, "1": 0.95})
    assert "| 2 | 1 | 0.500 |" in report(world, d)


def test_no_tau_means_no_alarm_numbers(world):
    d = run(world, pl.concat([oracle("a"), oracle("b")]))
    text = report(world, d)
    assert "τ not chosen" in text
    assert "lead q25" not in text


def test_missing_cv_match_fails_unless_partial(world):
    d = run(world, oracle("a"))
    with pytest.raises(ReportError, match="1 of 2 matches"):
        report(world, d)
    assert report(world, d, allow_partial=True).startswith("# PARTIAL RUN")


def test_cv_match_must_be_in_folds(world):
    make_match(world, "c", "pff")
    d = run(world, pl.concat([oracle("a"), oracle("b"), oracle("c")]))
    with pytest.raises(ReportError, match="c .*folds.json"):
        report(world, d)


def test_duplicate_and_stray_predictions_fail(world):
    both = pl.concat([oracle("a"), oracle("b")])
    d = run(world, pl.concat([both, oracle("a").head(1)]), name="dup")
    with pytest.raises(ReportError, match="duplicate"):
        report(world, d)
    stray = both.with_columns(t_s=pl.when(pl.col("t_s") == 0.0).then(99.0).otherwise("t_s"))
    d = run(world, stray, name="stray")
    with pytest.raises(ReportError, match="don't land"):
        report(world, d)


def test_null_p_on_a_scored_row_names_the_match(world):
    b = oracle("b").with_columns(p_h5=pl.when(pl.col("t_s") == 2.0).then(None).otherwise("p_h5"))
    d = run(world, pl.concat([oracle("a"), b]))
    with pytest.raises(ReportError, match=r"\['b'\]"):
        report(world, d)


def test_metrica_dropped_and_idsse_only_with_final(world):
    preds = pl.concat([oracle("a"), oracle("b"), oracle("met"), oracle("ids")])
    d = run(world, preds, tau={"0": 0.5, "1": 0.5, "final": 0.5})
    text = report(world, d)
    assert "Dropped 1 matches" in text
    assert "1 IDSSE matches present, not shown" in text and "### IDSSE" not in text
    final = report(world, d, final=True)
    assert "### IDSSE (1 matches)" in final and "Alarms at τ = 0.5" in final


def test_cli_writes_report_md(world, capsys):
    d = run(world, pl.concat([oracle("a"), oracle("b")]))
    args = [
        str(d),
        "--gamestate",
        str(world / "gamestate"),
        "--processed",
        str(world / "processed"),
        "--folds",
        str(world / "folds.json"),
    ]
    assert main(args) == 0
    assert (d / "report.md").read_text().startswith("# Report: r")
    assert main([*args, "--final"]) == 0
    partial = save_run(world / "runs" / "p", oracle("a"), {"model": "m", "horizons": ["h5"]})
    assert main([str(partial), *args[1:]]) == 1
    assert "1 of 2 matches" in capsys.readouterr().err
