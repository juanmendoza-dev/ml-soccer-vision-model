"""03 Revision v1h: the hysteresis output rule, the derived manifest and its fail-closed
checks, v1 staying byte-identical, parity with the inner-OOF sweep, and the gate reporting
every attempt."""

import ast
import json
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from test_possession_model import game, trained, write_game  # noqa: F401

from prediction import possession
from prediction import possession_gate as gate
from vision import possession_features as pf
from vision import possession_model as pm
from vision import possession_train as pt
from vision.state import StateConfig

V1, V1H = "lgbm-v1-nested5x4-s1", "lgbm-v1h-nested5x4-s1"
OUT = pm.HYSTERESIS
ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data/processed"
MODELS = ROOT / "data/models/possession"


def rl(p, fallback=None, rule=None, period=None, k=None):
    n = len(p)
    return pm.relabel(
        period or [1] * n,
        k or list(range(n)),
        p,
        fallback or [False] * n,
        rule or [None] * n,
        OUT,
    )


# --- the output rule, by hand ---


def test_in_band_keeps_the_team_and_crossing_a_threshold_switches():
    assert rl([0.8, 0.6, 0.4, 0.26, 0.24, 0.5, 0.74, 0.76]) == (
        ["home"] * 4 + ["away"] * 3 + ["home"]
    )


def test_exactly_at_a_threshold_switches():
    assert rl([0.2, 0.75, 0.25]) == ["away", "home", "away"]


def test_a_reset_row_in_band_takes_v1s_half_cut_and_a_tie_goes_home():
    assert rl([0.5]) == ["home"]
    assert rl([0.49]) == ["away"]
    assert rl([0.6, 0.3]) == ["home", "home"]


def test_fallback_holds_the_team_and_takes_the_rules_only_without_one():
    fb = [False, True, True, False]
    assert rl([0.9, None, None, 0.4], fb, [None, "away", "away", None]) == ["home"] * 4
    # no team yet: the rule's, null included, and the rule's team becomes the current one
    fb = [True, True, False]
    assert rl([None, None, 0.6], fb, [None, "away", None]) == [None, "away", "away"]


def test_period_start_and_grid_gap_reset():
    p = [0.9, 0.4, 0.4, 0.4]
    assert rl(p, period=[1, 1, 2, 2]) == ["home", "home", "away", "away"]
    assert rl(p, k=[0, 1, 3, 4]) == ["home", "home", "away", "away"]


def random_rows(seed, n=4000):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, 1, n)
    at = rng.uniform(size=n) < 0.1  # plenty of rows exactly on a threshold
    p[at] = rng.choice([0.25, 0.75], int(at.sum()))
    fb = rng.uniform(size=n) < 0.2
    rule = rng.choice(np.array(["home", "away", None], dtype=object), n)
    period = np.repeat([1, 2], n // 2)
    k = np.concatenate([np.arange(n // 2), np.arange(n // 2)])
    k[n // 4 :] += 1  # one grid gap inside period 1
    return period, k, p, fb, rule


def test_relabel_is_causal():
    period, k, p, fb, rule = random_rows(0)
    full = pm.relabel(period, k, p, fb, rule, OUT)
    p2 = p.copy()
    p2[3000:] = 1 - p2[3000:]
    assert pm.relabel(period, k, p2, fb, rule, OUT)[:3000] == full[:3000]
    assert pm.relabel(period[:3000], k[:3000], p[:3000], fb[:3000], rule[:3000], OUT) == full[:3000]


def test_mirrored_inputs_swap_every_label_away_from_the_half_tie():
    period, k, p, fb, rule = random_rows(1)
    p[p == 0.5] = 0.51
    swap = {"home": "away", "away": "home", None: None}
    a = pm.relabel(period, k, p, fb, rule, OUT)
    b = pm.relabel(period, k, 1 - p, fb, np.array([swap[r] for r in rule], dtype=object), OUT)
    assert b == [swap[t] for t in a]


# --- the derived manifest ---


@pytest.fixture
def derived(trained, tmp_path):  # noqa: F811
    models = tmp_path / "models"
    shutil.copytree(trained["models"], models)
    man = pt.derive(V1H, models, log=lambda _: None)
    return {**trained, "models": models, "dir": models / V1H, "v1": models / V1, "manifest": man}


def test_derive_copies_v1s_boosters_and_differs_only_in_the_output_rule(derived):
    v1 = json.loads((derived["v1"] / pm.MANIFEST).read_text())
    man = json.loads((derived["dir"] / pm.MANIFEST).read_text())
    assert man["output"] == OUT and man["derived_from"]["model_id"] == V1
    assert man["derived_from"]["manifest_sha256"] == pf.sha256_file(derived["v1"] / pm.MANIFEST)
    assert {k: v for k, v in man.items() if k not in pm.DERIVED_KEYS} == {
        k: v for k, v in v1.items() if k != "model_id"
    }
    for tr in man["trainings"].values():
        for f in ("model.txt", "fit.json"):
            a, b = derived["dir"] / tr["dir"] / f, derived["v1"] / tr["dir"] / f
            assert a.read_bytes() == b.read_bytes()
    with pytest.raises(ValueError, match="sealed"):
        pt.derive(V1H, derived["models"], log=lambda _: None)


def test_v1h_predicts_v1s_probabilities_with_the_hysteresis_labels(derived):
    gs, cache = derived["gs"], derived["cache"]
    a = pm.predict_match(derived["v1"], "outer_0", "g00", gs, cache)
    b = pm.predict_match(derived["dir"], "outer_0", "g00", gs, cache)
    assert b.columns == pm.STATE_COLUMNS
    assert a.drop("possession_team").equals(b.drop("possession_team"))
    p = b["p_home"].fill_null(np.nan).to_numpy()
    want = pm.relabel(b["period"], b["k"], p, b["fallback"], b["rule_possession_team"], OUT)
    assert b["possession_team"].to_list() == want


def write(path, man):
    path.write_text(json.dumps(man, indent=2) + "\n")


@pytest.mark.parametrize(
    "change, why",
    [
        (lambda m: m["output"].update(home_at=0.7), "isn't 03's v1h rule"),
        (lambda m: m["derived_from"].update(manifest_sha256="0" * 64), "missing or changed"),
        (lambda m: m.pop("derived_from"), "only with derived_from"),
        (lambda m: m.update(stride=2), "beyond the output rule"),
    ],
)
def test_a_changed_derived_manifest_fails_closed(derived, change, why):
    path = derived["dir"] / pm.MANIFEST
    man = json.loads(path.read_text())
    change(man)
    write(path, man)
    with pytest.raises(ValueError, match=why):
        pm.predict_match(derived["dir"], "outer_0", "g00", derived["gs"], derived["cache"])


def test_a_changed_source_manifest_fails_the_derived_one(derived):
    path = derived["v1"] / pm.MANIFEST
    man = json.loads(path.read_text())
    write(path, man | {"git_commit": "changed"})
    with pytest.raises(ValueError, match="missing or changed"):
        pm.predict_match(derived["dir"], "outer_0", "g00", derived["gs"], derived["cache"])


def test_the_driver_trains_nothing_for_a_derived_id(derived):
    with pytest.raises(SystemExit):
        pt.main(["--model-id", V1H, "--pilot", "--models", str(derived["models"])])


# --- real data: v1 unchanged, and parity with the sweep ---


def cached(model_id, role_glob):
    key = possession.config_key(StateConfig(possession_model=model_id))
    return sorted(PROCESSED.glob(f"*/state_inferred_age_{key}_{role_glob}.parquet"))


def v1_test_cache():
    """A v1 test-role cache, written by the code before v1h."""
    files = cached(V1, "outer*_test")
    return files[0] if files and (MODELS / V1 / pm.MANIFEST).exists() else None


@pytest.mark.skipif(v1_test_cache() is None, reason="needs v1's learned states and model")
def test_v1s_predictions_are_byte_identical_to_its_overnight_caches():
    path = v1_test_cache()
    m, outer = path.parent.name, path.stem.split("_outer")[1].split("_")[0]
    out = pm.predict_match(MODELS / V1, f"outer_{outer}", m)
    assert out.equals(pl.read_parquet(path))


def sweep_smooth():
    """smooth() from scripts/possession_smoothing_sweep.py, without running the sweep."""
    src = (ROOT / "scripts/possession_smoothing_sweep.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "smooth")
    scope = {}
    exec(compile(ast.Module([fn], []), "sweep", "exec"), scope)
    return scope["smooth"]


@pytest.mark.skipif(not cached(V1, "outer*_inner-oof*"), reason="needs v1's inner-OOF learned states")
def test_the_rule_matches_the_sweeps_chosen_setting_on_inner_oof():
    smooth = sweep_smooth()
    for path in cached(V1, "outer*_inner-oof*")[:20]:
        st = pl.read_parquet(path).sort("period", "k")
        per, team, p, fb = (st[c].to_list() for c in ("period", "rule_possession_team", "p_home", "fallback"))
        want = smooth(per, team, p, fb, OUT["home_at"] - 0.5, OUT["dwell"], True)
        pn = st["p_home"].fill_null(np.nan).to_numpy()
        assert pm.relabel(st["period"], st["k"], pn, st["fallback"], st["rule_possession_team"], OUT) == want, path


# --- the gate reports every attempt ---


def test_earlier_attempts_reads_the_source_gate_or_refuses(tmp_path):
    assert gate.earlier_attempts({"derived_from": None}, tmp_path) == []
    res = {"derived_from": {"model_id": V1}}
    with pytest.raises(FileNotFoundError, match="run it first"):
        gate.earlier_attempts(res, tmp_path)
    d = tmp_path / f"possession-gate-{V1}"
    d.mkdir()
    (d / "gate_h5.json").write_text(json.dumps({"model_id": V1}))
    assert gate.earlier_attempts(res, tmp_path) == [{"model_id": V1}]


def test_the_report_puts_every_attempt_side_by_side():
    names = ["overall", "first3", "positives", "late", "churn"]
    bars = [gate.Bar(n, 1.0, 2.0, True) for n in names]
    rows = pl.DataFrame(
        {"fold": [0], "pos": [True], "ball": ["0 visible"], "since_chg": ["a <1s"], "dis": [True],
         "dis_learned": [False], "match": ["m"]}
    )
    counts = {k: 1 for k in ("rows", "first3_rows", "positives", "late_rows", "overall", "first3",
                             "positive_dis", "late_dis")}
    res = {
        "rows": rows,
        "per_match": pl.DataFrame({"match": ["m"], "fold": [0], "churn_learned": [1], "churn_pff": [1],
                                   "churn_default": [1], "fallback_all": [0], "grid_rows": [1]}),
        "default": counts,
        "learned": counts,
        "fallback_scored": {"m": 0},
        "bars": bars,
        "passed": True,
    }
    v1 = {"model_id": V1, "passed": False, "bars": [vars(b) | {"value": 9.0} for b in bars]}
    text = gate.report(res, V1H, [v1])
    assert "Every attempt through this gate" in text
    assert V1 in text and V1H in text and "FAIL" in text and "PASS" in text
    assert "Every attempt" not in gate.report(res, V1)
