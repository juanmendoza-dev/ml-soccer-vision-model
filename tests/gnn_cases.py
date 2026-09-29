"""The frame GNN's tests. They need torch, so tests/test_gnn.py runs this file in its own
pytest process (see there). Run it directly with: python -m pytest tests/gnn_cases.py
"""

import json
import math

import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")

from test_cv import match as cv_match
from test_features import mirror, synth

from evaluation.metrics import pr_auc
from prediction import cv
from prediction.features import FEATURES, match_features
from prediction.gnn import GNNModel, fit_temperature, resolve_device, sigmoid
from prediction.gnn_net import EDGE_FEATURES, FrameGNN, TemporalGNN, edge_features
from prediction.graphs import IDX, MAX_NODES, GraphStore, match_graphs

TINY = {"d": 16, "layers": 2, "max_epochs": 8, "batch_size": 64, "lr": 3e-3, "patience": 8}


def model(features=FEATURES, graphs=None, **params):
    return GNNModel(features, {**TINY, **params}, device="cpu", graphs=graphs)


def world(seeds, n=120, change=None, label=None):
    """Random matches from test_features.synth (22 players and a ball, possession flipping
    every 3 s, a third of everything ESTIMATED) with graphs and v1 features. The toy label
    is the ball within 30 m of goal. change(frames, objects) edits a match first."""
    parts, datas = [], []
    for j, seed in enumerate(seeds):
        frames, objects = synth(seed, n)
        if change:
            frames, objects = change(frames, objects)
        mid = f"s{j}"
        parts.append((mid, match_graphs(frames, objects.lazy())))
        f = match_features(frames, objects.lazy())
        y = f["ball_dist"].fill_nan(99.0) < 30 if label is None else label(f)
        datas.append(
            f.with_columns(
                match_id=pl.lit(mid),
                ball_state=pl.lit("alive"),
                eligible=pl.lit(True),
                all_estimated=pl.lit(False),
                label_mask_h5=pl.lit(True),
                label_shot_h5=y,
            )
        )
    return GraphStore.from_parts(parts), pl.concat(datas).with_row_index("row")


@pytest.fixture(scope="module")
def fitted():
    store, data = world(range(30, 38))
    return model(graphs=store).fit(data, "h5"), store, data


def predict_with(m, store, data):
    m.graphs = store
    return m.predict(data).to_numpy()


def test_learns_the_toy_and_stays_near_the_base_rate(fitted):
    m, _, data = fitted
    held_store, held = world(range(50, 54))
    p = predict_with(m, held_store, held)
    y = held["label_shot_h5"].to_numpy()
    assert pr_auc(y, p) > 2 * y.mean()
    assert p[y].mean() > 3 * p[~y].mean()
    assert m.coef == {}
    info = m.fit_info
    assert info["es_matches"] and set(info["es_matches"]) <= set(data["match_id"])
    assert info["device"] == "cpu" and info["epochs_run"] == len(info["curve"])
    json.dumps(info)  # goes into run.json


def test_predictions_only_use_the_past(fitted):
    m, _, _ = fitted
    cut = 6.0

    def later(frames, objects):
        after = pl.col("t_s") > cut
        return (
            frames.with_columns(
                possession_team=pl.when(after).then(pl.lit("home")).otherwise("possession_team"),
                flipped=pl.when(after).then(True).otherwise("flipped"),
            ),
            objects.with_columns(
                x=pl.when(after).then(pl.col("x") * -3 + 7).otherwise("x"),
                y=pl.when(after).then(0.0).otherwise("y"),
                visible=pl.when(after).then(~pl.col("visible")).otherwise("visible"),
            ),
        )

    base_store, base = world(range(40, 43))
    moved_store, moved = world(range(40, 43), change=later)
    a, b = predict_with(m, base_store, base), predict_with(m, moved_store, moved)
    upto = (base["t_s"] <= cut).to_numpy()
    assert np.allclose(a[upto], b[upto], rtol=0, atol=1e-6)
    assert not np.allclose(a[~upto], b[~upto], rtol=0, atol=1e-6)  # later rows did change


def test_estimated_positions_never_reach_a_prediction(fitted):
    m, _, _ = fitted

    def scramble(frames, objects):
        est = ~pl.col("visible")
        return frames, objects.with_columns(
            x=pl.when(est).then(pl.col("x") * 7 - 20).otherwise("x"),
            y=pl.when(est).then(-pl.col("y") + 3).otherwise("y"),
            z=pl.when(est).then(9.0).otherwise("z"),
        )

    a = predict_with(m, *world(range(40, 43)))
    b = predict_with(m, *world(range(40, 43), change=scramble))
    assert np.array_equal(a, b)


def test_a_mirrored_match_gives_the_same_predictions(fitted):
    """Pitch rotated 180 degrees and the teams swapped: same graphs, same v1 features
    (every row has a team here), so the same p."""
    m, _, _ = fitted
    a = predict_with(m, *world(range(40, 43)))
    b = predict_with(m, *world(range(40, 43), change=mirror))
    assert np.allclose(a, b, rtol=0, atol=1e-6)


def test_a_rows_p_doesnt_depend_on_its_batch(fitted):
    m, store, data = fitted
    full = predict_with(m, store, data)
    some = data.filter(pl.col("row") % 7 == 3).reverse()
    alone = predict_with(m, store, some)
    assert np.allclose(alone, full[some["row"].to_numpy()], rtol=0, atol=1e-6)


def test_deterministic_on_cpu():
    store, data = world(range(60, 64))
    a = model(graphs=store).fit(data, "h5").predict(data).to_numpy()
    b = model(graphs=store).fit(data, "h5").predict(data).to_numpy()
    assert np.array_equal(a, b)


def test_predict_nulls_where_05_says_so(fitted):
    m, store, data = fitted
    rows = data.head(3).with_columns(
        eligible=pl.Series([True, False, True]), all_estimated=pl.Series([False, False, True])
    )
    p = predict_with(m, store, rows)
    assert 0 < p[0] < 1 and np.isnan(p[1]) and np.isnan(p[2])


def test_ignores_masked_and_all_estimated_rows():
    store, data = world(range(70, 76))
    n = data.height
    poison_store, poison = world(range(80, 84), label=lambda f: f["ball_dist"].fill_nan(0) > 30)
    both = GraphStore.from_parts(
        [(i, g) for i, g in zip(store.ids, split(store), strict=True)]
        + [(f"p{j}", g) for j, g in enumerate(split(poison_store))]
    )
    masked = poison["match_id"].is_in(["s0", "s1"])  # the other two are all ESTIMATED
    poison = poison.with_columns(
        match_id="p" + pl.col("match_id").str.slice(1),
        row=pl.col("row") + n,
        label_mask_h5=~masked,
        all_estimated=~masked,
    )
    a = model(graphs=store).fit(data, "h5")
    b = model(graphs=both).fit(pl.concat([data, poison]), "h5")
    assert a.n_train == b.n_train == n
    assert np.array_equal(a.predict(data).to_numpy(), b.predict(data).to_numpy())


def split(store: GraphStore) -> list[dict]:
    """A store back into its per-match graphs."""
    out = []
    for j in range(len(store.ids)):
        m = store.match == j
        out.append(
            {
                "nodes": store.nodes[m],
                "n": store.n[m],
                "period": store.period[m],
                "k": store.k[m],
                "capped": np.array(0),
                "team": store.team[m],
                "sign": store.sign[m],
                "poss": store.poss[m],
            }
        )
    return out


def test_graph_only_reads_no_hand_features():
    store, data = world(range(60, 64))
    m = model((), graphs=store).fit(data, "h5")
    a = m.predict(data).to_numpy()
    b = m.predict(data.drop(FEATURES)).to_numpy()
    assert np.array_equal(a, b)


def test_globals_only_arm_reads_no_graph():
    store, data = world(range(60, 64))
    m = model(graphs=store, layers=0).fit(data, "h5")
    shuffled = GraphStore.from_parts(list(zip(store.ids, split(store), strict=True)))
    shuffled.nodes = shuffled.nodes[::-1].copy()
    shuffled.n = shuffled.n[::-1].copy()
    assert np.array_equal(m.predict(data).to_numpy(), predict_with(m, shuffled, data))
    with pytest.raises(ValueError, match="nothing to read"):
        FrameGNN(15, 0, layers=0)


def test_edges_are_in_meters():
    x = torch.zeros(1, 3, len(IDX))
    x[0, 1, IDX["y"]] = 10.0 / 34  # 10 m across the pitch
    x[0, 2, IDX["x"]] = 10.0 / 52.5  # 10 m along it
    x[0, 1, IDX["vx"]] = 0.5  # 5 m/s
    x[0, [0, 1], IDX["has_vel"]] = 1.0
    e = edge_features(x)
    col = {f: i for i, f in enumerate(EDGE_FEATURES)}
    assert e.shape == (1, 3, 3, len(EDGE_FEATURES))
    assert float(e[0, 0, 1, col["dist"]]) * 20 == pytest.approx(10.0, abs=1e-4)
    assert float(e[0, 0, 2, col["dist"]]) * 20 == pytest.approx(10.0, abs=1e-4)
    assert float(e[0, 0, 1, col["dy"]]) * 20 == pytest.approx(10.0, abs=1e-4)
    assert float(e[0, 1, 0, col["dy"]]) * 20 == pytest.approx(-10.0, abs=1e-4)  # j seen from i
    assert float(e[0, 0, 1, col["dvx"]]) * 10 == pytest.approx(5.0, abs=1e-4)
    assert float(e[0, 0, 2, col["dvx"]]) == 0 and float(e[0, 0, 2, col["both_vel"]]) == 0
    assert float(e[0, 1, 1, col["dist"]]) == 0  # self-loop


def test_padding_and_empty_graphs_are_inert():
    net = FrameGNN(len(IDX), 0, d=8, layers=2).eval()
    x = torch.randn(3, MAX_NODES, len(IDX))
    n = torch.tensor([0, 2, MAX_NODES])
    g = torch.zeros(3, 0)
    out = net(x, n, g)
    assert torch.isfinite(out).all()
    y = x.clone()
    y[1, 2:] = torch.randn(MAX_NODES - 2, len(IDX)) * 100  # garbage in the padding
    assert torch.allclose(net(y, n, g)[1], out[1], atol=1e-6)


def test_temperature_recovers_a_known_scale():
    rng = np.random.default_rng(0)
    z = rng.normal(-3, 1.5, 200_000)
    y = (rng.random(len(z)) < sigmoid(z)).astype(float)
    assert fit_temperature(2 * z, y) == pytest.approx(2.0, rel=0.05)


def test_device_resolution():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in ("cuda", "mps", "cpu")
    with pytest.raises(ValueError):
        resolve_device("tpu")


# --- through prediction.cv, on test_cv's world (a shot at 20 s in every match) ---


def cv_world(n=15):
    """test_cv's toy (the ball walks toward goal, a shot at 20 s) with objects to build
    graphs from: the ball, its carrier and a defender, all visible."""
    ids = [f"m{i:02d}" for i in range(n)]
    folds = {
        "n_folds": 5,
        "seed": 7,
        "frozen": {},
        "matches": [
            {"match_id": i, "source": "pff", "fold": k % 5, "open_play_shots": 1}
            for k, i in enumerate(ids)
        ],
    }
    parts, datas = [], []
    for k, mid in enumerate(ids):
        d = cv_match(mid, k)
        bx = 52.5 - d["ball_dist"].to_numpy()
        t = d["t_s"].to_numpy()
        objs = []
        for i in range(d.height):
            for oid, team, x, y in (
                ("ball", None, bx[i], 0.0),
                ("h1", "home", bx[i] - 1.0, 1.0),
                ("a1", "away", 45.0, -3.0),
            ):
                objs.append(
                    {
                        "period": 1,
                        "t_s": t[i],
                        "object_id": oid,
                        "object_type": "ball" if oid == "ball" else "player",
                        "team": team,
                        "x": float(x),
                        "y": y,
                        "z": 0.0,
                        "visible": True,
                    }
                )
        frames = d.select("period", "t_s", "possession_team", flipped=pl.lit(False))
        parts.append((mid, match_graphs(frames, pl.DataFrame(objs).lazy())))
        datas.append(d)
    data = pl.concat(datas).with_row_index("row")
    data = data.with_columns(
        pl.lit(float("nan")).alias(f) for f in FEATURES if f not in data.columns
    )
    shots = pl.DataFrame(
        {
            "match_id": ids,
            "period": [1] * n,
            "t_s": [20.0] * n,
            "team": ["home"] * n,
            "open_play": [True] * n,
        }
    )
    return folds, data, shots, GraphStore.from_parts(parts)


CV_CONFIG = {"params": TINY, "device": "cpu"}


def test_gnn_runs_through_cv_and_records_its_fits():
    folds, data, shots, store = cv_world()
    preds, meta = cv.run_cv(
        "gnn",
        ["h5"],
        folds,
        data,
        shots,
        log=lambda *_: None,
        data_config={**CV_CONFIG, "features": [], "globals": "none"},
        model_kwargs={"graphs": store},
    )
    y = data["label_shot_h5"].to_numpy()
    p = preds["p_h5"].to_numpy()
    assert p[y].mean() > 2 * p[~y].mean()
    f0 = meta["folds"]["h5"]["0"]
    assert f0["coef"] == {} and f0["es_matches"] and f0["device"] == "cpu"
    assert meta["config"]["features"] == [] and meta["config"]["globals"] == "none"
    assert "graphs" not in meta["config"] and "log" not in meta["config"]
    assert set(meta["config"]["timing"]["h5"]) == {"0", "1", "2", "3", "4", "final"}
    fold_of = {m["match_id"]: m["fold"] for m in folds["matches"]}
    for k, info in meta["folds"]["h5"].items():
        assert all(fold_of[i] != int(k) for i in info["es_matches"])
    json.dumps(meta)


def test_every_gnn_prediction_is_out_of_fold(monkeypatch):
    folds, data, shots, store = cv_world()
    fits = []

    class Spy(GNNModel):
        def fit(self, df, h):
            self.trained_on = set(df["match_id"].unique())
            fits.append(self.trained_on)
            super().fit(df, h)
            assert set(self.fit_info["es_matches"]) <= self.trained_on
            return self

        def predict(self, df):
            assert not self.trained_on & set(df["match_id"].unique())
            return super().predict(df)

    monkeypatch.setitem(cv.MODELS, "spy", (Spy, {"features": [], **CV_CONFIG}))
    preds, _ = cv.run_cv(
        "spy", ["h5"], folds, data, shots, log=lambda *_: None, model_kwargs={"graphs": store}
    )
    assert len(fits) == 11  # 5 folds x (inner fit + outer fit) + the final inner fit
    assert preds["p_h5"].null_count() == 0


def test_one_fold_is_a_partial_timing_run():
    folds, data, shots, store = cv_world()
    preds, meta = cv.run_cv(
        "gnn",
        ["h5"],
        folds,
        data,
        shots,
        log=lambda *_: None,
        data_config={**CV_CONFIG, "features": []},
        model_kwargs={"graphs": store},
        only_folds=[2],
    )
    assert set(meta["tau"]["h5"]) == {"2"} and meta["partial"]["folds"] == [2]
    assert set(meta["config"]["timing"]["h5"]) == {"2"}
    in_fold = preds.join(
        pl.DataFrame([{"match_id": m["match_id"], "fold": m["fold"]} for m in folds["matches"]]),
        on="match_id",
    )
    assert in_fold.filter(pl.col("fold") == 2)["p_h5"].null_count() == 0
    assert (
        in_fold.filter(pl.col("fold") != 2)["p_h5"].null_count()
        == in_fold.filter(pl.col("fold") != 2).height
    )
    meta["config"]["timing"] |= {"load_s": 1.0, "graphs_s": 2.0, "cv_s": 3.0}  # as main adds
    lines = cv.one_fold_estimate(meta, 2, 5, 60.0)
    assert "Full run: about" in lines[0] and math.isfinite(
        meta["config"]["timing"]["h5"]["2"]["predict_s"]
    )


# --- the temporal GNN (05 model 3): windows of 3 steps 0.5 s apart ---

TEMPORAL = {"steps": 3, "step_rows": 5, "batch_size": 32}


@pytest.fixture(scope="module")
def temporal():
    store, data = world(range(30, 38))
    return model(graphs=store, **TEMPORAL).fit(data, "h5"), store, data


def test_temporal_learns_the_toy(temporal):
    m, _, _ = temporal
    held_store, held = world(range(50, 54))
    p = predict_with(m, held_store, held)
    y = held["label_shot_h5"].to_numpy()
    assert pr_auc(y, p) > 2 * y.mean()
    assert m.fit_info["n_params"] > model(graphs=held_store).params["d"]
    json.dumps(m.fit_info)


def test_temporal_learns_what_only_the_past_shows():
    """The label is whether the ball got closer to goal between 1.0 and 0.5 s ago. The
    ball is a random walk, so nothing at t shows that step (the velocity at t covers the
    last 0.5 s only): the window has it, the frame GNN without globals doesn't. One team
    keeps the ball, so every row's frame is the same."""

    def home(frames, objects):
        return frames.with_columns(possession_team=pl.lit("home"), flipped=pl.lit(False)), objects

    def moved(f):
        d = f["ball_dist"].to_numpy()
        lag = lambda n: np.r_[np.full(n, np.nan), d[:-n]]
        step = lag(10) - lag(5)
        return pl.Series(np.nan_to_num(step, nan=0.0) > 0.3)

    store, data = world(range(30, 38), n=400, change=home, label=moved)
    held_store, held = world(range(50, 53), n=400, change=home, label=moved)
    y = held["label_shot_h5"].to_numpy()
    scores = {}
    longer = {"max_epochs": 15, "patience": 15, "stride": 1}
    for name, extra in (("frame", {}), ("temporal", TEMPORAL)):
        m = model((), graphs=store, **longer, **extra).fit(data, "h5")
        scores[name] = pr_auc(y, predict_with(m, held_store, held))
    assert scores["temporal"] > scores["frame"] + 0.1, (scores, y.mean())


def test_temporal_only_uses_the_past(temporal):
    m, _, _ = temporal
    cut = 6.0

    def later(frames, objects):
        after = pl.col("t_s") > cut
        return frames.with_columns(
            possession_team=pl.when(after).then(pl.lit("home")).otherwise("possession_team"),
            flipped=pl.when(after).then(True).otherwise("flipped"),
        ), objects.with_columns(
            x=pl.when(after).then(pl.col("x") * -3 + 7).otherwise("x"),
            visible=pl.when(after).then(~pl.col("visible")).otherwise("visible"),
        )

    base_store, base = world(range(40, 43))
    a = predict_with(m, base_store, base)
    b = predict_with(m, *world(range(40, 43), change=later))
    upto = (base["t_s"] <= cut).to_numpy()
    assert np.allclose(a[upto], b[upto], rtol=0, atol=1e-6)
    assert not np.allclose(a[~upto], b[~upto], rtol=0, atol=1e-6)


def test_temporal_ignores_estimated_positions(temporal):
    m, _, _ = temporal

    def scramble(frames, objects):
        est = ~pl.col("visible")
        return frames, objects.with_columns(
            x=pl.when(est).then(pl.col("x") * 7 - 20).otherwise("x"),
            y=pl.when(est).then(-pl.col("y") + 3).otherwise("y"),
        )

    a = predict_with(m, *world(range(40, 43)))
    b = predict_with(m, *world(range(40, 43), change=scramble))
    assert np.array_equal(a, b)


def test_temporal_mirrored_match_gives_the_same_p(temporal):
    m, _, _ = temporal
    a = predict_with(m, *world(range(40, 43)))
    b = predict_with(m, *world(range(40, 43), change=mirror))
    assert np.allclose(a, b, rtol=0, atol=1e-6)


def test_temporal_rows_p_doesnt_depend_on_its_batch(temporal):
    m, store, data = temporal
    full = predict_with(m, store, data)
    some = data.filter(pl.col("row") % 5 == 1).reverse()
    assert np.allclose(predict_with(m, store, some), full[some["row"].to_numpy()], atol=1e-6)


def test_temporal_empty_steps_and_padding_are_inert():
    net = TemporalGNN(len(IDX), 0, d=8, layers=2).eval()
    x = torch.randn(2, 3, MAX_NODES, len(IDX))
    n = torch.tensor([[0, 4, 5], [0, 0, 3]])
    g = torch.zeros(2, 0)
    out = net(x, n, g)
    assert torch.isfinite(out).all()
    y = x.clone()
    y[:, 0] = torch.randn(2, MAX_NODES, len(IDX)) * 100  # a step with no row
    y[0, 1, 4:] = 50.0  # padding
    assert torch.allclose(net(y, n, g), out, atol=1e-6)
    with pytest.raises(ValueError, match="needs message passing"):
        TemporalGNN(15, 4, layers=0)


def test_tgnn_runs_through_cv_out_of_fold(monkeypatch):
    folds, data, shots, store = cv_world()
    seen = []

    class Spy(GNNModel):
        def fit(self, df, h):
            self.trained_on = set(df["match_id"].unique())
            seen.append(self.params["steps"])
            return super().fit(df, h)

        def predict(self, df):
            assert not self.trained_on & set(df["match_id"].unique())
            return super().predict(df)

    params = {**cv.MODELS["tgnn"][1]["params"], **TINY, **TEMPORAL}
    monkeypatch.setitem(cv.MODELS, "spy", (Spy, {"features": [], **CV_CONFIG, "params": params}))
    preds, meta = cv.run_cv(
        "spy", ["h5"], folds, data, shots, log=lambda *_: None, model_kwargs={"graphs": store}
    )
    assert seen == [3] * 11
    y = data["label_shot_h5"].to_numpy()
    p = preds["p_h5"].to_numpy()
    assert preds["p_h5"].null_count() == 0 and p[y].mean() > 2 * p[~y].mean()
    assert meta["config"]["params"]["steps"] == 3
    json.dumps(meta)


def test_tgnn_config_is_the_frame_gnn_with_a_window():
    frame, temporal = cv.MODELS["gnn"][1], cv.MODELS["tgnn"][1]
    assert frame["params"]["steps"] == 1 and temporal["params"]["steps"] == 6
    assert {k: v for k, v in temporal.items() if k != "params"} == {
        k: v for k, v in frame.items() if k != "params"
    }
    assert cv.gnn_overrides(["steps=4", "temperature=false"]) == {
        "steps": 4,
        "temperature": False,
    }
    with pytest.raises(ValueError):
        cv.gnn_overrides(["nope=1"])
