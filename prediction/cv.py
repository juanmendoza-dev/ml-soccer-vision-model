"""Out-of-fold predictions with 07's tau rule, written as a run (07, "Run format").

    python -m prediction.cv [--model floor|lgbm|gnn|tgnn] [--ball-source held|raw] [--run-id ID]
        [--no-report] [--features-version 1|2] [--degrade ARM:SEVERITY ...]
        [--degrade-arm both|test] [--degrade-seed N] [--device auto|cuda|mps|cpu]
        [--gnn-globals v1|none] [--gnn-layers N] [--gnn-param KEY=VALUE ...] [--one-fold N]
        [--possession provider|inferred] [--stale-possession none|unknown] [--stale-after S]
        [--possession-model ID]

Per horizon and outer fold: fit on the training matches minus the inner split, pick
tau on the inner matches, refit on all training matches, predict the held-out fold.
The "final" tau does the same over all CV matches with inner_split(None).

--ball-source raw feeds PFF's ESTIMATED ball, which uses later frames (05, Leakage). It's
only there to measure how much that leak inflates a result; held is the real number.

--features-version picks the lgbm's feature set (05): 1 is the baseline, 2 adds the
lead-time features. Each version has its own feature cache.

--degrade runs the vision sensitivity test (05): objects are degraded the way vision
fails before features are built. --degrade-arm both (default) degrades training and test
matches; test fits and picks tau on clean data and predicts degraded held-out folds.

--possession inferred builds the model's inputs from stage 8's possession (vision.state,
run on the game state) instead of PFF's (05, "Inferred possession"; 07 #6). Labels, scored
rows, tau and the alarms stay on PFF's. lgbm and floor only, not with --degrade.
--stale-possession unknown --stale-after S makes stage 8's possession null where its last
confirmed carrier is more than S seconds old (arm U, 05 "Stale possession").
--features-version 3 adds stage 8's carrier age as a feature, with either possession, so
it runs stage 8 too (arm S). lgbm and floor only, not with --degrade.
--possession-model ID (with --possession inferred) takes possession from a sealed learned
model (03 stage 8, vision.possession_train) under strict nested contexts: each outer fold's
data is assembled once with test possession for its fold and inner-OOF possession for the
training matches, then both horizons run on it; the final tau uses every match's outer
test possession. lgbm, v1, held ball only.

--model gnn is the frame GNN (05 model 2) on graphs cached per match. --gnn-globals none
and --gnn-layers 0 are its ablation arms. --one-fold N is a timing run: outer fold N only,
no final tau, saved as a partial run with an estimate of the full run's time.

--model tgnn is the temporal GNN (05 model 3): the same graphs, a 2.5 s window of them per
row. --gnn-param overrides one of gnn.PARAMS (steps, step_rows, batch_size, stride, ...),
so a timing run can shrink the window or the batch without a code edit.
"""

import argparse
import contextlib
import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

from evaluation.folds import FOLDS_PATH, GAMESTATE_DIR, inner_split, load
from evaluation.metrics import MAX_FALSE_PER_MATCH, choose_tau, shots_table
from evaluation.report import PROCESSED_DIR, render
from evaluation.runs import RUNS_DIR, save_run
from prediction import degrade, floor, gnn, graphs, lgbm, possession
from prediction.features import (
    BALL_SOURCES,
    FEATURE_SETS,
    FEATURES,
    FEATURES_VERSION,
    STAGE8_VERSIONS,
    load_match,
)
from prediction.floor import training_rows

MODELS = {
    "floor": (floor.LogisticFloor, {"features": list(floor.FEATURES), "l2": 1e-6}),
    "lgbm": (
        lgbm.LGBMModel,
        {
            "features": list(FEATURES),
            "features_version": FEATURES_VERSION,
            "params": lgbm.PARAMS,
            "max_rounds": 2000,
            "early_stopping": 100,
            "es_share": 0.15,
        },
    ),
    "gnn": (
        gnn.GNNModel,
        {
            "features": list(FEATURES),  # the global vector; [] with --gnn-globals none
            "features_version": FEATURES_VERSION,
            "globals": "v1",
            "graphs_version": graphs.GRAPHS_VERSION,
            "params": gnn.PARAMS,
            "es_share": 0.15,
            "device": "auto",
        },
    ),
}
MODELS["tgnn"] = (gnn.GNNModel, {**MODELS["gnn"][1], "params": gnn.TEMPORAL_PARAMS})
GNNS = {"gnn", "tgnn"}
# config keys about the data, not the model's constructor
DATA_KEYS = (
    "features",
    "features_version",
    "ball_source",
    "degrade",
    "degrade_arm",
    "degrade_seed",
    "degrade_realized",
    "possession",
    "state_config",
    "possession_scored",
    "stale_possession",
    "globals",
    "graphs_version",
)
# models whose feature list follows features_version (the floor's is fixed)
VERSIONED = {"lgbm", "gnn", "tgnn"}
KEYS = ["match_id", "period", "t_s"]


def load_data(
    ids: list[str],
    processed_dir: Path,
    gamestate_dir: Path,
    horizons,
    ball_source: str = "held",
    degrade_specs=(),
    degrade_seed: int = degrade.DEFAULT_SEED,
    degrade_stats: dict | None = None,
    features_version: str = FEATURES_VERSION,
    possession_source: str = "provider",
    possession_stats: dict | None = None,
    stale_after: float | None = None,
    state_config: possession.StateConfig | None = None,
    contexts: dict | None = None,
):
    """contexts {match_id: possession.Context} is for learned possession only: each match
    loads under its context, tagged with a possession_context column."""

    def one(i):
        df = load_match(
            i,
            processed_dir,
            horizons,
            ball_source,
            degrade=degrade_specs,
            degrade_seed=degrade_seed,
            degrade_stats=degrade_stats,
            features_version=features_version,
            possession=possession_source,
            gamestate_dir=gamestate_dir,
            possession_stats=possession_stats,
            stale_after=stale_after,
            state_config=state_config,
            context=contexts[i] if contexts else None,
        )
        return df.with_columns(possession_context=pl.lit(contexts[i].suffix)) if contexts else df

    data = pl.concat([one(i) for i in ids]).with_row_index("row")
    shots = pl.concat(
        [
            shots_table(
                pl.read_parquet(gamestate_dir / i / "events.parquet"),
                pl.read_parquet(
                    gamestate_dir / i / "frames.parquet",
                    columns=["frame_id", "period", "timestamp_s"],
                ),
                i,
            )
            for i in ids
        ]
    )
    return data, shots


def of(data: pl.DataFrame, ids) -> pl.DataFrame:
    return data.filter(pl.col("match_id").is_in(list(ids)))


def fit_and_choose(make, data, shots, fit_ids, val_ids, h, max_false):
    """Fit on fit_ids, pick tau on val_ids (07). Returns (tau choice, fitted model)."""
    if set(fit_ids) & set(val_ids):
        raise AssertionError(f"inner fit and validation overlap: {set(fit_ids) & set(val_ids)}")
    model = make().fit(of(data, fit_ids), h)
    val = of(data, val_ids)
    choice = choose_tau(val.with_columns(p=model.predict(val)), shots, max_false)
    return choice, model


def run_cv(
    model: str,
    horizons: list[str],
    folds: dict,
    data: pl.DataFrame,
    shots: pl.DataFrame,
    max_false: float = MAX_FALSE_PER_MATCH,
    log=print,
    ball_source: str = "held",
    test_data: pl.DataFrame | None = None,
    data_config: dict | None = None,
    model_kwargs: dict | None = None,
    only_folds: list[int] | None = None,
    test_attrs: dict | None = None,
    provider=None,
) -> tuple[pl.DataFrame, dict]:
    """test_data (same rows as data, e.g. degraded) replaces data for the held-out
    predictions only; every fit and tau choice uses data. test_attrs are set on each
    outer model just before it predicts the held-out fold (the GNN's degraded graph
    store, with --degrade-arm test). model_kwargs go to the model but not into run.json
    (the GNN's graph store). only_folds runs just those outer folds and skips the final
    tau: a partial run, for timing.

    provider (learned possession, 05) maps an outer fold k, or None for the final tau, to
    that context's table: the same rows and order as data, with a possession_context
    column. Every fit, tau and prediction of that context reads it; data stays the
    canonical provider table for the run's row index and statistics."""
    cls, config = MODELS[model]
    config = {**config, "ball_source": ball_source, **(data_config or {})}
    if test_data is None:
        test_data = data
    elif not test_data.select(KEYS).equals(data.select(KEYS)):
        raise ValueError("test_data must have the same rows as data")

    def make():
        kw = {k: v for k, v in config.items() if k not in DATA_KEYS}
        if model in VERSIONED:
            kw["features"] = config["features"]
        return cls(**kw, **(model_kwargs or {}))

    ids = sorted({m["match_id"] for m in folds["matches"]})
    fold_of = {m["match_id"]: m["fold"] for m in folds["matches"]}
    p_cols = {h: np.full(data.height, np.nan) for h in horizons}
    tau, met, detail, timing = ({h: {} for h in horizons} for _ in range(4))
    fold_list = range(folds["n_folds"]) if only_folds is None else only_folds

    def outer_fold(h, fold, data, test_data, context=None):
        test = [i for i in ids if fold_of[i] == fold]
        train = [i for i in ids if fold_of[i] != fold]
        inner = inner_split(fold, folds)
        if set(train) & set(test) or not set(inner) <= set(train):
            raise AssertionError(f"fold {fold}: train/test/inner don't nest")
        fit_ids = [i for i in train if i not in inner]
        t0 = time.perf_counter()
        choice, _ = fit_and_choose(make, data, shots, fit_ids, inner, h, max_false)
        t1 = time.perf_counter()
        m = make().fit(of(data, train), h)
        t2 = time.perf_counter()
        rows = of(test_data, test)
        for name, value in (test_attrs or {}).items():
            setattr(m, name, value)
        p_cols[h][rows["row"].to_numpy()] = m.predict(rows).fill_null(np.nan).to_numpy()
        t3 = time.perf_counter()
        timing[h][str(fold)] = {
            "inner_fit_and_tau_s": round(t1 - t0, 1),
            "outer_fit_s": round(t2 - t1, 1),
            "predict_s": round(t3 - t2, 1),
        }
        tau[h][str(fold)] = choice["tau"]
        met[h][str(fold)] = choice["met"]
        detail[h][str(fold)] = {
            "coef": m.coef,
            "train_rows": m.n_train,
            **getattr(m, "fit_info", {}),
            "inner": {
                k: choice[k] for k in ("miss_rate", "lead_s_median", "false_alarms_per_match")
            },
        } | ({"context": context} if context else {})
        log(
            f"{h} fold {fold}: tau {choice['tau']:.4f} (met {choice['met']}, inner miss "
            f"{choice['miss_rate']:.3f}, false/match {choice['false_alarms_per_match']:.2f}) "
            f"coef {', '.join(f'{k} {v:+.3f}' for k, v in top(m.coef))} ({t3 - t0:.0f} s)"
        )

    def final_tau(h, data):
        t0 = time.perf_counter()
        inner = inner_split(None, folds)
        choice, _ = fit_and_choose(
            make, data, shots, [i for i in ids if i not in inner], inner, h, max_false
        )
        tau[h]["final"], met[h]["final"] = choice["tau"], choice["met"]
        timing[h]["final"] = {"inner_fit_and_tau_s": round(time.perf_counter() - t0, 1)}
        log(f"{h} final: tau {choice['tau']:.4f} (met {choice['met']})")

    if provider is None:  # one table for every fit: horizons outside folds
        for h in horizons:
            for fold in fold_list:
                outer_fold(h, fold, data, test_data)
            if only_folds is None:
                final_tau(h, data)
    else:  # learned possession (05): one assembly per context, both horizons on it
        contexts = [*fold_list] + ([None] if only_folds is None else [])
        assembly_s = {}
        for outer in contexts:
            t0 = time.perf_counter()
            ctx = provider(outer)
            assembly_s["final" if outer is None else str(outer)] = round(
                time.perf_counter() - t0, 1
            )
            check_context(ctx, data, outer, fold_of)
            for h in horizons:
                if outer is None:
                    final_tau(h, ctx)
                else:
                    outer_fold(h, outer, ctx, ctx, context_roles(outer))
        timing["assembly_s"] = assembly_s
    preds = data.select(KEYS).with_columns(
        **{f"p_{h}": pl.Series(p_cols[h]).fill_nan(None) for h in horizons}
    )
    scored = training_rows(test_data, horizons[0])
    meta = {
        "model": model,
        "horizons": horizons,
        "config": {
            **config,
            "max_false_per_match": max_false,
            # PFF's ESTIMATED ball uses later frames (05, Leakage): with ball_source=raw
            # this share of scored rows reads it; with held they get a held ball or none
            "ball_not_visible_share_scored": round(
                float((~scored["ball_visible"].fill_null(True)).mean()), 4
            ),
            "ball_missing_share_scored": round(
                float(scored["ball_dist"].fill_nan(None).is_null().mean()), 5
            ),
            "timing": timing,
        },
        "tau": tau,
        "tau_met": met,
        "folds": detail,
    }
    if only_folds is not None:
        meta["partial"] = {
            "folds": list(only_folds),
            "note": "timing run: these outer folds only and no final tau, not a CV result",
        }
    return preds, meta


def context_roles(outer: int) -> dict:
    return {"outer": outer, "train": f"outer{outer}_inner-oof<j>", "test": f"outer{outer}_test"}


def check_context(ctx: pl.DataFrame, data: pl.DataFrame, outer: int | None, fold_of: dict):
    """A context's table must have the canonical rows in order, and every match the
    possession of its role: test (own fold k) or inner-oof (training match) in outer k,
    the test output of its own fold in the final context."""
    if not ctx.select(KEYS).equals(data.select(KEYS)):
        raise ValueError(f"context {outer}: rows differ from the canonical table")
    got = ctx.select("match_id", "possession_context").unique()
    for m, c in got.iter_rows():
        j = fold_of[m]
        if outer is None or j == outer:
            want = f"outer{j}_test"
        else:
            want = f"outer{outer}_inner-oof{j}"
        if c != want:
            raise ValueError(f"context {outer}: match {m} has {c}, want {want}")


def top(coef: dict[str, float], n: int = 6) -> list[tuple[str, float]]:
    return sorted(coef.items(), key=lambda kv: -abs(kv[1]))[:n]


@contextlib.contextmanager
def keep_awake():
    """Windows sleeps when idle, even halfway through a long run (caffeinate is macOS
    only): ask it not to while this process runs (09). Nothing elsewhere."""
    if sys.platform != "win32":
        yield
        return
    import ctypes

    continuous, system_required = 0x80000000, 0x00000001
    try:
        ok = ctypes.windll.kernel32.SetThreadExecutionState(continuous | system_required)
    except (AttributeError, OSError):
        ok = 0
    if not ok:
        print("warning: couldn't keep Windows awake, set sleep to never (09)", flush=True)
    try:
        yield
    finally:
        with contextlib.suppress(AttributeError, OSError):
            ctypes.windll.kernel32.SetThreadExecutionState(continuous)


def log(msg: str) -> None:
    print(msg, flush=True)


def one_fold_estimate(meta: dict, fold: int, n_folds: int, setup_s: float) -> list[str]:
    """Full-run time from one fold: n_folds folds plus the final tau's fit, per horizon."""
    out = []
    for h in meta["horizons"]:
        t = meta["config"]["timing"][h][str(fold)]
        fold_s = sum(t.values())
        full = n_folds * fold_s + t["inner_fit_and_tau_s"]
        out.append(
            f"{h}: fold {fold} took {fold_s / 60:.1f} min (inner fit + tau "
            f"{t['inner_fit_and_tau_s'] / 60:.1f}, outer fit {t['outer_fit_s'] / 60:.1f}, "
            f"predict {t['predict_s'] / 60:.1f}). Full run: about {full / 60:.0f} min "
            f"({full / 3600:.1f} h) for {h}"
        )
    out.append(f"plus {setup_s / 60:.1f} min loading data and graphs (cache builds excluded)")
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="floor", choices=sorted(MODELS))
    ap.add_argument("--ball-source", default="held", choices=BALL_SOURCES)
    ap.add_argument("--horizons", nargs="+", default=["h5", "h3"])
    ap.add_argument("--run-id")
    ap.add_argument("--processed", type=Path, default=PROCESSED_DIR)
    ap.add_argument("--gamestate", type=Path, default=GAMESTATE_DIR)
    ap.add_argument("--folds", type=Path, default=FOLDS_PATH)
    ap.add_argument("--runs", type=Path, default=RUNS_DIR)
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--features-version", default=FEATURES_VERSION, choices=sorted(FEATURE_SETS))
    ap.add_argument(
        "--degrade",
        action="append",
        default=[],
        metavar="ARM:SEVERITY",
        help=f"repeatable; arms {', '.join(degrade.ARMS)}, or a preset "
        f"({', '.join(degrade.PRESETS)})",
    )
    ap.add_argument("--degrade-arm", choices=["both", "test"], default="both")
    ap.add_argument("--degrade-seed", type=int, default=degrade.DEFAULT_SEED)
    ap.add_argument("--device", default="auto", choices=gnn.DEVICES, help="gnn: torch device")
    ap.add_argument(
        "--gnn-globals",
        default="v1",
        choices=["v1", "none"],
        help="gnn: the hand features as a global vector, or none (graph only)",
    )
    ap.add_argument(
        "--gnn-layers", type=int, help="gnn: message-passing layers, 0 = globals only (05)"
    )
    ap.add_argument(
        "--gnn-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="gnn/tgnn: override one of gnn.PARAMS, repeatable (e.g. steps=4, batch_size=64)",
    )
    ap.add_argument(
        "--possession",
        default="provider",
        choices=possession.POSSESSION,
        help="where the model's inputs get possession from: PFF's, or stage 8's (07 #6)",
    )
    ap.add_argument(
        "--stale-possession",
        default="none",
        choices=possession.STALE,
        help="unknown: stage 8's possession is null once its carrier is --stale-after s old",
    )
    ap.add_argument("--stale-after", type=float, metavar="S", help="seconds, with unknown")
    ap.add_argument(
        "--possession-model",
        metavar="ID",
        help="learned possession (03 stage 8): a sealed model under data/models/possession",
    )
    ap.add_argument(
        "--one-fold",
        type=int,
        metavar="N",
        help="timing run: outer fold N only, no final tau, saved as a partial run",
    )
    args = ap.parse_args(argv)
    specs = degrade.canonical(args.degrade)
    if specs and args.ball_source != "held":
        ap.error("--degrade needs --ball-source held")
    try:
        args.gnn_overrides = gnn_overrides(args.gnn_param)
    except ValueError as e:
        ap.error(str(e))
    if args.possession == "inferred":
        if args.model in GNNS:
            ap.error("--possession inferred is lgbm/floor only: graphs use PFF's possession (05)")
        if specs:
            ap.error("--possession inferred can't be combined with --degrade (05)")
    if args.stale_possession == "unknown":
        if args.possession != "inferred":
            ap.error("--stale-possession unknown needs --possession inferred (05)")
        if args.stale_after is None or not args.stale_after > 0:
            ap.error("--stale-possession unknown needs a positive --stale-after")
    elif args.stale_after is not None:
        ap.error("--stale-after needs --stale-possession unknown")
    if args.features_version in STAGE8_VERSIONS:
        if args.model in GNNS:
            ap.error(f"--features-version {args.features_version} is lgbm/floor only (05)")
        if specs:
            ap.error(f"--features-version {args.features_version} can't be combined with --degrade")
    if args.model == "tgnn" and args.gnn_layers == 0:
        ap.error("--gnn-layers 0 has no graph, so no temporal model: use --model gnn")
    folds = load(args.folds)
    if args.one_fold is not None and not 0 <= args.one_fold < folds["n_folds"]:
        ap.error(f"--one-fold must be 0..{folds['n_folds'] - 1}")
    state_config = None
    if args.possession_model is not None:
        why = learned_refusal(args, specs)
        if why:
            ap.error(f"--possession-model: {why} (05, Learned possession)")
        state_config = possession.StateConfig(possession_model=args.possession_model)
    with keep_awake():
        return run(args, specs, folds, state_config)


def learned_refusal(args, specs) -> str | None:
    """Why a --possession-model run is refused, or None: v1 learned is one arm (05)."""
    if args.possession != "inferred":
        return "needs --possession inferred"
    if args.model != "lgbm":
        return "lgbm only"
    if args.features_version != "1":
        return "features v1 only"
    if args.ball_source != "held":
        return "held ball only"
    if specs:
        return "no --degrade"
    if args.stale_possession != "none" or args.stale_after is not None:
        return "no stale-possession arm"
    return None


def gnn_overrides(pairs: list[str]) -> dict:
    """KEY=VALUE strings into gnn.PARAMS overrides, typed like the defaults."""
    out = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or key not in gnn.PARAMS:
            raise ValueError(f"--gnn-param {pair!r}: expected KEY=VALUE, KEY one of gnn.PARAMS")
        kind = type(gnn.PARAMS[key])
        out[key] = value.lower() in ("1", "true", "yes") if kind is bool else kind(value)
    return out


def run(args, specs, folds: dict, state_config: possession.StateConfig | None = None) -> int:
    """state_config is stage 8's config for inferred possession and v3 (default rule when
    None); run.json records the one that ran."""
    started = time.perf_counter()
    state_config = state_config or possession.StateConfig()
    ids = sorted({m["match_id"] for m in folds["matches"]})
    stats, pstats = {}, {}
    version = args.features_version
    learned = state_config.possession_model is not None
    # learned: the canonical table is the provider's (labels, row index); each context's
    # possession comes through the provider below
    data, shots = load_data(
        ids,
        args.processed,
        args.gamestate,
        args.horizons,
        args.ball_source,
        specs,
        args.degrade_seed,
        stats,
        version,
        "provider" if learned else args.possession,
        pstats,
        args.stale_after,
        None if learned else state_config,
    )
    provider = None
    if learned:
        man_path, man = possession.learned_manifest(state_config)
        fold_of = {m["match_id"]: m["fold"] for m in folds["matches"]}
        if man["folds"]["assignments"] != fold_of:
            raise ValueError(f"{man_path}: its folds aren't this run's folds")

        def provider(outer):
            ctx = {m: possession.context_for(outer, fold_of[m]) for m in ids}
            return load_data(
                ids,
                args.processed,
                args.gamestate,
                args.horizons,
                args.ball_source,
                features_version=version,
                possession_source="inferred",
                possession_stats=pstats,
                state_config=state_config,
                contexts=ctx,
            )[0]

    test_data, data_config, model_kwargs, test_attrs = None, {}, {}, None
    if args.model in VERSIONED:
        data_config = {"features": list(FEATURE_SETS[version]), "features_version": version}
    if args.possession == "inferred" or version in STAGE8_VERSIONS:
        data_config["state_config"] = state_config.to_dict()
    if args.possession == "inferred":
        data_config |= {
            "possession": "inferred",
            "possession_scored": possession.summarize(pstats),
        }
        if args.stale_after is not None:
            data_config["stale_possession"] = {"arm": "unknown", "after_s": args.stale_after}
    if args.possession == "inferred" and not learned:  # learned: counted as contexts load
        log(
            f"inferred possession: {data_config['possession_scored']}, stage 8 "
            f"{pstats.get('stage8_s', 0):.0f} s, features {pstats.get('features_s', 0):.0f} s "
            "(0 when cached)"
        )
    if specs:
        data_config |= {
            "degrade": specs,
            "degrade_arm": args.degrade_arm,
            "degrade_seed": args.degrade_seed,
            "degrade_realized": degrade.summarize(stats),
        }
        print(f"degraded {specs} ({args.degrade_arm}): {data_config['degrade_realized']}")
        if args.degrade_arm == "test":
            test_data = data
            data, _ = load_data(
                ids,
                args.processed,
                args.gamestate,
                args.horizons,
                args.ball_source,
                features_version=version,
            )
    load_s = time.perf_counter() - started
    graphs_s = build_s = 0.0
    if args.model in GNNS:
        t0 = time.perf_counter()
        # degraded graphs come from the same degraded objects as the features (05)
        fit_specs = specs if args.degrade_arm == "both" else []
        store = graphs.GraphStore.load(
            ids, args.processed, args.ball_source, log, fit_specs, args.degrade_seed
        )
        graphs_s = time.perf_counter() - t0
        build_s = graphs_s if store.built else 0.0
        log(
            f"graphs: {len(ids)} matches, {store.built} caches built, "
            f"{store.nodes.nbytes / 2**30:.2f} GB, {graphs_s:.0f} s"
        )
        model_kwargs = {"graphs": store, "log": log}
        if specs and args.degrade_arm == "test":
            t0 = time.perf_counter()
            test_attrs = {
                "graphs": graphs.GraphStore.load(
                    ids, args.processed, args.ball_source, log, specs, args.degrade_seed
                )
            }
            graphs_s += time.perf_counter() - t0
            log(f"degraded graphs for the held-out folds: {time.perf_counter() - t0:.0f} s")
        if args.gnn_globals == "none":
            data_config["features"] = []
        data_config |= {"globals": args.gnn_globals, "device": args.device}
        params = {**MODELS[args.model][1]["params"], **args.gnn_overrides}
        if args.gnn_layers is not None:
            params["layers"] = args.gnn_layers
        data_config["params"] = params
    log(f"{len(ids)} matches, {data.height:,} rows, {shots.height} shots")
    only = None if args.one_fold is None else [args.one_fold]
    t0 = time.perf_counter()
    preds, meta = run_cv(
        args.model,
        args.horizons,
        folds,
        data,
        shots,
        log=log,
        ball_source=args.ball_source,
        test_data=test_data,
        data_config=data_config,
        model_kwargs=model_kwargs,
        only_folds=only,
        test_attrs=test_attrs,
        provider=provider,
    )
    if learned:
        meta["config"]["possession_scored"] = possession.summarize(pstats)
        meta["config"]["possession_model"] = possession.learned_meta(
            state_config, man_path, man, ids, fold_of, args.processed, pstats
        )
        log(f"learned possession, outer-test rows: {meta['config']['possession_scored']}")
    meta["config"]["timing"] |= {
        **{k: round(pstats[k], 1) for k in ("stage8_s", "features_s") if k in pstats},
        "load_s": round(load_s, 1),
        "graphs_s": round(graphs_s, 1),
        "cv_s": round(time.perf_counter() - t0, 1),
    }
    first = next(iter(next(iter(meta["folds"].values())).values()))
    if "device" in first:
        meta["config"]["device_used"] = f"{first['device']} ({first['device_name']})"
    today = dt.datetime.now().astimezone().date().isoformat()
    run_id = args.run_id or f"{args.model}-{today}"
    if only is not None:
        fold_ids = [m["match_id"] for m in folds["matches"] if m["fold"] == args.one_fold]
        preds = preds.filter(pl.col("match_id").is_in(fold_ids))
        run_id = args.run_id or f"{args.model}-{today}-fold{args.one_fold}-partial"
    run_dir = save_run(args.runs / run_id, preds, meta)
    log(f"saved {run_dir}")
    if not args.no_report:
        text = render(
            run_dir, args.gamestate, args.processed, args.folds, allow_partial=only is not None
        )
        # the report has τ, ≥ and Δ in it: Windows would write cp1252 and fail
        (run_dir / "report.md").write_text(text, encoding="utf-8")
        log(f"wrote {run_dir / 'report.md'}")
    if only is not None:
        setup_s = load_s + graphs_s - build_s
        for line in one_fold_estimate(meta, args.one_fold, folds["n_folds"], setup_s):
            log(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
