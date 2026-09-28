"""Out-of-fold predictions with 07's tau rule, written as a run (07, "Run format").

    python -m prediction.cv [--model floor|lgbm] [--ball-source held|raw] [--run-id ID] [--no-report]
        [--degrade ARM:SEVERITY ...] [--degrade-arm both|test] [--degrade-seed N]

Per horizon and outer fold: fit on the training matches minus the inner split, pick
tau on the inner matches, refit on all training matches, predict the held-out fold.
The "final" tau does the same over all CV matches with inner_split(None).

--ball-source raw feeds PFF's ESTIMATED ball, which uses later frames (05, Leakage). It's
only there to measure how much that leak inflates a result; held is the real number.

--degrade runs the vision sensitivity test (05): objects are degraded the way vision
fails before features are built. --degrade-arm both (default) degrades training and test
matches; test fits and picks tau on clean data and predicts degraded held-out folds.
"""

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl

from evaluation.folds import FOLDS_PATH, GAMESTATE_DIR, inner_split, load
from evaluation.metrics import MAX_FALSE_PER_MATCH, choose_tau, shots_table
from evaluation.report import PROCESSED_DIR, render
from evaluation.runs import RUNS_DIR, save_run
from prediction import degrade, floor, lgbm
from prediction.features import BALL_SOURCES, FEATURES, FEATURES_VERSION, load_match
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
}
# config keys about the data, not the model's constructor
DATA_KEYS = (
    "features",
    "features_version",
    "ball_source",
    "degrade",
    "degrade_arm",
    "degrade_seed",
    "degrade_realized",
)
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
):
    data = pl.concat(
        [
            load_match(
                i,
                processed_dir,
                horizons,
                ball_source,
                degrade=degrade_specs,
                degrade_seed=degrade_seed,
                degrade_stats=degrade_stats,
            )
            for i in ids
        ]
    ).with_row_index("row")
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
) -> tuple[pl.DataFrame, dict]:
    """test_data (same rows as data, e.g. degraded) replaces data for the held-out
    predictions only; every fit and tau choice uses data."""
    cls, config = MODELS[model]
    config = {**config, "ball_source": ball_source, **(data_config or {})}
    if test_data is None:
        test_data = data
    elif not test_data.select(KEYS).equals(data.select(KEYS)):
        raise ValueError("test_data must have the same rows as data")

    def make():
        return cls(**{k: v for k, v in config.items() if k not in DATA_KEYS})

    ids = sorted({m["match_id"] for m in folds["matches"]})
    fold_of = {m["match_id"]: m["fold"] for m in folds["matches"]}
    p_cols = {h: np.full(data.height, np.nan) for h in horizons}
    tau, met, detail = {}, {}, {}
    for h in horizons:
        tau[h], met[h], detail[h] = {}, {}, {}
        for fold in range(folds["n_folds"]):
            test = [i for i in ids if fold_of[i] == fold]
            train = [i for i in ids if fold_of[i] != fold]
            inner = inner_split(fold, folds)
            if set(train) & set(test) or not set(inner) <= set(train):
                raise AssertionError(f"fold {fold}: train/test/inner don't nest")
            fit_ids = [i for i in train if i not in inner]
            choice, _ = fit_and_choose(make, data, shots, fit_ids, inner, h, max_false)
            m = make().fit(of(data, train), h)
            rows = of(test_data, test)
            p_cols[h][rows["row"].to_numpy()] = m.predict(rows).fill_null(np.nan).to_numpy()
            tau[h][str(fold)] = choice["tau"]
            met[h][str(fold)] = choice["met"]
            detail[h][str(fold)] = {
                "coef": m.coef,
                "train_rows": m.n_train,
                **getattr(m, "fit_info", {}),
                "inner": {
                    k: choice[k] for k in ("miss_rate", "lead_s_median", "false_alarms_per_match")
                },
            }
            log(
                f"{h} fold {fold}: tau {choice['tau']:.4f} (met {choice['met']}, inner miss "
                f"{choice['miss_rate']:.3f}, false/match {choice['false_alarms_per_match']:.2f}) "
                f"coef {', '.join(f'{k} {v:+.3f}' for k, v in top(m.coef))}"
            )
        inner = inner_split(None, folds)
        choice, m = fit_and_choose(
            make, data, shots, [i for i in ids if i not in inner], inner, h, max_false
        )
        tau[h]["final"], met[h]["final"] = choice["tau"], choice["met"]
        log(f"{h} final: tau {choice['tau']:.4f} (met {choice['met']})")
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
        },
        "tau": tau,
        "tau_met": met,
        "folds": detail,
    }
    return preds, meta


def top(coef: dict[str, float], n: int = 6) -> list[tuple[str, float]]:
    return sorted(coef.items(), key=lambda kv: -abs(kv[1]))[:n]


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
    args = ap.parse_args(argv)
    specs = degrade.canonical(args.degrade)
    if specs and args.ball_source != "held":
        ap.error("--degrade needs --ball-source held")
    folds = load(args.folds)
    ids = sorted({m["match_id"] for m in folds["matches"]})
    stats = {}
    data, shots = load_data(
        ids,
        args.processed,
        args.gamestate,
        args.horizons,
        args.ball_source,
        specs,
        args.degrade_seed,
        stats,
    )
    test_data, data_config = None, {}
    if specs:
        data_config = {
            "degrade": specs,
            "degrade_arm": args.degrade_arm,
            "degrade_seed": args.degrade_seed,
            "degrade_realized": degrade.summarize(stats),
        }
        print(f"degraded {specs} ({args.degrade_arm}): {data_config['degrade_realized']}")
        if args.degrade_arm == "test":
            test_data = data
            data, _ = load_data(
                ids, args.processed, args.gamestate, args.horizons, args.ball_source
            )
    print(f"{len(ids)} matches, {data.height:,} rows, {shots.height} shots", flush=True)
    preds, meta = run_cv(
        args.model,
        args.horizons,
        folds,
        data,
        shots,
        ball_source=args.ball_source,
        test_data=test_data,
        data_config=data_config,
    )
    run_id = args.run_id or f"{args.model}-{dt.datetime.now().astimezone().date().isoformat()}"
    run_dir = save_run(args.runs / run_id, preds, meta)
    print(f"saved {run_dir}")
    if not args.no_report:
        (run_dir / "report.md").write_text(
            render(run_dir, args.gamestate, args.processed, args.folds)
        )
        print(f"wrote {run_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
