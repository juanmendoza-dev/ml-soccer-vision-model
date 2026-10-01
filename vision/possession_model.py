"""Learned possession (03 stage 8): one LightGBM classifier, q = P(home has the ball), per
10 Hz grid row from the pfeat-v1 inputs (vision.possession_features).

Training rows are grid rows where PFF's ball is alive, PFF names home or away, and some
player/keeper is VISIBLE (not all_estimated). Every row is trained twice, as itself and
under the mirror M with the label flipped. The emitted probability is symmetrized,
p = (q(x) + 1 - q(Mx)) / 2, so p(Mx) = 1 - p(x). A fit is an early-stopping probe on
whole held-out matches, then a refit on all of the training matches for the probe's
best round count; it depends only on the sorted training-match set.

Vision never imports prediction: the LightGBM parameters are copied from prediction.lgbm.
"""

import hashlib
import json
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import polars as pl

from vision import possession_features as pf

SEED = 20260927
NUM_THREADS = 6  # the workstation's physical cores, fixed so `deterministic` reproduces
PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 500,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 255,
    "metric": "binary_logloss",
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
    "seed": SEED,
    "num_threads": NUM_THREADS,
}
MAX_ROUNDS = 2000
PATIENCE = 100
ES_SHARE = 0.15
LABEL = {"home": 1.0, "away": 0.0}
LAG_SFX = ("", *(f"_{name}" for name, _ in pf.LAGS))


def canonical(x: pl.DataFrame) -> pl.DataFrame:
    """The model's view of the cached features: every diff_* recomputed as Float32 home
    minus away, exactly as pf.mirror recomputes it. The cache subtracts in Float64 and
    rounds once, which can differ by an ulp, and then M(M(x)) != x bitwise."""
    return x.with_columns(
        (pl.col(f"home_{s}{sfx}") - pl.col(f"away_{s}{sfx}")).alias(f"diff_{s}{sfx}")
        for sfx in LAG_SFX
        for s in pf.SHAPES
    )


def labels(
    frames: pl.DataFrame, objects: pl.DataFrame | pl.LazyFrame, native_fps: float
) -> pl.DataFrame:
    """Per grid row (period, k, frame_id): all_estimated (no player/keeper on the native
    frame has visible true, whatever its team or coordinates, as the resampler defines
    it), label (PFF home 1, away 0, else null) and train (alive, labelled, not
    all_estimated). PFF's ball_state and possession only ever select and label rows."""
    g = pf.grid(pf.native(frames, native_fps), native_fps)
    vis = (
        objects.lazy()
        .filter(pl.col("object_type").is_in(pf.PLAYER_TYPES))
        .group_by("frame_id")
        .agg(any_visible=pl.col("visible").any())
        .collect()
    )
    return (
        g.select("period", "k", "frame_id")
        .join(
            frames.select("frame_id", "ball_state", "possession_team"),
            on="frame_id",
            how="left",
            maintain_order="left",
        )
        .join(vis, on="frame_id", how="left", maintain_order="left")
        .with_columns(
            all_estimated=~pl.col("any_visible").fill_null(False),
            label=pl.col("possession_team").replace_strict(
                LABEL, default=None, return_dtype=pl.Float64
            ),
        )
        .with_columns(
            train=(pl.col("ball_state") == "alive").fill_null(False)
            & pl.col("label").is_not_null()
            & ~pl.col("all_estimated")
        )
        .select("period", "k", "frame_id", "all_estimated", "label", "train")
    )


def match_rows(
    match_id: str, gamestate_dir: Path = pf.GAMESTATE_DIR, cache_dir: Path = pf.CACHE_DIR
) -> pl.DataFrame:
    """One match's cached inputs (hash-checked by pf.load) with all_estimated, label and
    train attached on the same grid keys."""
    x = pf.load(match_id, gamestate_dir, cache_dir)
    d = gamestate_dir / match_id
    fps = pl.read_parquet(d / "match.parquet")["native_fps"][0]
    lab = labels(pl.read_parquet(d / "frames.parquet"), pl.scan_parquet(d / "objects.parquet"), fps)
    keys = ["period", "k", "frame_id"]
    if not lab.select(keys).equals(x.select(keys)):
        raise ValueError(f"{match_id}: label grid differs from the cached input grid")
    return x.hstack(lab.select("all_estimated", "label", "train"))


def es_split(train_ids) -> list[str]:
    """The early-stopping matches for a training set: round(0.15 * n) of its sorted IDs,
    drawn with default_rng(SEED).choice, as prediction.lgbm draws them."""
    ids = np.array(sorted(train_ids))
    n = round(ES_SHARE * len(ids))
    if n < 1 or n >= len(ids):
        raise ValueError(f"{len(ids)} training matches can't be split for early stopping")
    return sorted(np.random.default_rng(SEED).choice(ids, n, replace=False).tolist())


def design(tables: list[pl.DataFrame], stride: int = 1) -> tuple[np.ndarray, np.ndarray, int]:
    """Training rows (train true, k % stride == 0) in the given table order, then the same
    rows mirrored with flipped labels. Returns X (Float32, pf.COLUMNS only), y and the
    number of original rows."""
    rows = pl.concat([t.filter(pl.col("train"), pl.col("k") % stride == 0) for t in tables])
    y = rows["label"].to_numpy()
    f = canonical(rows.select(pf.COLUMNS))
    x = np.vstack([f.to_numpy(), pf.mirror(f).to_numpy()]).astype(np.float32, copy=False)
    return x, np.r_[y, 1.0 - y], rows.height


def train_booster(x, y, rounds: int, params: dict, valid=None):
    import lightgbm as lgb

    ds = lgb.Dataset(x, y, feature_name=list(pf.COLUMNS))
    kw = {}
    if valid is not None:
        kw["valid_sets"] = [lgb.Dataset(*valid, reference=ds)]
        kw["callbacks"] = [lgb.early_stopping(PATIENCE, verbose=False)]
    return lgb.train(params, ds, num_boost_round=rounds, **kw)


def ids_sha256(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def peak_rss() -> int:
    """The process's peak resident memory so far, bytes."""
    try:
        import psutil

        mi = psutil.Process().memory_info()
        if hasattr(mi, "peak_wset"):
            return int(mi.peak_wset)
    except ImportError:
        pass
    import resource

    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def versions() -> dict:
    import lightgbm

    return {
        "python": sys.version.split()[0],
        "lightgbm": lightgbm.__version__,
        "numpy": np.__version__,
        "polars": pl.__version__,
    }


def fit(
    tables: Mapping[str, pl.DataFrame],
    train_ids,
    out_dir: Path,
    stride: int = 1,
    params: dict | None = None,
) -> dict:
    """Probe + refit on one training-match set; writes out_dir/model.txt and fit.json and
    returns the fit record. The probe trains on the non-ES matches (strided) and stops on
    the ES matches at full 10 Hz; the refit trains on all of them (strided) for
    max(1, best) rounds. Mirroring happens after the split, so a match's two copies
    stay on one side. `params` overrides are for tests only."""
    params = {**PARAMS, **(params or {})}
    ids = sorted(train_ids)
    es = es_split(ids)
    probe_ids = [m for m in ids if m not in es]
    t0 = time.perf_counter()
    xp, yp, np_rows = design([tables[m] for m in probe_ids], stride)
    xv, yv, nv_rows = design([tables[m] for m in es], 1)
    if not np_rows or not nv_rows:
        raise ValueError(f"no eligible rows: probe {np_rows}, early stopping {nv_rows}")
    t1 = time.perf_counter()
    probe = train_booster(xp, yp, MAX_ROUNDS, params, valid=(xv, yv))
    best = max(1, probe.best_iteration)
    del xp, yp, xv, yv, probe
    t2 = time.perf_counter()
    xr, yr, nr_rows = design([tables[m] for m in ids], stride)
    t3 = time.perf_counter()
    booster = train_booster(xr, yr, best, params)
    t4 = time.perf_counter()
    del xr, yr
    out_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(out_dir / "model.txt")
    eligible = {m: int(tables[m]["train"].sum()) for m in ids}
    record = {
        "contract": pf.CONTRACT,
        "columns_sha256": ids_sha256(pf.COLUMNS),
        "training_ids": ids,
        "training_sha256": ids_sha256(ids),
        "probe_ids": probe_ids,
        "es_ids": es,
        "refit_ids": ids,
        "stride": stride,
        "rows": {
            "eligible": sum(eligible.values()),
            "eligible_per_match": eligible,
            "probe": {"after_stride": np_rows, "mirrored": 2 * np_rows},
            "es": {"rows": nv_rows, "mirrored": 2 * nv_rows},
            "refit": {"after_stride": nr_rows, "mirrored": 2 * nr_rows},
        },
        "best_iteration": best,
        "max_rounds": MAX_ROUNDS,
        "patience": PATIENCE,
        "es_share": ES_SHARE,
        "params": params,
        "timing_s": {
            "probe_design": t1 - t0,
            "probe_train": t2 - t1,
            "refit_design": t3 - t2,
            "refit_train": t4 - t3,
        },
        "peak_rss_bytes": peak_rss(),
        "model_sha256": pf.sha256_file(out_dir / "model.txt"),
        "versions": versions(),
        "git_commit": git_commit(),
    }
    (out_dir / "fit.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def p_home(booster, feats: pl.DataFrame) -> np.ndarray:
    """Symmetrized P(home has the ball), Float64: (q(x) + 1 - q(Mx)) / 2 from one booster."""
    f = canonical(feats.select(pf.COLUMNS))
    q = booster.predict(f.to_numpy(), num_threads=NUM_THREADS)
    qm = booster.predict(pf.mirror(f).to_numpy(), num_threads=NUM_THREADS)
    return (q + 1.0 - qm) / 2.0
