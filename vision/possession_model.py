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
from vision.state import StateConfig, infer

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


N_FOLDS = 5
MANIFEST = "manifest.json"


def plan(folds: Mapping[str, int], dedup: bool = True) -> dict[str, dict]:
    """The 25 logical CV fits over fold assignments {match_id: fold}: outer_k trains on
    U - F_k and predicts F_k (role test); outer_k_inner_j trains on U - F_k - F_j and
    predicts F_j (role inner-oof). With dedup, both inner IDs of a fold pair share one
    training, pair_<a>_<b> (a < b); without it each inner ID has its own."""
    if sorted(set(folds.values())) != list(range(N_FOLDS)):
        raise ValueError(f"folds must be 0..{N_FOLDS - 1}, got {sorted(set(folds.values()))}")
    fold_ids = {j: sorted(m for m, f in folds.items() if f == j) for j in range(N_FOLDS)}
    out = {}
    for k in range(N_FOLDS):
        out[f"outer_{k}"] = {
            "training": f"outer_{k}",
            "role": "test",
            "outer": k,
            "fold": k,
            "training_ids": sorted(m for m, f in folds.items() if f != k),
            "predicts": fold_ids[k],
        }
        for j in range(N_FOLDS):
            if j == k:
                continue
            lid = f"outer_{k}_inner_{j}"
            out[lid] = {
                "training": f"pair_{min(j, k)}_{max(j, k)}" if dedup else lid,
                "role": "inner-oof",
                "outer": k,
                "fold": j,
                "training_ids": sorted(m for m, f in folds.items() if f not in (j, k)),
                "predicts": fold_ids[j],
            }
    return out


def trainings(fits: Mapping[str, dict]) -> dict[str, list[str]]:
    """Each stored training of a plan with its sorted training matches."""
    return {f["training"]: f["training_ids"] for f in fits.values()}


def check_manifest(man: dict) -> None:
    """Fail closed unless the manifest is sealed and every logical fit resolves to a
    training on exactly its allowed matches, excluding its outer fold and its own."""
    if man.get("sealed") is not True:
        raise ValueError("manifest: not sealed (an incomplete artifact can't serve a CV run)")
    folds = {m: int(f) for m, f in man["folds"]["assignments"].items()}
    want = plan(folds, man["dedup_pairs"])
    if set(man["logical"]) != set(want):
        raise ValueError("manifest: logical fit IDs differ from the 25 of the nested scheme")
    stored = man["trainings"]
    if set(stored) != set(trainings(want)):
        raise ValueError(f"manifest: trainings {sorted(stored)} differ from the plan")
    for lid, w in want.items():
        e = man["logical"][lid]
        for key in ("training", "role", "outer", "fold", "predicts"):
            if e.get(key) != w[key]:
                raise ValueError(f"manifest: {lid} {key} is {e.get(key)!r}, want {w[key]!r}")
        tr = stored[w["training"]]
        ids = tr["training_ids"]
        if ids != w["training_ids"] or tr["training_sha256"] != ids_sha256(ids):
            raise ValueError(f"manifest: {lid} training set differs from the plan")
        if e.get("training_sha256") != tr["training_sha256"]:
            raise ValueError(f"manifest: {lid} training hash differs from its training's")
        outer = {m for m, f in folds.items() if f == w["outer"]}
        if set(ids) & (outer | set(w["predicts"])):
            raise ValueError(f"manifest: {lid} trains on its outer or predicted fold")


def resolve(model_dir: Path, logical_id: str, match_id: str) -> tuple[dict, dict, Path]:
    """(manifest, logical entry, model.txt path) for predicting match_id under
    logical_id. Refuses a match outside the fit's prediction set, and a model or fit
    record whose hashes or match lists differ from the manifest's."""
    man = json.loads((model_dir / MANIFEST).read_text())
    check_manifest(man)
    if logical_id not in man["logical"]:
        raise ValueError(f"manifest: no logical fit {logical_id!r}")
    entry = man["logical"][logical_id]
    if match_id not in entry["predicts"]:
        raise ValueError(f"{logical_id} doesn't predict match {match_id}")
    tr = man["trainings"][entry["training"]]
    path = model_dir / tr["dir"] / "model.txt"
    rec = json.loads((path.parent / "fit.json").read_text())
    if pf.sha256_file(path) != tr["model_sha256"] or rec["model_sha256"] != tr["model_sha256"]:
        raise ValueError(f"{path}: model hash differs from the manifest")
    ids = tr["training_ids"]
    if rec["training_ids"] != ids or sorted(rec["probe_ids"] + rec["es_ids"]) != ids:
        raise ValueError(f"{path.parent / 'fit.json'}: match lists differ from the manifest")
    if match_id in ids:
        raise ValueError(f"{logical_id} trained on match {match_id}")
    return man, entry, path


GRID_INTERNAL = ("t_us", "u_us", "fi", "seg", "d")


def rule_grid(
    frames: pl.DataFrame, objects: pl.DataFrame | pl.LazyFrame, native_fps: float
) -> pl.DataFrame:
    """The 2D rule's outputs per grid row (period, k, frame_id): rule_possession_team,
    ball_carrier_id, ball_state and carrier_age_s (Float64 seconds since the period's
    last native frame with a carrier, null before the first). Stage 8's default rule on
    usable objects with every z null, as pf.rule_frames runs it."""
    nat = pf.native(frames, native_fps)
    flat = pf.usable(objects).with_columns(z=pl.lit(None, pl.Float64), visible=pl.lit(True))
    state = infer(frames.select("frame_id", "period", "timestamp_s"), flat, StateConfig())
    per = (
        nat.select("frame_id", "fi", "period", "ts_us")
        .join(state, on="frame_id")
        .sort("fi")
        .with_columns(
            carrier_age_s=(
                pl.col("ts_us")
                - pl.when(pl.col("ball_carrier_id").is_not_null())
                .then("ts_us")
                .forward_fill()
                .over("period")
            )
            / pf.US
        )
        .select(
            "fi",
            rule_possession_team="possession_team",
            ball_carrier_id="ball_carrier_id",
            ball_state="ball_state",
            carrier_age_s="carrier_age_s",
        )
    )
    return (
        pf.grid(nat, native_fps)
        .join(per, on="fi", how="left", maintain_order="left")
        .drop(GRID_INTERNAL)
    )


STATE_COLUMNS = [
    "period",
    "k",
    "t_s",
    "frame_id",
    "possession_team",
    "carrier_age_s",
    "rule_possession_team",
    "ball_carrier_id",
    "ball_state",
    "p_home",
    "fallback",
    "fit_id",
]
FRAME_TIME_COLUMNS = ["frame_id", "period", "timestamp_s", "home_attacks_positive_x"]


def predict_match(
    model_dir: Path,
    logical_fit_id: str,
    match_id: str,
    gamestate_dir: Path = pf.GAMESTATE_DIR,
    cache_dir: Path = pf.CACHE_DIR,
) -> pl.DataFrame:
    """One match's learned state on every valid grid row under one logical fit (03
    "Prediction plumbing"): possession home if p_home >= 0.5 else away; with no usable
    player on the native frame, the 2D rule's possession (null included), p_home null
    and fallback true. Carrier, ball state and carrier age are always the 2D rule's.
    Reads only the hash-checked inputs and the native frames' time columns (never PFF
    possession or ball state); writes nothing."""
    import lightgbm as lgb

    man, _, path = resolve(model_dir, logical_fit_id, match_id)
    feats = pf.load(match_id, gamestate_dir, cache_dir)
    _, side = pf.cache_paths(match_id, cache_dir)
    meta = json.loads(side.read_text())
    recorded = man["inputs"].get(match_id, {})
    for key in ("frames_sha256", "objects_sha256", "grid_sha256"):
        if recorded.get(key) != meta[key]:
            raise ValueError(f"{match_id}: {key} differs from the model's manifest")
    d = gamestate_dir / match_id
    fps = pl.read_parquet(d / "match.parquet")["native_fps"][0]
    frames = pl.read_parquet(d / "frames.parquet", columns=FRAME_TIME_COLUMNS)
    rule = rule_grid(frames, pl.scan_parquet(d / "objects.parquet"), fps)
    keys = ["period", "k", "frame_id"]
    if not rule.select(keys).equals(feats.select(keys)):
        raise ValueError(f"{match_id}: rule grid differs from the cached input grid")
    p = p_home(lgb.Booster(model_file=str(path)), feats)
    fallback = (feats["players_n"] == 0).to_numpy()
    learned = pl.Series(np.where(p >= 0.5, "home", "away"))
    return rule.with_columns(
        t_s=feats["t_s"],
        possession_team=pl.when(pl.Series(fallback))
        .then(pl.col("rule_possession_team"))
        .otherwise(learned),
        p_home=pl.Series(np.where(fallback, np.nan, p)).fill_nan(None),
        fallback=pl.Series(fallback),
        fit_id=pl.lit(logical_fit_id),
    ).select(STATE_COLUMNS)
