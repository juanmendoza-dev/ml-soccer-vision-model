"""Stage 8's possession as the model's input (05, "Inferred possession"; 07 #6).

The one place prediction calls into vision/: vision.state.infer, which 03 writes to run on
dataset tracking as well as live. Stage 8 runs on the native game state (02), and its
possession_team is joined onto the 10 Hz grid by frame_id. Each grid row already uses the
latest native frame at or before t, and infer is causal, so a row's inferred possession
uses frames <= t only.

Only the model's inputs take it. Labels, eligible, all_estimated and the table's own
possession_team / flipped / ball_state stay the provider's (prediction.features.load_match).

Carrier age (05, "Stale possession"): seconds since stage 8 last confirmed a carrier in
the period, from its ball_carrier_id on native frames <= t. It's the v3 feature
poss_carrier_age_s, and arm U turns possession unknown where it's over stale_after.

Learned possession (03 stage 8, "Prediction plumbing"): with StateConfig.possession_model
set, a match's possession comes from vision.possession_model.predict_match under one
nested CV context, joined on the full grid key (period, tick) with the native frame_id
checked. Each context has its own cache, and every read rechecks its hashes.
"""

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from prediction.resample import flipped_expr
from vision import possession_features as vpf
from vision import possession_model as vpm
from vision.state import StateConfig, infer, rule_only

POSSESSION = ("provider", "inferred")
STALE = ("none", "unknown")  # arm U: "unknown" nulls stage 8's possession once it's stale
# what infer reads from 02's objects
OBJECT_COLS = ["frame_id", "object_id", "object_type", "team", "x", "y", "z", "visible"]


def config_key(config: StateConfig) -> str:
    """Cache key for a StateConfig: any threshold change gets new caches."""
    blob = json.dumps(config.to_dict(), sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def carrier_age(frames: pl.DataFrame, state: pl.DataFrame) -> pl.Series:
    """Seconds since the last native frame of the same period with a confirmed carrier
    (0 on one, null before the period's first). frames (frame_id, period, timestamp_s) and
    state (infer's output) are joined by frame_id; the result is in frames' row order."""
    return (
        frames.select("frame_id", "period", "timestamp_s")
        .with_row_index("i")
        .join(state.select("frame_id", "ball_carrier_id"), on="frame_id", how="left")
        .sort("period", "timestamp_s", "frame_id")
        .with_columns(
            carrier_age_s=pl.col("timestamp_s")
            - pl.when(pl.col("ball_carrier_id").is_not_null())
            .then(pl.col("timestamp_s"))
            .forward_fill()
            .over("period")
        )
        .sort("i")["carrier_age_s"]
    )


def inferred_state(
    match_id: str,
    gamestate_dir: Path,
    processed_dir: Path,
    config: StateConfig | None = None,
    cache: bool = True,
    timing: dict | None = None,
) -> pl.DataFrame:
    """frame_id, stage 8's possession_team and carrier_age_s on every native frame of the
    match, cached next to the resampled tables. Seconds spent in infer are added to
    timing["stage8_s"]. The cache name changed when carrier_age_s joined it, so the old
    possession-only state_inferred_<key> files are never read."""
    config = config or StateConfig()
    rule_only(config)
    path = processed_dir / match_id / f"state_inferred_age_{config_key(config)}.parquet"
    if cache and path.exists():
        return pl.read_parquet(path)
    src = gamestate_dir / match_id
    t0 = time.perf_counter()
    frames = pl.read_parquet(src / "frames.parquet", columns=["frame_id", "period", "timestamp_s"])
    objects = pl.scan_parquet(src / "objects.parquet").select(OBJECT_COLS)
    state = infer(frames, objects, config)
    out = state.select("frame_id", "possession_team").with_columns(
        carrier_age_s=carrier_age(frames, state)
    )
    if timing is not None:
        timing["stage8_s"] = timing.get("stage8_s", 0.0) + time.perf_counter() - t0
    if cache:
        out.write_parquet(path)
    return out


def swap_possession(
    frames: pl.DataFrame, state: pl.DataFrame, stale_after: float | None = None
) -> pl.DataFrame:
    """Grid rows (with frame_id and home_attacks_positive_x) with possession_team replaced
    by state's at the row's native frame and flipped recomputed from it, the resampler's
    way. With stale_after (arm U), possession is null where carrier_age_s is over it.
    state's other columns (carrier_age_s) come along. Row order is kept."""
    missing = frames.join(state, on="frame_id", how="anti")
    if missing.height:
        raise ValueError(f"{missing.height} grid rows point at frames stage 8 didn't label")
    out = frames.drop("possession_team", "flipped").join(
        state, on="frame_id", how="left", validate="m:1", maintain_order="left"
    )
    if stale_after is not None:
        stale = pl.col("carrier_age_s") > stale_after
        out = out.with_columns(
            pl.when(stale.fill_null(False))
            .then(None)
            .otherwise(pl.col("possession_team"))
            .alias("possession_team")
        )
    return out.with_columns(flipped=flipped_expr())


def stale_rows(state_rows: pl.DataFrame, stale_after: float) -> pl.Series:
    """Rows arm U makes unknown: a team from stage 8, but its carrier is over stale_after
    seconds old."""
    return state_rows.select((pl.col("carrier_age_s") > stale_after).fill_null(False)).to_series()


def add_stats(
    stats: dict,
    provider: pl.DataFrame,
    inferred: pl.DataFrame,
    scored: pl.Series,
    unknown: pl.Series | None = None,
):
    """Counts over the scored rows, for run.json: inferred possession differs from the
    provider's, flipped differs, inferred is null, and (arm U) rows made unknown."""
    p, i = provider.filter(scored), inferred.filter(scored)
    counts = {
        "scored_rows": p.height,
        "possession_differs": int(p["possession_team"].ne_missing(i["possession_team"]).sum()),
        "flipped_differs": int(p["flipped"].ne_missing(i["flipped"]).sum()),
        "inferred_null": int(i["possession_team"].is_null().sum()),
    }
    if unknown is not None:
        counts["stale_unknown"] = int(unknown.filter(scored).sum())
    for k, v in counts.items():
        stats[k] = stats.get(k, 0) + v


def summarize(stats: dict) -> dict:
    n = stats.get("scored_rows", 0)
    share = lambda k: round(stats[k] / n, 4) if n else None
    return {
        "scored_rows": n,
        "possession_differs_share": share("possession_differs"),
        "flipped_differs_share": share("flipped_differs"),
        "inferred_null_share": share("inferred_null"),
    } | ({"stale_unknown_share": share("stale_unknown")} if "stale_unknown" in stats else {})


# --- learned possession (03 stage 8, "Prediction plumbing") ---

MODELS_DIR = Path("data/models/possession")
VISION_CACHE_DIR = vpf.CACHE_DIR


@dataclass(frozen=True)
class Context:
    """Which nested fit supplies a match's possession. For outer k, a match in fold j is
    `test` when j == k and `inner-oof` otherwise; `final` (outer None) is the final tau's
    out-of-fold possession, which reuses outer_j's test output."""

    role: str
    fold: int
    outer: int | None = None

    def __post_init__(self):
        ok = {
            "test": self.outer == self.fold,
            "inner-oof": self.outer is not None and self.outer != self.fold,
            "final": self.outer is None,
        }
        if not ok.get(self.role, False):
            raise ValueError(f"invalid context {self}")

    @property
    def fit_id(self) -> str:
        """The logical fit that predicts this match here."""
        if self.role == "inner-oof":
            return f"outer_{self.outer}_inner_{self.fold}"
        return f"outer_{self.fold}"

    @property
    def suffix(self) -> str:
        """Cache name suffix; final reads the test file of the match's own fold."""
        if self.role == "inner-oof":
            return f"outer{self.outer}_inner-oof{self.fold}"
        return f"outer{self.fold}_test"


def context_for(outer: int | None, fold: int) -> Context:
    """A fold-j match's context in outer k (None: final)."""
    if outer is None:
        return Context("final", fold)
    return Context("test" if fold == outer else "inner-oof", fold, outer)


def sha256_file(path: Path) -> str:
    return vpf.sha256_file(path)


def native_hashes(match_id: str, gamestate_dir: Path) -> dict:
    """SHA256 of the native frames/objects and of the grid rebuilt from those frames."""
    meta = vpf.native_inputs(match_id, gamestate_dir)
    frames = pl.read_parquet(gamestate_dir / match_id / "frames.parquet")
    fps = meta.pop("native_fps")
    meta["grid_sha256"] = vpf.grid_hash(vpf.grid(vpf.native(frames, fps), fps))
    return meta


def refuse_stale(path: Path, side: Path, want: dict) -> None:
    """Raise unless side's recorded values equal `want` and path's output hash."""
    meta = json.loads(side.read_text())
    now = want | {"output_sha256": sha256_file(path)}
    stale = [k for k, v in now.items() if meta.get(k) != v]
    if stale:
        raise ValueError(f"{path}: stale learned cache ({', '.join(stale)} changed); delete it")


def learned_manifest(config: StateConfig, models_dir: Path | None = None) -> tuple[Path, dict]:
    """The sealed, checked CV manifest of config's possession model."""
    models_dir = models_dir or MODELS_DIR
    if config.possession_model is None:
        raise ValueError("learned possession needs StateConfig.possession_model")
    path = models_dir / config.possession_model / vpm.MANIFEST
    if not path.exists():
        raise FileNotFoundError(
            f"{path}: no trained model; run python -m vision.possession_train first"
        )
    man = json.loads(path.read_text())
    vpm.check_manifest(man)
    if man["model_id"] != config.possession_model:
        raise ValueError(f"{path}: model id {man['model_id']!r}, not {config.possession_model!r}")
    return path, man


def learned_meta(
    config: StateConfig,
    man_path: Path,
    man: dict,
    ids: list[str],
    fold_of: dict,
    processed_dir: Path,
    stats: dict,
) -> dict:
    """run.json's learned-only block (05, Run metadata): the sealed artifact and its
    hashes, the scheme, the final alias and each match's outer-context cache hashes."""
    key = config_key(config)
    caches = {}
    for m in ids:
        for k in range(vpm.N_FOLDS):
            suffix = context_for(k, fold_of[m]).suffix
            side = processed_dir / m / f"state_inferred_age_{key}_{suffix}.json"
            if side.exists():
                caches.setdefault(m, {})[suffix] = json.loads(side.read_text())["output_sha256"]
    return {
        "model_id": config.possession_model,
        "scheme": man["scheme"],
        "stride": man["stride"],
        "contract": man["contract"],
        "rule_2d": man["rule_2d"],
        "mirror": man["mirror"],
        "threshold": man["threshold"],
        "manifest": str(man_path),
        "manifest_sha256": sha256_file(man_path),
        "folds_sha256": man["folds"]["sha256"],
        "dedup_pairs": man["dedup_pairs"],
        "final": "a match in fold j reads outer<j>_test (outer_j's test output)",
        "trainings": {
            n: {
                k: t[k]
                for k in ("model_sha256", "best_iteration", "timing_s", "peak_rss_bytes", "rows")
            }
            for n, t in man["trainings"].items()
        },
        "fallback_rows_test": stats.get("fallback_rows", 0),
        "grid_rows_test": stats.get("grid_rows", 0),
        "state_sha256": caches,
    }


def learned_state(
    match_id: str,
    config: StateConfig,
    context: Context,
    gamestate_dir: Path,
    processed_dir: Path,
    models_dir: Path | None = None,
    cache_dir: Path | None = None,
    cache: bool = True,
) -> pl.DataFrame:
    """The learned grid state of one match in one context (03's column contract), from
    state_inferred_age_<key>_<context>.parquet when its provenance still matches: the
    model manifest, the stored model, the native frames/objects and the grid. A mismatch
    is an error, never a reason to rebuild silently or fall back to another cache."""
    man_path, man = learned_manifest(config, models_dir)
    model_dir = man_path.parent
    fold = man["folds"]["assignments"].get(match_id)
    if fold != context.fold:
        raise ValueError(f"{match_id} is in fold {fold}, the context says {context.fold}")
    training = man["trainings"][man["logical"][context.fit_id]["training"]]
    want = {
        "model_id": config.possession_model,
        "context": context.suffix,
        "fit_id": context.fit_id,
        "manifest_sha256": sha256_file(man_path),
        "model_sha256": training["model_sha256"],
        **native_hashes(match_id, gamestate_dir),
    }
    stem = processed_dir / match_id / f"state_inferred_age_{config_key(config)}_{context.suffix}"
    path, side = stem.with_suffix(".parquet"), stem.with_suffix(".json")
    if cache and path.exists():
        if not side.exists():
            raise ValueError(f"{path}: no provenance; delete it")
        refuse_stale(path, side, want)
        return pl.read_parquet(path)
    cache_dir = cache_dir or VISION_CACHE_DIR
    out = vpm.predict_match(model_dir, context.fit_id, match_id, gamestate_dir, cache_dir)
    if cache:
        out.write_parquet(path)
        side.write_text(json.dumps(want | {"output_sha256": sha256_file(path)}, indent=2) + "\n")
    return out


def swap_learned(frames: pl.DataFrame, state: pl.DataFrame) -> pl.DataFrame:
    """Grid rows (period, t_s, frame_id, home_attacks_positive_x, ...) with possession_team
    from the learned state and flipped recomputed. Joined on the full key (period, tick),
    never frame_id alone (a sparse source can reuse a native frame at several ticks);
    the native frame_id must match too. Rows, keys and order must be identical."""
    tick = (pl.col("t_s") * 10).round().cast(pl.Int64)
    keys = frames.select("period", k=tick)
    if keys.height != state.height or not keys.equals(state.select("period", "k")):
        raise ValueError("learned state rows differ from the grid's (period, tick) keys")
    if not frames["frame_id"].equals(state["frame_id"]):
        raise ValueError("learned state's native frame_id differs from the grid's")
    out = frames.with_columns(possession_team=state["possession_team"])
    return out.with_columns(flipped=flipped_expr())


def fill(
    config: StateConfig,
    gamestate_dir: Path,
    processed_dir: Path,
    log=print,
) -> int:
    """Build every match's learned state in each of the 5 outer contexts (its test fit and
    4 inner-OOF fits; final reuses test), so later runs only read caches. Returns the
    number of states built or rechecked."""
    _, man = learned_manifest(config)
    n = 0
    for m, fold in sorted(man["folds"]["assignments"].items()):
        t0 = time.perf_counter()
        for k in range(vpm.N_FOLDS):
            learned_state(m, config, context_for(k, fold), gamestate_dir, processed_dir)
            n += 1
        log(f"{m}: {vpm.N_FOLDS} contexts, {time.perf_counter() - t0:.1f} s")
    return n


def main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Fill the learned-possession state caches for every outer context."
    )
    ap.add_argument("--possession-model", required=True, metavar="ID")
    ap.add_argument("--gamestate", type=Path, default=Path("data/gamestate"))
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    n = fill(StateConfig(possession_model=a.possession_model), a.gamestate, a.processed)
    print(f"{n} learned states in {time.perf_counter() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv[1:]))
