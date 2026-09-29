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
"""

import hashlib
import json
import time
from pathlib import Path

import polars as pl

from prediction.resample import flipped_expr
from vision.state import StateConfig, infer

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
