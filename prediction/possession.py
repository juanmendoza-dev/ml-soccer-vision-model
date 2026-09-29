"""Stage 8's possession as the model's input (05, "Inferred possession"; 07 #6).

The one place prediction calls into vision/: vision.state.infer, which 03 writes to run on
dataset tracking as well as live. Stage 8 runs on the native game state (02), and its
possession_team is joined onto the 10 Hz grid by frame_id. Each grid row already uses the
latest native frame at or before t, and infer is causal, so a row's inferred possession
uses frames <= t only.

Only the model's inputs take it. Labels, eligible, all_estimated and the table's own
possession_team / flipped / ball_state stay the provider's (prediction.features.load_match).
"""

import hashlib
import json
import time
from pathlib import Path

import polars as pl

from prediction.resample import flipped_expr
from vision.state import StateConfig, infer

POSSESSION = ("provider", "inferred")
# what infer reads from 02's objects
OBJECT_COLS = ["frame_id", "object_id", "object_type", "team", "x", "y", "z", "visible"]


def config_key(config: StateConfig) -> str:
    """Cache key for a StateConfig: any threshold change gets new caches."""
    blob = json.dumps(config.to_dict(), sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def inferred_state(
    match_id: str,
    gamestate_dir: Path,
    processed_dir: Path,
    config: StateConfig | None = None,
    cache: bool = True,
    timing: dict | None = None,
) -> pl.DataFrame:
    """frame_id and stage 8's possession_team on every native frame of the match, cached
    next to the resampled tables. Seconds spent in infer are added to timing["stage8_s"]."""
    config = config or StateConfig()
    path = processed_dir / match_id / f"state_inferred_{config_key(config)}.parquet"
    if cache and path.exists():
        return pl.read_parquet(path)
    src = gamestate_dir / match_id
    t0 = time.perf_counter()
    frames = pl.read_parquet(src / "frames.parquet", columns=["frame_id", "period", "timestamp_s"])
    objects = pl.scan_parquet(src / "objects.parquet").select(OBJECT_COLS)
    out = infer(frames, objects, config).select("frame_id", "possession_team")
    if timing is not None:
        timing["stage8_s"] = timing.get("stage8_s", 0.0) + time.perf_counter() - t0
    if cache:
        out.write_parquet(path)
    return out


def swap_possession(frames: pl.DataFrame, state: pl.DataFrame) -> pl.DataFrame:
    """Grid rows (with frame_id and home_attacks_positive_x) with possession_team replaced
    by state's at the row's native frame and flipped recomputed from it, the resampler's
    way. Row order is kept."""
    missing = frames.join(state, on="frame_id", how="anti")
    if missing.height:
        raise ValueError(f"{missing.height} grid rows point at frames stage 8 didn't label")
    return (
        frames.drop("possession_team", "flipped")
        .join(state, on="frame_id", how="left", validate="m:1", maintain_order="left")
        .with_columns(flipped=flipped_expr())
    )


def add_stats(stats: dict, provider: pl.DataFrame, inferred: pl.DataFrame, scored: pl.Series):
    """Counts over the scored rows, for run.json: inferred possession differs from the
    provider's, flipped differs, inferred is null."""
    p, i = provider.filter(scored), inferred.filter(scored)
    counts = {
        "scored_rows": p.height,
        "possession_differs": int(p["possession_team"].ne_missing(i["possession_team"]).sum()),
        "flipped_differs": int(p["flipped"].ne_missing(i["flipped"]).sum()),
        "inferred_null": int(i["possession_team"].is_null().sum()),
    }
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
    }
