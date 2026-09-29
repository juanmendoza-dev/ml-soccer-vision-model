"""Game state (02) -> 10 Hz frames, objects and shot/goal labels (05, "Resampled frames and labels").

    python -m prediction.resample [--games 10502 ...] [--gamestate DIR] [--out DIR]

Causal: each 0.1 s grid point takes the latest native frame at or before it, never
interpolates, and never bridges a period or a gap. Only `label_*` columns look ahead.
Reads the game state only; nothing here imports from converters/ or vision/.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import polars as pl

from prediction.labels import HORIZONS, US, add_labels, label_events, shots_without_positive, to_us

GAMESTATE_DIR = Path("data/gamestate")
OUT_DIR = Path("data/processed")

GRID_US = 100_000  # 10 Hz
MAX_STALENESS = 1.5  # native intervals; an older frame means a gap
PLAYER_TYPES = ("player", "goalkeeper")
FRAME_COLUMNS = [
    "home_attacks_positive_x",
    "ball_state",
    "possession_team",
    "ball_carrier_id",
    "view_polygon",
]


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


def build_grid(frames: pl.DataFrame, native_fps: float) -> tuple[pl.DataFrame, int]:
    """One row per 0.1 s grid point with the latest native frame at or before it.

    The staleness limit comes from the declared native_fps (02 match table), never from
    the frames themselves: estimating the cadence from the whole match let later frames
    decide whether an earlier grid row exists (detection review D8).
    Returns the grid and how many grid points were skipped for a gap.
    """
    if not native_fps or native_fps <= 0:
        raise ValueError(f"native_fps must be a positive declared rate, got {native_fps!r}")
    frames = frames.with_columns(ts_us=to_us("timestamp_s")).sort("ts_us")
    tol = round(MAX_STALENESS * US / native_fps)
    bounds = frames.group_by("period").agg(first=pl.col("ts_us").min(), last=pl.col("ts_us").max())
    grid = (
        bounds.select(
            "period",
            k=pl.int_ranges(
                (pl.col("first") + GRID_US - 1) // GRID_US, pl.col("last") // GRID_US + 1
            ),
        )
        .explode("k", empty_as_null=True)
        .drop_nulls("k")
        .select("period", grid_us=pl.col("k").cast(pl.Int64) * GRID_US)
        .sort("grid_us")
    )
    grid = grid.join_asof(
        frames,
        left_on="grid_us",
        right_on="ts_us",
        by="period",
        strategy="backward",
        check_sortedness=False,
    ).sort("period", "grid_us")
    fresh = pl.col("grid_us") - pl.col("ts_us") <= tol
    skipped = grid.filter(~fresh).height
    return grid.filter(fresh), skipped


def add_frame_flags(grid: pl.DataFrame, objects: pl.DataFrame) -> pl.DataFrame:
    """flipped, eligible, all_estimated: all from the native frame at t."""
    visible = (
        objects.filter(pl.col("object_type").is_in(PLAYER_TYPES))
        .group_by("frame_id")
        .agg(any_visible=pl.col("visible").any())
    )
    return (
        grid.join(visible, on="frame_id", how="left")
        .with_columns(
            flipped=pl.when(pl.col("possession_team").is_not_null()).then(
                (pl.col("possession_team") == "home") != pl.col("home_attacks_positive_x")
            ),
            eligible=(pl.col("ball_state") == "alive").fill_null(False)
            & pl.col("possession_team").is_not_null(),
            all_estimated=~pl.col("any_visible").fill_null(False),
        )
        .drop("any_visible")
    )


def finish_frames(grid: pl.DataFrame, match_id: str) -> pl.DataFrame:
    order = ["period", "t_s", "frame_id", "timestamp_s", *FRAME_COLUMNS]
    order += ["flipped", "eligible", "all_estimated"]
    order += ["label_set_play_phase"]
    order += [f"label_{k}_{h}" for h in HORIZONS for k in ("mask", "shot", "goal")]
    return grid.with_columns(
        match_id=pl.lit(match_id),
        t_s=pl.col("grid_us") / US,
        label_set_play_phase=pl.col("set_play_phase"),  # label-side only (02)
    ).select("match_id", *order)


def resample_objects(objects: pl.DataFrame, frames10: pl.DataFrame) -> pl.DataFrame:
    """Objects of each grid point's native frame, plus coordinates rotated to the attack (05)."""
    sign = pl.when(pl.col("flipped").fill_null(False)).then(-1.0).otherwise(1.0)
    cols = objects.columns
    out = frames10.select("frame_id", "period", "t_s", "flipped").join(
        objects, on="frame_id", how="inner"
    )
    return (
        out.with_columns(
            x_att=pl.col("x") * sign,
            y_att=pl.col("y") * sign,
            vx_att=pl.col("vx") * sign,
            vy_att=pl.col("vy") * sign,
        )
        .select(*cols, "period", "t_s", "x_att", "y_att", "vx_att", "vy_att")
        .sort("period", "t_s", "object_id")
    )


def resample_match(
    frames: pl.DataFrame,
    objects: pl.DataFrame,
    events: pl.DataFrame,
    match_id: str,
    native_fps: float,
) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """In-memory core: game state tables -> (frames_10hz, objects_10hz, stats)."""
    grid, skipped = build_grid(frames, native_fps)
    used = objects.filter(pl.col("frame_id").is_in(grid["frame_id"].unique().implode()))
    grid = add_frame_flags(grid, used)
    lev = label_events(events, frames)
    grid = add_labels(grid, lev)
    frames10 = finish_frames(grid, match_id)
    objects10 = resample_objects(used, frames10)
    return frames10, objects10, {"grid_points_skipped_gap": skipped, "label_events": lev}


def build_report(
    match_id: str,
    frames: pl.DataFrame,
    objects_in: int,
    frames10: pl.DataFrame,
    objects10: pl.DataFrame,
    stats: dict,
) -> dict:
    lev = stats["label_events"]
    n = frames10.height
    missing = shots_without_positive(frames10, lev)
    open_shots = lev.filter(pl.col("kind") == "shot").height
    return {
        "match_id": match_id,
        "resample_commit": git_commit(),
        "rows": {
            "frames": {"in": frames.height, "out": n},
            "objects": {"in": objects_in, "out": objects10.height},
        },
        "grid_points_skipped_gap": stats["grid_points_skipped_gap"],
        "events_without_frame": lev.filter(pl.col("period").is_null()).height,
        "eligible_share": round(frames10["eligible"].mean(), 4) if n else 0.0,
        "all_estimated_share": round(frames10["all_estimated"].mean(), 4) if n else 0.0,
        "eligible_not_estimated_share": round(
            (frames10["eligible"] & ~frames10["all_estimated"]).mean(), 4
        )
        if n
        else 0.0,
        "mask_true": {h: int(frames10[f"label_mask_{h}"].sum()) for h in HORIZONS},
        "positives": {
            f"label_{k}_{h}": int(frames10[f"label_{k}_{h}"].sum())
            for h in HORIZONS
            for k in ("shot", "goal")
        },
        "label_events": {
            k: lev.filter(pl.col("kind") == k).height for k in ("shot", "goal", "set_play")
        },
        "open_play_shots": open_shots,
        "open_play_shots_with_positive": {h: open_shots - len(m) for h, m in missing.items()},
        "open_play_shots_without_positive": missing,
    }


def process_game(
    match_id: str, gamestate_dir: Path = GAMESTATE_DIR, out_dir: Path = OUT_DIR
) -> dict:
    src = gamestate_dir / match_id
    frames = pl.read_parquet(src / "frames.parquet")
    events = pl.read_parquet(src / "events.parquet")
    objects = pl.read_parquet(src / "objects.parquet")
    objects_in = objects.height
    match = pl.read_parquet(src / "match.parquet")
    frames10, objects10, stats = resample_match(
        frames, objects, events, match_id, match["native_fps"].item()
    )
    del objects
    dst = out_dir / match_id
    dst.mkdir(parents=True, exist_ok=True)
    # cached features and graphs (prediction.features, prediction.graphs) were built from
    # the old tables
    for stale in [*dst.glob("features_v*.parquet"), *dst.glob("graphs_v*.npz")]:
        stale.unlink()
    frames10.write_parquet(dst / "frames_10hz.parquet")
    objects10.write_parquet(dst / "objects_10hz.parquet")
    report = build_report(match_id, frames, objects_in, frames10, objects10, stats)
    report = {"source": match["source"].item(), **report}
    (dst / "resample_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def summarize(reports: list[dict]) -> dict:
    """Totals per source; per-H positive rates are over unmasked rows."""
    sources = sorted({r["source"] for r in reports})
    return {src: summarize_source([r for r in reports if r["source"] == src]) for src in sources}


def summarize_source(reports: list[dict]) -> dict:
    rows = sum(r["rows"]["frames"]["out"] for r in reports)
    eligible = sum(r["eligible_share"] * r["rows"]["frames"]["out"] for r in reports)
    shares = sorted(r["eligible_share"] for r in reports)
    out = {
        "games": len(reports),
        "rows_10hz": rows,
        "objects_10hz": sum(r["rows"]["objects"]["out"] for r in reports),
        "eligible_share": round(eligible / rows, 4) if rows else 0.0,
        "eligible_share_per_game": [shares[0], shares[len(shares) // 2], shares[-1]],
        "open_play_shots": sum(r["open_play_shots"] for r in reports),
    }
    for h in HORIZONS:
        mask = sum(r["mask_true"][h] for r in reports)
        out[h] = {
            "mask_true": mask,
            "shot_positives": sum(r["positives"][f"label_shot_{h}"] for r in reports),
            "goal_positives": sum(r["positives"][f"label_goal_{h}"] for r in reports),
            "open_play_shots_with_positive": sum(
                r["open_play_shots_with_positive"][h] for r in reports
            ),
        }
        out[h]["shot_rate"] = round(out[h]["shot_positives"] / mask, 4) if mask else 0.0
        out[h]["goal_rate"] = round(out[h]["goal_positives"] / mask, 5) if mask else 0.0
    return out


def all_games(gamestate_dir: Path = GAMESTATE_DIR) -> list[str]:
    return sorted(p.parent.name for p in gamestate_dir.glob("*/frames.parquet"))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", nargs="+")
    ap.add_argument("--gamestate", type=Path, default=GAMESTATE_DIR)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    reports, failed = [], 0
    for g in args.games or all_games(args.gamestate):
        try:
            r = process_game(g, args.gamestate, args.out)
        except Exception as e:  # noqa: BLE001 -- one bad game shouldn't stop a batch
            print(f"FAIL {g}  {type(e).__name__}: {e}", flush=True)
            failed += 1
            continue
        reports.append(r)
        p = r["positives"]
        print(
            f"ok   {g}  rows {r['rows']['frames']['out']}  eligible {r['eligible_share']:.2f}"
            f"  shot_h5 {p['label_shot_h5']}  shot_h3 {p['label_shot_h3']}"
            f"  shots {r['open_play_shots']} w/o positive h5 "
            f"{len(r['open_play_shots_without_positive']['h5'])}",
            flush=True,
        )
    if reports and not args.games:
        summary = summarize(reports)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
