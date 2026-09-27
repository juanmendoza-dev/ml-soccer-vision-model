"""Which game-state objects land off the pitch, and what the view/homography looked like then.

A script over one run's outputs (03 Diagnostics: anomaly checks are scripts over
the cache). Written for the first 2060 smoke test, where the validator flagged
rows more than 15 m off the pitch.

    python -m vision.offpitch --match-id smoke01
"""

import argparse
from pathlib import Path

import polars as pl

from gamestate.schema import PITCH_LENGTH, PITCH_WIDTH
from gamestate.validate import POSITION_MARGIN_M

MAX_X = PITCH_LENGTH / 2 + POSITION_MARGIN_M
MAX_Y = PITCH_WIDTH / 2 + POSITION_MARGIN_M


def load(gamestate: Path, cache: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(objects, per-frame context) for one run."""
    objects = pl.read_parquet(gamestate / "objects.parquet")
    frames = pl.read_parquet(gamestate / "frames.parquet").select(
        "frame_id", "timestamp_s", h_ok=pl.col("view_polygon").is_not_null()
    )
    view = pl.read_parquet(cache / "view.parquet").select("frame_id", "view", "keypoints_found")
    det = pl.read_parquet(cache / "detections.parquet")
    h_err = det.group_by("frame_id").agg(pl.col("homography_err_m").first())
    ctx = (
        frames.join(view, on="frame_id", how="left")
        .join(h_err, on="frame_id", how="left")
        .sort("frame_id")
        .with_columns(
            # keypoints/err are only set on keypoint frames: carry the last one forward
            kp_last=pl.col("keypoints_found").forward_fill(),
            err_last=pl.col("homography_err_m").forward_fill(),
            # frames since the view last switched to match
            seg=(
                (pl.col("view") == "match") & (pl.col("view").shift(1) != "match").fill_null(True)
            ).cum_sum(),
        )
        .with_columns(since_switch=pl.int_range(pl.len()).over("seg"))
    )
    return objects, ctx


def off_pitch(objects: pl.DataFrame) -> pl.DataFrame:
    return objects.filter((pl.col("x").abs() > MAX_X) | (pl.col("y").abs() > MAX_Y))


def runs(off: pl.DataFrame, ctx: pl.DataFrame) -> pl.DataFrame:
    """Offending rows grouped into runs of consecutive frames."""
    frames = off.select("frame_id").unique().sort("frame_id")
    frames = frames.with_columns(run=(pl.col("frame_id").diff().fill_null(1) != 1).cum_sum())
    return (
        off.join(frames, on="frame_id")
        .join(ctx, on="frame_id", how="left")
        .group_by("run")
        .agg(
            first=pl.col("frame_id").min(),
            last=pl.col("frame_id").max(),
            rows=pl.len(),
            types=pl.col("object_type").unique().sort().str.join(","),
            interp=pl.col("interpolated").sum(),
            objects=pl.col("object_id").n_unique(),
            since_switch=pl.col("since_switch").min(),
            kp_last=pl.col("kp_last").first(),
            err_last=pl.col("err_last").first(),
            x_min=pl.col("x").min(),
            x_max=pl.col("x").max(),
            y_min=pl.col("y").min(),
            y_max=pl.col("y").max(),
        )
        .sort("first")
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--match-id", required=True)
    ap.add_argument("--gamestate-dir", type=Path, default=Path("data/gamestate"))
    ap.add_argument("--cache-dir", type=Path, default=Path("data/vision_cache"))
    args = ap.parse_args(argv)

    objects, ctx = load(args.gamestate_dir / args.match_id, args.cache_dir / args.match_id)
    off = off_pitch(objects)
    match = ctx.filter(pl.col("view") == "match")
    det = pl.read_parquet(args.cache_dir / args.match_id / "detections.parquet")
    # a valid homography but no position: the pipeline threw the projection away
    rejected = det.filter(pl.col("homography_ok") & pl.col("pitch_x").is_null()).height

    print(f"frames {ctx.height}, match {match.height}, homography ok {match['h_ok'].sum()}")
    print(f"objects {objects.height}, off pitch (>{POSITION_MARGIN_M:g} m) {off.height}")
    print(f"projections rejected by the pipeline {rejected}")
    if not off.height:
        return
    print("\noff-pitch rows by type / interpolated:")
    by = off.group_by("object_type", "interpolated").len().sort("len", descending=True)
    for r in by.iter_rows(named=True):
        print(f"  {r['object_type']:<11} interpolated={r['interpolated']!s:<5} {r['len']}")
    print("\nruns of consecutive frames:")
    # plain ASCII table so a redirected Windows console doesn't choke
    with pl.Config(
        tbl_rows=100,
        tbl_cols=20,
        tbl_width_chars=200,
        float_precision=1,
        tbl_formatting="ASCII_MARKDOWN",
    ):
        print(runs(off, ctx).drop("run"))


if __name__ == "__main__":
    main()
