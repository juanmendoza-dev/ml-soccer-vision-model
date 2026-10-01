"""Rerun stage 4's acceptance and the projection from a run's caches (03 Diagnostics).

    python -m vision.replay --match-id smoke04 --set max_homography_err_m=2 --set min_inliers=6

No detector or keypoint model runs: the keypoints cache has every stage 4 call and
the detections cache every box, so a threshold sweep takes seconds. With the run's
own config the output matches the detections cache (people exactly; the ball only
where it was detected, extrapolated rows are dropped).
"""

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import polars as pl

from vision.config import VisionConfig
from vision.pitch import HomographyFilter, fit_homography, on_pitch, project, to_02
from vision.types import BALL, GOALKEEPER, MATCH, PLAYER


def run_config(cache: Path) -> VisionConfig:
    return VisionConfig(**json.loads((cache / "run.json").read_text())["config"])


def with_overrides(config: VisionConfig, sets: list[str]) -> VisionConfig:
    fields = {f.name: f.type for f in dataclasses.fields(VisionConfig)}
    changes = {}
    for item in sets:
        name, _, value = item.partition("=")
        if name not in fields:
            raise SystemExit(f"unknown VisionConfig field {name!r}")
        changes[name] = type(getattr(config, name))(value)
    return dataclasses.replace(config, **changes)


def frame_homographies(
    keypoints: pl.DataFrame, views: pl.DataFrame, times: dict[int, float], config: VisionConfig
) -> dict[int, np.ndarray | None]:
    """frame_id -> the homography in use on that frame (None: not ok), as the pipeline
    decides it: fits offered in call order, the filter reset on each new segment."""
    filt = HomographyFilter(
        config.homography_window, config.max_homography_jump_m, config.homography_max_age_s
    )
    calls: dict[int, list[dict]] = {}
    for row in keypoints.iter_rows(named=True):
        calls.setdefault(row["frame_id"], []).append(row)
    segment = None
    out = {}
    for frame_id, view in views.select("frame_id", "view").sort("frame_id").iter_rows():
        for c in calls.get(frame_id, []):
            if not c["used"]:
                continue
            if c["segment"] != segment:
                filt.reset()
                segment = c["segment"]
            xy = np.column_stack([c["kp_x"], c["kp_y"]])
            fit = fit_homography(
                xy, np.array(c["kp_conf"]), config.min_keypoint_conf, config.ransac_m
            )
            if (
                fit.H is not None
                and fit.n_inliers >= config.min_inliers
                and fit.err_m <= config.max_homography_err_m
            ):
                filt.offer(fit, c["t"])
        out[frame_id] = filt.current(times[frame_id]) if view == MATCH else None
    return out


def replay(
    detections: pl.DataFrame,
    keypoints: pl.DataFrame,
    views: pl.DataFrame,
    times: dict[int, float],
    config: VisionConfig,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(detections with pitch_x / pitch_y / homography_ok / keeper team_cluster redone,
    one row per frame: frame_id, view, homography_ok)."""
    hs = frame_homographies(keypoints, views, times, config)
    det = detections.filter(~((pl.col("class") == BALL) & pl.col("tracked_only")))
    xs, ys = [], []
    for cls, x1, y1, x2, y2, frame_id in det.select(
        "class", "x1", "y1", "x2", "y2", "frame_id"
    ).iter_rows():
        H = hs[frame_id]
        if H is None:
            xs.append(None)
            ys.append(None)
            continue
        # feet for people, box center for the ball (as in the pipeline)
        px = ((x1 + x2) / 2, (y1 + y2) / 2 if cls == BALL else y2)
        xy = to_02(project(H, [px]), config.home_attacks_tv_right_p1)[0]
        ok = on_pitch(xy, config.max_off_pitch_m)
        xs.append(float(xy[0]) if ok else None)
        ys.append(float(xy[1]) if ok else None)
    det = det.with_columns(
        pitch_x=pl.Series(xs, dtype=pl.Float64),
        pitch_y=pl.Series(ys, dtype=pl.Float64),
        homography_ok=pl.col("frame_id").replace_strict(
            {f: H is not None for f, H in hs.items()}, return_dtype=pl.Boolean
        ),
    )
    det = _goalkeeper_clusters(det)
    frames = views.select("frame_id", "view").with_columns(
        homography_ok=pl.col("frame_id").replace_strict(
            {f: H is not None for f, H in hs.items()}, return_dtype=pl.Boolean
        )
    )
    return det, frames


def _goalkeeper_clusters(det: pl.DataFrame) -> pl.DataFrame:
    """A goalkeeper takes the cluster whose outfield players stand nearer on average in
    x, on frames where both clusters have a position (VisionPipeline._goalkeeper_teams)."""
    means = (
        det.filter(
            (pl.col("class") == PLAYER)
            & pl.col("team_cluster").is_not_null()
            & pl.col("pitch_x").is_not_null()
        )
        .group_by("frame_id")
        .agg(
            m0=pl.col("pitch_x").filter(pl.col("team_cluster") == 0).mean(),
            m1=pl.col("pitch_x").filter(pl.col("team_cluster") == 1).mean(),
        )
    )
    det = det.join(means, on="frame_id", how="left")
    keeper = (pl.col("class") == GOALKEEPER) & pl.col("pitch_x").is_not_null()
    both = pl.col("m0").is_not_null() & pl.col("m1").is_not_null()
    # ties go to cluster 0, as min() over the pipeline's dict does
    nearest = pl.when(
        (pl.col("pitch_x") - pl.col("m0")).abs() <= (pl.col("pitch_x") - pl.col("m1")).abs()
    )
    cluster = (
        pl.when(keeper & both)
        .then(nearest.then(0).otherwise(1))
        .when(pl.col("class") == GOALKEEPER)
        .then(None)
        .otherwise(pl.col("team_cluster"))
    )
    return det.with_columns(team_cluster=cluster.cast(pl.Int64)).drop("m0", "m1")


def load(cache: Path, gamestate: Path) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict]:
    times = dict(
        pl.read_parquet(
            gamestate / "frames.parquet", columns=["frame_id", "timestamp_s"]
        ).iter_rows()
    )
    return (
        pl.read_parquet(cache / "detections.parquet"),
        pl.read_parquet(cache / "keypoints.parquet"),
        pl.read_parquet(cache / "view.parquet"),
        times,
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="vision.replay")
    ap.add_argument("--match-id", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE")
    ap.add_argument("--out", type=Path, help="write the replayed detections here")
    ap.add_argument("--cache-dir", type=Path, default=Path("data/vision_cache"))
    ap.add_argument("--gamestate-dir", type=Path, default=Path("data/gamestate"))
    args = ap.parse_args(argv)

    cache = args.cache_dir / args.match_id
    base = run_config(cache)
    config = with_overrides(base, args.set)
    det, keypoints, views, times = load(cache, args.gamestate_dir / args.match_id)
    _, before = replay(det, keypoints, views, times, base)
    out, after = replay(det, keypoints, views, times, config)
    n_match = after.filter(pl.col("view") == MATCH).height
    for name, frames in [("run config", before), ("replayed", after)]:
        ok = frames["homography_ok"].sum()
        print(f"{name}: homography ok on {ok} of {n_match} match frames")
    if args.out:
        out.write_parquet(args.out)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
