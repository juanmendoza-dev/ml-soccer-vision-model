"""Rerun stage 4's acceptance and the projection from a run's caches (03 Diagnostics).

    python -m vision.replay --match-id smoke04 --set max_homography_err_m=2 --set min_inliers=6

No detector or keypoint model runs: the stage 4 cache (keypoints.parquet, or camera.parquet
for PnLCalib runs) has every stage 4 call and the detections cache every box, so a threshold
sweep takes seconds. With the run's own config the output matches the detections cache:
people exactly, and the ball exactly when the run wrote balls.parquet (stage 5 is rerun from
its candidates). Older runs keep only their detected ball rows, reprojected.

--variant TAG reruns stage 5 at the --set ball_* fields into data/variants/TAG/{vision_cache,
gamestate}/<match_id>, the run's people kept, so stage 8, inference and demo.video can run on
it without touching the run.

PnLCalib runs replay from the cached cameras; --revote votes them again from the cached peaks
on CPU, so pnl_kp_threshold / pnl_line_threshold can be swept (03 Diagnostics).
"""

import argparse
import dataclasses
import json
import math
import shutil
from pathlib import Path

import numpy as np
import polars as pl

from converters.common import causal_velocities
from gamestate.validate import validate_match
from vision.ball import BallTrack, expected_width
from vision.calib import accept, camera_fit, camera_from_peaks
from vision.config import VisionConfig
from vision.pipeline import frac_box
from vision.pitch import HomographyFilter, fit_homography, on_pitch, project, template, to_02
from vision.types import (
    BALL,
    GOALKEEPER,
    MATCH,
    PLAYER,
    CalibPeaks,
    Camera,
    Detection,
    VisionObject,
)
from vision.writer import DETECTIONS_SCHEMA, OBJECT_TYPES, OBJECTS_SCHEMA, on_screen

VARIANT_ROOT = Path("data/variants")
CACHE_FILES = ("view.parquet", "camera.parquet", "keypoints.parquet", "balls.parquet")
GS_FILES = ("match.parquet", "frames.parquet", "events.parquet", "players.parquet")


def run_config(cache: Path) -> VisionConfig:
    config = json.loads((cache / "run.json").read_text())["config"]
    config.setdefault("circle_kp_x_m", 9.15)  # runs before the field existed (03 Pitch template)
    config.setdefault("calib_backend", "roboflow")  # runs before PnLCalib (03 Pitch calibration)
    config.setdefault("pnl_blind_kp", 0)  # PnLCalib runs before it: always held
    # runs before the ball fields (10-ball 2): most confident, unfiltered, any speed
    config.setdefault("ball_picker", "max")
    config.setdefault("ball_cand_margin_m", config.get("max_off_pitch_m", 10.0))
    config.setdefault("ball_size_lo", 0.0)
    config.setdefault("ball_size_hi", math.inf)
    config.setdefault("ball_max_speed_mps", math.inf)
    config.setdefault("ball_air_ratio", math.inf)  # never airborne (10-ball 2g)
    return VisionConfig(**config)


def with_overrides(config: VisionConfig, sets: list[str]) -> VisionConfig:
    fields = {f.name: f.type for f in dataclasses.fields(VisionConfig)}
    changes = {}
    for item in sets:
        name, _, value = item.partition("=")
        if name not in fields:
            raise SystemExit(f"unknown VisionConfig field {name!r}")
        changes[name] = type(getattr(config, name))(value)
    return dataclasses.replace(config, **changes)


def cached_camera(c: dict) -> Camera | None:
    """The camera the run voted, from its camera.parquet row."""
    if not c["found"]:
        return None
    return Camera(
        c["fx"],
        c["fy"],
        c["cx"],
        c["cy"],
        np.array([c["cam_x"], c["cam_y"], c["cam_z"]]),
        np.array(c["rot"]).reshape(3, 3),
        c["err_px"],
        c["mode"],
    )


def revoted_camera(c: dict, config: VisionConfig) -> tuple[Camera | None, int]:
    """Vote the camera again from the cached peaks, at config's thresholds."""
    ends = [np.column_stack([c[f"line_x{e}"], c[f"line_y{e}"], c[f"line_s{e}"]]) for e in (1, 2)]
    peaks = CalibPeaks(
        np.column_stack([c["kp_x"], c["kp_y"], c["kp_score"]]), np.stack(ends, axis=1)
    )
    cam, n_kp, _ = camera_from_peaks(
        peaks, (c["img_w"], c["img_h"]), config.pnl_kp_threshold, config.pnl_line_threshold
    )
    return cam, n_kp


def _fit(c: dict, config: VisionConfig, pitch: np.ndarray, revote: bool):
    """One used stage 4 call -> (the Fit the pipeline offered the filter or None, whether
    the call saw no pitch and dropped the camera in use)."""
    if config.calib_backend == "pnlcalib":
        cam, n_kp = revoted_camera(c, config) if revote else (cached_camera(c), c["n_kp"])
        fit = None
        if accept(cam, config):
            fit = camera_fit(cam, (c["img_w"], c["img_h"]), n_kp, config.max_off_pitch_m)
        return fit, fit is None and n_kp < config.pnl_blind_kp
    xy = np.column_stack([c["kp_x"], c["kp_y"]])
    fit = fit_homography(
        xy, np.array(c["kp_conf"]), config.min_keypoint_conf, config.ransac_m, pitch
    )
    if (
        fit.H is not None
        and fit.n_inliers >= config.min_inliers
        and fit.err_m <= config.max_homography_err_m
    ):
        return fit, False
    return None, False


def frame_homographies(
    calls_df: pl.DataFrame,
    views: pl.DataFrame,
    times: dict[int, float],
    config: VisionConfig,
    revote: bool = False,
    run_every: int | None = None,
) -> dict[int, np.ndarray | None]:
    """frame_id -> the homography in use on that frame (None: not ok), as the pipeline
    decides it: fits offered in call order, the filter reset on each new segment.

    run_every: the run's keypoints_every. A config.keypoints_every that's a multiple of it
    keeps every k-th used call of a segment (the gate still saw them all: an approximation)."""
    filt = HomographyFilter(
        config.homography_window, config.max_homography_jump_m, config.homography_max_age_s
    )
    k = 1
    if run_every is not None and config.keypoints_every != run_every:
        if config.keypoints_every % run_every:
            raise ValueError(
                f"keypoints_every {config.keypoints_every} isn't a multiple of the run's "
                f"{run_every}"
            )
        k = config.keypoints_every // run_every
    calls: dict[int, list[dict]] = {}
    for row in calls_df.iter_rows(named=True):
        calls.setdefault(row["frame_id"], []).append(row)
    pitch = template(config.circle_kp_x_m)
    segment, n_seg = None, 0
    out = {}
    for frame_id, view in views.select("frame_id", "view").sort("frame_id").iter_rows():
        for c in calls.get(frame_id, []):
            if not c["used"]:
                continue
            if c["segment"] != segment:
                filt.reset()
                segment, n_seg = c["segment"], 0
            n_seg += 1
            if (n_seg - 1) % k:
                continue
            fit, blind = _fit(c, config, pitch, revote)
            if fit is not None:
                filt.offer(fit, c["t"])
            elif blind:
                filt.reset()
        out[frame_id] = filt.current(times[frame_id]) if view == MATCH else None
    return out


def replay(
    detections: pl.DataFrame,
    keypoints: pl.DataFrame,
    views: pl.DataFrame,
    times: dict[int, float],
    config: VisionConfig,
    revote: bool = False,
    run_every: int | None = None,
    balls: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(detections with pitch_x / pitch_y / homography_ok / keeper team_cluster redone,
    one row per frame: frame_id, view, homography_ok). keypoints is the stage 4 cache of
    either backend (load()). With balls (load_balls()), the ball rows are stage 5 rerun."""
    hs = frame_homographies(keypoints, views, times, config, revote, run_every)
    if balls is None:
        det = detections.filter(~((pl.col("class") == BALL) & pl.col("tracked_only")))
    else:
        det = detections.filter(pl.col("class") != BALL)
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
    if balls is not None:
        det = pl.concat([det, replay_ball(balls, views, times, hs, config)]).sort(
            "frame_id", maintain_order=True
        )
    frames = views.select("frame_id", "view").with_columns(
        homography_ok=pl.col("frame_id").replace_strict(
            {f: H is not None for f, H in hs.items()}, return_dtype=pl.Boolean
        )
    )
    return det, frames


def _ball_track(
    balls: pl.DataFrame,
    views: pl.DataFrame,
    times: dict[int, float],
    hs: dict[int, np.ndarray | None],
    config: VisionConfig,
):
    """Stage 5 rerun from balls.parquet: (match_id, frame_id, segment, H, Ball) for every
    frame with a ball. Segments and detection frames follow the view, as in
    VisionPipeline.step."""
    by_frame: dict[int, list[Detection]] = {}
    for frame_id, x1, y1, x2, y2, conf in balls.select(
        "frame_id", "x1", "y1", "x2", "y2", "det_confidence"
    ).iter_rows():
        by_frame.setdefault(frame_id, []).append(Detection((x1, y1, x2, y2), BALL, conf))
    track = BallTrack(config)
    segment, step, before = -1, 0, None
    for match_id, frame_id, view in (
        views.select("match_id", "frame_id", "view").sort("frame_id").iter_rows()
    ):
        if view != MATCH:
            before = view
            continue
        if before != MATCH:
            segment, step = segment + 1, 0
            track.reset()
        before = view
        detect_now = step % config.detect_every == 0
        step += 1
        H = hs[frame_id]

        def to_pitch(px, H=H):
            if H is None:
                return None, None
            xy = to_02(project(H, [px]), config.home_attacks_tv_right_p1)[0]
            if not on_pitch(xy, config.max_off_pitch_m):
                return None, None
            return float(xy[0]), float(xy[1])

        width_at = None
        if H is not None:
            Hinv = np.linalg.inv(H)

            def width_at(px, H=H, Hinv=Hinv):
                return expected_width(Hinv, H, px)

        b = track.update(
            times[frame_id],
            by_frame.get(frame_id, []) if detect_now else [],
            to_pitch,
            H is not None,
            width_at,
        )
        if b is not None:
            yield match_id, frame_id, segment, H, b


def replay_ball(
    balls: pl.DataFrame,
    views: pl.DataFrame,
    times: dict[int, float],
    hs: dict[int, np.ndarray | None],
    config: VisionConfig,
) -> pl.DataFrame:
    """Stage 5 rerun from balls.parquet: the ball rows of the detections cache."""
    rows = []
    for match_id, frame_id, segment, H, b in _ball_track(balls, views, times, hs, config):
        x1, y1, x2, y2 = b.box
        rows.append(
            {
                "match_id": match_id,
                "frame_id": frame_id,
                "object_id": f"{segment}-ball",
                "class": BALL,
                "team_cluster": None,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "det_confidence": None if b.interpolated else b.confidence,
                "tracked_only": b.interpolated,
                "pitch_x": b.x,
                "pitch_y": b.y,
                "homography_ok": H is not None,
                "homography_err_m": None,
            }
        )
    return pl.DataFrame(rows, schema=DETECTIONS_SCHEMA)


def ball_objects(
    balls: pl.DataFrame,
    views: pl.DataFrame,
    times: dict[int, float],
    hs: dict[int, np.ndarray | None],
    config: VisionConfig,
    image_size: tuple[int, int],
) -> pl.DataFrame:
    """The ball's 02 objects rows as GameStateWriter.add writes them (no vx / vy), from
    stage 5 rerun: rows without a pitch position never reach game state."""
    w, h = image_size
    rows = []
    for match_id, frame_id, segment, _, b in _ball_track(balls, views, times, hs, config):
        if b.x is None:
            continue
        o = VisionObject(
            f"{segment}-ball",
            BALL,
            None,
            None,
            b.x,
            b.y,
            b.confidence,
            b.interpolated,
            b.interpolated or b.airborne,
            b.box,
            frac_box(b.box, w, h),
            b.airborne,
        )
        rows.append(
            {
                "match_id": match_id,
                "frame_id": frame_id,
                "object_id": o.object_id,
                "object_type": OBJECT_TYPES[BALL],
                "team": None,
                "player_id": None,
                "x": b.x,
                "y": b.y,
                "z": None,
                "visible": on_screen(o),
                "interpolated": o.interpolated,
                "confidence": b.confidence,
            }
        )
    return pl.DataFrame(rows, schema=OBJECTS_SCHEMA)


def airborne_share(
    balls: pl.DataFrame,
    views: pl.DataFrame,
    times: dict[int, float],
    hs: dict[int, np.ndarray | None],
    config: VisionConfig,
) -> tuple[int, int]:
    """(ball rows written in the air, ball rows with a pitch position): 10-ball 2g."""
    rows = [b for *_, b in _ball_track(balls, views, times, hs, config) if b.x is not None]
    return sum(b.airborne for b in rows), len(rows)


def write_variant(
    match_id: str,
    config: VisionConfig,
    out_root: Path,
    cache_dir: Path,
    gamestate_dir: Path,
    image_size: tuple[int, int] | None = None,
) -> tuple[Path, Path]:
    """Stage 5 rerun at config into out_root/{vision_cache,gamestate}/match_id: the run's
    people as they are, its ball rows replaced, velocities and the 02 check redone. The
    run's own dirs are only read. image_size: the frames' (w, h), else camera.parquet's."""
    cache, gs = Path(cache_dir) / match_id, Path(gamestate_dir) / match_id
    base = run_config(cache)
    changed = [
        f.name
        for f in dataclasses.fields(VisionConfig)
        if getattr(base, f.name) != getattr(config, f.name)
    ]
    if any(not name.startswith("ball_") for name in changed):
        raise SystemExit(f"a ball variant changes ball_* fields only, not {changed}")
    balls = load_balls(cache)
    if balls is None:
        raise SystemExit(f"{match_id}: no balls.parquet, stage 5 can't be rerun")
    det, calls, views, times = load(cache, gs)
    if image_size is None:
        if "img_w" not in calls.columns:
            raise SystemExit(f"{match_id}: no camera.parquet to read the image size from")
        image_size = calls.select("img_w", "img_h").row(0)
    hs = frame_homographies(calls, views, times, config, run_every=base.keypoints_every)
    vcache = Path(out_root) / "vision_cache" / match_id
    vgs = Path(out_root) / "gamestate" / match_id
    for d in (vcache, vgs):
        d.mkdir(parents=True, exist_ok=True)
    for name in CACHE_FILES:
        if (cache / name).exists():
            shutil.copyfile(cache / name, vcache / name)
    for name in GS_FILES:
        shutil.copyfile(gs / name, vgs / name)
    people = det.filter(pl.col("class") != BALL)
    pl.concat([people, replay_ball(balls, views, times, hs, config)]).sort(
        "frame_id", maintain_order=True
    ).write_parquet(vcache / "detections.parquet")
    run = json.loads((cache / "run.json").read_text())
    run.update(config=config.__dict__, variant_of=match_id)
    (vcache / "run.json").write_text(json.dumps(run, indent=2, default=str))
    old = pl.read_parquet(gs / "objects.parquet").drop("vx", "vy")
    objects = pl.concat(
        [
            old.filter(pl.col("object_type") != OBJECT_TYPES[BALL]),
            ball_objects(balls, views, times, hs, config, image_size),
        ]
    ).sort("frame_id", maintain_order=True)
    frames = pl.read_parquet(vgs / "frames.parquet")
    fps = pl.read_parquet(vgs / "match.parquet")["native_fps"][0]
    causal_velocities(objects, frames, fps=fps).write_parquet(vgs / "objects.parquet")
    errors = validate_match(vgs)
    if errors:
        raise SystemExit(f"{match_id}: the variant fails 02 validation: {errors}")
    return vcache, vgs


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
        # PnLCalib runs write camera.parquet instead of keypoints.parquet (03 Diagnostics)
        pl.read_parquet(
            cache / "camera.parquet"
            if (cache / "camera.parquet").exists()
            else cache / "keypoints.parquet"
        ),
        pl.read_parquet(cache / "view.parquet"),
        times,
    )


def load_balls(cache: Path) -> pl.DataFrame | None:
    """balls.parquet, or None for runs before it (03 Diagnostics)."""
    path = cache / "balls.parquet"
    return pl.read_parquet(path) if path.exists() else None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="vision.replay")
    ap.add_argument("--match-id", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE")
    ap.add_argument("--revote", action="store_true", help="PnLCalib: vote cameras again")
    ap.add_argument("--out", type=Path, help="write the replayed detections here")
    ap.add_argument("--variant", help="stage 5 rerun at --set into <variant-root>/<this tag>")
    ap.add_argument("--variant-root", type=Path, default=VARIANT_ROOT)
    ap.add_argument("--cache-dir", type=Path, default=Path("data/vision_cache"))
    ap.add_argument("--gamestate-dir", type=Path, default=Path("data/gamestate"))
    args = ap.parse_args(argv)

    cache = args.cache_dir / args.match_id
    base = run_config(cache)
    config = with_overrides(base, args.set)
    if args.variant:
        out_root = args.variant_root / args.variant
        paths = write_variant(args.match_id, config, out_root, args.cache_dir, args.gamestate_dir)
        print("variant ->", *paths)
        return
    det, keypoints, views, times = load(cache, args.gamestate_dir / args.match_id)
    _, before = replay(det, keypoints, views, times, base)
    out, after = replay(
        det,
        keypoints,
        views,
        times,
        config,
        args.revote,
        run_every=base.keypoints_every,
        balls=load_balls(cache),
    )
    n_match = after.filter(pl.col("view") == MATCH).height
    for name, frames in [("run config", before), ("replayed", after)]:
        ok = frames["homography_ok"].sum()
        print(f"{name}: homography ok on {ok} of {n_match} match frames")
    if args.out:
        out.write_parquet(args.out)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
