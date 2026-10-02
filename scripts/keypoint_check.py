"""Per-keypoint template check on a bench clip (vision-bench review, fix 2).

    PYTHONPATH=. python scripts/keypoint_check.py --clip vb01-arg-fra

On each scored frame with a stage 4 call, fits a homography from the PFF players instead of
the pitch keypoints: matched vision feet (pixels) -> their PFF positions (TV frame). Each
confident keypoint is sent through it and compared with the landmark's real position (31/32
at the circle edge, 9.15 m, whatever circle_kp_x_m says, so clips compare). A landmark
that lands off by the same amount frame after frame is misplaced in the template (or the
keypoint model is biased on it); scattered offsets point at the fit instead.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import polars as pl

from scripts.team_crops import CACHE, matched_frames
from vision import bench, replay
from vision.pitch import CIRCLE_R, project, template, to_02

REAL = template(CIRCLE_R)  # offsets are against this, not the fitted template
MIN_PAIRS = 8  # matched players before a frame's PFF homography is trusted
RANSAC_M = 1.0
MAX_FIT_M = 0.6  # median residual of that fit
NEAR_M = 20.0  # keypoints this close to a matched player: inside what the players pin down

SIDE = ["far corner", "far box", "far 6-yd", "near 6-yd", "near box", "near corner"]
NAMES = (
    [f"L goal line {s}" for s in SIDE]
    + ["L 6-yd front far", "L 6-yd front near", "L pen spot"]
    + [f"L box front {s}" for s in ("far", "at 6-yd far", "at 6-yd near", "near")]
    + ["halfway far", "halfway circle far", "halfway circle near", "halfway near"]
    + [f"R box front {s}" for s in ("far", "at 6-yd far", "at 6-yd near", "near")]
    + ["R pen spot", "R 6-yd front far", "R 6-yd front near"]
    + [f"R goal line {s}" for s in SIDE]
    + ["circle left", "circle right"]
)


def offsets(clip: dict) -> pl.DataFrame:
    cache = CACHE / clip["clip_id"]
    config = replay.run_config(cache)
    det = pl.read_parquet(cache / "detections.parquet")
    kps = pl.read_parquet(cache / "keypoints.parquet").filter(pl.col("used"))
    kp_by = {r["frame_id"]: r for r in kps.iter_rows(named=True)}
    feet = {
        (f, o): ((x1 + x2) / 2, y2)
        for f, o, x1, x2, y2 in det.select("frame_id", "object_id", "x1", "x2", "y2").iter_rows()
    }
    rows = []
    for frame_id, v, t, pairs in matched_frames(clip):
        kp = kp_by.get(frame_id)
        if kp is None or len(pairs) < MIN_PAIRS:
            continue
        src = np.array([feet[(frame_id, v["object_id"][j])] for _, j, _ in pairs])
        dst = to_02(
            t.select("x", "y").to_numpy()[[i for i, _, _ in pairs]], config.home_attacks_tv_right_p1
        )
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_M)
        if H is None:
            continue
        inl = mask.ravel().astype(bool)
        res = np.linalg.norm(project(H, src[inl]) - dst[inl], axis=1)
        if inl.sum() < MIN_PAIRS - 2 or np.median(res) > MAX_FIT_M:
            continue
        conf = np.array(kp["kp_conf"])
        keep = np.flatnonzero(conf >= config.min_keypoint_conf)
        if not len(keep):
            continue
        px = np.column_stack([kp["kp_x"], kp["kp_y"]])[keep]
        at = project(H, px)
        near = np.linalg.norm(REAL[keep][:, None] - dst[inl][None], axis=2).min(axis=1)
        for k, (x, y), d in zip(keep, at, near):
            rows.append((frame_id, int(k), x - REAL[k, 0], y - REAL[k, 1], d, len(keep)))
    return pl.DataFrame(rows, schema=["frame_id", "kp", "dx", "dy", "near_m", "n_kp"], orient="row")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", type=Path, help="write the per-keypoint offsets as parquet")
    args = ap.parse_args()
    (clip,) = [c for c in bench.load_manifest(bench.MANIFEST) if c["clip_id"] == args.clip]
    off = offsets(clip)
    if args.out:
        off.write_parquet(args.out)
    print(f"{args.clip}: {off['frame_id'].n_unique()} frames with a PFF fit, TV frame meters")
    print("offset = where PFF's players put the keypoint minus the real landmark (31/32: 9.15 m)")
    near = off.filter(pl.col("near_m") <= NEAR_M)
    table = (
        near.group_by("kp")
        .agg(
            n=pl.len(),
            dx=pl.col("dx").median(),
            dy=pl.col("dy").median(),
            spread=(
                (pl.col("dx") - pl.col("dx").median()) ** 2
                + (pl.col("dy") - pl.col("dy").median()) ** 2
            )
            .sqrt()
            .median(),
        )
        .sort("kp")
    )
    print(f"keypoints within {NEAR_M:.0f} m of a matched player:")
    print(f"  {'#':>2} {'landmark':24s} {'n':>5} {'dx':>6} {'dy':>6} {'spread':>6}")
    for k, n, dx, dy, sp in table.iter_rows():
        print(f"  {k + 1:2d} {NAMES[k]:24s} {n:5d} {dx:6.2f} {dy:6.2f} {sp:6.2f}")
    print(json.dumps({"all_n": off.height, "near_n": near.height}))


if __name__ == "__main__":
    main()
