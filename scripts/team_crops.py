"""Offline kit-color team test (03 stage 3): feature variants scored the way the pipeline runs.

    PYTHONPATH=. python scripts/team_crops.py --clip vb01-arg-fra
    PYTHONPATH=. python scripts/team_crops.py --cache smoke03 --video <mp4>   # no truth

Rereads the run's video, takes the torso crop of every confident player detection in the
detections cache, and per variant: fits 2-means on the warmup crops only (the frames before
the run's first team prediction), predicts the rest, and gives each track the majority of
its first TEAM_VOTES predictions. With a bench clip, each track is labeled by its majority
PFF team over all frames where it matched a PFF player, matched after removing the frame's
affine (the homography's stretch pairs neighbours across teams otherwise). Without truth: cluster balance only, plus contact
sheets of crops per cluster to check by eye.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import polars as pl

from vision import bench, replay
from vision.pipeline import TEAM_VOTES, torso_crop
from vision.types import GOALKEEPER, PLAYER

CACHE = Path("data/vision_cache")
GS = Path("data/gamestate")
VIDEOS = Path("data/vision_bench/videos.json")
MIN_TRACK_MATCHES = 10  # matched frames before a track gets a PFF label
MIN_TRACK_SHARE = 0.7  # of those, on one team
AFFINE_ROUNDS = 2
LABEL_GATE_M = 1.5  # after the affine is removed
N_INIT = 10
SHEET_N = 48


def not_grass_lab(crop: np.ndarray) -> np.ndarray:
    """The non-grass pixels in Lab, as KitColorTeams.features takes them."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    keep = ~((hsv[..., 0] >= 30) & (hsv[..., 0] <= 90) & (hsv[..., 1] >= 40))
    return cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3)[keep.ravel()].astype(np.float64)


def crop_stats(cache: Path, video: Path, min_conf: float) -> tuple[pl.DataFrame, dict]:
    """Per confident player row: mean and median Lab of its torso crop, plus a few crops kept
    for the contact sheets."""
    run = json.loads((cache / "run.json").read_text())
    det = pl.read_parquet(cache / "detections.parquet")
    rows = det.filter((pl.col("class") == PLAYER) & (pl.col("det_confidence") >= min_conf))
    by_frame = rows.partition_by("frame_id", as_dict=True)
    cap = cv2.VideoCapture(str(video))
    for _ in range(round(run.get("video_start_s", 0.0) * cap.get(cv2.CAP_PROP_FPS))):
        cap.grab()  # frame exact, as vision.run --start-s
    out, crops = [], {}
    for frame_id in range(int(rows["frame_id"].max()) + 1):
        ok, image = cap.read()
        if not ok:
            raise SystemExit(f"{video} ends at frame {frame_id}")
        part = by_frame.get((frame_id,))
        if part is None:
            continue
        for oid, x1, y1, x2, y2 in part.select("object_id", "x1", "y1", "x2", "y2").iter_rows():
            crop = torso_crop(image, (x1, y1, x2, y2))
            px = not_grass_lab(crop) if crop.size else np.zeros((0, 3))
            if not len(px):
                continue
            out.append((frame_id, oid, *px.mean(axis=0), *np.median(px, axis=0)))
            crops[(frame_id, oid)] = cv2.resize(crop, (24, 32))
    cols = ["frame_id", "object_id", "L", "a", "b", "L_med", "a_med", "b_med"]
    return pl.DataFrame(out, schema=cols, orient="row"), crops


def remove_affine(v_xy: np.ndarray, t_xy: np.ndarray, pairs: list) -> np.ndarray:
    """Vision positions after the least-squares affine onto their matched PFF players. The
    homography is off by a per-frame stretch (bench review), which pairs neighbours across
    teams; with it removed the residual is about 0.5 m."""
    i, j = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    src = np.c_[v_xy[j], np.ones(len(j))]
    a, *_ = np.linalg.lstsq(src, t_xy[i], rcond=None)
    return np.c_[v_xy, np.ones(len(v_xy))] @ a


def track_truth(clip: dict) -> dict[str, str]:
    """object_id -> PFF team, by majority over the frames where the track matched someone."""
    cache = CACHE / clip["clip_id"]
    det, _, view, times = replay.load(cache, GS / clip["clip_id"])
    run = json.loads((cache / "run.json").read_text())
    offset = bench.sync_offset(clip)
    pff, truth = bench.load_truth(
        GS / clip["match_id"],
        clip["period"],
        clip["video_start_s"] + offset - 2,
        clip["video_end_s"] + offset + 2,
    )
    aligned = bench.align(view, times, run["video_start_s"], offset, pff)
    aligned = aligned.with_columns(
        mark=pl.Series(bench.mark_labels(aligned["video_s"].to_numpy(), clip["marks"]))
    ).filter(pl.col("mark").is_null() & pl.col("pff_frame_id").is_not_null())
    people = det.filter(
        pl.col("class").is_in([PLAYER, GOALKEEPER]) & pl.col("pitch_x").is_not_null()
    )
    vis_by = people.partition_by("frame_id", as_dict=True)
    truth_by = truth.partition_by("pff_frame_id", as_dict=True)
    seen: dict[str, Counter] = {}
    for frame_id, pff_id in aligned.select("frame_id", "pff_frame_id").iter_rows():
        t, v = truth_by.get((pff_id,)), vis_by.get((frame_id,))
        if t is None or v is None:
            continue
        t_xy, v_xy = t.select("x", "y").to_numpy(), v.select("pitch_x", "pitch_y").to_numpy()
        pairs = bench.match_people(t_xy, v_xy)
        for _ in range(AFFINE_ROUNDS):
            if len(pairs) < 4:
                break
            v_xy = remove_affine(v_xy, t_xy, pairs)
            pairs = [p for p in bench.match_people(t_xy, v_xy) if p[2] <= LABEL_GATE_M]
        for i, j, _ in pairs:
            seen.setdefault(v["object_id"][j], Counter())[t["team"][i]] += 1
    out = {}
    for oid, c in seen.items():
        team, n = c.most_common(1)[0]
        if c.total() >= MIN_TRACK_MATCHES and n / c.total() >= MIN_TRACK_SHARE:
            out[oid] = team
    return out


def two_means(x: np.ndarray, rng: np.random.Generator, iters: int = 20) -> np.ndarray:
    """KitColorTeams.fit's 2-means: N_INIT restarts, lowest inertia wins."""
    best = None
    for _ in range(N_INIT):
        c = x[rng.choice(len(x), 2, replace=False)]
        for _ in range(iters):
            lab = np.linalg.norm(x[:, None] - c[None], axis=2).argmin(axis=1)
            c = np.array([x[lab == k].mean(axis=0) if (lab == k).any() else c[k] for k in (0, 1)])
        inertia = (np.linalg.norm(x[:, None] - c[None], axis=2).min(axis=1) ** 2).sum()
        if best is None or inertia < best[1]:
            best = (c, inertia)
    return best[0]


VARIANTS = {
    # name: (columns, scale by warmup spread, weight on L after scaling)
    "ab mean (current)": (["a", "b"], False, 1.0),
    "ab mean, scaled": (["a", "b"], True, 1.0),
    "Lab mean, scaled": (["L", "a", "b"], True, 1.0),
    "Lab median, scaled": (["L_med", "a_med", "b_med"], True, 1.0),
    "Lab mean, scaled, L x0.5": (["L", "a", "b"], True, 0.5),
    "Lab median, scaled, L x0.5": (["L_med", "a_med", "b_med"], True, 0.5),
    "Lab mean, unscaled": (["L", "a", "b"], False, 1.0),
}


def run_variant(stats: pl.DataFrame, fit_before: int, cols, scaled, l_weight, seed):
    x = stats.select(cols).to_numpy()
    warm = stats["frame_id"].to_numpy() < fit_before
    if scaled:
        mu, sd = x[warm].mean(axis=0), x[warm].std(axis=0)
        x = (x - mu) / np.where(sd > 0, sd, 1.0)
    if cols[0].startswith("L"):
        x[:, 0] *= l_weight
    centers = two_means(x[warm], np.random.default_rng(seed))
    cl = np.linalg.norm(x[:, None] - centers[None], axis=2).argmin(axis=1)
    out = stats.select("frame_id", "object_id").with_columns(cluster=pl.Series(cl))
    after = out.filter(pl.col("frame_id") >= fit_before)
    votes = (
        after.sort("frame_id")
        .group_by("object_id", maintain_order=True)
        .head(TEAM_VOTES)
        .group_by("object_id")
        .agg(vote=pl.col("cluster").mode().sort().first())  # ties: lower number, deterministic
    )
    after = after.sort("frame_id").with_columns(
        ones=pl.col("cluster").cum_sum().over("object_id"),
        n=pl.int_range(1, pl.len() + 1).over("object_id"),
        first=pl.col("cluster").first().over("object_id"),
    )
    # majority of every prediction so far, causal; a tie keeps the track's first one
    running = (
        pl.when(2 * pl.col("ones") > pl.col("n"))
        .then(1)
        .when(2 * pl.col("ones") < pl.col("n"))
        .then(0)
        .otherwise(pl.col("first"))
    )
    after = after.with_columns(running=running).drop("ones", "n", "first")
    return out, after.join(votes, on="object_id", how="left"), int(warm.sum())


def agreement(df: pl.DataFrame, col: str, truth: dict[str, str]) -> tuple[float, int]:
    """Share agreeing with PFF under the better cluster -> home mapping."""
    lab = df.filter(pl.col("object_id").is_in(list(truth)) & pl.col(col).is_not_null())
    home = np.array([truth[o] == "home" for o in lab["object_id"]])
    c = lab[col].to_numpy()
    acc = max(((c == k) == home).mean() for k in (0, 1)) if len(c) else float("nan")
    return float(acc), len(c)


def contact_sheet(path: Path, out: pl.DataFrame, crops: dict) -> None:
    rows = []
    for k in (0, 1):
        sub = out.filter(pl.col("cluster") == k)
        pick = sub.sample(min(SHEET_N, sub.height), seed=0) if sub.height else sub
        tiles = [crops[(f, o)] for f, o in pick.select("frame_id", "object_id").iter_rows()]
        tiles += [np.zeros((32, 24, 3), np.uint8)] * (SHEET_N - len(tiles))
        rows.append(np.hstack(tiles))
        rows.append(np.full((4, 24 * SHEET_N, 3), 255, np.uint8))
    cv2.imwrite(str(path), np.vstack(rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", help="bench clip_id: video and truth from the manifest")
    ap.add_argument("--cache", help="cache dir name without truth (e.g. smoke03)")
    ap.add_argument("--video", type=Path)
    ap.add_argument("--sheets", type=Path, help="write per-variant contact sheets here")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    name = args.clip or args.cache
    cache = CACHE / name
    config = replay.run_config(cache)
    truth = None
    if args.clip:
        (clip,) = [c for c in bench.load_manifest(bench.MANIFEST) if c["clip_id"] == args.clip]
        video = Path(json.loads(VIDEOS.read_text())[args.clip])
        truth = track_truth(clip) if clip["match_id"] else None
    else:
        video = args.video
    det = pl.read_parquet(cache / "detections.parquet")
    fit_before = int(det.filter(pl.col("team_cluster").is_not_null())["frame_id"].min())
    stats, crops = crop_stats(cache, video, config.min_det_conf)
    print(f"{name}: {stats.height} crops, fit before frame {fit_before}")
    if truth is not None:
        teams = Counter(truth.values())
        rows = stats.filter(pl.col("object_id").is_in(list(truth))).height
        print(f"labeled tracks {len(truth)} ({dict(teams)}), {rows} crops on them")

    for vname, (cols, scaled, lw) in VARIANTS.items():
        res = []
        for seed in range(args.seeds):
            out, after, n_warm = run_variant(stats, fit_before, cols, scaled, lw, seed)
            small = min(np.bincount(after["cluster"].to_numpy(), minlength=2)) / after.height
            if truth is None:
                res.append((small, np.nan, np.nan, np.nan))
            else:
                res.append(
                    (
                        small,
                        *(agreement(after, c, truth)[0] for c in ("cluster", "vote", "running")),
                    )
                )
        small, crop_acc, vote_acc, run_acc = np.array(res).T
        print(
            f"  {vname:28s} warmup {n_warm:4d}  smaller cluster {small.mean():.1%}"
            f"  per crop {np.mean(crop_acc):.1%} (min {np.min(crop_acc):.1%})"
            f"  first {TEAM_VOTES} votes {np.mean(vote_acc):.1%} (min {np.min(vote_acc):.1%})"
            f"  running majority {np.mean(run_acc):.1%} (min {np.min(run_acc):.1%})"
        )
        if args.sheets:
            args.sheets.mkdir(parents=True, exist_ok=True)
            slug = "".join(ch if ch.isalnum() else "_" for ch in vname)
            out, _, _ = run_variant(stats, fit_before, cols, scaled, lw, 0)
            contact_sheet(args.sheets / f"{name}_{slug}.png", out, crops)


if __name__ == "__main__":
    main()
