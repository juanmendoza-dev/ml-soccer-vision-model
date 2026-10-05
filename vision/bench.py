"""Vision benchmark scorecard on PFF truth (07 "Vision benchmark (W0)").

    python -m vision.bench                                   # every clip, run config
    python -m vision.bench --set max_homography_err_m=2      # replayed with an override
    python -m vision.bench --grid min_inliers=4,6 --grid max_homography_err_m=1,2,3

Each clip's vision run lives in data/vision_cache/<clip_id> (vision.run with
--start-s). People truth is PFF's VISIBLE players and keepers at the PFF frame
nearest each vision frame, after the clip's sync offset. Frames inside the hand
marks (replay, closeup, other) aren't scored; they count toward false live.
Everything is replayed from the caches (vision.replay), so the run config and a
sweep are scored the same way.
"""

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment

from vision import replay
from vision.pitch import project, to_02
from vision.types import BALL, GOALKEEPER, MATCH, PLAYER

MANIFEST = Path("data/splits/vision_benchmark.json")
MARK_LABELS = ("replay", "closeup", "other")
PFF_FPS = 29.97
GATE_M = 5.0  # one-to-one matches further apart than this aren't the same person
NEAR_M = 2.0
MAX_DRIFT_S = 0.1  # sync pairs disagreeing by more: the source video's fps is off
GEOMETRY_TARGET = 0.17  # roadmap: frames without geometry at or under smoke04's
MAX_PREROLL_S = 30.0  # vision may start this much before the clip, unscored
TIE_PT = 0.005  # within this much of the best within-2 m, less geometry missing wins
# 07 Ball labels / Ball score
BALL_LABELS = Path("data/splits/vision_ball_labels.json")
BALL_EVERY = 5  # label every 5th source frame
BALL_NONE, BALL_UNSURE = "none", "unsure"
BALL_R_PX = 15.0  # a hit: about one ball diameter at 1080p on the bench
BALL_RS = (10.0, 25.0)  # printed next to it
# per clip, source frames: auto-accepted by ball_click --assist, auto frames shown in the
# spot check, flagged frames a person looked at again (10-ball 1c)
LABEL_LISTS = ("auto", "spot_checked", "flag_checked")


def load_manifest(path: Path) -> list[dict]:
    m = json.loads(path.read_text())
    clips = m["clips"]
    ids = [c["clip_id"] for c in clips]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate clip_id in the manifest")
    for c in clips:
        name = c["clip_id"]
        if not c["video_start_s"] < c["video_end_s"]:
            raise ValueError(f"{name}: video_start_s must be before video_end_s")
        # match_id null: footage PFF doesn't have, scored on geometry and marks only
        if c["match_id"] is not None and not c["sync"]:
            raise ValueError(f"{name}: needs at least one sync pair")
        for mk in c["marks"]:
            if mk["label"] not in MARK_LABELS:
                raise ValueError(f"{name}: mark label {mk['label']!r} not in {MARK_LABELS}")
            if not mk["start_s"] < mk["end_s"]:
                raise ValueError(f"{name}: mark {mk} ends before it starts")
    return clips


def sync_offset(clip: dict) -> float:
    """PFF timestamp_s = source video seconds + offset."""
    offsets = [s["timestamp_s"] - s["video_s"] for s in clip["sync"]]
    if max(offsets) - min(offsets) > MAX_DRIFT_S:
        raise ValueError(
            f"{clip['clip_id']}: sync pairs drift {max(offsets) - min(offsets):.2f} s; "
            "the video's frame rate is off"
        )
    return float(np.mean(offsets))


def mark_labels(video_s: np.ndarray, marks: list[dict]) -> list[str | None]:
    labels = [None] * len(video_s)
    for mk in marks:
        for i in np.flatnonzero((video_s >= mk["start_s"]) & (video_s < mk["end_s"])):
            labels[i] = mk["label"]
    return labels


def ball_label_frames(clip: dict, fps: float, every: int = BALL_EVERY) -> list[int]:
    """Source-video frame indices to label for the ball (07 Ball labels): multiples of
    every inside the clip, outside the marks."""
    start, end = clip["video_start_s"], clip["video_end_s"]
    idx = np.array(
        [
            i
            for i in range(math.floor(start * fps), math.ceil(end * fps) + 1)
            if i % every == 0 and start <= i / fps < end
        ]
    )
    marked = mark_labels(idx / fps, clip["marks"])
    return [int(i) for i, m in zip(idx, marked, strict=True) if m is None]


def load_ball_labels(path: Path = BALL_LABELS) -> dict:
    """clip_id -> {video_sha256, every, labels: {source frame index: [x, y] | none | unsure}}."""
    if not path.exists():
        return {}
    clips = json.loads(path.read_text())["clips"]
    for c in clips.values():
        c["labels"] = {int(k): v for k, v in c["labels"].items()}
        for v in c["labels"].values():
            if not (v in (BALL_NONE, BALL_UNSURE) or (isinstance(v, list) and len(v) == 2)):
                raise ValueError(f"bad ball label {v!r}")
        for k in LABEL_LISTS:
            if k in c and not all(isinstance(i, int) for i in c[k]):
                raise ValueError(f"{k} must list source frame indices")
    return clips


def save_ball_labels(clips: dict, path: Path = BALL_LABELS) -> None:
    """One label per line, so a diff shows what was clicked. Written whole, then swapped in."""
    lines = ['{"version": 1, "clips": {']
    for n, (clip_id, c) in enumerate(sorted(clips.items())):
        extra = "".join(f'"{k}": {json.dumps(sorted(c[k]))}, ' for k in LABEL_LISTS if k in c)
        if c.get("auto_off"):
            extra += '"auto_off": true, '
        lines.append(
            f'  "{clip_id}": {{"video_sha256": "{c["video_sha256"]}", "every": {c["every"]}, '
            f'{extra}"labels": {{'
        )
        items = sorted(c["labels"].items())
        for k, (idx, v) in enumerate(items):
            comma = "," if k < len(items) - 1 else ""
            lines.append(f'    "{idx}": {json.dumps(v)}{comma}')
        lines.append("  }}" + ("," if n < len(clips) - 1 else ""))
    lines.append("}}")
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(path)


def load_truth(
    gamestate: Path, period: int, t0: float, t1: float
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """PFF frames of the period between t0 and t1 (timestamp_s), and their VISIBLE
    players and keepers."""
    frames = (
        pl.scan_parquet(gamestate / "frames.parquet")
        .filter((pl.col("period") == period) & pl.col("timestamp_s").is_between(t0, t1))
        .select(pff_frame_id="frame_id", pff_t="timestamp_s")
        .sort("pff_t")
        .collect()
    )
    objects = (
        pl.scan_parquet(gamestate / "objects.parquet")
        .filter(
            pl.col("frame_id").is_in(frames["pff_frame_id"].implode())
            & pl.col("object_type").is_in([PLAYER, GOALKEEPER])
            & pl.col("visible")
        )
        .select(pff_frame_id="frame_id", kind="object_type", team="team", x="x", y="y")
        .collect()
    )
    return frames, objects


def align(
    frames: pl.DataFrame,
    times: dict[int, float],
    video_start_s: float,
    offset: float,
    pff: pl.DataFrame | None,
) -> pl.DataFrame:
    """Vision frames + video_s and the nearest PFF frame (null if none within half a frame)."""
    out = frames.with_columns(
        video_s=pl.col("frame_id").replace_strict(times, return_dtype=pl.Float64) + video_start_s
    ).with_columns(pff_t=pl.col("video_s") + offset)
    if pff is None:
        return out.with_columns(pff_frame_id=pl.lit(None, pl.Int64))
    return (
        out.sort("pff_t")
        .join_asof(pff, on="pff_t", strategy="nearest", tolerance=0.5 / PFF_FPS + 1e-6)
        .sort("frame_id")
    )


def match_people(truth: np.ndarray, vis: np.ndarray) -> list[tuple[int, int, float]]:
    """One-to-one by distance (Hungarian), pairs within GATE_M: (truth i, vision j, meters)."""
    if not len(truth) or not len(vis):
        return []
    d = np.linalg.norm(truth[:, None, :] - vis[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(np.where(d <= GATE_M, d, 1e6))
    return [(int(i), int(j), float(d[i, j])) for i, j in zip(rows, cols) if d[i, j] <= GATE_M]


def frame_pairs(aligned: pl.DataFrame, det: pl.DataFrame, truth: pl.DataFrame | None) -> dict:
    """Per scored frame: truth count, vision count and the matched pairs. Without truth
    (no PFF match), every unmarked clip frame is scored for geometry only."""
    scored = aligned.filter(pl.col("in_clip") & pl.col("mark").is_null())
    if truth is None:
        return {
            "n_scored": scored.height,
            "frames": [
                {
                    "geometry": ok,
                    "match_view": view == MATCH,
                    "truth": False,
                    "n_truth": 0,
                    "n_vis": 0,
                    "pairs": [],
                }
                for ok, view in scored.select("geometry", "view").iter_rows()
            ],
        }
    scored = scored.filter(pl.col("pff_frame_id").is_not_null())
    people = det.filter(
        pl.col("class").is_in([PLAYER, GOALKEEPER]) & pl.col("pitch_x").is_not_null()
    )
    vis_by = people.partition_by("frame_id", as_dict=True)
    truth_by = truth.partition_by("pff_frame_id", as_dict=True)
    out = {"n_scored": scored.height, "frames": []}
    for frame_id, pff_id, ok, view in scored.select(
        "frame_id", "pff_frame_id", "geometry", "view"
    ).iter_rows():
        t = truth_by.get((pff_id,))
        v = vis_by.get((frame_id,))
        t_xy = t.select("x", "y").to_numpy() if t is not None else np.zeros((0, 2))
        v_xy = v.select("pitch_x", "pitch_y").to_numpy() if v is not None else np.zeros((0, 2))
        pairs = match_people(t_xy, v_xy)
        out["frames"].append(
            {
                "geometry": ok,
                "match_view": view == MATCH,
                "truth": True,
                "n_truth": len(t_xy),
                "n_vis": len(v_xy),
                "pairs": [
                    (
                        d,
                        t["kind"][i],
                        t["team"][i],
                        v["class"][j],
                        v["team_cluster"][j],
                    )
                    for i, j, d in pairs
                ],
            }
        )
    return out


def team_scores(pairs: list[tuple], home_cluster: int | None) -> dict:
    """Outfield and keepers apart: accuracy where vision has a cluster, and coverage.
    Without home_cluster, the mapping that agrees more on outfield players is used."""
    by_kind = {
        kind: [(team, c) for _, k, team, _, c in pairs if k == kind]
        for kind in (PLAYER, GOALKEEPER)
    }

    def correct(sub, hc):
        return sum((team == "home") == (c == hc) for team, c in sub if c is not None)

    hc = home_cluster
    if hc is None:
        hc = max((0, 1), key=lambda k: correct(by_kind[PLAYER], k))
    out = {
        kind: {
            "pairs": len(sub),
            "assigned": sum(c is not None for _, c in sub),
            "correct": correct(sub, hc),
        }
        for kind, sub in by_kind.items()
    }
    out["home_cluster"] = hc
    out["home_cluster_picked"] = home_cluster is None
    return out


def score_clip(
    clip: dict,
    det: pl.DataFrame,
    frames: pl.DataFrame,
    times: dict,
    run_start_s: float,
    gamestate: Path,
    fps: float,
    offset_check: bool = False,
) -> dict:
    if gamestate is None:
        pff = truth = None
        offset = 0.0
    else:
        offset = sync_offset(clip)
        t0 = clip["video_start_s"] + offset - 2
        t1 = clip["video_end_s"] + offset + 2
        pff, truth = load_truth(gamestate, clip["period"], t0, t1)

    def pairs_at(off):
        aligned = align(frames, times, run_start_s, off, pff)
        aligned = aligned.with_columns(
            mark=pl.Series(
                mark_labels(aligned["video_s"].to_numpy(), clip["marks"]), dtype=pl.String
            ),
            geometry=(pl.col("view") == MATCH) & pl.col("homography_ok"),
            # pre-roll before video_start_s lets the gate and team warmup settle unscored
            in_clip=pl.col("video_s").is_between(
                clip["video_start_s"], clip["video_end_s"], closed="left"
            ),
        )
        return aligned, frame_pairs(aligned, det, truth)

    aligned, fp = pairs_at(offset)
    out = summarize(clip, aligned, fp, fps)
    if offset_check and truth is not None:
        out["offset_check_s"] = best_offset(pairs_at, offset) - offset
    return out


def best_offset(pairs_at, offset: float) -> float:
    """The offset within ±1 s (one PFF frame steps) with the lowest median matched error
    on frames with geometry. A check on the sync, never used to score (07)."""
    best, best_err = offset, float("inf")
    for k in range(-30, 31):
        off = offset + k / PFF_FPS
        _, fp = pairs_at(off)
        d = [p[0] for f in fp["frames"] if f["geometry"] for p in f["pairs"]]
        if d and np.median(d) < best_err:
            best, best_err = off, float(np.median(d))
    return best


def summarize(clip: dict, aligned: pl.DataFrame, fp: dict, fps: float) -> dict:
    fr = fp["frames"]
    pairs = [p for f in fr for p in f["pairs"]]
    geo = [f for f in fr if f["geometry"]]
    d_all = np.array([p[0] for p in pairs])
    d_geo = np.array([p[0] for f in geo for p in f["pairs"]])
    false_live = (
        aligned.filter(pl.col("in_clip") & pl.col("mark").is_not_null() & pl.col("geometry"))
        .group_by("mark")
        .len()
    )
    return {
        "clip_id": clip["clip_id"],
        "sync_coarse": clip.get("sync_coarse", False),
        "n_scored": fp["n_scored"],
        "n_truth_frames": sum(f["truth"] for f in fr),
        "n_match_view": sum(f["match_view"] for f in fr),
        "n_geometry": len(geo),
        "n_truth": sum(f["n_truth"] for f in fr),
        "n_truth_geo": sum(f["n_truth"] for f in geo),
        "n_vis": sum(f["n_vis"] for f in fr),
        "n_pairs": len(pairs),
        "n_near": int((d_all <= NEAR_M).sum()),
        "n_near_geo": int((d_geo <= NEAR_M).sum()),
        "errors": d_all.tolist(),
        "errors_geo": d_geo.tolist(),
        "teams": team_scores(pairs, clip.get("home_cluster")),
        "false_live_s": {k: n / fps for k, n in false_live.iter_rows()},
    }


def pooled(scores: list[dict]) -> dict:
    s = {
        k: sum(c[k] for c in scores)
        for k in (
            "n_scored",
            "n_truth_frames",
            "n_match_view",
            "n_geometry",
            "n_truth",
            "n_truth_geo",
            "n_vis",
            "n_pairs",
            "n_near",
            "n_near_geo",
        )
    }
    s["errors"] = [e for c in scores for e in c["errors"]]
    s["errors_geo"] = [e for c in scores for e in c["errors_geo"]]
    return s


def headline(s: dict) -> dict:
    def q(v, p):
        return float(np.quantile(v, p)) if v else float("nan")

    def frac(a, b):
        return a / b if b else float("nan")

    return {
        "geometry_missing": 1 - frac(s["n_geometry"], s["n_scored"]),
        # the two parts: the view gate said other, or match view with no accepted fit.
        # The second is smoke04's 17% (65 of 375 match-view frames) and what the sweep moves
        "view_other": 1 - frac(s["n_match_view"], s["n_scored"]),
        "homography_rejected": 1 - frac(s["n_geometry"], s["n_match_view"]),
        "within_2m": frac(s["n_near"], s["n_truth"]),
        "median_m": q(s["errors"], 0.5),
        "p90_m": q(s["errors"], 0.9),
        "within_2m_geo": frac(s["n_near_geo"], s["n_truth_geo"]),
        "median_geo_m": q(s["errors_geo"], 0.5),
        "p90_geo_m": q(s["errors_geo"], 0.9),
        "unmatched_per_frame": frac(s["n_vis"] - s["n_pairs"], s["n_truth_frames"]),
    }


def pick(results: list[tuple[dict, dict]]) -> tuple[dict, dict] | None:
    """(config overrides, pooled headline) with the highest within-2 m and match-view
    frames without an accepted homography at or under the target; within TIE_PT of it,
    the fewest of those."""
    ok = [r for r in results if r[1]["homography_rejected"] <= GEOMETRY_TARGET]
    if not ok:
        return None
    best = max(r[1]["within_2m"] for r in ok)
    close = [r for r in ok if r[1]["within_2m"] >= best - TIE_PT]
    return min(close, key=lambda r: r[1]["homography_rejected"])


def ball_px(row: dict, H: np.ndarray | None, home_right: bool) -> tuple[float, float]:
    """Where a vision ball row is in the image (07 Ball score): a detection's box center; an
    extrapolated row's pitch x/y back through the homography (its box is the last detection's)."""
    if not row["tracked_only"]:
        return (row["x1"] + row["x2"]) / 2, (row["y1"] + row["y2"]) / 2
    xy = to_02(np.array([[row["pitch_x"], row["pitch_y"]]]), home_right)  # its own inverse
    u, v = project(np.linalg.inv(H), xy)[0]
    return float(u), float(v)


def ball_frames(
    labels: dict[int, list | str],
    skip: int,
    det: pl.DataFrame,
    balls: pl.DataFrame,
    frames: pl.DataFrame,
    hs: dict,
    config,
) -> list[dict]:
    """Per labeled frame inside the run ("unsure" left out): the label, vision's ball row and
    the nearest candidates, as pixel distances. ball_headline scores them at any radius."""
    rows = {r["frame_id"]: r for r in det.filter(pl.col("class") == BALL).iter_rows(named=True)}
    cands: dict[int, list] = {}
    for f, x1, y1, x2, y2, conf in balls.select(
        "frame_id", "x1", "y1", "x2", "y2", "det_confidence"
    ).iter_rows():
        cands.setdefault(f, []).append(((x1 + x2) / 2, (y1 + y2) / 2, conf))
    views = dict(frames.select("frame_id", "view").iter_rows())
    out = []
    for idx, lab in sorted(labels.items()):
        f = idx - skip
        if lab == BALL_UNSURE or f not in views:
            continue
        H = hs.get(f)
        rec = {
            "frame": idx,
            "visible": lab != BALL_NONE,
            "match_view": views[f] == MATCH,
            "geometry": H is not None,
            "row": None,  # det / extrap
            "has_xy": False,
            "d_px": None,
            "d_m": None,
            "cand_hi_px": None,  # nearest candidate >= min_det_conf
            "cand_px": None,  # nearest candidate at any confidence
        }
        r = rows.get(f)
        if r is not None:
            rec["row"] = "extrap" if r["tracked_only"] else "det"
            rec["has_xy"] = r["pitch_x"] is not None
        if rec["visible"]:
            click = np.array(lab, dtype=float)
            if r is not None:  # an extrapolated row always has x/y and a homography
                at_px = ball_px(r, H, config.home_attacks_tv_right_p1)
                rec["d_px"] = float(np.linalg.norm(np.array(at_px) - click))
                if rec["has_xy"]:
                    at = to_02(project(H, [click]), config.home_attacks_tv_right_p1)[0]
                    rec["d_m"] = float(np.linalg.norm(at - [r["pitch_x"], r["pitch_y"]]))
            near = [
                (float(np.hypot(x - click[0], y - click[1])), c) for x, y, c in cands.get(f, [])
            ]
            if near:
                rec["cand_px"] = min(d for d, _ in near)
                hi = [d for d, c in near if c >= config.min_det_conf]
                rec["cand_hi_px"] = min(hi) if hi else None
        out.append(rec)
    return out


MISS_BUCKETS = ("view_other", "no_geometry", "wrong_pick", "low_conf", "drift", "not_detected")


def ball_headline(recs: list[dict], r_px: float = BALL_R_PX) -> dict:
    """07 Ball score at radius r_px: recall, precision, misses by bucket, error in meters."""

    def near(d):
        return d is not None and d <= r_px

    def hit(x):
        return x["visible"] and x["has_xy"] and near(x["d_px"])

    vis = [x for x in recs if x["visible"]]
    usable = [x for x in vis if x["match_view"] and x["geometry"]]
    shown = [x for x in recs if x["has_xy"]]  # rows that reach game state
    misses = dict.fromkeys(MISS_BUCKETS, 0)
    for x in vis:
        if hit(x):
            continue
        if not x["match_view"]:
            misses["view_other"] += 1
        elif not x["geometry"] or (x["row"] and not x["has_xy"] and near(x["d_px"])):
            misses["no_geometry"] += 1
        elif near(x["cand_hi_px"]):
            misses["wrong_pick"] += 1
        elif near(x["cand_px"]):
            misses["low_conf"] += 1
        elif x["row"] == "extrap":
            misses["drift"] += 1
        else:
            misses["not_detected"] += 1

    def frac(a, b):
        return a / b if b else float("nan")

    d_m = [x["d_m"] for x in vis if hit(x)]
    out = {
        "r_px": r_px,
        "n_labeled": len(recs),
        "n_visible": len(vis),
        "n_usable": len(usable),
        "recall": frac(sum(map(hit, vis)), len(vis)),
        "recall_usable": frac(sum(map(hit, usable)), len(usable)),
        "precision": frac(sum(map(hit, shown)), len(shown)),
        "false_on_none": sum(1 for x in shown if not x["visible"]),
        "misses": misses,
        "median_m": float(np.median(d_m)) if d_m else float("nan"),
        "p90_m": float(np.quantile(d_m, 0.9)) if d_m else float("nan"),
    }
    for kind in ("det", "extrap"):
        rows = [x for x in shown if x["row"] == kind]
        out[f"hits_{kind}"] = sum(map(hit, rows))
        out[f"rows_{kind}"] = len(rows)
        out[f"precision_{kind}"] = frac(out[f"hits_{kind}"], len(rows))
    return out


def fmt_ball(recs: list[dict]) -> str:
    h = ball_headline(recs)
    other = ", ".join(f"{r:.0f} px {ball_headline(recs, r)['recall']:.1%}" for r in BALL_RS)
    misses = ", ".join(f"{k} {v}" for k, v in h["misses"].items() if v)
    return (
        f"ball ({h['n_visible']} visible of {h['n_labeled']} labeled): recall {h['recall']:.1%} "
        f"(usable frames {h['recall_usable']:.1%}; at {other})  precision {h['precision']:.1%} "
        f"(detected {h['hits_det']}/{h['rows_det']}, extrapolated "
        f"{h['hits_extrap']}/{h['rows_extrap']}, on 'none' frames {h['false_on_none']})  "
        f"error {h['median_m']:.2f} / {h['p90_m']:.2f} m  misses: {misses or 'none'}"
    )


PFF_R_PX = 40.0  # 10-ball 1d: a hit against PFF's projected ball
PFF_RS = 25.0  # printed next to it
AGREE_PX = 25.0  # a good label this close to the projection agrees with PFF
AGREE_MIN = 0.70  # below this on a labeled clip the PFF score is withheld (vb02: 84.4%)
AGREE_MIN_N = 20  # labels needed before the agreement can withhold anything


def pff_frames(
    truth: pl.DataFrame,
    labels: dict | None,
    skip: int,
    det: pl.DataFrame,
    balls: pl.DataFrame,
    frames: pl.DataFrame,
    hs: dict,
    config,
) -> list[dict]:
    """Per PFF reference frame inside the run (10-ball 1d): its kind, vision's ball row and
    the nearest candidate as pixel distances to PFF's projection, and, where the frame has
    an [x, y] label, the label's distance to the projection and whether the row hits it."""
    rows = {r["frame_id"]: r for r in det.filter(pl.col("class") == BALL).iter_rows(named=True)}
    cands: dict[int, list] = {}
    for f, x1, y1, x2, y2 in balls.select("frame_id", "x1", "y1", "x2", "y2").iter_rows():
        cands.setdefault(f, []).append(((x1 + x2) / 2, (y1 + y2) / 2))
    views = dict(frames.select("frame_id", "view").iter_rows())
    out = []
    for src, kind, u, v in truth.select("src", "kind", "u", "v").iter_rows():
        f = src - skip
        if f not in views:
            continue
        rec = {
            "frame": src,
            "kind": kind,
            "match_view": views[f] == MATCH,
            "row": None,
            "has_xy": False,
            "d_px": None,
            "cand_px": None,
            "lab_px": None,
            "click_hit": None,
        }
        r = rows.get(f)
        at_px = None
        if r is not None:
            rec["row"] = "extrap" if r["tracked_only"] else "det"
            rec["has_xy"] = r["pitch_x"] is not None
            if rec["has_xy"] or not r["tracked_only"]:
                at_px = np.array(ball_px(r, hs.get(f), config.home_attacks_tv_right_p1))
        if u is not None:
            proj = np.array([u, v])
            if at_px is not None:
                rec["d_px"] = float(np.linalg.norm(at_px - proj))
            near = [float(np.hypot(x - u, y - v)) for x, y in cands.get(f, [])]
            rec["cand_px"] = min(near) if near else None
            lab = (labels or {}).get(src)
            if isinstance(lab, list):
                rec["lab_px"] = float(np.linalg.norm(np.array(lab) - proj))
                rec["click_hit"] = bool(
                    rec["has_xy"]
                    and at_px is not None
                    and np.linalg.norm(at_px - np.array(lab)) <= BALL_R_PX
                )
        out.append(rec)
    return out


def pff_headline(recs: list[dict], r_px: float = PFF_R_PX) -> dict:
    """PFF score (10-ball 1d) on `pff` frames: recall, precision, the candidates' ceiling,
    recall at 25 px, and how many vision rows fall on `estimated` frames."""

    def hit(x, r=r_px):
        return x["has_xy"] and x["d_px"] is not None and x["d_px"] <= r

    pff = [x for x in recs if x["kind"] == "pff"]
    shown = [x for x in pff if x["has_xy"]]

    def frac(a, b):
        return a / b if b else float("nan")

    return {
        "n_pff": len(pff),
        "recall": frac(sum(map(hit, pff)), len(pff)),
        "recall_25": frac(sum(hit(x, PFF_RS) for x in pff), len(pff)),
        "precision": frac(sum(map(hit, shown)), len(shown)),
        "ceiling": frac(
            sum(x["cand_px"] is not None and x["cand_px"] <= r_px for x in pff), len(pff)
        ),
        "rows_estimated": sum(x["has_xy"] for x in recs if x["kind"] == "estimated"),
    }


def agreement(recs: list[dict]) -> dict:
    """10-ball 1d's check on a labeled clip, over [x, y] labels on `pff` frames: the share
    within 25 px of the projection, and how often a hit under the click (15 px) and under
    PFF (40 px) agree."""
    lab = [x for x in recs if x["kind"] == "pff" and x["lab_px"] is not None]
    if not lab:
        return {"n": 0, "within_25": float("nan"), "verdict_40": float("nan")}
    pff_hit = [x["has_xy"] and x["d_px"] is not None and x["d_px"] <= PFF_R_PX for x in lab]
    return {
        "n": len(lab),
        "within_25": sum(x["lab_px"] <= AGREE_PX for x in lab) / len(lab),
        "verdict_40": sum(x["click_hit"] == h for x, h in zip(lab, pff_hit, strict=True))
        / len(lab),
    }


def fmt_pff(recs: list[dict]) -> str:
    h, a = pff_headline(recs), agreement(recs)
    agree = ""
    if a["n"]:
        agree = (
            f"; agreement: labels within 25 px {a['within_25']:.1%} of {a['n']}, "
            f"verdict at 40 px {a['verdict_40']:.1%}"
        )
        if a["n"] >= AGREE_MIN_N and a["within_25"] < AGREE_MIN:
            return f"PFF score withheld: agreement {a['within_25']:.1%} (sync or camera?){agree}"
    return (
        f"PFF score ({h['n_pff']} pff frames): recall {h['recall']:.1%} at 40 px "
        f"({h['recall_25']:.1%} at 25 px), precision {h['precision']:.1%}, "
        f"ceiling {h['ceiling']:.1%}, rows on estimated frames {h['rows_estimated']}{agree}"
    )


class Clip:
    """One clip's caches and truth, loaded once so a sweep only replays."""

    def __init__(
        self, clip: dict, cache_dir: Path, gamestate_dir: Path, ball_labels: dict | None = None
    ):
        self.clip = clip
        self.cache = cache_dir / clip["clip_id"]
        run = json.loads((self.cache / "run.json").read_text())
        if run.get("video_sha256") != clip["video_sha256"]:
            raise SystemExit(f"{clip['clip_id']}: the run's video isn't the manifest's (sha256)")
        self.run_start_s = run.get("video_start_s", 0.0)
        if not 0 <= clip["video_start_s"] - self.run_start_s <= MAX_PREROLL_S:
            raise SystemExit(
                f"{clip['clip_id']}: run starts at {self.run_start_s} s; the clip starts at "
                f"{clip['video_start_s']} s (run 0-{MAX_PREROLL_S:.0f} s before it)"
            )
        self.config = replay.run_config(self.cache)
        if self.config.home_attacks_tv_right_p1 != clip["home_attacks_tv_right_p1"]:
            raise SystemExit(
                f"{clip['clip_id']}: the run's attacking direction isn't the manifest's"
            )
        vision_gs = gamestate_dir / clip["clip_id"]
        self.inputs = replay.load(self.cache, vision_gs)
        # None: a run before balls.parquet, no ball score
        self.balls = replay.load_balls(self.cache)
        self.ball_labels = None
        if self.balls is not None and ball_labels and clip["clip_id"] in ball_labels:
            lab = ball_labels[clip["clip_id"]]
            if lab["video_sha256"] != clip["video_sha256"]:
                raise SystemExit(f"{clip['clip_id']}: the ball labels are on another video")
            self.ball_labels = lab["labels"]
        self.fps = pl.read_parquet(vision_gs / "match.parquet")["native_fps"][0]
        self.pff = gamestate_dir / clip["match_id"] if clip["match_id"] is not None else None
        self.truth = None  # 10-ball 1b, built by python -m vision.ball_truth
        if self.balls is not None and clip["match_id"] is not None:
            from vision import ball_truth  # it imports bench

            self.truth = ball_truth.load(clip)
        self._check_replay()

    def _check_replay(self) -> None:
        """Every score is a replay, so the run config's replay must be what the run wrote."""
        det, _ = replay.replay(*self.inputs, self.config, balls=self.balls)
        cols = ["frame_id", "object_id", "pitch_x", "pitch_y", "homography_ok", "team_cluster"]
        ball_cols = [*cols[:4], "x1", "y1", "x2", "y2", "det_confidence", "tracked_only"]

        def rows(d, ball):
            d = d.filter((pl.col("class") == BALL) == ball)
            return d.select(ball_cols if ball else cols).sort("frame_id", "object_id")

        cached = pl.read_parquet(self.cache / "detections.parquet")
        for ball in [False] if self.balls is None else [False, True]:
            if not rows(det, ball).equals(rows(cached, ball)):
                what = "ball" if ball else "people"
                raise SystemExit(
                    f"{self.clip['clip_id']}: replay doesn't reproduce the run's cache ({what})"
                )

    def score(self, sets: list[str], offset_check: bool = False, revote: bool = False) -> dict:
        config = replay.with_overrides(self.config, sets)
        det, frames = replay.replay(
            *self.inputs,
            config,
            revote,
            run_every=self.config.keypoints_every,
            balls=self.balls,
        )
        out = score_clip(
            self.clip,
            det,
            frames,
            self.inputs[3],
            self.run_start_s,
            self.pff,
            self.fps,
            offset_check,
        )
        if self.ball_labels is not None or self.truth is not None:
            _, keypoints, views, times = self.inputs
            hs = replay.frame_homographies(
                keypoints, views, times, config, revote, self.config.keypoints_every
            )
            skip = round(self.run_start_s * self.fps)
        if self.ball_labels is not None:
            out["ball"] = ball_frames(self.ball_labels, skip, det, self.balls, frames, hs, config)
        if self.truth is not None:
            out["pff_ball"] = pff_frames(
                self.truth, self.ball_labels, skip, det, self.balls, frames, hs, config
            )
        return out


def fmt(h: dict) -> str:
    return (
        f"geometry missing {h['geometry_missing']:.1%} (view other {h['view_other']:.1%}, "
        f"rejected {h['homography_rejected']:.1%} of match view)  within 2 m {h['within_2m']:.1%}  "
        f"median {h['median_m']:.2f} m  p90 {h['p90_m']:.2f} m  "
        f"(with geometry: {h['within_2m_geo']:.1%}, {h['median_geo_m']:.2f} / {h['p90_geo_m']:.2f} m)  "
        f"unmatched/frame {h['unmatched_per_frame']:.2f}"
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="vision.bench")
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--clip", action="append", help="score only these clip_ids")
    ap.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE")
    ap.add_argument("--grid", action="append", default=[], metavar="FIELD=V1,V2,...")
    ap.add_argument("--out", type=Path, help="write the scorecard as JSON")
    ap.add_argument(
        "--revote", action="store_true", help="PnLCalib runs: vote cameras again from the peaks"
    )
    ap.add_argument(
        "--offset-check", action="store_true", help="report the best sync offset within ±1 s"
    )
    ap.add_argument("--cache-dir", type=Path, default=Path("data/vision_cache"))
    ap.add_argument("--gamestate-dir", type=Path, default=Path("data/gamestate"))
    ap.add_argument("--ball-labels", type=Path, default=BALL_LABELS)
    args = ap.parse_args(argv)

    clips = [c for c in load_manifest(args.manifest) if not args.clip or c["clip_id"] in args.clip]
    labels = load_ball_labels(args.ball_labels)
    loaded = [Clip(c, args.cache_dir, args.gamestate_dir, labels) for c in clips]
    axes = [(f, vals.split(",")) for f, _, vals in (g.partition("=") for g in args.grid)]
    combos = [
        [f"{f}={v}" for (f, _), v in zip(axes, vs)]
        for vs in itertools.product(*[a[1] for a in axes])
    ]

    results = []
    for combo in combos:
        sets = args.set + combo
        per_clip = [
            c.score(sets, args.offset_check and len(combos) == 1, args.revote) for c in loaded
        ]
        h = headline(pooled(per_clip))
        ball = [x for c in per_clip for x in c.get("ball", [])]
        if ball:
            h["ball"] = ball_headline(ball)
        results.append(({"sets": sets, "clips": per_clip}, h))
        if len(combos) > 1:
            print(" ".join(combo) or "(run config)", "|", fmt(h))
            if ball:
                print("   ", fmt_ball(ball))
            pff_ball = [x for c in per_clip for x in c.get("pff_ball", [])]
            if pff_ball:
                print("   ", fmt_pff(pff_ball))

    if len(combos) == 1:
        ((run, h),) = results
        for c in run["clips"]:
            t = c["teams"]
            flag = " [coarse sync]" if c["sync_coarse"] else ""
            if not c["n_truth_frames"]:
                flag = " [no PFF: geometry and marks only]"
            print(f"{c['clip_id']}{flag}: {fmt(headline(c))}")
            if "offset_check_s" in c:
                print(f"    best offset is {c['offset_check_s']:+.3f} s from the sync's")
            for kind in (PLAYER, GOALKEEPER):
                k = t[kind]
                acc = k["correct"] / k["assigned"] if k["assigned"] else float("nan")
                cov = k["assigned"] / k["pairs"] if k["pairs"] else float("nan")
                picked = " (home cluster picked by agreement)" if t["home_cluster_picked"] else ""
                print(f"    {kind} team accuracy {acc:.1%}, coverage {cov:.1%}{picked}")
            if c["false_live_s"]:
                print(
                    "    false live:",
                    ", ".join(f"{k} {v:.1f} s" for k, v in c["false_live_s"].items()),
                )
            if c.get("ball"):
                print("   ", fmt_ball(c["ball"]))
            if c.get("pff_ball"):
                print("   ", fmt_pff(c["pff_ball"]))
            elif c["n_truth_frames"]:
                print(f"    no PFF reference (python -m vision.ball_truth --clip {c['clip_id']})")
        print(f"pooled ({len(loaded)} clips): {fmt(h)}")
        ball = [x for c in run["clips"] for x in c.get("ball", [])]
        if ball:
            print("   ", fmt_ball(ball))
        pff_ball = [x for c in run["clips"] for x in c.get("pff_ball", [])]
        if pff_ball:
            print("   ", fmt_pff(pff_ball))
    else:
        chosen = pick(results)
        if chosen is None:
            print(f"no setting keeps geometry missing at or under {GEOMETRY_TARGET:.0%}")
        else:
            print("pick:", " ".join(chosen[0]["sets"]), "|", fmt(chosen[1]))
    if args.out:
        args.out.write_text(
            json.dumps(
                [{"sets": r["sets"], "headline": h, "clips": r["clips"]} for r, h in results],
                indent=1,
            )
        )


if __name__ == "__main__":
    main()
