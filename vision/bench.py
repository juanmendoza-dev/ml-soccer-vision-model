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
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment

from vision import replay
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
        if not c["sync"]:
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
    pff: pl.DataFrame,
) -> pl.DataFrame:
    """Vision frames + video_s and the nearest PFF frame (null if none within half a frame)."""
    out = frames.with_columns(
        video_s=pl.col("frame_id").replace_strict(times, return_dtype=pl.Float64) + video_start_s
    ).with_columns(pff_t=pl.col("video_s") + offset)
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


def frame_pairs(aligned: pl.DataFrame, det: pl.DataFrame, truth: pl.DataFrame) -> dict:
    """Per scored frame: truth count, vision count and the matched pairs."""
    scored = aligned.filter(
        pl.col("in_clip") & pl.col("mark").is_null() & pl.col("pff_frame_id").is_not_null()
    )
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
    if offset_check:
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
        "unmatched_per_frame": frac(s["n_vis"] - s["n_pairs"], s["n_scored"]),
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


class Clip:
    """One clip's caches and truth, loaded once so a sweep only replays."""

    def __init__(self, clip: dict, cache_dir: Path, gamestate_dir: Path):
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
        self.fps = pl.read_parquet(vision_gs / "match.parquet")["native_fps"][0]
        self.pff = gamestate_dir / clip["match_id"]
        self._check_replay()

    def _check_replay(self) -> None:
        """Every score is a replay, so the run config's replay must be what the run wrote."""
        det, _ = replay.replay(*self.inputs, self.config)
        cols = ["frame_id", "object_id", "pitch_x", "pitch_y", "homography_ok", "team_cluster"]

        def people(d):
            return d.filter(pl.col("class") != BALL).select(cols).sort("frame_id", "object_id")

        cached = pl.read_parquet(self.cache / "detections.parquet")
        if not people(det).equals(people(cached)):
            raise SystemExit(f"{self.clip['clip_id']}: replay doesn't reproduce the run's cache")

    def score(self, sets: list[str], offset_check: bool = False) -> dict:
        det, frames = replay.replay(*self.inputs, replay.with_overrides(self.config, sets))
        return score_clip(
            self.clip,
            det,
            frames,
            self.inputs[3],
            self.run_start_s,
            self.pff,
            self.fps,
            offset_check,
        )


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
        "--offset-check", action="store_true", help="report the best sync offset within ±1 s"
    )
    ap.add_argument("--cache-dir", type=Path, default=Path("data/vision_cache"))
    ap.add_argument("--gamestate-dir", type=Path, default=Path("data/gamestate"))
    args = ap.parse_args(argv)

    clips = [c for c in load_manifest(args.manifest) if not args.clip or c["clip_id"] in args.clip]
    loaded = [Clip(c, args.cache_dir, args.gamestate_dir) for c in clips]
    axes = [(f, vals.split(",")) for f, _, vals in (g.partition("=") for g in args.grid)]
    combos = [
        [f"{f}={v}" for (f, _), v in zip(axes, vs)]
        for vs in itertools.product(*[a[1] for a in axes])
    ]

    results = []
    for combo in combos:
        sets = args.set + combo
        per_clip = [c.score(sets, args.offset_check and len(combos) == 1) for c in loaded]
        h = headline(pooled(per_clip))
        results.append(({"sets": sets, "clips": per_clip}, h))
        if len(combos) > 1:
            print(" ".join(combo) or "(run config)", "|", fmt(h))

    if len(combos) == 1:
        ((run, h),) = results
        for c in run["clips"]:
            t = c["teams"]
            flag = " [coarse sync]" if c["sync_coarse"] else ""
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
        print(f"pooled ({len(loaded)} clips): {fmt(h)}")
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
