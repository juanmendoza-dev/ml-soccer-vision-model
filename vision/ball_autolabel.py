"""Ball auto-labels from PFF (10-ball 3a). Training data only: it reads PFF, so nothing in
the pipeline may import it.

    PYTHONPATH=. python scripts/ball_autolabel.py --sync-check    # vb01-vb03 first (F1)

The sync check: shift PFF's clock by +-1.5 s in PFF-frame steps around the manifest's
offset and score each shift by the share of PFF-VISIBLE moving-ball frames (> 5 m/s) with
a ball candidate >= 0.5 within 15 px of PFF's projected ball. On the bench clips the peak
must sit at the manifest's offset to within one PFF frame, or the method can't be trusted
to sync the six auto-label matches.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import polars as pl

from vision import ball_truth as bt
from vision import bench, replay
from vision.bench import PFF_FPS
from vision.types import MATCH, Camera

SWEEP_S = 1.5
HIT_PX = 15.0
MIN_CONF = 0.5
MIN_SPEED = 5.0  # m/s; a still ball can't tell one offset from the next
BENCH_MATCHES = ("10517", "10511", "3854")  # never auto-labeled (10-ball 3a)
MANIFESTS = (bench.MANIFEST, Path("data/splits/demo_clips.json"))
# the six 2022 matches with PFF that aren't on the bench (10-ball 3a)
MATCHES = ("3857", "10507", "3816", "10510", "10508", "10514")


@dataclass
class SyncFrame:
    src: int  # source video frame
    cam: Camera
    cands: list[tuple[float, float, float]]  # ball candidates: center u, v and confidence


def refuse(match_id: str, manifests=MANIFESTS) -> None:
    """Stop on a bench match or any match a manifest names: its frames must never train."""
    names = set(BENCH_MATCHES)
    for m in manifests:
        names |= {c["match_id"] for c in json.loads(Path(m).read_text())["clips"]}
    if match_id in names:
        raise SystemExit(f"{match_id} is a bench match (or in a manifest): never auto-labeled")


def refuse_all(match_ids, manifests=MANIFESTS) -> None:
    for m in match_ids:
        refuse(m, manifests)


def score(
    frames: list[SyncFrame],
    pff: pl.DataFrame,
    offset: float,
    fps: float,
    home_right: bool,
    size: tuple[int, int],
) -> tuple[int, int]:
    """(hits, n) at one offset: n PFF-VISIBLE balls over MIN_SPEED inside the image, hits
    with a candidate >= MIN_CONF within HIT_PX of the projection."""
    hits = n = 0
    for f in frames:
        b = bt.ball_at(pff, f.src / fps + offset)
        if b is None or not b["visible"] or b["speed"] <= MIN_SPEED:
            continue
        p = bt.project_ball(f.cam, (b["x"], b["y"], b["z"]), home_right)
        if p is None or not (0 <= p[0] < size[0] and 0 <= p[1] < size[1]):
            continue
        n += 1
        hits += any(c >= MIN_CONF and np.hypot(u - p[0], v - p[1]) <= HIT_PX for u, v, c in f.cands)
    return hits, n


def sweep(frames, pff, offset, fps, home_right, size, half_s=SWEEP_S) -> list[dict]:
    """score() at offset + k PFF frames, k over +-half_s."""
    k = round(half_s * PFF_FPS)
    rows = []
    for i in range(-k, k + 1):
        shift = i / PFF_FPS
        hits, n = score(frames, pff, offset + shift, fps, home_right, size)
        rows.append({"shift": shift, "hits": hits, "n": n, "share": hits / n if n else 0.0})
    return rows


def peak(rows: list[dict]) -> dict:
    """The best share; a plateau of equal shares gives its middle shift."""
    best = max(r["share"] for r in rows)
    tied = [r for r in rows if r["share"] == best]
    return tied[(len(tied) - 1) // 2]


def sync_ok(p: dict) -> bool:
    return abs(p["shift"]) <= 1 / PFF_FPS + 1e-9


def camera(row: dict) -> Camera:
    """A camera from a vision.ball_truth row."""
    return Camera(
        row["fx"],
        row["fy"],
        row["cx"],
        row["cy"],
        np.array([row["cam_x"], row["cam_y"], row["cam_z"]]),
        np.array(row["rot"]).reshape(3, 3),
        float("nan"),
        "",
    )


def bench_frames(clip: dict, truth: pl.DataFrame, cache: Path, fps: float) -> list[SyncFrame]:
    """The PFF reference's frames with a camera, inside the run and in match view, with the
    run's ball candidates. A run that didn't detect on every frame can't tell a frame with
    no candidate from one it skipped, so it's refused."""
    run = json.loads((cache / "run.json").read_text())
    if replay.run_config(cache).detect_every != 1:
        raise SystemExit(f"{clip['clip_id']}: the run doesn't detect on every frame")
    skip = round(run.get("video_start_s", 0.0) * fps)
    views = dict(pl.read_parquet(cache / "view.parquet", columns=["frame_id", "view"]).iter_rows())
    balls = replay.load_balls(cache)
    if balls is None:
        raise SystemExit(f"{clip['clip_id']}: the run has no balls.parquet")
    cands: dict[int, list] = {}
    for f, x1, y1, x2, y2, c in balls.select(
        "frame_id", "x1", "y1", "x2", "y2", "det_confidence"
    ).iter_rows():
        cands.setdefault(f, []).append(((x1 + x2) / 2, (y1 + y2) / 2, c))
    out = []
    for row in truth.filter(pl.col("camera_ok")).iter_rows(named=True):
        f = row["src"] - skip
        if views.get(f) == MATCH:
            out.append(SyncFrame(row["src"], camera(row), cands.get(f, [])))
    return out


def check_clip(clip: dict, cache_dir: Path, gamestate_dir: Path) -> dict:
    """The sweep on one bench clip around its manifest offset."""
    truth = bt.load(clip)
    if truth is None:
        raise SystemExit(f"{clip['clip_id']}: no PFF reference (python -m vision.ball_truth)")
    fps = pl.read_parquet(gamestate_dir / clip["clip_id"] / "match.parquet")["native_fps"][0]
    frames = bench_frames(clip, truth, cache_dir / clip["clip_id"], fps)
    image = next((bt.FRAMES / clip["clip_id"]).glob("*.jpg"), None)
    if image is None:
        raise SystemExit(f"{clip['clip_id']}: no label frames in {bt.FRAMES}")
    h, w = cv2.imread(str(image)).shape[:2]
    pff = bt.pff_ball(gamestate_dir / clip["match_id"], clip["period"])
    rows = sweep(
        frames, pff, bench.sync_offset(clip), fps, clip["home_attacks_tv_right_p1"], (w, h)
    )
    return {"clip_id": clip["clip_id"], "frames": len(frames), "rows": rows, "peak": peak(rows)}


def fmt_check(r: dict) -> str:
    p = r["peak"]
    at0 = next(x for x in r["rows"] if abs(x["shift"]) < 1e-9)
    return (
        f"{r['clip_id']}: peak {p['shift'] * PFF_FPS:+.0f} PFF frames ({p['shift']:+.3f} s), "
        f"share {p['share']:.1%} ({p['hits']}/{p['n']}); at the manifest's offset "
        f"{at0['share']:.1%} ({at0['hits']}/{at0['n']}); {r['frames']} frames with a camera"
        f" -> {'ok' if sync_ok(p) else 'FAIL'}"
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="ball_autolabel")
    ap.add_argument("--sync-check", action="store_true", help="the sweep on the bench clips")
    ap.add_argument("--manifest", type=Path, default=bench.MANIFEST)
    ap.add_argument("--cache-dir", type=Path, default=bt.CACHE)
    ap.add_argument("--gamestate-dir", type=Path, default=Path("data/gamestate"))
    ap.add_argument("--json", type=Path, help="write the sweep rows here")
    args = ap.parse_args(argv)
    refuse_all(MATCHES)
    if not args.sync_check:
        raise SystemExit("only --sync-check is built (F1); the labels come in F2")
    results = [
        check_clip(c, args.cache_dir, args.gamestate_dir)
        for c in bench.load_manifest(args.manifest)
        if c["match_id"] is not None
    ]
    for r in results:
        print(fmt_check(r))
    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
    if not all(sync_ok(r["peak"]) for r in results):
        raise SystemExit("the sync check failed: stop before any auto-labels (10-ball 3a)")


if __name__ == "__main__":
    main()
