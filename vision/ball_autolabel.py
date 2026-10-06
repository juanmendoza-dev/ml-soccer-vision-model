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
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import polars as pl

from vision import ball_truth as bt
from vision import bench, replay
from vision.bench import PFF_FPS
from vision.pitch import to_02
from vision.types import MATCH, Camera

SWEEP_S = 1.5
HIT_PX = 15.0
MIN_CONF = 0.5
MIN_SPEED = 5.0  # m/s; a still ball can't tell one offset from the next
MIN_FRAMES = bench.AGREE_MIN_N  # scored frames the peak needs before it can pass
BENCH_MATCHES = ("10517", "10511", "3854")  # never auto-labeled (10-ball 3a)
MANIFESTS = (bench.MANIFEST, Path("data/splits/demo_clips.json"))
# the six 2022 matches with PFF that aren't on the bench (10-ball 3a)
MATCHES = ("3857", "10507", "3816", "10510", "10508", "10514")

# labels (10-ball 3a)
POS_PX = 25.0  # the nearest candidate this close to the projection is the ball ...
ALONE_PX = 40.0  # ... if no other candidate is this close
NEG_CONF, NEG_PX = 0.3, 60.0  # hard negatives: this confident, further than this
SPARE_CONF, SPARE_M = 0.5, 3.0  # a second ball this close to a touchline: spare balls
BIAS_CONF, BIAS_PX = 0.5, 40.0  # PFF drift: confident candidates this close ...
BIAS_WIN_S, BIAS_SKIP_S = 1.0, 0.2  # ... within 1 s of the frame, not within 0.2 s
HALF_WIDTH_M = 34.0  # 02 touchlines at y = +-34
EVERY = 3  # every 3rd source frame of live wide play
PIECES = Path("data/splits/ball_autolabel_pieces.json")
OFFSET_FROM = ("scoreboard", "stitched")  # where a piece's starting offset came from


@dataclass
class SyncFrame:
    src: int  # source video frame
    cam: Camera
    cands: list[tuple[float, float, float]]  # ball candidates: center u, v and confidence


Box = tuple[float, float, float, float, float]  # x1, y1, x2, y2, confidence (source px)


@dataclass
class LabelFrame:
    src: int
    t: float  # source video seconds
    kind: str  # vision.ball_truth's kind at the synced offset: only "pff" can be labeled
    proj: tuple[float, float] | None  # PFF's ball in the image, before the bias correction
    cam: Camera | None
    boxes: list[Box]  # the ball model's candidates


@dataclass
class Label:
    kind: str  # "positive", or why the frame is left out
    box: tuple[float, float, float, float] | None = None
    negatives: list = field(default_factory=list)


def ground_point(cam: Camera, u: float, v: float, home_right: bool) -> tuple[float, float]:
    """A pixel back to 02 (x, y) on the plane of a ball's center on the grass, the inverse of
    vision.ball_truth.project_ball at z = 0."""
    ray = cam.rotation.T @ np.array([(u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, 1.0])
    s = (-bt.BALL_D / 2 - cam.position[2]) / ray[2]
    w = cam.position + s * ray
    x, y = to_02(np.array([w[0], -w[1]]), home_right)
    return float(x), float(y)


def _center(b) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def bias(frames: list[LabelFrame], i: int) -> tuple[float, float]:
    """PFF's drift around frame i: the median offset (candidate - projection) of the
    nearest confident candidate within BIAS_PX on the projected frames 0.2-1 s away.
    (0, 0) when there's none."""
    t = frames[i].t
    offs = []
    for f in frames:
        if f.proj is None or not BIAS_SKIP_S < abs(f.t - t) <= BIAS_WIN_S:
            continue
        near = [
            (np.hypot(u - f.proj[0], v - f.proj[1]), u - f.proj[0], v - f.proj[1])
            for u, v in (_center(b) for b in f.boxes if b[4] >= BIAS_CONF)
        ]
        near = [n for n in near if n[0] <= BIAS_PX]
        if near:
            offs.append(min(near)[1:])
    if not offs:
        return 0.0, 0.0
    du, dv = np.median(np.array(offs), axis=0)
    return float(du), float(dv)


def label(f: LabelFrame, b: tuple[float, float], home_right: bool) -> Label:
    """10-ball 3a's rules on one frame, its projection moved by the bias b."""
    if f.kind != "pff" or f.proj is None:
        return Label(f.kind)
    pu, pv = f.proj[0] + b[0], f.proj[1] + b[1]
    dist = [float(np.hypot(*np.subtract(_center(x), (pu, pv)))) for x in f.boxes]
    order = np.argsort(dist)
    if not len(order) or dist[order[0]] > POS_PX:
        return Label("missed")
    ball = int(order[0])
    for j, x in enumerate(f.boxes):
        if j != ball and x[4] >= SPARE_CONF and f.cam is not None:
            _, y = ground_point(f.cam, *_center(x), home_right)
            if abs(abs(y) - HALF_WIDTH_M) <= SPARE_M:
                return Label("spare_ball")
    if len(order) > 1 and dist[order[1]] <= ALONE_PX:
        return Label("rival")
    negatives = [
        x[:4] for j, x in enumerate(f.boxes) if j != ball and x[4] >= NEG_CONF and dist[j] > NEG_PX
    ]
    return Label("positive", f.boxes[ball][:4], negatives)


def source_frames(start_s: float, end_s: float, fps: float, every: int = EVERY) -> list[int]:
    """Source frames in [start_s, end_s) that are multiples of every."""
    first = math.ceil(start_s * fps / every) * every
    return [i for i in range(first, math.ceil(end_s * fps) + 1, every) if i / fps < end_s]


def pff_players(gs_dir: Path, period: int) -> pl.DataFrame:
    """PFF's player rows of the period: frame_id, t, visible."""
    frames = pl.scan_parquet(gs_dir / "frames.parquet").filter(pl.col("period") == period)
    obj = pl.scan_parquet(gs_dir / "objects.parquet").filter(pl.col("object_type") == "player")
    return (
        obj.join(frames.select("frame_id", t="timestamp_s"), on="frame_id")
        .select("frame_id", "t", "visible")
        .collect()
    )


def cutaways(players: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per PFF frame, in time order: its t, and whether every player is ESTIMATED."""
    per = players.group_by("frame_id").agg(pl.col("t").first(), pl.col("visible").any()).sort("t")
    return per["t"].to_numpy(), ~per["visible"].to_numpy()


def is_cutaway(cut: tuple[np.ndarray, np.ndarray], t: float) -> bool:
    """The nearest PFF frame's flag; no PFF frame within one frame counts as a cutaway."""
    ts, flags = cut
    if not len(ts):
        return True
    i = int(np.abs(ts - t).argmin())
    return bool(abs(ts[i] - t) > 1 / PFF_FPS or flags[i])


def load_pieces(path: Path = PIECES) -> list[dict]:
    """The auto-label pieces: one match period's stretch of footage each, with the offset to
    start the sync sweep from (PFF timestamp_s = video seconds + offset_s)."""
    pieces = json.loads(Path(path).read_text())["pieces"]
    seen = set()
    for p in pieces:
        name = p.get("piece_id", "?")
        refuse(p["match_id"])
        if p["match_id"] not in MATCHES:
            raise SystemExit(f"{name}: match {p['match_id']} isn't one of the six (10-ball 3a)")
        if not name or name.strip() != name or "/" in name:
            raise SystemExit(f"{name!r}: a piece_id is a plain name")
        if name in seen:
            raise SystemExit(f"{name}: piece_id twice")
        seen.add(name)
        if not p["video_start_s"] < p["video_end_s"]:
            raise SystemExit(f"{name}: ends before it starts")
        if p["offset_from"] not in OFFSET_FROM:
            raise SystemExit(f"{name}: offset_from must be one of {OFFSET_FROM}")
        if p["period"] not in (1, 2, 3, 4):
            raise SystemExit(f"{name}: period must be 1-4")
    return pieces


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


def verdict(p: dict) -> str:
    """ok, or why not. An empty sweep ties at 0 everywhere and its middle is shift 0, so a
    peak needs hits and MIN_FRAMES scored frames before its shift counts."""
    if p["hits"] == 0 or p["n"] < MIN_FRAMES:
        return f"FAIL: no scorable frames ({p['hits']}/{p['n']} at the peak, need {MIN_FRAMES})"
    if abs(p["shift"]) > 1 / PFF_FPS + 1e-9:
        return f"FAIL: peak at {p['shift'] * PFF_FPS:+.0f} PFF frames"
    return "ok"


def sync_ok(p: dict) -> bool:
    return verdict(p) == "ok"


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
    by_k = {round(x["shift"] * PFF_FPS): x for x in r["rows"]}
    at0 = by_k[0]
    # a sharp peak drops off by +-5 frames and at the edges; a flat curve doesn't
    around = " ".join(f"{k:+d}: {by_k[k]['share']:.0%}" for k in (min(by_k), -5, 5, max(by_k)))
    return (
        f"{r['clip_id']}: peak {p['shift'] * PFF_FPS:+.0f} PFF frames ({p['shift']:+.3f} s), "
        f"share {p['share']:.1%} ({p['hits']}/{p['n']}); at the manifest's offset "
        f"{at0['share']:.1%} ({at0['hits']}/{at0['n']}); {around}; "
        f"{r['frames']} frames with a camera -> {verdict(p)}"
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
