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
import random
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import polars as pl

from converters.common import git_commit, sha256
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
ALONE_PX = 40.0  # ... if no other candidate is this close ...
RIVAL_CONF = 0.3  # ... at this confidence (the user's call 2026-10-06: weak boots are common)
NEG_CONF, NEG_PX = 0.3, 60.0  # hard negatives: this confident, further than this
SPARE_CONF, SPARE_M = 0.5, 3.0  # a second ball this close to a touchline: spare balls
BIAS_CONF, BIAS_PX = 0.5, 40.0  # PFF drift: confident candidates this close ...
BIAS_WIN_S, BIAS_SKIP_S = 1.0, 0.2  # ... within 1 s of the frame, not within 0.2 s
HALF_WIDTH_M = 34.0  # 02 touchlines at y = +-34
EVERY = 3  # every 3rd source frame of live wide play
DET_CONF = 0.05  # the ball model's threshold for auto-labels
PIECES = Path("data/splits/ball_autolabel_pieces.json")
OFFSET_FROM = ("scoreboard", "stitched")  # where a piece's starting offset came from
OUT = Path("data/ball_train")  # gitignored
SYNC_JSON = Path("data/vision_bench/sync_check.json")
VIDEOS = OUT / "videos.json"  # piece_id -> local video path, like ball_truth's videos.json


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
    if not len(order) or dist[order[0]] > ALONE_PX:
        return Label("missed")  # nothing near: the model missed it (F3's pool)
    if dist[order[0]] > POS_PX:
        return Label("drift")  # seen 25-40 px off: PFF drift or a near miss, not F3's
    ball = int(order[0])
    for j, x in enumerate(f.boxes):
        if j != ball and x[4] >= SPARE_CONF and f.cam is not None:
            _, y = ground_point(f.cam, *_center(x), home_right)
            if abs(abs(y) - HALF_WIDTH_M) <= SPARE_M:
                return Label("spare_ball")
    if any(j != ball and x[4] >= RIVAL_CONF and dist[j] <= ALONE_PX for j, x in enumerate(f.boxes)):
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


def surely_cutaway(cut: tuple[np.ndarray, np.ndarray], lo: float, hi: float) -> bool:
    """is_cutaway at every PFF time in [lo, hi]: every PFF frame within one frame of the
    range is flagged (none there counts too)."""
    ts, flags = cut
    a, b = np.searchsorted(ts, [lo - 1 / PFF_FPS, hi + 1 / PFF_FPS], side="left")
    return bool(flags[a:b].all())


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


EDGE_FRAMES = 2  # a peak this close to the sweep's end: the curve may still be rising


@dataclass
class Seen:
    """A kept frame of a piece: an accepted camera and the ball model's candidates."""

    src: int
    cam: Camera
    boxes: list[Box]


def collect(frames, srcs, camera_fn, detect_fn, stage: Path, skip=lambda src: False):
    """Pass 1 over (src, image) in order: the srcs with an accepted camera, each image saved
    to stage/<src>.jpg for pass 2. Most cutaways can't be told yet: PFF's flags are on its
    clock, and the offset comes from the sync; skip(src) drops the sure ones before the
    camera runs. Returns the kept frames, the counts without a camera and skipped, and the
    image size (w, h)."""
    want, kept, size, no_camera, skipped = set(srcs), [], None, 0, 0
    stage.mkdir(parents=True, exist_ok=True)
    for src, image in frames:
        if src not in want:
            continue
        if skip(src):
            skipped += 1
            continue
        cam = camera_fn(src, image)
        if cam is None:
            no_camera += 1
            continue
        size = (image.shape[1], image.shape[0])
        cv2.imwrite(str(stage / f"{src}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        kept.append(Seen(src, cam, detect_fn(src, image)))
    return kept, no_camera, skipped, size


def sync_piece(seen, pff, offset, fps, home_right, size) -> tuple[float | None, dict, str]:
    """The sweep around the piece's starting offset: (offset or None, peak, status)."""
    frames = [SyncFrame(x.src, x.cam, [(*_center(b), b[4]) for b in x.boxes]) for x in seen]
    rows = sweep(frames, pff, offset, fps, home_right, size)
    p = peak(rows)
    k, last = round(p["shift"] * PFF_FPS), round(rows[-1]["shift"] * PFF_FPS)
    if p["hits"] == 0 or p["n"] < MIN_FRAMES:
        return None, p, f"FAIL: no scorable frames ({p['hits']}/{p['n']} at the peak)"
    if abs(k) >= last - EDGE_FRAMES:
        return None, p, f"FAIL: peak at {k:+d} PFF frames, the sweep's edge"
    return offset + p["shift"], p, "ok"


def label_frames(seen, pff, offset, fps, home_right, size) -> list[LabelFrame]:
    out = []
    for x in seen:
        b = bt.ball_at(pff, x.src / fps + offset)
        proj = bt.project_ball(x.cam, (b["x"], b["y"], b["z"]), home_right) if b else None
        kind = bt.kind_of(b, True, proj, size)
        out.append(
            LabelFrame(
                x.src, x.src / fps, kind, proj[:2] if kind == "pff" else None, x.cam, x.boxes
            )
        )
    return out


def yolo_line(b, size) -> str:
    """Class 0 (the ball model's only class), center and size as fractions of the image."""
    w, h = size
    cx, cy = _center(b)
    return f"0 {cx / w:.6f} {cy / h:.6f} {(b[2] - b[0]) / w:.6f} {(b[3] - b[1]) / h:.6f}\n"


def run_piece(piece, frames, srcs, fps, pff, cut, camera_fn, detect_fn, out: Path) -> dict:
    """One piece: collect, sync by the ball, label, write. Positives go to images/ and
    labels/ (YOLO), their hard negatives to negatives/<piece>.json, missed balls to missed/
    for ball_click --assist (F3); everything else is deleted. Returns the manifest entry."""
    pid, home_right = piece["piece_id"], piece["home_attacks_tv_right_p1"]
    clear_piece(out, pid)
    stage = out / "stage" / pid
    off0 = piece["offset_s"]

    def sure(src):  # a cutaway at every offset the sweep can pick: no camera needed
        t = src / fps + off0
        return surely_cutaway(cut, t - SWEEP_S, t + SWEEP_S)

    seen, no_camera, pre_cut, size = collect(frames, srcs, camera_fn, detect_fn, stage, sure)
    entry = {"match_id": piece["match_id"], "start_offset_s": off0}
    entry |= {"frames": len(srcs), "skipped": {"no_camera": no_camera, "cutaway": pre_cut}}
    offset, p, status = (
        (None, None, "FAIL: no frames")
        if not seen
        else sync_piece(seen, pff, piece["offset_s"], fps, home_right, size)
    )
    entry |= {"status": status, "offset_s": offset, "peak": p}
    if offset is None:
        for f in stage.glob("*.jpg"):
            f.unlink()
        stage.rmdir()
        return entry
    cutaway = {x.src for x in seen if is_cutaway(cut, x.src / fps + offset)}
    for src in cutaway:
        (stage / f"{src}.jpg").unlink()
    entry["skipped"]["cutaway"] += len(cutaway)
    seen = [x for x in seen if x.src not in cutaway]
    lfs = label_frames(seen, pff, offset, fps, home_right, size)
    counts: dict[str, int] = {}
    negatives, missed = {}, {}
    for d in ("images", "labels", "negatives", "missed"):
        (out / d).mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(lfs):
        b = bias(lfs, i)
        lab = label(f, b, home_right)
        counts[lab.kind] = counts.get(lab.kind, 0) + 1
        name, staged = f"{pid}_{f.src}.jpg", stage / f"{f.src}.jpg"
        if lab.kind == "positive":
            staged.replace(out / "images" / name)
            (out / "labels" / f"{pid}_{f.src}.txt").write_text(yolo_line(lab.box, size))
            negatives[name] = [list(n) for n in lab.negatives]
        elif lab.kind == "missed":
            staged.replace(out / "missed" / name)
            diam = bt.project_ball(f.cam, _xyz(pff, f.t + offset), home_right)[2]
            missed[str(f.src)] = {"u": f.proj[0] + b[0], "v": f.proj[1] + b[1], "diam_px": diam}
        else:
            staged.unlink()
    stage.rmdir()
    (out / "negatives" / f"{pid}.json").write_text(json.dumps(negatives))
    (out / "missed" / f"{pid}.json").write_text(json.dumps(missed, indent=1))
    entry |= {"counts": counts, "hard_negatives": sum(len(v) for v in negatives.values())}
    return entry


def clear_piece(out: Path, pid: str) -> None:
    """Remove what an earlier or crashed run of the piece left, so a rerun can't keep stale
    labels. The missed-ball clicks joined from it go too: join them again."""
    own = re.compile(rf"{re.escape(pid)}_\d+\.(jpg|txt)")
    for d in ("images", "labels", "missed"):
        for f in (out / d).glob("*") if (out / d).exists() else []:
            if own.fullmatch(f.name):
                f.unlink()
    for f in (out / "negatives" / f"{pid}.json", out / "missed" / f"{pid}.json"):
        f.unlink(missing_ok=True)
    shutil.rmtree(out / "stage" / pid, ignore_errors=True)


def _xyz(pff, t) -> tuple[float, float, float]:
    b = bt.ball_at(pff, t)
    return b["x"], b["y"], b["z"]


def require_sync_check(path: Path, clips: list[str] | None = None) -> None:
    """Auto-labels only after --sync-check passed on every bench clip with PFF (10-ball 3a)."""
    if not Path(path).exists():
        raise SystemExit(f"no {path}: run --sync-check --json {path} first")
    results = {r["clip_id"]: r for r in json.loads(Path(path).read_text())}
    if clips is None:
        clips = [c["clip_id"] for c in bench.load_manifest(bench.MANIFEST) if c["match_id"]]
    for c in clips:
        if c not in results:
            raise SystemExit(f"{c} isn't in {path}: rerun --sync-check")
        if not sync_ok(results[c]["peak"]):
            raise SystemExit(f"{c}: {verdict(results[c]['peak'])}; no auto-labels")
    for c, r in results.items():
        if not sync_ok(r["peak"]):
            raise SystemExit(f"{c}: {verdict(r['peak'])}; no auto-labels")


def update_manifest(out: Path, key: str, entry: dict, section: str = "pieces") -> None:
    """out/manifest.json: the rules and one entry per piece (or per section key), each with
    the git commit."""
    path = out / "manifest.json"
    m = json.loads(path.read_text()) if path.exists() else {}
    m["rules"] = {
        k: globals()[k]
        for k in (
            "POS_PX",
            "ALONE_PX",
            "RIVAL_CONF",
            "NEG_CONF",
            "NEG_PX",
            "SPARE_CONF",
            "SPARE_M",
            "BIAS_CONF",
            "BIAS_PX",
            "BIAS_WIN_S",
            "BIAS_SKIP_S",
            "EVERY",
            "DET_CONF",
        )
    }
    m.setdefault(section, {})[key] = entry | {"git_commit": git_commit()}
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(m, indent=2))


def video_frames(path: Path, srcs):
    """(src, image) for the wanted source frames, decoded in order (never seeking, as
    vision.run and vision.ball_truth do)."""
    want = set(srcs)
    cap = cv2.VideoCapture(str(path))
    try:
        for i in range(max(want) + 1):
            if not cap.grab():
                raise SystemExit(f"{path} ends at frame {i}")
            if i in want:
                ok, image = cap.retrieve()
                if not ok:
                    raise SystemExit(f"can't decode frame {i} of {path}")
                yield i, image
    finally:
        cap.release()


def models(weights_dir: Path, pnl_dir: Path, device: str):
    """The workstation's camera_fn and detect_fn: PnLCalib with vision's per-call checks, and
    the current ball model at DET_CONF."""
    from vision.calib import accept, camera_from_peaks
    from vision.config import VisionConfig
    from vision.stages import PnLCalibCamera, YoloDetector
    from vision.types import BALL

    config = VisionConfig()
    net = PnLCalibCamera(pnl_dir, config.pnl_weights, device=device)
    ball = YoloDetector(
        weights_dir / "football-ball-detection.pt", device, conf=DET_CONF, required=(BALL,)
    )

    def camera_fn(src, image):
        size = (image.shape[1], image.shape[0])
        cam, _, _ = camera_from_peaks(
            net.detect(image), size, config.pnl_kp_threshold, config.pnl_line_threshold
        )
        return cam if accept(cam, config) else None

    def detect_fn(src, image):
        return [(*d.box, d.confidence) for d in ball.detect(image) if d.cls == BALL]

    return camera_fn, detect_fn


MISSED_N = 300  # 10-ball 3a: the missed-ball sample the user labels


def sample_missed(out: Path, n: int = MISSED_N, seed: int = 0) -> list[dict]:
    """A fixed random n of every piece's missed-ball frames (PFF VISIBLE, no candidate within
    POS_PX), each with PFF's corrected projection and ball size for the click tool's ring."""
    pool = []
    for path in sorted((out / "missed").glob("*.json")):
        pid = path.stem
        for src, e in sorted(json.loads(path.read_text()).items(), key=lambda kv: int(kv[0])):
            pool.append({"piece_id": pid, "src": int(src), "image": f"{pid}_{src}.jpg", **e})
    return random.Random(seed).sample(pool, min(n, len(pool)))


def join_missed(out: Path, sample: list[dict], clicks: dict) -> dict:
    """The user's verified clicks join the training set: a box of PFF's projected ball size
    at the click. "none", "unsure" and frames not yet clicked never become labels."""
    counts = {"joined": 0, "none": 0, "unsure": 0, "todo": 0}
    for f in sample:  # a rejoin starts over: a click changed to "none" must not stay
        (out / "images" / f["image"]).unlink(missing_ok=True)
        (out / "labels" / f["image"].replace(".jpg", ".txt")).unlink(missing_ok=True)
    for f in sample:
        c = clicks.get(f["image"])
        if c is None:
            counts["todo"] += 1
            continue
        if not isinstance(c, list):
            counts[c] += 1
            continue
        image = cv2.imread(str(out / "missed" / f["image"]))
        size = (image.shape[1], image.shape[0])
        r = f["diam_px"] / 2
        for d in ("images", "labels"):
            (out / d).mkdir(parents=True, exist_ok=True)
        shutil.copy(out / "missed" / f["image"], out / "images" / f["image"])
        b = (c[0] - r, c[1] - r, c[0] + r, c[1] + r)
        (out / "labels" / f["image"].replace(".jpg", ".txt")).write_text(yolo_line(b, size))
        counts["joined"] += 1
    return counts


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
    ap.add_argument("--json", type=Path, help="--sync-check: write the sweep rows here")
    ap.add_argument("--pieces", type=Path, default=PIECES)
    ap.add_argument("--piece", action="append", help="only these piece_ids (default: all)")
    ap.add_argument("--sync-json", type=Path, default=SYNC_JSON, help="--sync-check's --json")
    ap.add_argument("--videos", type=Path, default=VIDEOS, help="piece_id -> video path")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--weights-dir", type=Path, default=Path("../sports/examples/soccer/data"))
    ap.add_argument("--pnl-weights-dir", type=Path, default=bt.PNL_DIR)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sample-missed", action="store_true", help="F3: draw the 300 to click")
    ap.add_argument("--join-missed", action="store_true", help="F3: add the verified clicks")
    args = ap.parse_args(argv)
    refuse_all(MATCHES)
    if args.sync_check:
        sync_check(args)
        return
    sample_path, clicks_path = args.out / "missed_sample.json", args.out / "missed_clicks.json"
    if args.sample_missed:
        if sample_path.exists():
            raise SystemExit(f"{sample_path} exists: the sample is drawn once")
        sample = sample_missed(args.out)
        sample_path.write_text(json.dumps(sample, indent=1))
        print(f"{len(sample)} missed-ball frames in {sample_path}")
        return
    if args.join_missed:
        clicks = json.loads(clicks_path.read_text()) if clicks_path.exists() else {}
        counts = join_missed(args.out, json.loads(sample_path.read_text()), clicks)
        update_manifest(args.out, "missed", counts, section="clicks")
        print(counts)
        return
    pieces = load_pieces(args.pieces)  # refuses bench matches first
    if args.piece:
        pieces = [p for p in pieces if p["piece_id"] in args.piece]
    require_sync_check(args.sync_json)
    videos = json.loads(args.videos.read_text())
    for p in pieces:
        video = Path(videos[p["piece_id"]])
        if sha256(video) != p["video_sha256"]:
            raise SystemExit(f"{video} isn't {p['piece_id']}'s video (sha256)")
    camera_fn, detect_fn = models(args.weights_dir, args.pnl_weights_dir, args.device)
    for p in pieces:
        video = Path(videos[p["piece_id"]])
        fps = cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FPS)
        srcs = source_frames(p["video_start_s"], p["video_end_s"], fps)
        gs = args.gamestate_dir / p["match_id"]
        pff = bt.pff_ball(gs, p["period"])
        cut = cutaways(pff_players(gs, p["period"]))
        print(f"{p['piece_id']}: {len(srcs)} frames from {video.name}...")
        entry = run_piece(
            p, video_frames(video, srcs), srcs, fps, pff, cut, camera_fn, detect_fn, args.out
        )
        update_manifest(args.out, p["piece_id"], entry)
        print(f"  {entry['status']}; {entry.get('counts')}; skipped {entry['skipped']}")


def sync_check(args) -> None:
    results = [
        check_clip(c, args.cache_dir, args.gamestate_dir)
        for c in bench.load_manifest(args.manifest)
        if c["match_id"] is not None
    ]
    for r in results:
        print(fmt_check(r))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2))
    if not all(sync_ok(r["peak"]) for r in results):
        raise SystemExit("the sync check failed: stop before any auto-labels (10-ball 3a)")


if __name__ == "__main__":
    main()
