"""08 ball marker on real video: ring, velocity arrow and trail on the footage itself.

    python -m demo.video --video clip.mp4 --match-id smoke04 --out ball.mp4 [--frames 100-400]

Reads a vision run's detections cache (data/vision_cache/<id>, 03) for the ball's box and
its game state (data/gamestate/<id>, 02) for the ball's pitch position and vx, vy. The
cache keeps no homography, so each frame's pitch->screen mapping is refit from that
frame's people (08 "Ball marker on video"). Like debug mode, internal only.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import polars as pl

from demo import overlay as ov
from demo.render import open_writer

FIT_MAX_PX = 2.0  # worst refit error on the people it was fit to
MIN_POINTS = 4
MIN_SPREAD_M = 1.0  # people nearly on one line leave the mapping undetermined


def to_screen(H: np.ndarray, x: float, y: float) -> tuple[float, float] | None:
    """Pitch meters -> pixels, None for a point behind the camera (w <= 0)."""
    q = H @ np.array([x, y, 1.0])
    if q[2] <= 1e-9:
        return None
    return q[0] / q[2], q[1] / q[2]


def screen_mapping(rows: pl.DataFrame) -> np.ndarray | None:
    """One frame's pitch->screen homography, refit from its people rows (box bottom-center
    against pitch_x, pitch_y), or None when it can't be trusted."""
    p = rows.filter(pl.col("class") != "ball", pl.col("pitch_x").is_not_null())
    if p.height < MIN_POINTS:
        return None
    src = np.column_stack([p["pitch_x"].to_numpy(), p["pitch_y"].to_numpy()])
    dst = np.column_stack([((p["x1"] + p["x2"]) / 2).to_numpy(), p["y2"].to_numpy()])
    if np.linalg.svd(src - src.mean(axis=0), compute_uv=False)[-1] < MIN_SPREAD_M:
        return None
    H, _ = cv2.findHomography(src, dst, 0)
    if H is None or not np.isfinite(H).all():
        return None
    back = [to_screen(H, x, y) for x, y in src]
    if any(b is None for b in back):
        return None
    if np.hypot(*(np.array(back) - dst).T).max() > FIT_MAX_PX:
        return None
    return H


def as_point(p: tuple[float, float] | None, w: int, h: int) -> ov.Point | None:
    """Pixel point for drawing, None if missing or more than a frame's size off the image
    (cv2 needs int32, and a point near the horizon can be huge)."""
    if p is None or not (-w <= p[0] <= 2 * w and -h <= p[1] <= 2 * h):
        return None
    return round(p[0]), round(p[1])


class BallVideo:
    """A vision run loaded for drawing the ball marker frame by frame."""

    def __init__(self, cache: Path, gamestate: Path):
        det = pl.read_parquet(cache / "detections.parquet")
        views = pl.read_parquet(cache / "view.parquet")
        self.first, self.last = int(views["frame_id"].min()), int(views["frame_id"].max())
        view = dict(views.select("frame_id", "view").iter_rows())
        fps = float(pl.read_parquet(gamestate / "match.parquet")["native_fps"][0])
        self.trail_n = round(ov.TRAIL_S * fps)
        gs = pl.read_parquet(gamestate / "objects.parquet").filter(pl.col("object_type") == "ball")
        self.ball_gs = {(r["frame_id"], r["object_id"]): r for r in gs.iter_rows(named=True)}
        self.ball_det = {
            r["frame_id"]: r
            for r in det.filter(pl.col("class") == "ball")
            .unique("frame_id", keep="first")
            .iter_rows(named=True)
        }
        self.mapping: dict[int, np.ndarray | None] = {}
        for (fid,), rows in det.partition_by("frame_id", as_dict=True).items():
            usable = view.get(fid) == "match" and bool(rows["homography_ok"][0])
            self.mapping[fid] = screen_mapping(rows) if usable else None

    def ball_xy(self, frame_id: int) -> dict | None:
        d = self.ball_det.get(frame_id)
        return None if d is None else self.ball_gs.get((frame_id, d["object_id"]))

    def draw(self, image: np.ndarray, frame_id: int) -> np.ndarray:
        out = image.copy()
        h, w = out.shape[:2]
        d = self.ball_det.get(frame_id)
        if d is None:
            return out
        H = self.mapping.get(frame_id)
        g = self.ball_xy(frame_id)
        if H is not None:
            trail = []
            for k in range(frame_id, frame_id - self.trail_n, -1):
                if self.mapping.get(k) is None:
                    break  # a cut or close-up: the trail doesn't reach past it
                b = self.ball_xy(k)
                trail.append(None if b is None else as_point(to_screen(H, b["x"], b["y"]), w, h))
            ov.ball_trail(out, trail[::-1])
        center = (round((d["x1"] + d["x2"]) / 2), round((d["y1"] + d["y2"]) / 2))
        radius = max(round((d["x2"] - d["x1"]) * 0.9), 6)
        tip = None
        if H is not None and g is not None and g["vx"] is not None and g["vy"] is not None:
            tip = as_point(
                to_screen(H, g["x"] + ov.ARROW_S * g["vx"], g["y"] + ov.ARROW_S * g["vy"]), w, h
            )
        style = (
            ov.tint(g["interpolated"], g["confidence"])
            if g
            else ("guessed" if d["tracked_only"] else "solid")
        )
        ov.ball_marker(out, center, radius, style, tip)
        return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="demo.video")
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--match-id", required=True)
    ap.add_argument("--cache-dir", type=Path, default=Path("data/vision_cache"))
    ap.add_argument("--gamestate-dir", type=Path, default=Path("data/gamestate"))
    ap.add_argument("--frames", help="first-last inside the processed range (default: all of it)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    bv = BallVideo(args.cache_dir / args.match_id, args.gamestate_dir / args.match_id)
    first, last = bv.first, bv.last
    if args.frames:
        a, b = (int(v) for v in args.frames.split("-"))
        first, last = max(a, first), min(b, last)
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = open_writer(args.out, fps, size)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    n, arrows = 0, 0
    for frame_id in range(first, last + 1):
        ok, image = cap.read()
        if not ok:
            break
        writer.write(bv.draw(image, frame_id))
        n += 1
        arrows += bv.mapping.get(frame_id) is not None and frame_id in bv.ball_det
    writer.release()
    print(
        f"wrote {args.out} ({n} frames {first}-{first + n - 1}, ball with a usable mapping on {arrows})"
    )


if __name__ == "__main__":
    main()
