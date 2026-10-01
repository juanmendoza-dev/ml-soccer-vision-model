"""08 debug mode: boxes, IDs, teams and a minimap from the detections cache (03).

    python -m demo.debug --video clip.mp4 --cache data/vision_cache/<match_id> --out debug.mp4
    python -m demo.debug ... --frames 1200-1500
    python -m demo.debug ... --shape box

For finding where vision went wrong. Internal only, never published (08).
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import polars as pl

from demo.pitch import BALL, REFEREE, UNKNOWN, Pitch

# BGR, vision's team clusters
CLUSTER_COLORS = {0: (60, 60, 230), 1: (230, 140, 40)}
MINIMAP = Pitch(scale=3, pad=8)  # px per meter, px of grass around


def draw_minimap(rows: pl.DataFrame) -> np.ndarray:
    img = MINIMAP.blank()
    MINIMAP.draw_lines(img)
    for r in rows.filter(pl.col("pitch_x").is_not_null()).iter_rows(named=True):
        size = 3 if r["class"] == "ball" else 5
        cv2.circle(img, MINIMAP.px(r["pitch_x"], r["pitch_y"]), size, _color(r), -1)
    return img


def _color(r: dict) -> tuple[int, int, int]:
    if r["class"] == "ball":
        return BALL
    if r["class"] == "referee":
        return REFEREE
    return CLUSTER_COLORS.get(r["team_cluster"], UNKNOWN)


def annotate(image: np.ndarray, rows: pl.DataFrame, view: str | None, shape: str = "ring") -> np.ndarray:
    out = image.copy()
    for r in rows.iter_rows(named=True):
        x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
        color = _color(r)
        if r["class"] == "ball":
            cx = int((x1 + x2) / 2)
            cv2.drawMarker(out, (cx, int(y1) - 6), color, cv2.MARKER_TRIANGLE_DOWN, 14, 2)
            continue
        thickness = 1 if r["tracked_only"] else 2
        if shape == "box":
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            label_pos = (int(x1), int(y1) - 6)
        else:
            center = (int((x1 + x2) / 2), int(y2))
            axes = (max(int((x2 - x1) * 0.7), 6), max(int((x2 - x1) * 0.25), 3))
            cv2.ellipse(out, center, axes, 0, -45, 235, color, thickness)
            label_pos = (center[0] - 8, center[1] + 18)
        label = r["object_id"].split("-")[-1]
        cv2.putText(out, label, label_pos, 0, 0.45, color, 1, cv2.LINE_AA)

    minimap = draw_minimap(rows)
    mh, mw = minimap.shape[:2]
    h, w = out.shape[:2]
    if mw < w and mh < h:
        y0, x0 = h - mh - 16, w - mw - 16
        if view != "match":
            minimap = (minimap * 0.35).astype(np.uint8)
        out[y0 : y0 + mh, x0 : x0 + mw] = minimap
    if view != "match":
        cv2.putText(
            out, "PAUSED (not a match view)", (24, 48), 0, 1.0, (255, 255, 255), 2, cv2.LINE_AA
        )
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="demo.debug")
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True, help="data/vision_cache/<match_id>")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", help="first-last, e.g. 1200-1500")
    ap.add_argument("--shape", choices=["ring", "box"], default="ring")
    args = ap.parse_args(argv)

    det = pl.read_parquet(args.cache / "detections.parquet")
    views = dict(
        pl.read_parquet(args.cache / "view.parquet").select("frame_id", "view").iter_rows()
    )
    first, last = 0, None
    if args.frames:
        first, last = (int(v) for v in args.frames.split("-"))

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    by_frame = det.partition_by("frame_id", as_dict=True)
    empty = det.clear()
    frame_id = first
    while last is None or frame_id <= last:
        ok, image = cap.read()
        if not ok:
            break
        rows = by_frame.get((frame_id,), empty)
        writer.write(annotate(image, rows, views.get(frame_id), args.shape))
        frame_id += 1
    writer.release()
    print(f"wrote {args.out} ({frame_id - first} frames)")


if __name__ == "__main__":
    main()
