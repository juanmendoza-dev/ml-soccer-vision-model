"""Ball labels for the vision bench (07 Ball labels): click the ball on every 5th frame.

    PYTHONPATH=. python scripts/ball_click.py --clip vb02-ned-arg

First run per clip decodes the video once, in order (no seeking: it can land on the wrong
frame), and saves the frames to label as JPGs under data/vision_bench/ball_frames/<clip>/.
If the clip's run cache is there, each decoded frame's grass share is checked against
view.parquet, so the frame index is known to be the run's. Then the window opens.

    left click   ball here (center), next frame
    space / n    not visible (off screen, hidden), next
    u            can't tell, next
    a            back one        d   forward one, label unchanged
    f            flick to the previous labeled frame and back (to see the ball move)
    x            clear this frame's label
    q / Esc      quit (every label is saved as you go; rerun to resume)

Yellow ring: the previous frame's label. Green: this frame's. The box in the corner
magnifies the area under the mouse 4x.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import polars as pl

from converters.common import sha256
from vision import bench
from vision.view_gate import grass_share

VIDEOS = Path("data/vision_bench/videos.json")
FRAMES = Path("data/vision_bench/ball_frames")
CACHE = Path("data/vision_cache")
ZOOM, ZOOM_HALF = 4, 30  # magnifier: 61 x 61 source px at 4x
RING_PX = 15  # 07's hit radius, drawn around labels


def extract(video: Path, idx: list[int], out: Path, cache: Path | None) -> None:
    """Sequential decode, as vision.run does, saving the frames in idx."""
    out.mkdir(parents=True, exist_ok=True)
    todo = [i for i in idx if not (out / f"{i}.jpg").exists()]
    if not todo:
        return
    check = None
    if cache is not None and (cache / "view.parquet").exists():
        run = json.loads((cache / "run.json").read_text())
        view = pl.read_parquet(cache / "view.parquet", columns=["frame_id", "grass_share"])
        check = (run["video_start_s"], dict(view.iter_rows()))
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    want, last = set(todo), max(todo)
    checked = mismatched = 0
    print(f"decoding {video.name} up to frame {last} ({len(todo)} to save)...")
    for i in range(last + 1):
        if not cap.grab():
            raise SystemExit(f"{video} ends at frame {i}")
        if i not in want:
            continue
        ok, image = cap.retrieve()
        if not ok:
            raise SystemExit(f"can't decode frame {i}")
        if check is not None:
            run_frame = i - round(check[0] * fps)
            if run_frame in check[1]:
                checked += 1
                mismatched += grass_share(image) != check[1][run_frame]
        cv2.imwrite(str(out / f"{i}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cap.release()
    if check is None:
        print("no run cache to check the frame index against")
    elif mismatched:
        for p in out.glob("*.jpg"):
            p.unlink()
        raise SystemExit(f"{mismatched} of {checked} frames don't match the run: frame offset?")
    else:
        print(f"frame index checked against the run on {checked} frames")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="ball_click")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--scale", type=float, default=0.85, help="window size vs 1080p")
    ap.add_argument("--labels", type=Path, default=bench.BALL_LABELS)
    args = ap.parse_args(argv)

    clip = next((c for c in bench.load_manifest(bench.MANIFEST) if c["clip_id"] == args.clip), None)
    if clip is None:
        raise SystemExit(f"{args.clip} isn't in the manifest")
    video = Path(json.loads(VIDEOS.read_text())[args.clip])
    if sha256(video) != clip["video_sha256"]:
        raise SystemExit(f"{video} isn't the manifest's video (sha256)")
    fps = cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FPS)
    idx = bench.ball_label_frames(clip, fps)
    frames = FRAMES / args.clip
    cache = CACHE / args.clip
    extract(video, idx, frames, cache if cache.exists() else None)

    all_labels = bench.load_ball_labels(args.labels)
    entry = all_labels.setdefault(
        args.clip, {"video_sha256": clip["video_sha256"], "every": bench.BALL_EVERY, "labels": {}}
    )
    if entry["video_sha256"] != clip["video_sha256"] or entry["every"] != bench.BALL_EVERY:
        raise SystemExit("the labels file was made on another video or another N")
    labels = entry["labels"]

    def save():
        bench.save_ball_labels(all_labels, args.labels)

    pos = next((k for k, i in enumerate(idx) if i not in labels), len(idx) - 1)
    mouse = [None]
    s = args.scale
    flick = False
    win = f"ball_click {args.clip}"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    clicked = []

    def on_mouse(event, x, y, flags, param):
        mouse[0] = (x / s, y / s)
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append((round(x / s, 1), round(y / s, 1)))

    cv2.setMouseCallback(win, on_mouse)
    cache_img: dict[int, np.ndarray] = {}

    def load(i):
        if i not in cache_img:
            if len(cache_img) > 20:
                cache_img.clear()
            cache_img[i] = cv2.imread(str(frames / f"{i}.jpg"))
        return cache_img[i]

    while True:
        i = idx[pos]
        shown = idx[pos - 1] if flick and pos > 0 else i
        image = load(shown).copy()
        prev = labels.get(idx[pos - 1]) if pos > 0 else None
        if isinstance(prev, list):
            cv2.circle(image, (round(prev[0]), round(prev[1])), RING_PX, (0, 220, 255), 2)
        cur = labels.get(i)
        if isinstance(cur, list):
            cv2.circle(image, (round(cur[0]), round(cur[1])), RING_PX, (0, 255, 0), 2)
        view = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if mouse[0] is not None:  # magnifier from the full-resolution frame
            mx, my = (round(v) for v in mouse[0])
            src = load(shown)
            pad = cv2.copyMakeBorder(src, *[ZOOM_HALF] * 4, cv2.BORDER_CONSTANT)
            crop = pad[my : my + 2 * ZOOM_HALF + 1, mx : mx + 2 * ZOOM_HALF + 1]
            if crop.shape[:2] == (2 * ZOOM_HALF + 1,) * 2:
                big = cv2.resize(crop, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)
                c = ZOOM_HALF * ZOOM + ZOOM // 2
                cv2.line(big, (c, 0), (c, big.shape[0]), (0, 0, 255), 1)
                cv2.line(big, (0, c), (big.shape[1], c), (0, 0, 255), 1)
                view[: big.shape[0], view.shape[1] - big.shape[1] :] = big
        done = sum(j in labels for j in idx)
        text = (
            f"{args.clip}  {pos + 1}/{len(idx)}  frame {i}  {i / fps:.2f} s  "
            f"labeled {done}  now: {cur if cur is not None else '-'}"
            + ("  [PREVIOUS FRAME]" if flick else "")
        )
        cv2.rectangle(view, (0, 0), (view.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(view, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow(win, view)
        key = cv2.waitKey(30) & 0xFF
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if clicked and flick:
            clicked.clear()  # the label is for this frame, not the one flicked to
        elif clicked:
            labels[i] = list(clicked[-1])
            clicked.clear()
            flick = False
            save()
            pos = min(pos + 1, len(idx) - 1)
        elif key in (ord(" "), ord("n"), ord("u")):
            labels[i] = bench.BALL_UNSURE if key == ord("u") else bench.BALL_NONE
            flick = False
            save()
            pos = min(pos + 1, len(idx) - 1)
        elif key in (ord("a"), 8):
            pos, flick = max(pos - 1, 0), False
        elif key == ord("d"):
            pos, flick = min(pos + 1, len(idx) - 1), False
        elif key == ord("f"):
            flick = not flick
        elif key == ord("x"):
            labels.pop(i, None)
            save()
        elif key in (ord("q"), 27):
            break
    cv2.destroyAllWindows()
    done = sum(j in labels for j in idx)
    print(f"{args.clip}: {done} of {len(idx)} frames labeled -> {args.labels}")


if __name__ == "__main__":
    main()
