"""Ball labels for the vision bench (07 Ball labels): click the ball on every 5th frame.

    PYTHONPATH=. python scripts/ball_click.py --clip vb02-ned-arg
    PYTHONPATH=. python scripts/ball_click.py --clip vb02-ned-arg --frames 240-280,1505-1590   # fix some
    PYTHONPATH=. python scripts/ball_click.py --clip vb01-arg-fra --assist     # PFF pre-placement
    PYTHONPATH=. python scripts/ball_click.py --clip vb02-ned-arg --flag       # second look
    PYTHONPATH=. python scripts/ball_click.py --clip demo01-arg-fra-81 \
        --manifest data/splits/demo_clips.json --assist

Only click a ball you can see. If you can't see it (in the air against the crowd, hidden
in a group of players, off screen), press space; if you can't tell, press u. Don't click
where it probably is: a guess scores vision against a ball that isn't in the image.

First run per clip decodes the video once, in order (no seeking: it can land on the wrong
frame), and saves the frames to label as JPGs under data/vision_bench/ball_frames/<clip>/.
If the clip's run cache is there, each decoded frame's grass share is checked against
view.parquet, so the frame index is known to be the run's. Then the window opens.

    left click   ball here (center), next frame
    Enter        accept the suggestion (cyan ring); on a spot-check or flagged frame with
                 no suggestion, keep the label as it is
    k            keep this frame's label as it is (spot check, flags), next
    space / n    not visible (off screen, hidden), next
    u            can't tell, next
    a            back one        d   forward one, label unchanged
    f            flick to the previous labeled frame and back (to see the ball move)
    x            clear this frame's label
    q / Esc      quit (every label is saved as you go; rerun to resume)

Yellow ring: the previous frame's label. Green: this frame's. The box in the corner
magnifies the area under the mouse 4x.

--assist (10-ball 1c) needs the clip's PFF reference (python -m vision.ball_truth). Magenta:
PFF's ball, solid where PFF tracks it (VISIBLE), dashed where it only estimates it. Cyan: the
suggestion, the ball candidate nearest PFF's ball. Frames where a confident candidate sits
right on PFF's ball, alone, are labeled without being shown (auto-accept, listed under
"auto"); a random tenth of them is shown first as a spot check. If more than 3% of those
get changed, auto-accept turns off for the clip and its auto frames come back to you.
--no-auto keeps the suggestions without auto-accept. --candidates FILE takes the candidates
from another model's run (balls.parquet's columns) instead of the clip's balls.parquet.

--flag opens the frames whose label disagrees with PFF: a ball more than 30 px from PFF's,
or "none" where a candidate sits on PFF's ball. A flag is a prompt, not an error: PFF's
ball drifts by up to 1 m for seconds, so press k when the label is right.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import polars as pl

from converters.common import sha256
from vision import ball_assist as ba
from vision import bench
from vision.ball_truth import extract

VIDEOS = Path("data/vision_bench/videos.json")
FRAMES = Path("data/vision_bench/ball_frames")
CACHE = Path("data/vision_cache")
ZOOM, ZOOM_HALF = 4, 30  # magnifier: 61 x 61 source px at 4x
RING_PX = 15  # 07's hit radius, drawn around labels
SAME_PX = 3.0  # a decision this close to the auto label didn't change it
ENTER = 13
PFF_COLOR, SUGGEST_COLOR = (255, 0, 255), (255, 255, 0)


def dashed_circle(image, center, radius, color):
    for start in range(0, 360, 30):
        cv2.ellipse(image, center, (radius, radius), 0, start, start + 15, color, 2)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="ball_click")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--manifest", type=Path, default=bench.MANIFEST)
    ap.add_argument("--scale", type=float, default=0.85, help="window size vs 1080p")
    ap.add_argument("--labels", type=Path, default=bench.BALL_LABELS)
    ap.add_argument(
        "--frames", help="step through only these source frames, to fix labels: 240-280,1060"
    )
    ap.add_argument("--assist", action="store_true", help="PFF pre-placement (10-ball 1c)")
    ap.add_argument("--no-auto", action="store_true", help="--assist without auto-accept")
    ap.add_argument("--candidates", type=Path, help="another run's balls.parquet")
    ap.add_argument("--flag", action="store_true", help="labels that disagree with PFF")
    args = ap.parse_args(argv)

    clip = next((c for c in bench.load_manifest(args.manifest) if c["clip_id"] == args.clip), None)
    if clip is None:
        raise SystemExit(f"{args.clip} isn't in {args.manifest}")
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

    truth, cands = None, {}
    if args.assist or args.flag:
        truth, cands = ba.clip_inputs(clip)
        if truth is None:
            raise SystemExit(
                f"no PFF reference for {args.clip}: python -m vision.ball_truth --clip {args.clip}"
                + (f" --manifest {args.manifest}" if args.manifest != bench.MANIFEST else "")
            )
        if args.candidates:
            start = json.loads((cache / "run.json").read_text())["video_start_s"]
            cands = ba.cands_by_src(pl.read_parquet(args.candidates), round(start * fps))
    proj = {} if truth is None else ba._proj(truth)

    sample: list[int] = []
    if args.assist and not args.no_auto and not entry.get("auto_off"):
        auto = entry.setdefault("auto", [])
        n = 0
        for i in idx:
            kind, u, v = proj.get(i, (None, None, None))
            if i in labels or u is None:
                continue
            hit = ba.auto_accept(cands.get(i, []), (u, v), kind)
            if hit is not None:
                labels[i] = [round(hit[0], 1), round(hit[1], 1)]
                auto.append(i)
                n += 1
        save()
        print(f"auto-accepted {n} frames ({len(auto)} auto on this clip)")
        sample = ba.spot_sample(auto)
    spot_todo = [i for i in sample if i not in entry.get("spot_checked", [])]

    def order() -> list[int]:
        if args.flag:
            return ba.unresolved_flags(entry, truth, cands)
        if args.frames:
            spans = [[int(v) for v in part.split("-")] for part in args.frames.split(",")]
            return [i for i in idx if any(sp[0] <= i <= sp[-1] for sp in spans)]
        if args.assist:
            pending = [i for i in spot_todo if i not in entry.get("spot_checked", [])]
            return pending + [i for i in idx if i not in labels]
        return idx

    steps = order()
    if not steps:
        print("nothing to look at: no unresolved flags" if args.flag else "nothing to label")
        return
    pos = (
        0
        if (args.flag or args.frames or args.assist)
        else next((k for k, i in enumerate(steps) if i not in labels), len(steps) - 1)
    )
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

    def decide(i, new) -> bool:
        """Record a person's decision on frame i. True when the spot check turned
        auto-accept off (the stepping list changes)."""
        old = labels.get(i)
        if new is not None:
            labels[i] = new
        auto = entry.get("auto", [])
        same = (
            isinstance(old, list)
            and isinstance(labels.get(i), list)
            and np.hypot(old[0] - labels[i][0], old[1] - labels[i][1]) <= SAME_PX
        )
        if i in auto and not same:
            auto.remove(i)  # a person's label now
        if args.flag:
            entry.setdefault("flag_checked", []).append(i)
        off = False
        if i in sample and i not in entry.get("spot_checked", []):
            entry.setdefault("spot_checked", []).append(i)
            off = ba.apply_spot(entry, sample)
            if off:
                checked, changed, _ = ba.spot_result(entry)
                print(
                    f"spot check: {changed} of {checked} changed, auto-accept off for this "
                    "clip; the auto frames are back in the list"
                )
            elif set(sample) <= set(entry["spot_checked"]):
                checked, changed, share = ba.spot_result(entry)
                print(f"spot check done: {changed} of {checked} changed ({share:.0%}), kept")
        save()
        return off

    while True:
        i = steps[pos]
        shown = steps[pos - 1] if flick and pos > 0 else i
        image = load(shown).copy()
        prev = labels.get(steps[pos - 1]) if pos > 0 else None
        if isinstance(prev, list):
            cv2.circle(image, (round(prev[0]), round(prev[1])), RING_PX, (0, 220, 255), 2)
        kind, u, v = proj.get(i, (None, None, None))
        sug = None
        if truth is not None and u is not None and not flick:
            d = truth.filter(pl.col("src") == i)["diam_px"][0]
            r = max(round((d or 12) / 2), 6)
            if kind == "pff":
                cv2.circle(image, (round(u), round(v)), r, PFF_COLOR, 2)
            else:
                dashed_circle(image, (round(u), round(v)), r, PFF_COLOR)
            sug = ba.suggest(cands.get(i, []), (u, v), kind)
            if sug is not None:
                cv2.circle(image, (round(sug[0]), round(sug[1])), RING_PX + 4, SUGGEST_COLOR, 2)
                cv2.putText(
                    image,
                    f"{sug[2]:.2f}",
                    (round(sug[0]) + RING_PX + 6, round(sug[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    SUGGEST_COLOR,
                    2,
                )
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
        tags = []
        if i in sample and i not in entry.get("spot_checked", []):
            tags.append("[SPOT CHECK]")
        if args.flag:
            tags.append(f"[FLAG: {ba.flag_reason(labels, truth, cands, i) or 'resolved'}]")
        if kind is not None:
            tags.append(f"PFF {kind}")
        text = (
            f"{args.clip}  {pos + 1}/{len(steps)}  frame {i}  {i / fps:.2f} s  "
            f"labeled {done}/{len(idx)}  now: {cur if cur is not None else '-'}  "
            + " ".join(tags)
            + ("  [PREVIOUS FRAME]" if flick else "")
        )
        cv2.rectangle(view, (0, 0), (view.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(view, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow(win, view)
        key = cv2.waitKey(30) & 0xFF
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        decided, off = False, False
        if clicked and flick:
            clicked.clear()  # the label is for this frame, not the one flicked to
        elif clicked:
            off = decide(i, list(clicked[-1]))
            clicked.clear()
            decided = True
        elif key in (ord(" "), ord("n"), ord("u")):
            off = decide(i, bench.BALL_UNSURE if key == ord("u") else bench.BALL_NONE)
            decided = True
        elif key == ENTER and sug is not None:
            off = decide(i, [round(sug[0], 1), round(sug[1], 1)])
            decided = True
        elif (key == ord("k") or key == ENTER) and cur is not None and (args.flag or i in sample):
            off = decide(i, None)  # keep the label as it is
            decided = True
        elif key in (ord("a"), 8):
            pos, flick = max(pos - 1, 0), False
        elif key == ord("d"):
            pos, flick = min(pos + 1, len(steps) - 1), False
        elif key == ord("f"):
            flick = not flick
        elif key == ord("x"):
            labels.pop(i, None)
            if i in entry.get("auto", []):
                entry["auto"].remove(i)
            save()
        elif key in (ord("q"), 27):
            break
        if decided:
            flick = False
            if off:
                steps, pos = order(), 0
                if not steps:
                    break
            else:
                pos = min(pos + 1, len(steps) - 1)
    cv2.destroyAllWindows()
    done = sum(j in labels for j in idx)
    print(f"{args.clip}: {done} of {len(idx)} frames labeled -> {args.labels}")
    if truth is not None:
        print(*ba.status(args.labels), sep="\n")


if __name__ == "__main__":
    main()
