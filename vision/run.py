"""Video -> game state (02) + detections cache (03), offline.

    python -m vision.run --video clip.mp4 --weights-dir <roboflow sports data/> --match-id demo1

Weights are roboflow/sports' (examples/soccer/setup.sh downloads them).
Then look at it with: python -m demo.debug --video clip.mp4 --cache data/vision_cache/demo1
"""

import argparse
import math
import time
from pathlib import Path

import cv2

from converters.common import sha256
from vision.config import VisionConfig
from vision.pipeline import Stages, VisionPipeline
from vision.stages import (
    BALL_WEIGHTS,
    PITCH_WEIGHTS,
    PLAYER_WEIGHTS,
    BallAndPeopleDetector,
    ByteTracker,
    KitColorTeams,
    YoloDetector,
    YoloKeypoints,
)
from vision.types import BALL
from vision.writer import GameStateWriter


def build_stages(weights_dir: Path, device: str, fps: float, config: VisionConfig) -> Stages:
    conf = config.track_min_conf  # the pipeline filters the ball and new tracks higher
    people = YoloDetector(weights_dir / PLAYER_WEIGHTS, device=device, conf=conf)
    detector = people
    if (weights_dir / BALL_WEIGHTS).exists():
        ball = YoloDetector(weights_dir / BALL_WEIGHTS, device=device, conf=conf, required=(BALL,))
        detector = BallAndPeopleDetector(people, ball)
    return Stages(
        detector=detector,
        tracker=ByteTracker(fps / config.detect_every, config.lost_track_s),
        keypoints=YoloKeypoints(weights_dir / PITCH_WEIGHTS, device=device),
        teams=KitColorTeams(),
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="vision.run")
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--weights-dir", type=Path, required=True)
    ap.add_argument("--match-id", required=True)
    ap.add_argument("--home", default="home")
    ap.add_argument("--away", default="away")
    ap.add_argument("--home-cluster", type=int, choices=[0, 1], help="kit cluster that is home")
    ap.add_argument(
        "--home-attacks-left", action="store_true", help="home attacks TV-left in period 1"
    )
    ap.add_argument("--period", type=int, default=1)
    ap.add_argument("--device", default="cuda", help="cuda, cpu or mps")
    ap.add_argument("--detect-every", type=int, default=1)
    ap.add_argument("--fps", type=float, help="override when the video reports none or a bad one")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--gamestate-dir", type=Path, default=Path("data/gamestate"))
    ap.add_argument("--cache-dir", type=Path, default=Path("data/vision_cache"))
    args = ap.parse_args(argv)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"can't open {args.video}")
    fps = args.fps or cap.get(cv2.CAP_PROP_FPS)
    if not (math.isfinite(fps) and fps > 0):
        raise SystemExit(f"{args.video} reports fps {fps}; pass --fps")
    try:
        config = VisionConfig(
            detect_every=args.detect_every,
            home_cluster=args.home_cluster,
            home_attacks_tv_right_p1=not args.home_attacks_left,
            period=args.period,
        )
    except ValueError as e:
        raise SystemExit(f"bad config: {e}") from None
    pipe = VisionPipeline(config, build_stages(args.weights_dir, args.device, fps, config))
    writer = GameStateWriter(
        args.match_id,
        args.home,
        args.away,
        fps,
        config,
        args.gamestate_dir,
        args.cache_dir,
        run_info={
            "video": str(args.video),
            "video_sha256": sha256(args.video),
            "weights": {p.name: sha256(p) for p in sorted(args.weights_dir.glob("*.pt"))},
            "device": args.device,
        },
    )

    start = time.perf_counter()
    frame_id = 0
    while args.max_frames is None or frame_id < args.max_frames:
        ok, image = cap.read()
        if not ok:
            break
        writer.add(pipe.step(frame_id, frame_id / fps, image))
        frame_id += 1
        if frame_id % 100 == 0:
            rate = frame_id / (time.perf_counter() - start)
            print(f"  {frame_id} frames, {rate:.1f} fps, view {pipe.gate.view}")
    cap.release()
    if frame_id == 0:
        raise SystemExit(f"no frames decoded from {args.video}")
    out = writer.close()
    rate = frame_id / max(time.perf_counter() - start, 1e-9)
    print(f"{frame_id} frames at {rate:.1f} fps -> {out}")
    if args.home_cluster is None:
        print("team is null until you pass --home-cluster (check the colors in demo.debug)")
    if writer.errors:
        print("02 validation failed:", *writer.errors, sep="\n  ")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
