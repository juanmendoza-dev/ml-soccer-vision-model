"""08 pitch view: a top-down render of one stretch of a match from game state alone.

    python -m demo.render --match 10502 --goal 1 --out goal1.mp4
    python -m demo.render --match 10502 --frames 22000-22600 --out clip.mp4 [--step 2]

--goal N renders the N-th goal (1-based) from --before s ahead of it to --after s past it.
Draws the ball ring, arrow and trail, players with speed rings and the uncertainty tint,
the shooting-lane cone, event markers, the possession panel and the ticker (08 "Element
definitions"). Every frame uses frames <= t only. A PFF render is internal until PFF's
terms of use are read (roadmap), like any other PFF output.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import polars as pl

from demo import overlay as ov
from demo import tally
from demo.pitch import AWAY, HOME, UNKNOWN, Pitch

HEADER, FOOTER = 110, 50
PITCH = Pitch(scale=10, pad=30, y0=HEADER)
PLAYER_R, BALL_R = 9, 6
FLASH_S = 2.0  # a goal flashes the ticker for this long, 4 times a second
COLORS = {"home": HOME, "away": AWAY}
OBJECT_COLS = [
    "frame_id",
    "object_id",
    "object_type",
    "team",
    "x",
    "y",
    "vx",
    "vy",
    "visible",
    "interpolated",
    "confidence",
]


class Scene:
    """One match loaded for rendering: frames, events, every ball row (for the tallies
    and the trail), the possession tally, and the players of frames [first - trail, last]."""

    def __init__(self, match_dir: Path, first: int, last: int):
        match = pl.read_parquet(match_dir / "match.parquet").row(0, named=True)
        self.names = {"home": match["home_team"], "away": match["away_team"]}
        self.fps = float(match["native_fps"])
        self.trail_n = round(ov.TRAIL_S * self.fps)
        self.frames = pl.read_parquet(match_dir / "frames.parquet").sort("frame_id")
        self.events = pl.read_parquet(match_dir / "events.parquet")
        objects = pl.scan_parquet(match_dir / "objects.parquet").select(OBJECT_COLS)
        self.ball = (
            objects.filter(pl.col("object_type") == "ball")
            .collect()
            .unique("frame_id", keep="first")
        )
        self.tally = tally.possession_tally(self.frames, self.ball)
        players = objects.filter(
            pl.col("object_type").is_in(tally.PLAYER_TYPES),
            pl.col("frame_id").is_between(first - self.trail_n, last),
        ).collect()
        self.players = players.partition_by("frame_id", as_dict=True)
        self.empty = players.clear()
        self.ball_at = {r["frame_id"]: r for r in self.ball.iter_rows(named=True)}
        self.frame_at = {
            r["frame_id"]: r
            for r in self.frames.filter(pl.col("frame_id").is_between(first, last)).iter_rows(
                named=True
            )
        }
        self.tally_at = {
            r["frame_id"]: r
            for r in self.tally.filter(pl.col("frame_id").is_between(first, last)).iter_rows(
                named=True
            )
        }
        self.size = (PITCH.size[0], HEADER + PITCH.size[1] + FOOTER)

    def frame_ids(self, first: int, last: int, step: int = 1) -> list[int]:
        return sorted(f for f in self.frame_at if first <= f <= last)[::step]

    def draw(self, frame_id: int) -> np.ndarray:
        f = self.frame_at[frame_id]
        w, h = self.size
        img = np.full((h, w, 3), ov.PANEL_BG, dtype=np.uint8)
        img[HEADER : HEADER + PITCH.size[1]] = PITCH.blank()
        PITCH.draw_lines(img, goals=True)
        players = self.players.get((frame_id,), self.empty)
        ball = self.ball_at.get(frame_id)

        if ball is not None and f["ball_state"] == "alive":
            cone = tally.lane(
                players, ball["x"], ball["y"], f["possession_team"], f["home_attacks_positive_x"]
            )
            if cone is not None:
                tri, n = cone
                ov.lane_cone(img, [PITCH.px(x, y) for x, y in tri], n)

        items = tally.ticker(self.events, frame_id, self.fps)
        for e in items:
            if e["x"] is not None and e["y"] is not None:
                ov.event_marker(img, PITCH.px(e["x"], e["y"]), e["event_type"] == "goal")

        trail = []
        for k in range(frame_id - self.trail_n + 1, frame_id + 1):
            b = self.ball_at.get(k)
            trail.append(None if b is None else PITCH.px(b["x"], b["y"]))
        ov.ball_trail(img, trail)

        for p in players.sort("object_id").iter_rows(named=True):
            ov.player_marker(
                img,
                PITCH.px(p["x"], p["y"]),
                PLAYER_R,
                COLORS.get(p["team"], UNKNOWN),
                ov.speed_band(p["vx"], p["vy"]),
                ov.tint(p["interpolated"], p["confidence"]),
                p["object_id"].split("_")[-1],
            )

        if ball is not None:
            tip = None
            if ball["vx"] is not None and ball["vy"] is not None:
                tip = PITCH.px(
                    ball["x"] + ov.ARROW_S * ball["vx"], ball["y"] + ov.ARROW_S * ball["vy"]
                )
            center = PITCH.px(ball["x"], ball["y"])
            ov.ball_marker(
                img, center, BALL_R, ov.tint(ball["interpolated"], ball["confidence"]), tip
            )

        ov.possession_panel(
            img,
            (0, 0, w, HEADER),
            self.names,
            COLORS,
            tally.shares(self.tally_at[frame_id]),
            tally.score(self.events, frame_id),
            tally.clock(f["period"], f["timestamp_s"]),
        )
        lines = [(tally.event_text(e, self.names), e["event_type"] == "goal") for e in items]
        goal = next((e for e in items if e["event_type"] == "goal"), None)
        flash = False
        if goal is not None:
            since = frame_id - goal["frame_id"]
            quarter = max(round(self.fps / 4), 1)
            flash = since < FLASH_S * self.fps and (since // quarter) % 2 == 0
        ov.ticker_strip(img, (0, h - FOOTER, w, FOOTER), lines, flash)
        return img


def goal_window(match_dir: Path, n: int, before: float, after: float) -> tuple[int, int]:
    """Frames from `before` s ahead of the n-th goal (1-based) to `after` s past it, kept
    inside the goal's period."""
    goals = pl.read_parquet(match_dir / "events.parquet").filter(pl.col("event_type") == "goal")
    goals = goals.sort("frame_id")
    if not 1 <= n <= goals.height:
        raise SystemExit(f"--goal {n}: the match has {goals.height} goals")
    g = goals["frame_id"][n - 1]
    fps = pl.read_parquet(match_dir / "match.parquet")["native_fps"][0]
    frames = pl.read_parquet(match_dir / "frames.parquet", columns=["frame_id", "period"])
    period = frames.filter(pl.col("frame_id") == g)["period"][0]
    ids = frames.filter(pl.col("period") == period)["frame_id"]
    return max(g - round(before * fps), ids.min()), min(g + round(after * fps), ids.max())


def open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """H.264 where OpenCV has it (plays in QuickTime), else MPEG-4 part 2."""
    for code in ("avc1", "mp4v"):
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*code), fps, size)
        if w.isOpened():
            return w
    raise SystemExit(f"no video encoder for {path}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="demo.render")
    ap.add_argument("--match", required=True, help="match_id under --gamestate")
    ap.add_argument("--gamestate", type=Path, default=Path("data/gamestate"))
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--frames", help="first-last frame_id, e.g. 22000-22600")
    where.add_argument("--goal", type=int, help="render around the n-th goal (1-based)")
    ap.add_argument("--before", type=float, default=15.0, help="s before the goal")
    ap.add_argument("--after", type=float, default=5.0, help="s after the goal")
    ap.add_argument("--step", type=int, default=1, help="draw every n-th native frame")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    d = args.gamestate / args.match
    if args.goal is not None:
        first, last = goal_window(d, args.goal, args.before, args.after)
    else:
        first, last = (int(v) for v in args.frames.split("-"))
    scene = Scene(d, first, last)
    ids = scene.frame_ids(first, last, args.step)
    if not ids:
        raise SystemExit(f"no frames in {first}-{last}")
    writer = open_writer(args.out, scene.fps / args.step, scene.size)
    for frame_id in ids:
        writer.write(scene.draw(frame_id))
    writer.release()
    print(f"wrote {args.out} ({len(ids)} frames, {first}-{last})")


if __name__ == "__main__":
    main()
