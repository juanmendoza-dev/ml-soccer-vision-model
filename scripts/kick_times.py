"""Kicks in PFF's ball track, for syncing a bench clip without cut edges (07 Sync, rung 3).

    PYTHONPATH=. python scripts/kick_times.py --match-id 3854 --period 1 --from-s 265 --to-s 354

Lists the moments the ball goes from slow to fast on rows that are `visible` and not
`interpolated`. The kick is put at the last position before the jump. PFF's ball moves every
other frame (~15 Hz), so each kick is good to about one update (~67 ms). With `--offset`
(PFF time minus video time, from the scoreboard), it also prints the video time to look at.
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl

SLOW_MS = 7.0  # ball speed before the kick, below this
FAST_MS = 9.0  # first step after it, above this
AFTER_MS = 10.0  # and the mean over the next WINDOW_S, above this (PFF smooths the jump)
WINDOW_S = 0.2  # how long each side has to hold
MAX_GAP_S = 0.1  # a longer gap between real positions isn't one stretch


def ball_updates(gamestate_dir: Path, match_id: str, period: int, t0: float, t1: float):
    frames = pl.read_parquet(gamestate_dir / match_id / "frames.parquet").filter(
        (pl.col("period") == period) & pl.col("timestamp_s").is_between(t0, t1)
    )
    ball = (
        pl.read_parquet(gamestate_dir / match_id / "objects.parquet")
        .filter((pl.col("object_type") == "ball") & pl.col("visible") & ~pl.col("interpolated"))
        .join(frames.select("frame_id", "timestamp_s"), on="frame_id")
        .sort("timestamp_s")
    )
    # PFF repeats each position on two frames; keep the first frame of each new position
    moved = (pl.col("x") != pl.col("x").shift()) | (pl.col("y") != pl.col("y").shift())
    return ball.filter(moved.fill_null(True)).select("timestamp_s", "x", "y", "z")


def kicks(ball: pl.DataFrame) -> list[dict]:
    t = ball["timestamp_s"].to_numpy()
    xy = ball.select("x", "y").to_numpy()
    out = []
    for i in range(1, len(t) - 1):
        before = (t >= t[i] - WINDOW_S) & (t <= t[i])
        after = (t >= t[i]) & (t <= t[i] + WINDOW_S)
        if before.sum() < 2 or after.sum() < 3:
            continue
        span = np.r_[t[before], t[after]]
        if np.diff(span).max() > MAX_GAP_S:
            continue
        a = np.flatnonzero(after)
        v_after = np.linalg.norm(xy[a[-1]] - xy[a[0]]) / (t[a[-1]] - t[a[0]])
        step_before = np.linalg.norm(xy[i] - xy[i - 1]) / (t[i] - t[i - 1])
        step_after = np.linalg.norm(xy[i + 1] - xy[i]) / (t[i + 1] - t[i])
        if step_before < SLOW_MS and step_after > FAST_MS and v_after > AFTER_MS:
            out.append(
                {"timestamp_s": t[i], "x": xy[i, 0], "y": xy[i, 1], "v_after": v_after}
            )
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match-id", required=True)
    ap.add_argument("--period", type=int, required=True)
    ap.add_argument("--from-s", type=float, required=True, help="PFF timestamp_s")
    ap.add_argument("--to-s", type=float, required=True)
    ap.add_argument("--offset", type=float, help="PFF time minus video time, roughly")
    ap.add_argument("--gamestate-dir", type=Path, default=Path("data/gamestate"))
    args = ap.parse_args(argv)

    ball = ball_updates(args.gamestate_dir, args.match_id, args.period, args.from_s, args.to_s)
    for k in kicks(ball):
        line = f"{k['timestamp_s']:9.3f}  x {k['x']:6.1f}  y {k['y']:6.1f}  {k['v_after']:4.1f} m/s"
        if args.offset is not None:
            line += f"  video ~{k['timestamp_s'] - args.offset:7.2f}"
        print(line)


if __name__ == "__main__":
    main()
