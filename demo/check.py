"""Demo regression check (ball fix plan, Phase 0): on a demo clip, the meter before the
goal next to PFF tracking's, false peaks, predicted share, possession agreement and the
vision ball against PFF's.

    python -m demo.check --clip demo01-arg-fra-81 --model-id goal-f0-2026-10-05
    python -m demo.check ... --rerun --variant gate --set ball_picker=gate   # stage 5 variant
    python -m demo.check ... --variant gate --render                         # private render

A variant lives in data/variants/<tag> (vision.replay --variant): its own vision cache, game
state and predictions. Vision's grid time t_s is PFF period time t_s + video_start_s +
the clip's sync offset, as in scripts/demo_compare.py. The PFF side is lgbm-held's
out-of-fold p_goal_cal_h5, the same fold-0 model and map as the demo model.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

from vision.bench import load_manifest, sync_offset
from vision.replay import VARIANT_ROOT

PFF_PGOAL = Path("data/runs/lgbm-held-2026-09-27/pgoal.parquet")
MODELS = Path("data/models/goal")
RENDERS = Path("C:/footage/renders")
VIDEOS = Path("data/vision_bench/videos.json")
BEFORES = (5, 3, 2, 1, 0.5)
PEAK_P = 0.02  # a row above this far from any shot is a false peak
PEAK_AWAY_S = 3.0
TABLE_S = 3.0  # per-row table over the last seconds before the goal


def join_rows(vis: pl.DataFrame, pff: pl.DataFrame, shift: float) -> pl.DataFrame:
    """Vision's grid rows on PFF's 10 Hz grid key k: vision and PFF p and possession."""
    v = vis.with_columns(k=((pl.col("t_s") + shift) * 10).round().cast(pl.Int64)).select(
        "k", "frame_id", "predicted", vis_poss="possession_team", vis_p="p_goal_h5"
    )
    p = pff.select("k", pff_poss="possession_team", pff_p="p_goal_cal_h5")
    return v.join(p, on="k", how="inner").sort("k").with_columns(t=pl.col("k") / 10)


def meter_at(j: pl.DataFrame, goal_t: float, befores=BEFORES) -> list[tuple]:
    out = []
    for b in befores:
        r = j.filter(pl.col("k") == round((goal_t - b) * 10))
        out.append((b, r["vis_p"][0] if r.height else None, r["pff_p"][0] if r.height else None))
    return out


def false_peaks(
    j: pl.DataFrame, shot_ts: list[float], p_min: float = PEAK_P, away_s: float = PEAK_AWAY_S
) -> pl.DataFrame:
    near = pl.lit(False)
    for s in shot_ts:
        near = near | ((pl.col("t") - s).abs() <= away_s)
    return j.filter((pl.col("vis_p") > p_min) & ~near)


def _dist_stats(d: np.ndarray) -> dict:
    if not len(d):
        return {"n": 0, "median_m": float("nan"), "p90_m": float("nan"), "within_2m": float("nan")}
    return {
        "n": len(d),
        "median_m": float(np.median(d)),
        "p90_m": float(np.quantile(d, 0.9)),
        "within_2m": float((d <= 2.0).mean()),
    }


def ball_distance(vis_ball: pl.DataFrame, pff_ball: pl.DataFrame) -> dict:
    """Vision ball (k, x, y) against PFF's (k, x, y, visible), split by PFF visibility."""
    j = vis_ball.join(pff_ball, on="k", suffix="_pff").with_columns(
        d=((pl.col("x") - pl.col("x_pff")) ** 2 + (pl.col("y") - pl.col("y_pff")) ** 2).sqrt()
    )
    return {
        name: _dist_stats(j.filter(pl.col("visible") == vis)["d"].to_numpy())
        for name, vis in (("visible", True), ("estimated", False))
    }


def pff_ball_grid(gs: Path, period: int) -> pl.DataFrame:
    """PFF's ball and ball state on its 10 Hz grid: the PFF frame nearest each k."""
    frames = pl.read_parquet(gs / "frames.parquet").filter(pl.col("period") == period)
    ball = pl.read_parquet(gs / "objects.parquet").filter(pl.col("object_type") == "ball")
    return (
        frames.select("frame_id", "timestamp_s", "ball_state")
        .join(ball.select("frame_id", "x", "y", "visible"), on="frame_id", how="left")
        .with_columns(k=(pl.col("timestamp_s") * 10).round().cast(pl.Int64))
        .with_columns(off=(pl.col("timestamp_s") - pl.col("k") / 10).abs())
        .sort("k", "off")
        .unique("k", keep="first")
        .select("k", "x", "y", "visible", pff_state="ball_state")
    )


def event_times(gs: Path, period: int, kinds: tuple[str, ...]) -> list[float]:
    ev = pl.read_parquet(gs / "events.parquet").filter(pl.col("event_type").is_in(kinds))
    t = pl.read_parquet(gs / "frames.parquet", columns=["frame_id", "period", "timestamp_s"])
    return sorted(ev.join(t, on="frame_id").filter(pl.col("period") == period)["timestamp_s"])


def _pct(p) -> str:
    return "    -" if p is None else f"{100 * p:5.2f}%"


def _xy(x, y) -> str:
    return "      -      " if x is None else f"({x:5.1f}, {y:5.1f})"


def report(clip: dict, model_id: str, root: Path | None) -> None:
    clip_id = clip["clip_id"]
    gs_root = root / "gamestate" if root else Path("data/gamestate")
    pred_root = root / "predictions" if root else Path("data/predictions")
    gs = gs_root / clip_id
    pff_gs = Path("data/gamestate") / clip["match_id"]
    run = json.loads((Path("data/vision_cache") / clip_id / "run.json").read_text())
    shift = run["video_start_s"] + sync_offset(clip)
    period = clip["period"]

    vis = pl.read_parquet(pred_root / clip_id / f"{model_id}.parquet")
    pff_p = pl.read_parquet(PFF_PGOAL).filter(
        pl.col("match_id") == clip["match_id"], pl.col("period") == period
    )
    j = join_rows(vis, pff_p, shift)
    ball = (
        pl.read_parquet(gs / "objects.parquet")
        .filter(pl.col("object_type") == "ball")
        .select("frame_id", "x", "y", "interpolated")
    )
    state = pl.read_parquet(gs / "frames.parquet", columns=["frame_id", "ball_state"])
    pff_ball = pff_ball_grid(pff_gs, period)
    rows = (
        j.join(ball, on="frame_id", how="left")
        .join(state, on="frame_id", how="left")
        .join(pff_ball, on="k", how="left", suffix="_pff")
        .with_columns(
            d=((pl.col("x") - pl.col("x_pff")) ** 2 + (pl.col("y") - pl.col("y_pff")) ** 2).sqrt()
        )
    )
    goals = event_times(pff_gs, period, ("goal",))
    shots = event_times(pff_gs, period, ("shot", "goal"))
    in_clip = [g for g in goals if j["t"].min() <= g <= j["t"].max()]
    print(f"{clip_id}  {model_id}  root {root or 'data'}  grid rows {j.height}")

    for g in in_clip:
        print(f"\ngoal at PFF t {g:.2f} (video {g - sync_offset(clip):.2f} s)")
        print(
            "  last 3 s:  t-goal  vision     pff    vision ball      interp  pff ball      vis  dist"
        )
        last = rows.filter((pl.col("t") >= g - TABLE_S) & (pl.col("t") <= g + 1e-9))
        for r in last.iter_rows(named=True):
            interp = "-" if r["interpolated"] is None else ("y" if r["interpolated"] else "n")
            vis_flag = "-" if r["visible"] is None else ("V" if r["visible"] else "E")
            d = "  -  " if r["d"] is None else f"{r['d']:5.1f}"
            print(
                f"            {r['t'] - g:+6.1f}  {_pct(r['vis_p'])}  {_pct(r['pff_p'])}  "
                f"{_xy(r['x'], r['y'])}  {interp:^6}  {_xy(r['x_pff'], r['y_pff'])}  "
                f"{vis_flag:^3}  {d}"
            )
        print(
            "  meter:",
            "  ".join(
                f"{b} s {_pct(v).strip()} (pff {_pct(p).strip()})" for b, v, p in meter_at(j, g)
            ),
        )
        after = rows.filter((pl.col("t") > g + 1e-9) & (pl.col("vis_p") > PEAK_P))
        print(
            f"  after the goal: {after.height} rows over {PEAK_P:.0%}"
            + (f", max {_pct(after['vis_p'].max()).strip()}" if after.height else "")
        )

    peaks = false_peaks(rows, shots)
    print(
        f"\nfalse peaks (p > {PEAK_P:.0%}, more than {PEAK_AWAY_S:.0f} s from a PFF shot): "
        f"{peaks.height} rows"
    )
    for r in peaks.iter_rows(named=True):
        rel = f"goal {r['t'] - in_clip[-1]:+.1f} s" if in_clip else f"t {r['t']:.1f}"
        print(
            f"  {rel:>14}  vision {_pct(r['vis_p'])}  pff {_pct(r['pff_p'])}  "
            f"ball {_xy(r['x'], r['y'])} interp {r['interpolated']}"
        )
    if peaks.height:
        print(f"  max {_pct(peaks['vis_p'].max())}")

    video_s = pl.col("t") - sync_offset(clip)
    end = in_clip[-1] - sync_offset(clip) if in_clip else clip["video_end_s"]
    scored = rows.filter(video_s.is_between(clip["video_start_s"], end))
    print(
        f"\npredicted share: whole clip {vis['predicted'].mean():.1%}, "
        f"scored video {clip['video_start_s']:.0f}-{end:.1f} s {scored['predicted'].mean():.1%}"
    )
    both = rows.filter(pl.col("vis_poss").is_not_null() & pl.col("pff_poss").is_not_null())
    print(
        f"possession agreement where both have one: "
        f"{(both['vis_poss'] == both['pff_poss']).mean():.1%} of {both.height} rows"
    )

    live = rows.filter(
        pl.col("x").is_not_null(), pl.col("ball_state") == "alive", pl.col("pff_state") == "alive"
    )
    print("vision ball vs PFF ball on rows both call alive:")
    for name, sel in (("all rows", live), ("detected rows", live.filter(~pl.col("interpolated")))):
        d = ball_distance(sel.select("k", "x", "y"), pff_ball.select("k", "x", "y", "visible"))
        print(
            f"  {name}: "
            + "; ".join(
                f"PFF {k} n {s['n']}, median {s['median_m']:.1f} m, p90 {s['p90_m']:.1f} m, "
                f"within 2 m {s['within_2m']:.1%}"
                for k, s in d.items()
            )
        )


def run_module(*args: str) -> None:
    print("$", "python -m", *args)
    subprocess.run([sys.executable, "-m", *args], check=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="demo.check")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--manifest", type=Path, default=Path("data/splits/demo_clips.json"))
    ap.add_argument("--variant", help="a ball variant's tag: data/variants/<tag>")
    ap.add_argument("--rerun", action="store_true", help="rebuild the variant first (--set)")
    ap.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE")
    ap.add_argument("--render", action="store_true", help="render to C:/footage/renders")
    args = ap.parse_args(argv)
    clip = {c["clip_id"]: c for c in load_manifest(args.manifest)}[args.clip]
    root = VARIANT_ROOT / args.variant if args.variant else None
    if args.rerun:
        if root is None:
            raise SystemExit("--rerun needs --variant")
        sets = [a for s in args.set for a in ("--set", s)]
        run_module("vision.replay", "--match-id", args.clip, "--variant", args.variant, *sets)
        run_module(
            "vision.stage8",
            "--match-id",
            args.clip,
            "--gamestate-dir",
            str(root / "gamestate"),
            "--cache-dir",
            str(root / "vision_cache"),
        )
        run_module(
            "prediction.infer",
            "--match-id",
            args.clip,
            "--model",
            str(MODELS / args.model_id),
            "--gamestate-dir",
            str(root / "gamestate"),
            "--out-dir",
            str(root / "predictions"),
        )
    report(clip, args.model_id, root)
    if args.render:
        base = root or Path("data")
        cache = base / "vision_cache" if root else Path("data/vision_cache")
        out = RENDERS / f"{args.clip}-{args.variant or 'base'}.mp4"
        run_module(
            "demo.video",
            "--video",
            json.loads(VIDEOS.read_text())[args.clip],
            "--match-id",
            args.clip,
            "--cache-dir",
            str(cache),
            "--gamestate-dir",
            str(base / "gamestate"),
            "--predictions",
            str(base / "predictions" / args.clip / f"{args.model_id}.parquet"),
            "--truth-clip",
            args.clip,
            "--clip-manifest",
            str(args.manifest),
            "--out",
            str(out),
        )


if __name__ == "__main__":
    main()
