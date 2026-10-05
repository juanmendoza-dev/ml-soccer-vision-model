"""Vision vs PFF-tracking P(goal) on a demo clip (roadmap Phase 3, "compare predictor
accuracy on vision vs dataset tracking", one clip).

    PYTHONPATH=. python scripts/demo_compare.py --clip demo01-arg-fra-81 --model-id goal-f0-2026-10-05

Vision's grid time t (period time from the run's start) is PFF period time
t + video_start_s(run.json) + the clip's sync offset. The PFF side is the base run's
out-of-fold p_goal_cal_h5, the same fold-0 model and map as the demo model.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from vision.bench import load_manifest, sync_offset

BASE = Path("data/runs/lgbm-held-2026-09-27/pgoal.parquet")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--manifest", type=Path, default=Path("data/splits/demo_clips.json"))
    args = ap.parse_args()
    clip = {c["clip_id"]: c for c in load_manifest(args.manifest)}[args.clip]
    start = json.loads(Path(f"data/vision_cache/{args.clip}/run.json").read_text())["video_start_s"]
    shift = start + sync_offset(clip)
    vis = (
        pl.read_parquet(f"data/predictions/{args.clip}/{args.model_id}.parquet")
        .with_columns(k=((pl.col("t_s") + shift) * 10).round().cast(pl.Int64))
        .select("k", "predicted", vis_poss="possession_team", vis_p="p_goal_h5")
    )
    pff = pl.read_parquet(BASE).filter(
        pl.col("match_id") == clip["match_id"], pl.col("period") == clip["period"]
    ).select("k", pff_poss="possession_team", pff_p="p_goal_cal_h5")
    j = vis.join(pff, on="k", how="inner").sort("k")
    goals = pl.read_parquet(f"data/gamestate/{clip['match_id']}/events.parquet").filter(pl.col("event_type") == "goal")
    times = pl.read_parquet(f"data/gamestate/{clip['match_id']}/frames.parquet", columns=["frame_id", "period", "timestamp_s"])
    goal_t = goals.join(times, on="frame_id").filter(pl.col("period") == clip["period"])["timestamp_s"]
    both = j.drop_nulls(["vis_p", "pff_p"])
    print(f"rows {j.height}, both predicted {both.height}")
    print(f"possession agreement {(j['vis_poss'] == j['pff_poss']).mean():.3f}")
    if both.height > 1:
        print(f"correlation of log p: {np.corrcoef(np.log(both['vis_p']), np.log(both['pff_p']))[0, 1]:.3f}")
    for g in goal_t:
        print(f"goal at {g:.1f} s")
        for before in (5, 3, 2, 1, 0.5):
            k = round((g - before) * 10)
            r = j.filter(pl.col("k") == k)
            v = r["vis_p"].to_list()[0] if r.height else None
            p = r["pff_p"].to_list()[0] if r.height else None
            print(f"  {before:>4} s before: vision {v}, pff {p}")
    j.with_columns(t=pl.col("k") / 10).select("t", "vis_poss", "pff_poss", "vis_p", "pff_p").gather_every(10).write_csv(
        f"data/predictions/{args.clip}/compare_1s.csv"
    )


if __name__ == "__main__":
    main()
