"""Lead-time view of one or more runs (05, lead-time features): what the report doesn't show.

    python scripts/lead_time.py data/runs/<A> [data/runs/<B> ...] [--horizon h5]

Per run, at each fold's own tau (07), over PFF open-play shots:
- shots caught, median lead over caught shots, and shots with a lead of at least 1 / 2 s
  as a share of *all* open-play shots (the median only covers caught shots, so it moves
  whenever coverage does)
- median p on the grid row at or before 5 / 2 / 1 / 0.2 s before each shot. Paired: a
  shot counts at an offset only if every run has a non-null p on that row, so all runs
  are read on the same shots and rows. Diagnostic only; the test is the paired compare.
- gain share by feature group, mean over folds (not an ablation)
- alarms at matched false-alarm levels: one tau grid over pooled OOF for every run, and at
  each budget the lowest miss rate within it. Descriptive (tau isn't picked on the
  evaluation folds, 07), but it compares runs at the same false alarms, which the per-fold
  tau doesn't when a run's budget transfers badly
- PR-AUC with only the positives whose shot is 0-1, 1-2, 2-3, 3-5 s away (every negative
  kept), pooled and per fold: how well each run ranks rows early in an attack
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.folds import FOLDS_PATH, GAMESTATE_DIR, load
from evaluation.metrics import alarm_summary, pr_auc, score_alarms
from evaluation.report import (
    PROCESSED_DIR,
    alarms_at_tau,
    join_predictions,
    load_shots,
    taus_for,
)
from evaluation.runs import load_run

OFFSETS = (5.0, 2.0, 1.0, 0.2)
SWEEP = np.round(np.arange(0.40, 0.755, 0.01), 2)
BUDGETS = (2.0, 3.0, 5.0, 12.0)
TTS_BINS = ((0, 1), (1, 2), (2, 3), (3, 5))
KEYS = ["match_id", "period", "t_s"]
GROUPS = {
    "ball position": [
        "ball_x",
        "ball_y",
        "ball_z",
        "ball_dist",
        "ball_angle",
        "ball_in_box",
    ],
    "likely carrier": ["carrier_dist", "carrier_speed", "carrier_vgoal", "carrier_goal_dist"],
    "defence": [
        "press_dist",
        "def_within_5m",
        "lane_defenders",
        "gk_in_lane",
        "gk_off_line",
        "gk_ball_dist",
        "def_in_box",
        "def_goal_side",
    ],
    "ball motion and history (v1)": [
        "ball_age_s",
        "ball_speed",
        "ball_vgoal",
        "ball_dist_change_1s",
        "ball_dist_change_3s",
        "ball_final_third_s",
        "possession_s",
    ],
    "attackers in box / goal side": ["att_in_box", "att_goal_side"],
    "visible counts": ["n_visible_att", "n_visible_def"],
    "v2: ball progress 2-5 s": [
        "ball_dist_change_5s",
        "ball_dist_max_5s",
        "ball_dist_min_5s",
        "ball_vgoal_mean_2s",
        "ball_vgoal_mean_5s",
        "ball_box_s",
    ],
    "v2: possession so far": [
        "poss_start_dist",
        "poss_min_dist",
        "poss_final_third_s",
        "poss_advance_rate",
    ],
    "v2: carrier progress": ["carrier_goal_dist_change_3s", "carrier_goal_dist_change_5s"],
    "v2: near the box": [
        "att_near_box",
        "def_near_box",
        "near_box_diff",
        "att_near_box_change_3s",
        "def_near_box_change_3s",
    ],
    "v2: defensive line": [
        "def_line_dist",
        "def_line_change_3s",
        "ball_line_gap",
        "att_beyond_line",
    ],
}


def pff_rows(preds: pl.DataFrame, folds: dict, h: str) -> pl.DataFrame:
    fold_of = {m["match_id"]: m["fold"] for m in folds["matches"] if m["source"] == "pff"}
    ids = sorted(fold_of)
    info = pl.DataFrame({"match_id": ids, "fold": [fold_of[i] for i in ids]})
    rows = join_predictions(preds, ids, [h], PROCESSED_DIR).join(info, on="match_id")
    return rows.rename({f"p_{h}": "p"})


def p_before(rows: pl.DataFrame, shots: pl.DataFrame, off: float) -> pl.DataFrame:
    """p on the grid row at or before (shot time - off), same period; null if none."""
    k = shots.with_row_index("shot").with_columns(
        k=((pl.col("t_s") - off) * 10 + 1e-6).floor().cast(pl.Int64)
    )
    grid = rows.select("match_id", "period", "p", k=(pl.col("t_s") * 10).round().cast(pl.Int64))
    return k.join(grid, on=["match_id", "period", "k"], how="left").sort("shot")


def matched_budget(rows: pl.DataFrame, shots: pl.DataFrame) -> dict[float, tuple]:
    """Per budget: (tau, summary) with the lowest miss rate at or under it; ties go to the
    higher tau, as in 07."""
    sweep = [(float(t), alarm_summary(rows, shots, float(t))) for t in SWEEP]
    out = {}
    for b in BUDGETS:
        ok = [x for x in sweep if x[1]["false_alarms_per_match"] <= b]
        if not ok:
            out[b] = None
            continue
        tau, sm = min(ok, key=lambda x: (x[1]["miss_rate"], -x[0]))
        lead = score_alarms(rows, shots, tau)[0]["lead_s"].drop_nulls()
        out[b] = (tau, sm | {"lead_2s": int((lead >= 2).sum())})
    return out


def time_to_shot(rows: pl.DataFrame, shots: pl.DataFrame, h: str) -> pl.DataFrame:
    """Scored rows (07) with tts: seconds to the next open-play shot by possession_team."""
    op = (
        shots.filter("open_play")
        .select("match_id", "period", possession_team="team", shot_t="t_s")
        .sort("shot_t")
    )
    scored = rows.filter(pl.col(f"label_mask_{h}"), ~pl.col("all_estimated")).sort("t_s")
    return (
        scored.join_asof(
            op,
            left_on="t_s",
            right_on="shot_t",
            by=["match_id", "period", "possession_team"],
            strategy="forward",
            check_sortedness=False,
        )
        .with_columns(tts=pl.col("shot_t") - pl.col("t_s"))
        .sort(KEYS)
    )


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--horizon", default="h5")
    args = ap.parse_args(argv)
    h = args.horizon
    folds = load(FOLDS_PATH)
    ids = sorted(m["match_id"] for m in folds["matches"] if m["source"] == "pff")
    shots = load_shots(ids, GAMESTATE_DIR)
    open_shots = shots.filter("open_play")
    names, leads, ps, gains, budgets, early = [], [], {}, [], [], []
    horizon_s = float(h[1:])
    bins = [(lo, hi) for lo, hi in TTS_BINS if lo < horizon_s]
    base = None
    for run in args.runs:
        meta, preds = load_run(run)
        names.append(meta["run_id"])
        rows = pff_rows(preds, folds, h)
        taus = taus_for(meta, h, sorted(rows["fold"].unique().to_list()))
        scored, _, _ = alarms_at_tau(rows, shots, taus)
        lead = scored["lead_s"]
        n = scored.height
        caught = lead.drop_nulls()
        leads.append(
            [
                meta["run_id"],
                n,
                caught.len(),
                float(caught.median()) if caught.len() else float("nan"),
                int((caught >= 1).sum()),
                int((caught >= 2).sum()),
                (caught >= 2).sum() / n,
            ]
        )
        for off in OFFSETS:
            ps.setdefault(off, []).append(p_before(rows, open_shots, off)["p"].to_numpy())
        coef = [f["coef"] for f in meta["folds"][h].values()]
        mean = {f: np.mean([c.get(f, 0.0) for c in coef]) for f in coef[0]}
        unknown = set(mean) - {f for fs in GROUPS.values() for f in fs}
        if unknown:
            raise SystemExit(f"features in no group: {sorted(unknown)}")
        gains.append({g: sum(mean.get(f, 0.0) for f in fs) for g, fs in GROUPS.items()})
        budgets.append(matched_budget(rows, shots))
        tts = time_to_shot(rows, shots, h)
        if base is None:
            base = tts.select(*KEYS, "fold", "tts", y=pl.col(f"label_shot_{h}"))
        elif not tts.select(KEYS).equals(base.select(KEYS)):
            raise SystemExit("runs don't share scored rows")
        early.append(tts["p"].to_numpy())

    print(f"## Lead time at each fold's tau ({h}, PFF open-play shots)\n")
    print("| run | shots | caught | median lead (s) | lead ≥ 1 s | lead ≥ 2 s | ≥ 2 s share |")
    print("|---|---|---|---|---|---|---|")
    for r in leads:
        print(f"| {r[0]} | {r[1]:,} | {r[2]} | {r[3]:.2f} | {r[4]} | {r[5]} | {r[6]:.3f} |")

    print("\n## Median p before open-play shots (paired: same shots and rows in every run)\n")
    print("| s before shot | shots | " + " | ".join(names) + " |")
    print("|---|---|" + "---|" * len(names))
    for off, arrs in ps.items():
        ok = np.all([~np.isnan(a) for a in arrs], axis=0)
        meds = " | ".join(f"{np.median(a[ok]):.3f}" for a in arrs)
        print(f"| {off:g} | {ok.sum():,} | {meds} |")
    print(
        "\n(p above 0.3 two seconds out: "
        + ", ".join(
            f"{n} {np.mean(a[np.all([~np.isnan(x) for x in ps[2.0]], axis=0)] > 0.3):.2f}"
            for n, a in zip(names, ps[2.0], strict=True)
        )
        + ")"
    )

    print(f"\n## Alarms at matched false alarms per match ({h}, pooled OOF, descriptive)\n")
    print(
        "| budget | run | τ | false / match | caught | miss rate | median lead (s) | lead ≥ 2 s |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for b in BUDGETS:
        for name, res in zip(names, budgets, strict=True):
            if res[b] is None:
                print(f"| ≤ {b:g} | {name} | – | – | – | – | – | – |")
                continue
            tau, sm = res[b]
            caught = round((1 - sm["miss_rate"]) * leads[0][1])
            print(
                f"| ≤ {b:g} | {name} | {tau:.2f} | {sm['false_alarms_per_match']:.2f} | "
                f"{caught} | {sm['miss_rate']:.3f} | {sm['lead_s_median']:.2f} | {sm['lead_2s']} |"
            )

    y, fold, tts = base["y"].to_numpy(), base["fold"].to_numpy(), base["tts"].to_numpy()
    print(f"\n## PR-AUC by the positives' time to shot ({h}; every negative kept)\n")
    head = " | ".join(names)
    last = f" | folds {names[-1]} better" if len(names) > 1 else ""
    print(f"| time to shot | positives | base rate | {head}{last} |")
    print("|---|---|---|" + "---|" * len(names) + ("---|" if last else ""))
    for lo, hi in bins:
        keep = ~y | ((tts > lo) & (tts <= hi))
        vals = [pr_auc(y[keep], p[keep]) for p in early]
        cells = " | ".join(f"{v:.4f}" for v in vals)
        if last:
            wins = sum(
                pr_auc(y[keep & (fold == f)], early[-1][keep & (fold == f)])
                > pr_auc(y[keep & (fold == f)], early[0][keep & (fold == f)])
                for f in np.unique(fold)
            )
            cells += f" | {wins}/{len(np.unique(fold))}"
        print(f"| {lo}–{hi} s | {int(y[keep].sum()):,} | {y[keep].mean():.4f} | {cells} |")

    print(f"\n## Gain share by group ({h}, mean over folds)\n")
    print("| group | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for g in GROUPS:
        print(f"| {g} | " + " | ".join(f"{x[g]:.3f}" for x in gains) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
