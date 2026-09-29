"""Where the inferred-possession arm loses (07 #6, Docs/reviews/possession-inferred-*.md).

    python scripts/possession_split.py data/runs/<provider> data/runs/<inferred>

Pooled PR-AUC of both runs on the scored rows (07), split by whether stage 8's
possession (stage 8's cached state, made null past the run's stale_after for arm U)
agrees with PFF's on the row. Descriptive:
the split is picked from stage 8's output, not from the labels, so it's a diagnosis of
the paired compare, not a replacement for it.

On the rows where they disagree it also asks who is right: how often each team (PFF's,
stage 8's) takes an open-play shot in (t, t+H]. If stage 8's team shot about as often as
PFF's, much of the disagreement would be PFF's lag (07, Floor from the labels).
"""

import argparse
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.folds import FOLDS_PATH, GAMESTATE_DIR
from evaluation.metrics import pr_auc, shots_table
from evaluation.report import PROCESSED_DIR
from evaluation.runs import load_run
from prediction.features import tenths
from prediction.labels import HORIZONS
from prediction.possession import inferred_state, stale_rows
from vision.state import StateConfig


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("provider", type=Path)
    ap.add_argument("inferred", type=Path)
    ap.add_argument("--processed", type=Path, default=PROCESSED_DIR)
    ap.add_argument("--gamestate", type=Path, default=GAMESTATE_DIR)
    args = ap.parse_args(argv)
    runs = {"provider": load_run(args.provider), "inferred": load_run(args.inferred)}
    config = StateConfig(**runs["inferred"][0]["config"]["state_config"])
    # arm U (05, Stale possession): the possession the run saw is null past after_s
    stale = runs["inferred"][0]["config"].get("stale_possession")
    horizons = runs["inferred"][0]["horizons"]
    ids = sorted(m["match_id"] for m in json.loads(FOLDS_PATH.read_text())["matches"])
    cols = ["match_id", "period", "t_s", "frame_id", "possession_team", "all_estimated"]
    cols += [f"label_{x}_{h}" for h in horizons for x in ("mask", "shot")]
    rows, shots = [], []
    for m in ids:
        d = args.processed / m
        f = pl.read_parquet(d / "frames_10hz.parquet", columns=cols)
        s = inferred_state(m, args.gamestate, args.processed, config)
        if stale:
            s = s.with_columns(
                possession_team=pl.when(stale_rows(s, stale["after_s"]))
                .then(None)
                .otherwise(pl.col("possession_team"))
            )
        s = s.select("frame_id", inf="possession_team")
        rows.append(f.join(s, on="frame_id", how="left"))
        g = args.gamestate / m
        shots.append(
            shots_table(
                pl.read_parquet(g / "events.parquet"),
                pl.read_parquet(
                    g / "frames.parquet", columns=["frame_id", "period", "timestamp_s"]
                ),
                m,
            )
        )
    f = pl.concat(rows).with_columns(k=tenths())
    open_play = (
        pl.concat(shots)
        .filter("open_play")
        .select("match_id", "period", "team", shot_t="t_s")
        .sort("shot_t")
    )
    # each row's next open-play shot by PFF's team and by stage 8's, strictly after t
    for who, team in (("pff", "possession_team"), ("inf", "inf")):
        nxt = open_play.rename({"team": team, "shot_t": f"{who}_next"})
        f = (
            f.sort("t_s")
            .join_asof(
                nxt,
                left_on="t_s",
                right_on=f"{who}_next",
                by=["match_id", "period", team],
                strategy="forward",
                allow_exact_matches=False,
                check_sortedness=False,
            )
            .sort("match_id", "period", "t_s")
        )
    for name, (_, preds) in runs.items():
        p = preds.select(
            "match_id", "period", k=tenths(), **{f"{name}_{h}": f"p_{h}" for h in horizons}
        )
        f = f.join(p, on=["match_id", "period", "k"], how="left")
    f = f.with_columns(agree=pl.col("inf").eq_missing(pl.col("possession_team")))
    for h in horizons:
        sc = f.filter(pl.col(f"label_mask_{h}"), ~pl.col("all_estimated"))
        pos = sc.filter(pl.col(f"label_shot_{h}"))
        print(f"\n## {h}\n")
        print(
            f"stage 8 disagrees with PFF on {1 - sc['agree'].mean():.1%} of {sc.height:,} scored "
            f"rows and {1 - pos['agree'].mean():.1%} of {pos.height:,} positive rows\n"
        )
        print("| rows | n | base rate | PR-AUC provider | PR-AUC inferred |")
        print("|---|---|---|---|---|")
        for name, sub in (
            ("all scored", sc),
            ("stage 8 agrees", sc.filter(pl.col("agree"))),
            ("stage 8 disagrees", sc.filter(~pl.col("agree"))),
            ("  names the other team", sc.filter(~pl.col("agree"), pl.col("inf").is_not_null())),
            ("  unknown (null)", sc.filter(pl.col("inf").is_null())),
        ):
            y = sub[f"label_shot_{h}"].to_numpy().astype(float)
            a = pr_auc(y, sub[f"provider_{h}"].to_numpy())
            b = pr_auc(y, sub[f"inferred_{h}"].to_numpy())
            print(f"| {name} | {sub.height:,} | {y.mean():.4f} | {a:.3f} | {b:.3f} |")
        dis = sc.filter(~pl.col("agree"), pl.col("inf").is_not_null())
        secs = HORIZONS[h] + 1e-9
        rates = dis.select(
            **{
                who: ((pl.col(f"{who}_next") - pl.col("t_s")) <= secs).fill_null(False).mean()
                for who in ("pff", "inf")
            }
        ).row(0, named=True)
        print(
            f"\nOn the {dis.height:,} scored rows where stage 8 names the other team: an open-play "
            f"shot within {HORIZONS[h]} s by PFF's team on {rates['pff']:.4f} of them, by stage 8's "
            f"team on {rates['inf']:.4f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
