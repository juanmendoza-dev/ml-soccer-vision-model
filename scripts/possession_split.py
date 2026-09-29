"""Where the inferred-possession arm loses (07 #6, Docs/reviews/possession-inferred-*.md).

    python scripts/possession_split.py data/runs/<provider> data/runs/<inferred>

Pooled PR-AUC of both runs on the scored rows (07), split by whether stage 8's
possession (the inferred run's cached state) agrees with PFF's on the row. Descriptive:
the split is picked from stage 8's output, not from the labels, so it's a diagnosis of
the paired compare, not a replacement for it.
"""

import argparse
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.folds import FOLDS_PATH
from evaluation.metrics import pr_auc
from evaluation.report import PROCESSED_DIR
from evaluation.runs import load_run
from prediction.features import tenths
from prediction.possession import config_key
from vision.state import StateConfig


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("provider", type=Path)
    ap.add_argument("inferred", type=Path)
    ap.add_argument("--processed", type=Path, default=PROCESSED_DIR)
    args = ap.parse_args(argv)
    runs = {"provider": load_run(args.provider), "inferred": load_run(args.inferred)}
    config = StateConfig(**runs["inferred"][0]["config"]["state_config"])
    key = config_key(config)
    horizons = runs["inferred"][0]["horizons"]
    ids = sorted(m["match_id"] for m in json.loads(FOLDS_PATH.read_text())["matches"])
    cols = ["match_id", "period", "t_s", "frame_id", "possession_team", "all_estimated"]
    cols += [f"label_{x}_{h}" for h in horizons for x in ("mask", "shot")]
    rows = []
    for m in ids:
        d = args.processed / m
        f = pl.read_parquet(d / "frames_10hz.parquet", columns=cols)
        s = pl.read_parquet(d / f"state_inferred_{key}.parquet").rename({"possession_team": "inf"})
        rows.append(f.join(s, on="frame_id", how="left"))
    f = pl.concat(rows).with_columns(k=tenths())
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
        ):
            y = sub[f"label_shot_{h}"].to_numpy().astype(float)
            a = pr_auc(y, sub[f"provider_{h}"].to_numpy())
            b = pr_auc(y, sub[f"inferred_{h}"].to_numpy())
            print(f"| {name} | {sub.height:,} | {y.mean():.4f} | {a:.3f} | {b:.3f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
