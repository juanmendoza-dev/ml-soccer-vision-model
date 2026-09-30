"""Where stage 8's possession disagrees with PFF: by ball visibility and time since PFF's change.

    PYTHONPATH=. python scripts/possession_split.py [--scored h5|h3] [--state KEY=VALUE ...]
        [match_id ...]

Every provider frame with a team, default StateConfig unless --state overrides a field
(e.g. --state carrier_min_s=0.1, --state team_near_s=0.1). --scored keeps only the frames that
are 07's scored rows at that horizon (label mask on, not all ESTIMATED) and repeats the
tables for its positive rows; the gap and since-change bookkeeping still runs on every frame.
A diagnostic for the carry-forward fix
(stage 8 review), not a run: the bookkeeping columns are never model inputs.
"""

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import polars as pl

from vision.state import StateConfig, infer
from vision.state_check import OBJECT_COLS

GS = Path("data/gamestate")
PROCESSED = Path("data/processed")
ap = argparse.ArgumentParser()
ap.add_argument("--scored", choices=["h5", "h3"])
ap.add_argument("--state", action="append", default=[], metavar="KEY=VALUE")
ap.add_argument("ids", nargs="*")
opts = ap.parse_args()
scored = opts.scored
names = {f.name for f in fields(StateConfig)}
over = {}
for pair in opts.state:
    key, _, value = pair.partition("=")
    if key not in names:
        ap.error(f"--state {pair!r}: not a StateConfig field")
    over[key] = float(value)
config = StateConfig(**over)
print(f"StateConfig overrides: {over or 'none'}")
if opts.ids:
    ids = opts.ids
elif scored:
    ids = sorted(m["match_id"] for m in json.loads(Path("data/splits/folds.json").read_text())["matches"])
else:
    ids = sorted(p.parent.name for p in GS.glob("*/frames.parquet"))
parts = []
for i in ids:
    d = GS / i
    frames = pl.read_parquet(d / "frames.parquet")
    if frames["possession_team"].is_null().all():
        continue
    objects = pl.scan_parquet(d / "objects.parquet").select(OBJECT_COLS)
    inf = infer(frames, objects, config)
    vis = (
        objects.filter(pl.col("object_type") == "ball", pl.col("visible").fill_null(False),
                       pl.col("x").is_not_null())
        .select("frame_id").unique().with_columns(seen=pl.lit(True)).collect()
    )
    j = (
        frames.select("frame_id", "period", "timestamp_s", p_state="ball_state", p_poss="possession_team")
        .join(inf.select("frame_id", "possession_team", "ball_carrier_id"), on="frame_id")
        .join(vis, on="frame_id", how="left")
        .sort("period", "timestamp_s")
        .with_columns(seen=pl.col("seen").fill_null(False))
        .with_columns(
            # causal-looking bookkeeping, fine for a diagnostic
            last_seen=pl.when("seen").then("timestamp_s").forward_fill().over("period"),
            chg=pl.when(pl.col("p_poss") != pl.col("p_poss").forward_fill().shift(1).over("period"))
            .then("timestamp_s").forward_fill().over("period"),
        )
        .filter(pl.col("p_poss").is_not_null())
        .with_columns(
            match=pl.lit(i),
            dis=(pl.col("p_poss") != pl.col("possession_team")).fill_null(True),
            gap=pl.col("timestamp_s") - pl.col("last_seen"),
            since=pl.col("timestamp_s") - pl.col("chg"),
        )
    )
    if scored:
        rows = (
            pl.read_parquet(PROCESSED / i / "frames_10hz.parquet",
                            columns=["frame_id", "all_estimated", f"label_mask_{scored}", f"label_shot_{scored}"])
            .filter(pl.col(f"label_mask_{scored}"), ~pl.col("all_estimated"))
            .select("frame_id", pos=f"label_shot_{scored}")
            .unique("frame_id")
        )
        j = j.join(rows, on="frame_id")
    parts.append(j)
    print(i, file=sys.stderr, end=" ", flush=True)

a = pl.concat(parts)


def bucket(col, edges, labels, none):
    e = pl.when(pl.col(col).is_null()).then(pl.lit(none))
    for hi, lab in zip(edges, labels):
        e = e.when(pl.col(col) < hi).then(pl.lit(lab))
    return e.otherwise(pl.lit(labels[-1]))


a = a.with_columns(
    ball=pl.when("seen").then(pl.lit("0 visible")).otherwise(
        bucket("gap", [0.5, 2, 10, 1e9], ["1 gap<0.5s", "2 gap0.5-2s", "3 gap2-10s", "4 gap>10s"], "5 never")
    ),
    since_chg=bucket("since", [1, 3, 10, 1e9], ["a <1s", "b 1-3s", "c 3-10s", "d >10s"], "e none"),
    state=pl.col("p_state").fill_null("null"),
)


def summary(a, what):
    """Counts, not shares of disagreement: the scored rows are fixed by PFF, so configs
    compare on these directly."""
    early = pl.col("since_chg").is_in(["a <1s", "b 1-3s"])
    parts = {
        "all": a,
        "< 3 s since change": a.filter(early),
        "> 10 s since change": a.filter(pl.col("since_chg") == "d >10s"),
    }
    if "pos" in a.columns:
        parts["positives"] = a.filter("pos")
    print(f"\nsummary, {what}")
    for name, b in parts.items():
        n, d = b.height, int(b["dis"].sum())
        print(f"  {name}: {d} / {n} disagree ({d / n:.4f})")


def tables(a, what):
    tot = int(a["dis"].sum())
    print(f"\n\n{a.height} {what}, disagree {tot / a.height:.3f} ({tot})\n")
    for keys in (["ball"], ["since_chg"], ["state"], ["ball", "since_chg"]):
        t = (
            a.group_by(keys)
            .agg(n=pl.len(), dis_n=pl.col("dis").sum())
            .with_columns(
                frame_share=(pl.col("n") / a.height).round(3),
                dis_rate=(pl.col("dis_n") / pl.col("n")).round(3),
                share_of_dis=(pl.col("dis_n") / tot).round(3),
            )
            .sort(keys)
            .drop("dis_n")
        )
        with pl.Config(tbl_rows=40, tbl_formatting="MARKDOWN", tbl_hide_dataframe_shape=True,
                       tbl_hide_column_data_types=True):
            print(t, "\n")


summary(a, f"scored frames ({scored})" if scored else "provider frames")
if scored:
    tables(a, f"scored frames ({scored})")
    tables(a.filter("pos"), f"positive scored frames ({scored})")
else:
    tables(a, "provider frames")
