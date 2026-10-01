"""Touch vs turnover check for learned possession (03 stage 8, "Hypothesis check").

    PYTHONPATH=. python scripts/possession_touch_check.py

On H = 5 scored grid rows where team_near_s = 0.05 switches away from the team both rules
had and the default rule keeps it, split by PFF over the next 3 s into turnover (PFF names
the new team) and touch (PFF keeps the old team throughout). Reports each pfeat-v1
current-block feature's separation AUC, oriented to the new team. Definitions are fixed in
Docs/reviews/possession-touch-check-2026-09-30.md. PFF's later labels only sort the groups.
"""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

from vision import possession_features as pf
from vision.state import StateConfig, infer
from vision.state_check import OBJECT_COLS

GS = Path("data/gamestate")
PROCESSED = Path("data/processed")
AHEAD_S = 3.0
BAR = 0.70
MIN_FINITE = 0.5
TEAMS = ("home", "away")
SHAPE = [f"{t}_{s}" for s in pf.SHAPES for t in TEAMS] + [f"diff_{s}" for s in pf.SHAPES]
CONTACT = [
    "near_home_m",
    "near_away_m",
    *(c for c in pf.BLOCK if c.startswith("nearest_")),
    "last_contact_team",
]
HEADING = [c for c in pf.BLOCK if c.startswith("heads_")]
ELIGIBLE = (
    {c: "shape" for c in SHAPE} | {c: "contact" for c in CONTACT} | {c: "heading" for c in HEADING}
)
assert set(ELIGIBLE) <= set(pf.BLOCK)


def triggers(m: str) -> pl.DataFrame:
    """Trigger rows of one match with their group and oriented current-block features."""
    d = GS / m
    frames = pl.read_parquet(d / "frames.parquet")
    objects = pl.scan_parquet(d / "objects.parquet").select(OBJECT_COLS)
    rules = (
        infer(frames, objects, StateConfig())
        .select("frame_id", slow="possession_team")
        .join(
            infer(frames, objects, StateConfig(team_near_s=0.05)).select(
                "frame_id", fast="possession_team"
            ),
            on="frame_id",
        )
    )
    scored = pl.read_parquet(
        PROCESSED / m / "frames_10hz.parquet",
        columns=["period", "t_s", "frame_id", "all_estimated", "label_mask_h5", "label_shot_h5"],
    ).with_columns(k=(pl.col("t_s") * 10).round().cast(pl.Int64))
    feats = pf.load(m)
    g = (
        feats.join(rules, on="frame_id", how="left")
        .sort("period", "k")
        .with_columns(
            prev_ok=(pl.col("k").shift(1) == pl.col("k") - 1)
            & (pl.col("period").shift(1) == pl.col("period")),
            old=pl.col("fast").shift(1),
            slow_prev=pl.col("slow").shift(1),
        )
        .join(scored, on=["period", "k"], how="inner", suffix="_10hz")
    )
    assert (g["frame_id"] == g["frame_id_10hz"]).all(), m
    trig = g.filter(
        pl.col("label_mask_h5"),
        ~pl.col("all_estimated"),
        pl.col("prev_ok").fill_null(False),
        pl.col("old").is_not_null(),
        pl.col("fast").is_not_null(),
        pl.col("fast") != pl.col("old"),
        pl.col("slow") == pl.col("old"),
        pl.col("slow_prev") == pl.col("old"),
    ).rename({"fast": "new"})
    if not trig.height:
        return trig
    nat = frames.select("period", "timestamp_s", "possession_team").sort("period", "timestamp_s")
    per = {
        p: (g_["timestamp_s"].to_numpy(), g_["possession_team"].to_list())
        for (p,), g_ in nat.partition_by("period", as_dict=True).items()
    }
    group = []
    for period, t, new, old in trig.select("period", "t_s", "new", "old").iter_rows():
        ts, poss = per[period]
        a, b = (
            np.searchsorted(ts, t + 1e-9, "left"),
            np.searchsorted(ts, t + AHEAD_S + 1e-9, "right"),
        )
        ahead = poss[a:b]
        if new in ahead:
            group.append("turnover")
        elif ahead and all(p == old for p in ahead):
            group.append("touch")
        else:
            group.append(None)
    trig = trig.with_columns(group=pl.Series(group, dtype=pl.String), match=pl.lit(m)).drop_nulls(
        "group"
    )
    away = trig.filter(pl.col("new") == "away")
    oriented = pl.concat([trig.filter(pl.col("new") == "home"), pf.mirror(away)])
    return oriented.select("match", "period", "t_s", "new", "group", pos="label_shot_h5", *pf.BLOCK)


def auc(x: np.ndarray, y: np.ndarray) -> float:
    """P(score of a turnover > score of a touch), ties count half; y = 1 for turnover."""
    r = pl.Series(x).rank("average").to_numpy()
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if not n1 or not n0:
        return float("nan")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def table(rows: pl.DataFrame) -> pl.DataFrame:
    y_all = (rows["group"] == "turnover").to_numpy().astype(int)
    out = []
    for c in pf.BLOCK:
        x = rows[c].to_numpy().astype(np.float64)
        fin = np.isfinite(x)
        a = auc(x[fin], y_all[fin])
        t, h = x[fin & (y_all == 1)], x[fin & (y_all == 0)]
        q = lambda v, p: float(np.quantile(v, p)) if len(v) else float("nan")  # noqa: E731
        fin_t, fin_h = fin[y_all == 1].mean(), fin[y_all == 0].mean()
        kind = ELIGIBLE.get(c, "-")
        out.append(
            {
                "feature": c,
                "kind": kind,
                "auc": round(a, 3),
                "sep": round(max(a, 1 - a), 3),
                "finite_turnover": round(float(fin_t), 3),
                "finite_touch": round(float(fin_h), 3),
                "eligible": bool(kind != "-" and min(fin_t, fin_h) >= MIN_FINITE),
                "med_turnover": round(q(t, 0.5), 3),
                "iqr_turnover": f"{q(t, 0.25):.2f}..{q(t, 0.75):.2f}",
                "med_touch": round(q(h, 0.5), 3),
                "iqr_touch": f"{q(h, 0.25):.2f}..{q(h, 0.75):.2f}",
            }
        )
    return pl.DataFrame(out).sort("sep", descending=True, nulls_last=True)


def show(t):
    with pl.Config(
        tbl_rows=80,
        tbl_cols=20,
        tbl_formatting="MARKDOWN",
        tbl_hide_dataframe_shape=True,
        tbl_hide_column_data_types=True,
        tbl_width_chars=400,
        fmt_str_lengths=40,
    ):
        print(t, "\n")


ids = sorted(
    m["match_id"] for m in json.loads(Path("data/splits/folds.json").read_text())["matches"]
)
parts = []
for m in ids:
    parts.append(triggers(m))
    print(m, file=sys.stderr, end=" ", flush=True)
rows = pl.concat([p for p in parts if p.height])
print(f"# Touch vs turnover, {len(ids)} matches, H = 5 scored trigger rows\n")
for name, sel in (
    ("all trigger rows", pl.lit(True)),
    ("positives only (descriptive)", pl.col("pos")),
):
    r = rows.filter(sel)
    n = r.group_by("group").agg(n=pl.len()).sort("group")
    print(f"## {name}: {r.height} rows\n")
    show(n)
    show(r.group_by("group", "new").agg(n=pl.len()).sort("group", "new"))
    t = table(r)
    show(t)
    if name == "all trigger rows":
        best = t.filter("eligible").head(1)
        passed = best.height and best["sep"][0] >= BAR
        print(
            f"Bar: best eligible feature {best['feature'][0] if best.height else None} at separation "
            f"{best['sep'][0] if best.height else None} -> {'PASS' if passed else 'FAIL (stop before any fit)'}\n"
        )
print(
    f"Matches with triggers: {rows['match'].n_unique()}; per-match rows: median "
    f"{rows.group_by('match').len()['len'].median()}"
)
