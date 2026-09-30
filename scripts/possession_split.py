"""Where stage 8's possession disagrees with PFF: by ball visibility and time since PFF's change.

    PYTHONPATH=. python scripts/possession_split.py [--scored h5|h3] [--state KEY=VALUE ...]
        [match_id ...]

Every provider frame with a team, default StateConfig unless --state overrides a field
(e.g. --state carrier_min_s=0.1, --state team_near_s=0.1). --scored keeps only the frames that
are 07's scored rows at that horizon (label mask on, not all ESTIMATED) and repeats the
tables for its positive rows; the gap and since-change bookkeeping still runs on every frame.
A diagnostic for the carry-forward fix
(stage 8 review), not a run: the bookkeeping columns are never model inputs.

--residual (with --scored h5 --state team_near_s=0.05) adds 03 stage 8's residual
diagnosis for learned possession: the ball-visible rows in the first second after PFF's
change where that rule still disagrees, sorted by why, plus change timing.
"""

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import polars as pl

from vision.state import PLAYER_TYPES, StateConfig, StateMachine, infer, is_out
from vision.state_check import OBJECT_COLS

GS = Path("data/gamestate")
PROCESSED = Path("data/processed")
ap = argparse.ArgumentParser()
ap.add_argument("--scored", choices=["h5", "h3"])
ap.add_argument("--state", action="append", default=[], metavar="KEY=VALUE")
ap.add_argument("--residual", action="store_true")
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
if opts.residual and (scored != "h5" or over != {"team_near_s": 0.05}):
    ap.error("--residual is 03's diagnosis: --scored h5 --state team_near_s=0.05 and nothing else")
REACH = config.carrier_radius_m
WINDOW = (-1.0, 2.0)  # timing window around PFF's change, s
SEEN_SHARE = 0.5  # ball visible on at least this share of the window's frames to call "no contact"


def replay(frames: pl.DataFrame, obj: pl.DataFrame, c: StateConfig) -> pl.DataFrame:
    """Stage 8 stepped exactly as infer() steps it, keeping what infer() doesn't return: the
    rule's candidate and team streak after each frame, and whether the ball was out."""
    vis = obj.lazy().filter(pl.col("visible").fill_null(False), pl.col("x").is_not_null())
    ball = (
        vis.filter(pl.col("object_type") == "ball")
        .sort("frame_id", "object_id")
        .unique("frame_id", keep="first", maintain_order=True)
        .select("frame_id", bx="x", by="y", bz="z")
    )
    cand = (
        vis.filter(pl.col("object_type").is_in(PLAYER_TYPES))
        .select("frame_id", "object_id", "team", "x", "y")
        .join(ball, on="frame_id")
        .with_columns(d=((pl.col("x") - pl.col("bx")) ** 2 + (pl.col("y") - pl.col("by")) ** 2).sqrt())
        .sort("frame_id", "d", "object_id")
        .unique("frame_id", keep="first", maintain_order=True)
        .filter(pl.col("d") <= c.carrier_radius_m, pl.col("bz").is_null() | (pl.col("bz") < c.carrier_max_z_m))
        .select("frame_id", cand="object_id", cand_team="team")
    )
    rows = (
        frames.lazy()
        .select("frame_id", "period", "timestamp_s")
        .join(ball, on="frame_id", how="left")
        .join(cand, on="frame_id", how="left")
        .sort("period", "timestamp_s", "frame_id")
        .collect()
    )
    sm = StateMachine(c)
    rec = {k: [] for k in ("r_poss", "r_cand", "r_team", "r_since", "r_out")}
    cols = ("period", "timestamp_s", "bx", "by", "bz", "cand", "cand_team")
    for period, t, x, y, z, who, team in zip(*(rows[k].to_list() for k in cols), strict=True):
        b = None if x is None else (x, y, z)
        _, p, _ = sm.step(period, t, b, who, team)
        rec["r_poss"].append(p)
        rec["r_cand"].append(sm.cand)
        rec["r_team"].append(sm.team_cand)
        rec["r_since"].append(sm.team_since)
        rec["r_out"].append(b is not None and is_out(x, y, c.out_margin_m))
    return rows.select("frame_id", "period", "timestamp_s", "bx", "by", "bz").with_columns(
        pl.Series(k, v, dtype=pl.Boolean if k == "r_out" else pl.Float64 if k == "r_since" else pl.String)
        for k, v in rec.items()
    )


def first_at(times: np.ndarray, hit: np.ndarray, lo: float, hi: float) -> float | None:
    """Time of the first hit in [lo, hi], times sorted."""
    a, b = np.searchsorted(times, lo, "left"), np.searchsorted(times, hi, "right")
    k = np.flatnonzero(hit[a:b])
    return float(times[a + k[0]]) if len(k) else None


def residual(i: str, frames: pl.DataFrame, obj: pl.DataFrame, j: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Every ball-visible first-second scored row of one match with its evidence flags
    (03, "First, diagnose the residual"), and the timing of each PFF change they follow."""
    rp = replay(frames, obj, config)
    chk = j.select("frame_id", "possession_team").join(rp.select("frame_id", "r_poss"), on="frame_id")
    assert chk.height == j.height and (chk["possession_team"].eq_missing(chk["r_poss"])).all(), i
    base = j.filter("seen", pl.col("since") < 1)
    no_z = obj.with_columns(z=pl.lit(None, pl.Float64))
    alt = (
        infer(frames, no_z, StateConfig()).select("frame_id", poss_2d="possession_team")
        .join(infer(frames, no_z, config).select("frame_id", poss_nh="possession_team"), on="frame_id")
    )
    ball = rp.filter(pl.col("bx").is_not_null()).select("frame_id", "bx", "by")
    near = (
        obj.lazy()
        .filter(pl.col("object_type").is_in(PLAYER_TYPES), pl.col("x").is_not_null())
        .select("frame_id", "object_id", "team", "x", "y", vis=pl.col("visible").fill_null(False))
        .join(ball.lazy(), on="frame_id")
        .with_columns(d=((pl.col("x") - pl.col("bx")) ** 2 + (pl.col("y") - pl.col("by")) ** 2).sqrt())
        .collect()
    )
    in_reach = pl.col("d") <= REACH
    ev = (
        near.join(base.select("frame_id"), on="frame_id")
        .sort("frame_id", "d", "object_id")
        .group_by("frame_id", maintain_order=True)
        .agg(
            d1=pl.col("d").filter("vis").first(),
            t1=pl.col("team").filter("vis").first(),
            d2=pl.col("d").filter("vis").slice(1, 1).first(),
            t2=pl.col("team").filter("vis").slice(1, 1).first(),
            est_home=(~pl.col("vis") & in_reach & (pl.col("team") == "home")).any(),
            est_away=(~pl.col("vis") & in_reach & (pl.col("team") == "away")).any(),
        )
    )
    p = pl.col("p_poss")
    rows = (
        base.join(rp.select("frame_id", "bz", "r_cand", "r_team", "r_since", "r_out"), on="frame_id")
        .join(alt, on="frame_id")
        .join(ev, on="frame_id", how="left")
        .with_columns(
            z=pl.when(pl.col("bz").is_null()).then(pl.lit("unknown"))
            .when(pl.col("bz") < config.carrier_max_z_m).then(pl.lit("low")).otherwise(pl.lit("high")),
            high=pl.col("bz").fill_null(0) >= config.carrier_max_z_m,
            no_vis=pl.col("d1").is_null() | (pl.col("d1") > REACH),
            est=pl.col("est_home").fill_null(False) | pl.col("est_away").fill_null(False),
            est_pff=pl.when(p == "home").then("est_home").otherwise("est_away").fill_null(False),
            cand_other=pl.col("r_team").is_not_null() & (pl.col("r_team") != p),
            cand_unknown=pl.col("r_cand").is_not_null() & pl.col("r_team").is_null(),
            short=(pl.col("r_team") == p).fill_null(False)
            & (pl.col("timestamp_s") - pl.col("r_since") < config.team_near_s - 1e-9),
            tie=(pl.col("d1") == pl.col("d2")).fill_null(False) & (pl.col("t1") != pl.col("t2")).fill_null(False),
            ok_2d=(pl.col("poss_2d") == p).fill_null(False),
            ok_nh=(pl.col("poss_nh") == p).fill_null(False),
        )
        .with_columns(
            bucket=pl.when("high").then(pl.lit("1 height gate"))
            .when(pl.col("no_vis") & pl.col("est")).then(pl.lit("2 ESTIMATED in reach, no VISIBLE"))
            .when("no_vis").then(pl.lit("3 no visible player in reach"))
            .when(pl.col("r_cand").is_not_null() & (pl.col("r_team") != p).fill_null(True))
            .then(pl.lit("4 candidate unknown/other team"))
            .when("short").then(pl.lit("5 PFF's team, streak < 0.05 s"))
            .otherwise(pl.lit("6 unexplained")),
            why=pl.when("r_out").then(pl.lit("ball out")).when("tie").then(pl.lit("exact tie"))
            .otherwise(pl.lit("other")),
        )
        .select(
            "match", "frame_id", "period", "timestamp_s", "chg", "p_poss", "pos", "dis", "z", "high", "no_vis",
            "est", "est_pff", "cand_other", "cand_unknown", "short", "tie", "r_out", "ok_2d", "ok_nh", "bucket",
            "why", "d1",
        )
    )

    # timing, per change the residual rows follow
    contact = (
        near.filter("vis", in_reach)
        .group_by("frame_id")
        .agg(c_home=(pl.col("team") == "home").any(), c_away=(pl.col("team") == "away").any())
    )
    tl = (
        rp.select("frame_id", "period", "timestamp_s", "r_poss", seen=pl.col("bx").is_not_null())
        .join(contact, on="frame_id", how="left")
        .with_columns(pl.col("c_home", "c_away").fill_null(False))
    )
    per = {k: g for (k,), g in tl.partition_by("period", as_dict=True).items()}
    out = []
    for period, chg, new in rows.filter("dis").select("period", "chg", "p_poss").unique().sort("period", "chg").iter_rows():
        g = per[period]
        t = g["timestamp_s"].to_numpy()
        lo, hi = chg + WINDOW[0], chg + WINDOW[1]
        a, b = np.searchsorted(t, lo, "left"), np.searchsorted(t, hi, "right")
        touch = first_at(t, g[f"c_{new}"].to_numpy(), lo, hi)
        has = (g["r_poss"] == new).fill_null(False).to_numpy()
        switch = first_at(t, has & ~np.r_[False, has[:-1]], lo, hi)  # a switch, not a hold
        out.append({
            "match": i, "period": period, "chg": chg, "new": new,
            "contact": None if touch is None else touch - chg,
            "switch": None if switch is None else switch - chg,
            "held": bool(a < len(has) and has[a]),
            "seen_share": float(g["seen"][a:b].mean()) if b > a else 0.0,
        })
    return rows, pl.DataFrame(out, schema={"match": pl.String, "period": pl.Int64, "chg": pl.Float64,
                                           "new": pl.String, "contact": pl.Float64, "switch": pl.Float64,
                                           "held": pl.Boolean, "seen_share": pl.Float64})

print(f"StateConfig overrides: {over or 'none'}")
if opts.ids:
    ids = opts.ids
elif scored:
    ids = sorted(m["match_id"] for m in json.loads(Path("data/splits/folds.json").read_text())["matches"])
else:
    ids = sorted(p.parent.name for p in GS.glob("*/frames.parquet"))
parts = []
res_rows, res_chgs = [], []
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
    if opts.residual:
        r, ch = residual(i, frames, objects.collect(), j)
        res_rows.append(r)
        res_chgs.append(ch)
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


def show(t):
    with pl.Config(tbl_rows=60, tbl_cols=20, tbl_formatting="MARKDOWN", tbl_hide_dataframe_shape=True,
                   tbl_hide_column_data_types=True, fmt_str_lengths=40, tbl_width_chars=400):
        print(t, "\n")


def residual_report(rows: pl.DataFrame, chgs: pl.DataFrame):
    """03 stage 8, "First, diagnose the residual". Rows are H = 5 scored, ball visible,
    < 1 s after PFF's change; the residual is those where team_near_s = 0.05 disagrees."""
    key = ["match", "period", "chg"]

    chgs = chgs.with_columns(
        timing=pl.when(pl.col("contact").is_not_null()).then(pl.lit("contact"))
        .when(pl.col("seen_share") >= SEEN_SHARE).then(pl.lit("no contact, ball seen"))
        .otherwise(pl.lit("no contact, ball mostly unseen"))
    )
    res = rows.filter("dis").join(chgs.select(*key, "timing", "contact", "switch"), on=key, how="left")
    assert res["timing"].null_count() == 0
    print("\n\n# Residual diagnosis (03 stage 8), H = 5 scored rows, ball visible, < 1 s after PFF's change")
    print(f"team_near_s = 0.05, reach {REACH} m, timing window {WINDOW} s, 'ball seen' = visible on >= {SEEN_SHARE:.0%}"
          " of the window's frames\n")
    for name, sel in (("all", pl.lit(True)), ("positives", pl.col("pos"))):
        b, r = rows.filter(sel), res.filter(sel)
        print(f"\n## {name}: {r.height} / {b.height} ball-visible first-second rows disagree ({r.height / b.height:.4f})\n")
        print("Disjoint buckets, in 03's order (share = of the residual; rate = of all ball-visible first-second rows):\n")
        show(
            r.group_by("bucket").agg(n=pl.len())
            .with_columns(share=(pl.col("n") / r.height).round(3), rate=(pl.col("n") / b.height).round(4))
            .sort("bucket")
        )
        show(r.filter(pl.col("bucket") == "6 unexplained").group_by("why").agg(n=pl.len()).sort("why"))
        print("Overlapping evidence flags (count and share of the residual):\n")
        flags = ["high", "no_vis", "est", "est_pff", "cand_other", "cand_unknown", "short", "tie", "r_out"]
        show(pl.DataFrame({"flag": flags, "n": [int(r[f].sum()) for f in flags]})
             .with_columns(share=(pl.col("n") / r.height).round(3)))
        show(r.group_by("z").agg(n=pl.len()).sort("z"))
        print("Flag combinations (top 15):\n")
        show(r.group_by(["high", "no_vis", "est", "cand_other", "cand_unknown", "short"]).agg(n=pl.len())
             .sort("n", descending=True).head(15))
        nv = r.filter(pl.col("bucket") == "3 no visible player in reach", pl.col("d1").is_not_null())
        print("Bucket 3, nearest VISIBLE player's distance to the ball, m (rows with any visible player):\n")
        show(nv.select(pl.len().alias("n"), *(pl.col("d1").quantile(q).round(2).alias(f"q{int(q * 100)}")
                                           for q in (0.1, 0.25, 0.5, 0.75, 0.9)),
                       within_2m=(pl.col("d1") <= 2).mean().round(3), within_3m=(pl.col("d1") <= 3).mean().round(3)))
        h = r.filter(pl.col("bucket") == "1 height gate")
        print(f"Height gate rows right under the 2D rule (default, z null): {int(h['ok_2d'].sum())} / {h.height};"
              f" under team_near_s = 0.05 with z null (extra): {int(h['ok_nh'].sum())} / {h.height}\n")
        print("Timing class of the change each residual row follows, by bucket:\n")
        show(r.group_by("bucket", "timing").agg(n=pl.len()).sort("bucket", "timing"))
        est = pl.col("bucket") == "2 ESTIMATED in reach, no VISIBLE"
        tim = pl.col("timing") == "no contact, ball seen"
        n_est, n_tim, n_both = (int(r.select(e.sum()).item()) for e in (est, tim, est & tim))
        gone = int(r.select((est | tim).sum()).item())
        print(f"Unreachable: ESTIMATED bucket {n_est}, timing-mismatch candidates {n_tim}, both {n_both};"
              f" union {gone}")
        print(f"Reachable residual: {r.height - gone} / {r.height} ({(r.height - gone) / r.height:.3f})\n")

    c = chgs.filter(pl.col("contact").is_not_null())
    print(f"\n## changes: {chgs.height} distinct PFF changes behind the residual\n")
    show(chgs.group_by("timing").agg(n=pl.len()).sort("timing"))
    print("Signed first visible new-team contact minus change time, s (changes with contact):\n")
    show(c.select(pl.col("contact").quantile(q).round(2).alias(f"q{int(q * 100)}") for q in (0.1, 0.25, 0.5, 0.75, 0.9)))
    s = chgs.filter(pl.col("switch").is_not_null())
    print(f"First switch of the rule to the new team minus change time, s ({s.height} of {chgs.height} within the window;"
          f" {int(chgs['held'].sum())} already had the new team at the window start):\n")
    show(s.select(pl.col("switch").quantile(q).round(2).alias(f"q{int(q * 100)}") for q in (0.1, 0.25, 0.5, 0.75, 0.9)))

    print("\n## first 20 distinct changes per bucket (match, period, time)\n")
    per_bucket = (
        res.group_by("bucket", *key).agg(rows=pl.len(), pos=pl.col("pos").any(), d1=pl.col("d1").min())
        .join(chgs, on=key)
        .sort("bucket", *key)
        .group_by("bucket", maintain_order=True).head(20)
    )
    for (bucket,), g in per_bucket.partition_by("bucket", as_dict=True, maintain_order=True).items():
        print(f"### {bucket}\n")
        show(g.select("match", "period", pl.col("chg").round(2), "new", "rows", "pos", pl.col("d1").round(2),
                      pl.col("contact").round(2), pl.col("switch").round(2), "held",
                      pl.col("seen_share").round(2), "timing"))


summary(a, f"scored frames ({scored})" if scored else "provider frames")
if opts.residual:
    residual_report(pl.concat(res_rows), pl.concat(res_chgs))
    sys.exit()
if scored:
    tables(a, f"scored frames ({scored})")
    tables(a.filter("pos"), f"positive scored frames ({scored})")
else:
    tables(a, "provider frames")
