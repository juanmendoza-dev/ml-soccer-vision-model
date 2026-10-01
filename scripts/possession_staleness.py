"""When is stage 8's possession wrong? (07 #6 follow-up, Docs/reviews/possession-stale-*.md)

    python scripts/possession_staleness.py data/runs/<provider> data/runs/<inferred> [--out FILE]

Stage 8 (vision.state.infer, the inferred run's StateConfig) runs on every match's native
game state. From its ball_carrier_id, per native frame: carrier_age_s = seconds since the
last frame of the same period with a confirmed carrier (0 on a carrier frame, null before
the period's first one). That's causal and reads only the schema column stage 8 fills, so
it's a legal live input. Joined onto the 10 Hz grid by frame_id and restricted to the
scored rows (label_mask_h5 and not all_estimated).

Per carrier_age_s bin: share of scored and positive rows, how often stage 8 disagrees with
PFF, who shoots next on the disagreeing rows (as possession_split.py), and PR-AUC of both
runs there (descriptive). Also split by a VISIBLE ball on the row, by seconds since PFF's
own last possession change, and by who changed last on the disagreeing rows (PFF changed
and stage 8 hasn't followed, or stage 8 changed and PFF didn't).

Uses PFF possession on every match but no labels or predictions to pick anything: the
PR-AUC and shot columns are descriptive only.
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
from prediction.possession import OBJECT_COLS, config_key, inferred_state
from vision.state import StateConfig, infer

H = "h5"
# (label, lower bound exclusive, upper bound inclusive); null and 0 are their own rows
AGE_BINS = [
    ("0", None, None),
    ("(0, 0.5]", 0.0, 0.5),
    ("(0.5, 1]", 0.5, 1.0),
    ("(1, 2]", 1.0, 2.0),
    ("(2, 3]", 2.0, 3.0),
    ("(3, 5]", 3.0, 5.0),
    ("(5, 10]", 5.0, 10.0),
    ("> 10", 10.0, None),
]
CHANGE_BINS = [
    ("(0, 1]", 0.0, 1.0),
    ("(1, 2]", 1.0, 2.0),
    ("(2, 3]", 2.0, 3.0),
    ("(3, 5]", 3.0, 5.0),
    ("(5, 10]", 5.0, 10.0),
    ("> 10", 10.0, None),
]
THRESHOLDS = (0.5, 1.0, 2.0, 3.0, 5.0, 10.0)


def since(t: pl.Expr, mark: pl.Expr) -> pl.Expr:
    """Seconds since the last row of the period where mark is true (0 there), null
    before the first one. Rows must be sorted by (period, timestamp_s)."""
    last = pl.when(mark).then(t).forward_fill().over("period")
    return t - last


def change(col: str) -> pl.Expr:
    """True where col differs from the previous row of the period (null counts as a
    value), and on the period's first row."""
    prev = pl.col(col).shift().over("period")
    first = pl.col("period").ne_missing(pl.col("period").shift())
    return first | pl.col(col).ne_missing(prev)


def staleness(match_dir: Path, config: StateConfig) -> pl.DataFrame:
    """Per native frame: stage 8's possession and carrier age, PFF's possession, and
    seconds since each side's possession last changed."""
    frames = pl.read_parquet(
        match_dir / "frames.parquet",
        columns=["frame_id", "period", "timestamp_s", "possession_team"],
    )
    objects = pl.scan_parquet(match_dir / "objects.parquet").select(OBJECT_COLS)
    state = infer(frames, objects, config)
    teams = objects.select("object_id", carrier_team="team").unique("object_id").collect()
    f = (
        frames.rename({"possession_team": "pff"})
        .join(state.rename({"possession_team": "inf"}), on="frame_id")
        .join(teams, left_on="ball_carrier_id", right_on="object_id", how="left")
        .sort("period", "timestamp_s", "frame_id")
        .with_columns(
            carrier_age_s=since(pl.col("timestamp_s"), pl.col("ball_carrier_id").is_not_null()),
            last_carrier_team=pl.col("carrier_team").forward_fill().over("period"),
            pff_change_s=since(pl.col("timestamp_s"), change("pff")),
            inf_change_s=since(pl.col("timestamp_s"), change("inf")),
        )
    )
    # possession is the latest carrier's team, so the two must match wherever a carrier
    # has been seen this period (every PFF player has a team)
    seen = f.filter(pl.col("carrier_age_s").is_not_null())
    bad = seen.filter(pl.col("last_carrier_team").ne_missing(pl.col("inf")))
    if bad.height:
        raise AssertionError(
            f"{match_dir.name}: {bad.height} frames where possession isn't the "
            "latest carrier's team"
        )
    if f.filter(pl.col("carrier_age_s").is_null())["inf"].is_not_null().any():
        raise AssertionError(f"{match_dir.name}: possession before the period's first carrier")
    return f.select("frame_id", "inf", "carrier_age_s", "pff_change_s", "inf_change_s")


def in_bin(col: str, lo, hi) -> pl.Expr:
    c = pl.col(col)
    if lo is None and hi is None:
        return c == 0
    e = c > lo
    return e if hi is None else e & (c <= hi)


def table(sc: pl.DataFrame, groups: list[tuple[str, pl.Expr]], first: str) -> list[str]:
    n, n_pos = sc.height, int(sc["y"].sum())
    out = [
        (
            f"| {first} | scored rows | positives | disagree | "
            "shot next, PFF's team | shot next, stage 8's | PR-AUC provider | PR-AUC inferred |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, cond in groups:
        sub = sc.filter(cond)
        if not sub.height:
            out.append(f"| {name} | 0 | | | | | | |")
            continue
        y = sub["y"].to_numpy().astype(bool)
        dis = sub.filter(~pl.col("agree"), pl.col("inf").is_not_null())
        pff = dis["pff_shot"].mean() if dis.height else float("nan")
        inf = dis["inf_shot"].mean() if dis.height else float("nan")
        a = pr_auc(y, sub["p_provider"].to_numpy()) if y.any() else float("nan")
        b = pr_auc(y, sub["p_inferred"].to_numpy()) if y.any() else float("nan")
        out.append(
            f"| {name} | {sub.height / n:.1%} ({sub.height:,}) | {y.sum() / n_pos:.1%} | "
            f"{1 - sub['agree'].mean():.1%} | {pff:.4f} | {inf:.4f} | {a:.3f} | {b:.3f} |"
        )
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("provider", type=Path)
    ap.add_argument("inferred", type=Path)
    ap.add_argument("--processed", type=Path, default=PROCESSED_DIR)
    ap.add_argument("--gamestate", type=Path, default=GAMESTATE_DIR)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    runs = {"provider": load_run(args.provider), "inferred": load_run(args.inferred)}
    config = StateConfig(**runs["inferred"][0]["config"]["state_config"])
    if config.possession_model is not None:
        ap.error(
            f"{args.inferred} is a learned-possession run: carrier staleness is the rule's; "
            "use possession_split.py --possession-model (03)"
        )
    key = config_key(config)
    ids = sorted(m["match_id"] for m in json.loads(FOLDS_PATH.read_text())["matches"])
    cols = ["match_id", "period", "t_s", "frame_id", "possession_team", "all_estimated"]
    cols += [f"label_mask_{H}", f"label_shot_{H}"]
    rows, shots = [], []
    for m in ids:
        d = args.processed / m
        f = pl.read_parquet(d / "frames_10hz.parquet", columns=cols)
        st = staleness(args.gamestate / m, config)
        # the same stage 8 the model arms read (its cache), and the same carrier age
        cached = inferred_state(m, args.gamestate, args.processed, config)
        chk = cached.join(st, on="frame_id", how="left")
        if chk["possession_team"].ne_missing(chk["inf"]).any():
            raise AssertionError(f"{m}: stage 8 differs from the cached state")
        if not chk["carrier_age_s"].equals(chk["carrier_age_s_right"], check_names=False):
            raise AssertionError(f"{m}: carrier age differs from prediction.possession's")
        ball = (
            pl.scan_parquet(d / "objects_10hz.parquet")
            .filter(pl.col("object_type") == "ball", pl.col("visible").fill_null(False))
            .select("period", "t_s", ball_visible=pl.lit(True))
            .unique(["period", "t_s"])
            .collect()
        )
        f = (
            f.join(st, on="frame_id", how="left")
            .join(ball, on=["period", "t_s"], how="left")
            .with_columns(pl.col("ball_visible").fill_null(False))
        )
        rows.append(f)
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
    secs = HORIZONS[H] + 1e-9
    for who, team in (("pff", "possession_team"), ("inf", "inf")):
        nxt = open_play.rename({"team": team, "shot_t": "next"})
        f = (
            f.sort("t_s")
            .join_asof(
                nxt,
                left_on="t_s",
                right_on="next",
                by=["match_id", "period", team],
                strategy="forward",
                allow_exact_matches=False,
                check_sortedness=False,
            )
            .with_columns(
                **{f"{who}_shot": ((pl.col("next") - pl.col("t_s")) <= secs).fill_null(False)}
            )
            .drop("next")
        )
    for name, (_, preds) in runs.items():
        p = preds.select("match_id", "period", k=tenths(), **{f"p_{name}": f"p_{H}"})
        f = f.join(p, on=["match_id", "period", "k"], how="left")
    f = f.with_columns(agree=pl.col("inf").eq_missing(pl.col("possession_team")))
    sc = f.filter(pl.col(f"label_mask_{H}"), ~pl.col("all_estimated")).with_columns(
        y=pl.col(f"label_shot_{H}")
    )
    if sc["carrier_age_s"].is_null().sum() != sc["inf"].is_null().sum():
        raise AssertionError("carrier age null on rows with an inferred team")

    out = [
        f"# When stage 8's possession is wrong ({len(ids)} matches, {H})",
        "",
        f"Stage 8 config `{key}`: `{config.to_dict()}`",
        f"Runs: provider `{args.provider.name}`, inferred `{args.inferred.name}`.",
        "",
        (
            f"Scored rows: {sc.height:,}, positives {int(sc['y'].sum()):,}, stage 8 disagrees on "
            f"{1 - sc['agree'].mean():.1%} (nulls count as disagreeing)."
        ),
        "",
        (
            'Shares of scored rows and positives are of all scored rows. "disagree" is within the '
            "bin. The shot columns are on the bin's disagreeing rows where stage 8 names a team: "
            f"the share with an open-play shot by that team within {HORIZONS[H]} s. PR-AUC is "
            "descriptive (the bins come from stage 8's output)."
        ),
        "",
        "## By carrier age (seconds since stage 8 last confirmed a carrier)",
        "",
    ]
    age = [("null (no carrier yet)", pl.col("carrier_age_s").is_null())]
    age += [(n, in_bin("carrier_age_s", lo, hi)) for n, lo, hi in AGE_BINS]
    out += table(sc, age, "carrier age (s)")

    out += ["", '## Stale beyond S (candidate thresholds for "unknown")', ""]
    n_dis = int((~sc["agree"]).sum())
    out += [
        "| age > S | scored rows | positives | disagree there | share of all disagreeing rows |",
        "|---|---|---|---|---|",
    ]
    for s in THRESHOLDS:
        sub = sc.filter(pl.col("carrier_age_s") > s)
        d = int((~sub["agree"]).sum())
        out.append(
            f"| {s:g} s | {sub.height / sc.height:.1%} | {sub['y'].sum() / sc['y'].sum():.1%} | "
            f"{d / max(sub.height, 1):.1%} | {d / n_dis:.1%} |"
        )

    out += ["", "## By a VISIBLE ball on the row", ""]
    out += table(
        sc,
        [("ball visible", pl.col("ball_visible")), ("no visible ball", ~pl.col("ball_visible"))],
        "ball",
    )
    out += ["", "## Carrier age × ball visible (disagreement rate, share of scored rows)", ""]
    out += ["| carrier age (s) | ball visible | no visible ball |", "|---|---|---|"]
    for name, cond in age:
        cells = []
        for vis in (True, False):
            sub = sc.filter(cond, pl.col("ball_visible") == vis)
            cells.append(
                f"{1 - sub['agree'].mean():.1%} ({sub.height / sc.height:.1%})"
                if sub.height
                else "-"
            )
        out.append(f"| {name} | {cells[0]} | {cells[1]} |")

    out += ["", "## By seconds since PFF's own last possession change", ""]
    out += table(
        sc, [(n, in_bin("pff_change_s", lo, hi)) for n, lo, hi in CHANGE_BINS], "PFF change (s)"
    )

    out += ["", "## On the disagreeing rows: who changed last", ""]
    dis = sc.filter(~pl.col("agree"), pl.col("inf").is_not_null())
    lag = pl.col("pff_change_s") < pl.col("inf_change_s")
    out += [
        (
            "PFF changed more recently: PFF's team switched and stage 8 hasn't followed (stage 8 "
            "lags, or PFF switched wrongly). Stage 8 changed more recently: stage 8 switched and PFF "
            "didn't (stage 8 changed wrongly, or PFF lags)."
        ),
        "",
        (
            "| disagreeing rows | share | median carrier age (s) | shot next, PFF's team | "
            "shot next, stage 8's |"
        ),
        "|---|---|---|---|---|",
    ]
    for name, cond in (
        ("PFF changed more recently", lag),
        ("stage 8 changed more recently", ~lag),
    ):
        sub = dis.filter(cond)
        out.append(
            f"| {name} | {sub.height / dis.height:.1%} ({sub.height:,}) | "
            f"{sub['carrier_age_s'].median():.1f} | {sub['pff_shot'].mean():.4f} | "
            f"{sub['inf_shot'].mean():.4f} |"
        )
    q = dis.select(
        pl.col("carrier_age_s").quantile(0.25).alias("q25"),
        pl.col("carrier_age_s").median().alias("median"),
        pl.col("carrier_age_s").quantile(0.75).alias("q75"),
    ).row(0, named=True)
    agr = sc.filter(pl.col("agree"))["carrier_age_s"]
    out += [
        "",
        (
            f"Carrier age on disagreeing rows: quartiles {q['q25']:.1f} / {q['median']:.1f} / "
            f"{q['q75']:.1f} s. On agreeing rows: {agr.quantile(0.25):.1f} / {agr.median():.1f} / "
            f"{agr.quantile(0.75):.1f} s."
        ),
    ]
    text = "\n".join(out) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
