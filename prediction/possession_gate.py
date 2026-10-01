"""The learned-possession gate (03 stage 8, "Checks and fixed bars", step 2).

Learned possession's `test` roles from all five outer folds are scored against PFF's
possession on 07's H = 5 scored rows, next to the historical default rule. Five pooled bars
decide; nothing here reads shot metrics or picks a variant. The bars and the baseline they
were cut from are constants below, written down before any learned result existed.

The native-frame bookkeeping (ball visibility, time since PFF's change) is the old
scripts/possession_split.py's, kept as `native_rows` so both read the same rows. The grid
join is new: learned state is on the 10 Hz grid, joined on the full key (period, tick) with
the native frame_id checked, and rows are counted on the grid, never deduplicated.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from prediction import possession
from vision.state import StateConfig, infer
from vision.state_check import OBJECT_COLS

# What the historical default rule scores on the 64 matches, H = 5 (03, "Checks and fixed
# bars"). The gate reproduces these first; a mismatch stops it rather than moving a bar.
BASELINE = {
    "rows": 1_977_379,
    "positives": 49_812,
    "overall": 281_606,
    "first3": 154_806,
    "positive_dis": 3_862,
    "late_rows": 1_162_041,
    "late_dis": 52_455,
    "churn_pff": 14_546,
    "churn_default": 10_074,
}
# The fixed bars. The late bar is relative: the default's unrounded rate plus a point.
BARS = {
    "overall": 253_445,  # at least 10% below 281,606
    "first3": 131_585,  # at least 15% below 154,806
    "positive_dis": 3_862,  # no worse than the default on positives
    "late_margin": 0.010,
    "churn": 18_182,  # 1.25 x PFF's 14,546
}

EARLY = ["a <1s", "b 1-3s"]
LATE = "d >10s"


def bucket(col, edges, labels, none):
    e = pl.when(pl.col(col).is_null()).then(pl.lit(none))
    for hi, lab in zip(edges, labels):
        e = e.when(pl.col(col) < hi).then(pl.lit(lab))
    return e.otherwise(pl.lit(labels[-1]))


def label_rows(a: pl.DataFrame) -> pl.DataFrame:
    """The ball-visibility, since-change and ball-state buckets. First 3 s is [0, 3) after
    PFF's last change; `> 10 s` includes exactly 10 s; rows before any change are `none`."""
    return a.with_columns(
        ball=pl.when("seen")
        .then(pl.lit("0 visible"))
        .otherwise(
            bucket(
                "gap",
                [0.5, 2, 10, 1e9],
                ["1 gap<0.5s", "2 gap0.5-2s", "3 gap2-10s", "4 gap>10s"],
                "5 never",
            )
        ),
        since_chg=bucket("since", [1, 3, 10, 1e9], ["a <1s", "b 1-3s", "c 3-10s", LATE], "e none"),
        state=pl.col("p_state").fill_null("null"),
    )


def native_rows(
    match_id: str, frames: pl.DataFrame, objects: pl.LazyFrame, config: StateConfig
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(rows, rule): rows is every native frame with a PFF team, with the rule's possession,
    PFF's, whether the ball was seen, the gap since it last was and the time since PFF's last
    change (bookkeeping only, never a model input); rule is the rule's possession on every
    native frame, for churn."""
    inf = infer(frames, objects, config)
    vis = (
        objects.filter(
            pl.col("object_type") == "ball",
            pl.col("visible").fill_null(False),
            pl.col("x").is_not_null(),
        )
        .select("frame_id")
        .unique()
        .with_columns(seen=pl.lit(True))
        .collect()
    )
    j = (
        frames.select(
            "frame_id", "period", "timestamp_s", p_state="ball_state", p_poss="possession_team"
        )
        .join(inf.select("frame_id", "possession_team", "ball_carrier_id"), on="frame_id")
        .join(vis, on="frame_id", how="left")
        .sort("period", "timestamp_s")
        .with_columns(seen=pl.col("seen").fill_null(False))
        .with_columns(
            # causal-looking bookkeeping, fine for a diagnostic
            last_seen=pl.when("seen").then("timestamp_s").forward_fill().over("period"),
            chg=pl.when(pl.col("p_poss") != pl.col("p_poss").forward_fill().shift(1).over("period"))
            .then("timestamp_s")
            .forward_fill()
            .over("period"),
        )
        .filter(pl.col("p_poss").is_not_null())
        .with_columns(
            match=pl.lit(match_id),
            dis=(pl.col("p_poss") != pl.col("possession_team")).fill_null(True),
            gap=pl.col("timestamp_s") - pl.col("last_seen"),
            since=pl.col("timestamp_s") - pl.col("chg"),
        )
    )
    return j, inf.select("frame_id", rule="possession_team")


def tick(period_t: str = "t_s") -> pl.Expr:
    """The grid tick of a row, as learned state stores it (swap_learned's key)."""
    return (pl.col(period_t) * 10).round().cast(pl.Int64)


def scored_grid(grid: pl.DataFrame, horizon: str = "h5") -> pl.DataFrame:
    """07's scored rows on the grid (label mask on, not all ESTIMATED): period, k, frame_id
    and pos (the shot label). One row per grid tick, so a native frame reused at two ticks
    appears twice."""
    return grid.filter(pl.col(f"label_mask_{horizon}"), ~pl.col("all_estimated")).select(
        "period", k=tick(), frame_id=pl.col("frame_id"), pos=f"label_shot_{horizon}"
    )


def join_learned(scored: pl.DataFrame, state: pl.DataFrame) -> pl.DataFrame:
    """Scored grid rows with the learned possession at the same (period, tick). The native
    frame_id must agree; a row the state lacks, or a repeated key, is an error."""
    got = scored.join(
        state.select("period", "k", "frame_id", learned="possession_team", fallback="fallback"),
        on=["period", "k"],
        how="left",
        validate="1:1",
        suffix="_state",
    )
    if got["frame_id_state"].null_count():
        raise ValueError("scored grid rows the learned state doesn't cover")
    if not got["frame_id"].equals(got["frame_id_state"]):
        raise ValueError("learned state's native frame_id differs from the grid's")
    return got.drop("frame_id_state")


def match_rows(
    match_id: str, fold: int, scored: pl.DataFrame, state: pl.DataFrame, native: pl.DataFrame
) -> pl.DataFrame:
    """One match's gate rows: every scored grid row with PFF's team, the default rule's
    disagreement (the old script's, by native frame), the learned disagreement and the
    buckets. Null counts as disagreement on both sides. native_rows' frames are unique per
    frame_id, so the join is many-to-one."""
    rows = (
        join_learned(scored, state)
        .join(native, on="frame_id", how="inner", validate="m:1")
        .with_columns(
            match=pl.lit(match_id),
            fold=pl.lit(fold),
            dis_learned=(pl.col("p_poss") != pl.col("learned")).fill_null(True),
        )
    )
    return label_rows(rows)


def tally(rows: pl.DataFrame, dis: str) -> dict:
    """Disagreement counts over the pooled rows for one possession source."""
    early = rows.filter(pl.col("since_chg").is_in(EARLY))
    late = rows.filter(pl.col("since_chg") == LATE)
    pos = rows.filter("pos")
    return {
        "rows": rows.height,
        "positives": pos.height,
        "overall": int(rows[dis].sum()),
        "first3_rows": early.height,
        "first3": int(early[dis].sum()),
        "positive_dis": int(pos[dis].sum()),
        "late_rows": late.height,
        "late_dis": int(late[dis].sum()),
    }


def churn(grid: pl.DataFrame, team: str = "team") -> int:
    """Home/away switches per period over every grid row, carrying the last known team over
    nulls so a null gap isn't a change. A period's first team is never a change, and
    nothing is counted across periods."""
    d = (
        grid.sort("period", "k")
        .with_columns(t=pl.col(team).forward_fill().over("period"))
        .with_columns(prev=pl.col("t").shift(1).over("period"))
    )
    return int(
        d.select((pl.col("prev").is_not_null() & (pl.col("t") != pl.col("prev"))).sum()).item()
    )


def check_baseline(default: dict, churns: dict, baseline: dict | None = None) -> None:
    """Stop unless the default rule reproduces the baseline counts the bars were cut from."""
    want = baseline or BASELINE
    got = default | churns
    bad = {k: (got.get(k), v) for k, v in want.items() if got.get(k) != v}
    if bad:
        raise ValueError(
            "baseline mismatch, the gate stops (counts, not bars, changed): "
            + ", ".join(f"{k} {g} != {w}" for k, (g, w) in bad.items())
        )


@dataclass(frozen=True)
class Bar:
    name: str
    value: float
    limit: float
    passed: bool


def evaluate(
    learned: dict, learned_churn: int, default: dict, bars: dict | None = None
) -> list[Bar]:
    """The five bars, all inclusive: learned must be at most the limit. Same denominators
    as the default (checked), null already counted as disagreement."""
    b = bars or BARS
    for k in ("rows", "positives", "late_rows", "first3_rows"):
        if learned[k] != default[k]:
            raise ValueError(
                f"learned and default denominators differ on {k}: {learned[k]} vs {default[k]}"
            )
    # no late rows is only possible off the real data (the baseline check demands 1,162,041)
    rate = lambda t: t["late_dis"] / t["late_rows"] if t["late_rows"] else 0.0
    late_rate, late_limit = rate(learned), rate(default) + b["late_margin"]
    out = [
        ("overall disagreement", learned["overall"], b["overall"]),
        ("first-3-s disagreement", learned["first3"], b["first3"]),
        ("positive disagreement", learned["positive_dis"], b["positive_dis"]),
        ("late-change rate (>= 10 s)", late_rate, late_limit),
        ("possession changes (churn)", learned_churn, b["churn"]),
    ]
    return [Bar(n, v, lim, v <= lim) for n, v, lim in out]


def folds_of(manifest: dict, folds_path: Path) -> dict[str, int]:
    """Match -> fold from the manifest, which must equal the frozen folds.json."""
    frozen = {m["match_id"]: m["fold"] for m in json.loads(folds_path.read_text())["matches"]}
    got = manifest["folds"]["assignments"]
    if got != frozen:
        raise ValueError(f"manifest folds differ from {folds_path}")
    return frozen


def run(
    config: StateConfig,
    gamestate_dir: Path,
    processed_dir: Path,
    folds_path: Path,
    ids: list[str] | None = None,
    baseline: dict | None = None,
    bars: dict | None = None,
    models_dir: Path | None = None,
    log=print,
) -> dict:
    """Score every match's `test` context, check the baseline, then the bars. Returns the
    rows, the counts, the per-match churn and fallbacks, and the bar results."""
    _, man = possession.learned_manifest(config, models_dir)
    fold_of = folds_of(man, folds_path)
    ids = ids or sorted(fold_of)
    parts, per = [], []
    for m in ids:
        t0 = time.perf_counter()
        fold = fold_of[m]
        frames = pl.read_parquet(gamestate_dir / m / "frames.parquet")
        if frames["possession_team"].is_null().all():
            continue
        objects = pl.scan_parquet(gamestate_dir / m / "objects.parquet").select(OBJECT_COLS)
        native, rule = native_rows(m, frames, objects, StateConfig())
        grid = pl.read_parquet(processed_dir / m / "frames_10hz.parquet")
        # only the match's own `test` context: never an inner-OOF or the live model
        state = possession.learned_state(
            m, config, possession.context_for(fold, fold), gamestate_dir, processed_dir, models_dir
        )
        scored = scored_grid(grid)
        parts.append(
            match_rows(
                m,
                fold,
                scored,
                state,
                native.select("frame_id", "p_poss", "p_state", "seen", "gap", "since", "dis"),
            )
        )
        every = grid.select("period", k=tick(), frame_id="frame_id", pff="possession_team")
        every = every.join(rule, on="frame_id", how="left", validate="m:1")
        per.append(
            {
                "match": m,
                "fold": fold,
                "churn_learned": churn(state.select("period", "k", team="possession_team")),
                "churn_pff": churn(every.select("period", "k", team="pff")),
                "churn_default": churn(every.select("period", "k", team="rule")),
                "fallback_all": int(state["fallback"].sum()),
                "grid_rows": state.height,
            }
        )
        log(f"{m}: fold {fold}, {time.perf_counter() - t0:.1f} s")
    rows = pl.concat(parts)
    per_match = pl.DataFrame(per)
    default = tally(rows, "dis")
    check_baseline(
        default,
        {
            "churn_pff": int(per_match["churn_pff"].sum()),
            "churn_default": int(per_match["churn_default"].sum()),
        },
        baseline,
    )
    learned = tally(rows, "dis_learned")
    results = evaluate(learned, int(per_match["churn_learned"].sum()), default, bars)
    fallback_scored = {
        m: int(g["fallback"].sum()) for (m,), g in rows.partition_by("match", as_dict=True).items()
    }
    return {
        "rows": rows,
        "default": default,
        "learned": learned,
        "per_match": per_match,
        "fallback_scored": fallback_scored,
        "bars": results,
        "passed": all(b.passed for b in results),
    }


def table(t: pl.DataFrame) -> str:
    with pl.Config(
        tbl_rows=80,
        tbl_cols=20,
        tbl_formatting="MARKDOWN",
        tbl_hide_dataframe_shape=True,
        tbl_hide_column_data_types=True,
        tbl_width_chars=300,
    ):
        return str(t)


def by(rows: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Rows and both disagreement counts per group."""
    return (
        rows.group_by(keys)
        .agg(n=pl.len(), default_dis=pl.col("dis").sum(), learned_dis=pl.col("dis_learned").sum())
        .with_columns(
            default_rate=(pl.col("default_dis") / pl.col("n")).round(4),
            learned_rate=(pl.col("learned_dis") / pl.col("n")).round(4),
        )
        .sort(keys)
    )


def report(res: dict, model_id: str) -> str:
    """The gate's markdown report: the five bars, then the saved tables (03: overall,
    per-fold, positives, ball visibility x change age, fallbacks and churn per match)."""
    rows, pm = res["rows"], res["per_match"]
    d, le = res["default"], res["learned"]
    out = [f"# Possession gate, {model_id}, H = 5\n"]
    out.append(
        f"**{'PASS' if res['passed'] else 'FAIL'}**: {sum(b.passed for b in res['bars'])} of "
        f"{len(res['bars'])} bars. Only these five pooled bars decide.\n"
    )
    bars = pl.DataFrame(
        {
            "bar": [b.name for b in res["bars"]],
            "learned": [round(float(b.value), 5) for b in res["bars"]],
            "limit (<=)": [round(float(b.limit), 5) for b in res["bars"]],
            "result": ["pass" if b.passed else "FAIL" for b in res["bars"]],
        }
    )
    out += [table(bars), ""]
    cmp = pl.DataFrame(
        {
            "": ["all", "first 3 s", "positives", ">= 10 s"],
            "rows": [d["rows"], d["first3_rows"], d["positives"], d["late_rows"]],
            "default": [d["overall"], d["first3"], d["positive_dis"], d["late_dis"]],
            "learned": [le["overall"], le["first3"], le["positive_dis"], le["late_dis"]],
        }
    )
    out += ["## Disagreement counts, learned vs the default rule", "", table(cmp), ""]
    out += ["## Per fold", "", table(by(rows, ["fold"])), ""]
    out += ["## Positives", "", table(by(rows.filter("pos"), ["fold"])), ""]
    out += [
        "## Ball visibility x time since PFF's change",
        "",
        table(by(rows, ["ball", "since_chg"])),
        "",
    ]
    fb = pl.DataFrame(
        {
            "match": list(res["fallback_scored"]),
            "fallback_scored": list(res["fallback_scored"].values()),
        }
    )
    per = pm.join(fb, on="match", how="left").with_columns(pl.col("fallback_scored").fill_null(0))
    head = (
        f"## Fallbacks and churn per match (totals: churn learned {int(pm['churn_learned'].sum())}, "
        f"PFF {int(pm['churn_pff'].sum())}, default {int(pm['churn_default'].sum())}; fallback rows on "
        f"scored rows {int(per['fallback_scored'].sum())}, on all grid rows {int(pm['fallback_all'].sum())})"
    )
    out += [head, "", table(per.sort("match")), ""]
    out.append(
        "Not reported: the S10 oracle-floor windows (03). No code or definition of them exists "
        "in the repo beyond 07's prose, so none was invented.\n"
    )
    return "\n".join(out)


def save(res: dict, model_id: str, runs_dir: Path) -> Path:
    """data/runs/possession-gate-<model>/: the report and the numbers behind it."""
    d = runs_dir / f"possession-gate-{model_id}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "gate_h5.md").write_text(report(res, model_id))
    (d / "gate_h5.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "passed": res["passed"],
                "bars": [vars(b) for b in res["bars"]],
                "default": res["default"],
                "learned": res["learned"],
                "per_match": res["per_match"].to_dicts(),
            },
            indent=2,
        )
        + "\n"
    )
    return d
