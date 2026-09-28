"""Markdown report for one model run (07, "Report").

    python -m evaluation.report data/runs/<run_id> [--final] [--allow-partial]

Joins the run's predictions onto frames_10hz for labels, looks up each match's fold
and source itself, and writes <run_dir>/report.md. IDSSE only renders with --final.
"""

import argparse
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import polars as pl

from evaluation.folds import FOLDS_PATH, GAMESTATE_DIR, load
from evaluation.metrics import (
    calibration,
    score_alarms,
    shots_table,
    tau_sweep,
    threshold_free,
)
from evaluation.runs import HORIZONS, git_state, load_run

PROCESSED_DIR = Path("data/processed")
CV_SOURCES = ("pff", "skillcorner")
EXTERNAL_SOURCES = ("idsse",)
SWEEP_TAUS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
LEAD_BINS = (0, 1, 2, 3, 4, 5)  # seconds; last bin is >= 5
FRAME_COLS = ["match_id", "period", "t_s", "possession_team", "ball_state", "all_estimated"]


class ReportError(ValueError):
    pass


def tenths(col: str = "t_s") -> pl.Expr:
    return (pl.col(col) * 10).round().cast(pl.Int64).alias("k")


def match_sources(ids: list[str], gamestate_dir: Path) -> dict[str, str]:
    out = {}
    for i in ids:
        path = gamestate_dir / i / "match.parquet"
        if not path.exists():
            raise ReportError(f"match {i}: no {path}")
        out[i] = pl.read_parquet(path)["source"].item()
    return out


def assign_matches(
    ids: list[str], sources: dict[str, str], folds: dict, allow_partial: bool
) -> tuple[dict[str, int | None], list[str], list[str]]:
    """({match: fold, None for external}, dropped matches, problems for a partial run)."""
    fold_of = {m["match_id"]: m["fold"] for m in folds["matches"]}
    kept, dropped = {}, []
    for i in ids:
        src = sources[i]
        if src in CV_SOURCES:
            if i not in fold_of:
                raise ReportError(f"match {i} ({src}) isn't in folds.json")
            kept[i] = fold_of[i]
        elif src in EXTERNAL_SOURCES:
            kept[i] = None
        else:
            dropped.append(i)
    problems = []
    for src in sorted({sources[i] for i in kept} & set(CV_SOURCES)):
        listed = {m["match_id"] for m in folds["matches"] if m["source"] == src}
        missing = sorted(listed - set(kept))
        if missing:
            problems.append(f"{src}: {len(missing)} of {len(listed)} matches have no predictions")
    if problems and not allow_partial:
        raise ReportError("; ".join(problems) + " (--allow-partial to render anyway)")
    return kept, dropped, problems


def join_predictions(
    preds: pl.DataFrame, ids: list[str], horizons: list[str], processed_dir: Path
) -> pl.DataFrame:
    """frames_10hz rows of `ids` with p_<h> joined on (match_id, period, tenths of t_s)."""
    cols = FRAME_COLS + [f"label_{x}_{h}" for h in horizons for x in ("mask", "shot")]
    frames = pl.concat(
        [pl.read_parquet(processed_dir / i / "frames_10hz.parquet", columns=cols) for i in ids]
    ).with_columns(tenths())
    p = preds.filter(pl.col("match_id").is_in(ids)).with_columns(tenths()).drop("t_s")
    dup = p.filter(pl.struct("match_id", "period", "k").is_duplicated())
    if dup.height:
        raise ReportError(f"{dup.height} duplicate prediction keys, e.g. {dup.row(0, named=True)}")
    stray = p.join(frames, on=["match_id", "period", "k"], how="anti")
    if stray.height:
        raise ReportError(
            f"{stray.height} predictions don't land on a frames_10hz row, "
            f"e.g. {stray.row(0, named=True)}"
        )
    return frames.join(p, on=["match_id", "period", "k"], how="left").drop("k")


def load_shots(ids: list[str], gamestate_dir: Path) -> pl.DataFrame:
    return pl.concat(
        [
            shots_table(
                pl.read_parquet(gamestate_dir / i / "events.parquet"),
                pl.read_parquet(
                    gamestate_dir / i / "frames.parquet",
                    columns=["frame_id", "period", "timestamp_s"],
                ),
                i,
            )
            for i in ids
        ]
    )


def null_scored(rows: pl.DataFrame, h: str) -> list[str]:
    bad = rows.filter(pl.col(f"label_mask_{h}"), ~pl.col("all_estimated"), pl.col("p").is_null())
    return sorted(bad["match_id"].unique().to_list())


def fmt(x, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return f"{x:,}" if isinstance(x, int) else str(x)


def table(header: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return out + ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows]


def metric_row(m: dict) -> list:
    return [m["rows"], m["positives"], m["base_rate"], m["pr_auc"], m["roc_auc"], m["brier"]]


METRIC_HEADER = ["rows", "positives", "base rate", "PR-AUC", "ROC-AUC", "Brier"]


def alarms_at_tau(
    rows: pl.DataFrame, shots: pl.DataFrame, taus: dict[int, float]
) -> tuple[pl.DataFrame, pl.DataFrame, dict[int, dict]]:
    """Each fold scored with its own tau on its own matches, then pooled."""
    scored, al, per_fold = [], [], {}
    for fold, tau in sorted(taus.items()):
        fr = rows.filter(pl.col("fold") == fold)
        s, a = score_alarms(fr, shots, tau)
        scored.append(s)
        al.append(a)
        n = fr["match_id"].n_unique()
        false = a.filter(~pl.col("true")).height
        per_fold[fold] = {
            "miss_rate": s["lead_s"].null_count() / s.height if s.height else float("nan"),
            "false_per_match": false / n if n else float("nan"),
        }
    return pl.concat(scored), pl.concat(al), per_fold


def lead_table(scored: pl.DataFrame) -> list[str]:
    lead = scored["lead_s"].drop_nulls().to_numpy()
    edges = [*LEAD_BINS, np.inf]
    rows = []
    for lo, hi in pairwise(edges):
        label = f"≥ {lo} s" if np.isinf(hi) else f"{lo}–{hi} s"
        rows.append([label, int(((lead >= lo) & (lead < hi)).sum())])
    rows.append(["missed", scored["lead_s"].null_count()])
    return table(["lead time", "shots"], rows)


def taus_for(meta: dict, h: str, folds_present: list[int]) -> dict[int, float] | None:
    given = (meta.get("tau") or {}).get(h) or {}
    taus = {int(k): float(v) for k, v in given.items() if k.isdigit()}
    return taus if taus and set(folds_present) <= set(taus) else None


def source_section(
    rows: pl.DataFrame, shots: pl.DataFrame, meta: dict, h: str, src: str
) -> list[str]:
    out = [f"### {src.upper()}", ""]
    bad = null_scored(rows, h)
    if bad:
        raise ReportError(f"p_{h} is null on scored rows in matches {bad}")
    n_matches = rows["match_id"].n_unique()
    folds = sorted(rows["fold"].unique().to_list())
    out += [f"{n_matches} matches, folds {folds}.", "", "**Pooled out-of-fold**", ""]
    out += table(METRIC_HEADER, [metric_row(threshold_free(rows, h))])

    taus = taus_for(meta, h, folds)
    alarm_fold = {}
    if taus:
        scored, al, alarm_fold = alarms_at_tau(rows, shots, {f: taus[f] for f in folds})
        lead = scored["lead_s"].drop_nulls()
        false = al.filter(~pl.col("true")).height
        out += ["", "**Alarms at each fold's chosen τ** (pooled)", ""]
        out += table(
            [
                "open-play shots",
                "missed",
                "miss rate",
                "lead median (s)",
                "lead q25–q75 (s)",
                "alarms",
                "false alarms",
                "false / match",
            ],
            [
                [
                    scored.height,
                    scored["lead_s"].null_count(),
                    scored["lead_s"].null_count() / scored.height if scored.height else None,
                    float(lead.median()) if len(lead) else None,
                    f"{lead.quantile(0.25):.2f}–{lead.quantile(0.75):.2f}" if len(lead) else None,
                    al.height,
                    false,
                    false / n_matches,
                ]
            ],
        )
        out += ["", *lead_table(scored)]
    else:
        out += ["", "**Alarms:** τ not chosen (no per-fold τ in run.json for every fold)."]

    header = ["fold", "matches", *METRIC_HEADER]
    if taus:
        header += ["miss rate", "false / match"]
    per_fold, stats = [], []
    for f in folds:
        fr = rows.filter(pl.col("fold") == f)
        m = threshold_free(fr, h)
        stats.append([m["pr_auc"], m["roc_auc"], m["brier"]])
        r = [f, fr["match_id"].n_unique(), *metric_row(m)]
        if taus:
            r += [alarm_fold[f]["miss_rate"], alarm_fold[f]["false_per_match"]]
        per_fold.append(r)
    s = np.array(stats, dtype=float)
    ddof = 1 if len(s) > 1 else 0
    spread = ["mean ± std", "", "", "", ""] + [
        f"{np.nanmean(s[:, j]):.3f} ± {np.nanstd(s[:, j], ddof=ddof):.3f}" for j in range(3)
    ]
    if taus:
        spread += ["", ""]
    out += ["", "**Per fold**", "", *table(header, [*per_fold, spread])]

    y = rows.filter(pl.col(f"label_mask_{h}"), ~pl.col("all_estimated"))
    cal = calibration(y[f"label_shot_{h}"].to_numpy().astype(bool), y["p"].to_numpy())
    out += ["", "**Calibration** (pooled, 10 quantile bins)", ""]
    out += table(["bin", "mean p", "positive rate", "rows"], cal.rows())

    sweep = tau_sweep(rows, shots, SWEEP_TAUS)
    out += [
        "",
        "**τ sweep** (pooled out-of-fold; descriptive only, τ is picked on training matches)",
        "",
    ]
    out += table(
        ["τ", "miss rate", "lead median (s)", "alarms", "false / match"],
        sweep.select(
            "tau", "miss_rate", "lead_s_median", "alarms", "false_alarms_per_match"
        ).rows(),
    )
    return out + [""]


def external_section(rows: pl.DataFrame, shots: pl.DataFrame, meta: dict, h: str) -> list[str]:
    bad = null_scored(rows, h)
    if bad:
        raise ReportError(f"p_{h} is null on scored rows in matches {bad}")
    out = [f"### IDSSE ({rows['match_id'].n_unique()} matches)", ""]
    out += table(METRIC_HEADER, [metric_row(threshold_free(rows, h))])
    tau = ((meta.get("tau") or {}).get(h) or {}).get("final")
    if tau is None:
        out += ["", "**Alarms:** no `final` τ in run.json."]
    else:
        s, a = score_alarms(rows, shots, float(tau))
        false = a.filter(~pl.col("true")).height
        out += ["", f"**Alarms at τ = {tau}**", ""]
        out += table(
            ["open-play shots", "missed", "lead median (s)", "false / match"],
            [
                [
                    s.height,
                    s["lead_s"].null_count(),
                    float(s["lead_s"].median()) if s["lead_s"].drop_nulls().len() else None,
                    false / rows["match_id"].n_unique(),
                ]
            ],
        )
    return out + [""]


def render(
    run_dir: Path,
    gamestate_dir: Path = GAMESTATE_DIR,
    processed_dir: Path = PROCESSED_DIR,
    folds_path: Path = FOLDS_PATH,
    final: bool = False,
    allow_partial: bool = False,
) -> str:
    meta, preds = load_run(run_dir)
    horizons = [h for h in HORIZONS if h in meta["horizons"]]
    ids = sorted(preds["match_id"].unique().to_list())
    sources = match_sources(ids, gamestate_dir)
    kept, dropped, problems = assign_matches(ids, sources, load(folds_path), allow_partial)
    cv = [i for i, f in kept.items() if f is not None]
    ext = [i for i, f in kept.items() if f is None]
    commit, dirty = git_state()

    out = []
    if problems:
        out += [
            "# PARTIAL RUN — not comparable to other runs",
            "",
            *(f"- {p}" for p in problems),
            "",
        ]
    out += [
        f"# Report: {meta['run_id']}",
        "",
        f"- Model: `{meta['model']}`",
        f"- Run commit: `{meta.get('git_commit')}`" + (" (dirty)" if meta.get("git_dirty") else ""),
        f"- Report commit: `{commit}`" + (" (dirty)" if dirty else ""),
        f"- Created: {meta.get('created')}",
        f"- Horizons: {', '.join(horizons)}",
        f"- Config: `{meta.get('config', {})}`",
    ]
    if dropped:
        out.append(f"- Dropped {len(dropped)} matches from other sources (not CV or external)")
    if ext and not final:
        out.append(f"- {len(ext)} IDSSE matches present, not shown (use --final)")
    out.append("")

    fold_of = pl.DataFrame(
        {"match_id": list(kept), "fold": list(kept.values())},
        schema={"match_id": pl.String, "fold": pl.Int64},
    )
    shown = cv + (ext if final else [])
    if not shown:
        raise ReportError("no CV matches in the run" + ("" if final else " (IDSSE needs --final)"))
    frames = join_predictions(preds, shown, horizons, processed_dir).join(
        fold_of, on="match_id", how="left"
    )
    shots = load_shots(shown, gamestate_dir)
    src = pl.DataFrame({"match_id": shown, "source": [sources[i] for i in shown]})
    frames = frames.join(src, on="match_id", how="left")

    for h in horizons:
        rows_h = frames.rename({f"p_{h}": "p"})
        out += [f"## {h} ({h[1:]} s horizon)", ""]
        for s in CV_SOURCES:
            part = rows_h.filter(pl.col("source") == s)
            if part.height:
                out += source_section(part, shots, meta, h, s)
        if final and ext:
            out += external_section(rows_h.filter(pl.col("fold").is_null()), shots, meta, h)
    return "\n".join(out).rstrip() + "\n"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--gamestate", type=Path, default=GAMESTATE_DIR)
    ap.add_argument("--processed", type=Path, default=PROCESSED_DIR)
    ap.add_argument("--folds", type=Path, default=FOLDS_PATH)
    ap.add_argument("--out", type=Path, help="default: <run_dir>/report.md")
    ap.add_argument("--final", action="store_true", help="render IDSSE (the final check)")
    ap.add_argument("--allow-partial", action="store_true", help="render a run missing CV matches")
    args = ap.parse_args(argv)
    try:
        text = render(
            args.run_dir, args.gamestate, args.processed, args.folds, args.final, args.allow_partial
        )
    except ReportError as e:
        print(f"report: {e}", file=sys.stderr)
        return 1
    out = args.out or args.run_dir / "report.md"
    out.write_text(text)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
