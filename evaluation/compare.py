"""Paired comparison of two runs by fold (07: A beats B only if it wins on most folds).

    python -m evaluation.compare data/runs/<A> data/runs/<B> [--out FILE]

Both runs are scored on the same rows (07's scored rows come from the labels, never
from p) and must cover the same CV matches. Alarms use each run's own per-fold tau. CV
sources only: IDSSE is the final check, never a model comparison.
"""

import argparse
import sys
from pathlib import Path

import polars as pl

from evaluation.folds import FOLDS_PATH, GAMESTATE_DIR, load
from evaluation.metrics import threshold_free
from evaluation.report import (
    CV_SOURCES,
    PROCESSED_DIR,
    ReportError,
    alarms_at_tau,
    assign_matches,
    join_predictions,
    load_shots,
    match_sources,
    null_scored,
    table,
    taus_for,
)
from evaluation.runs import HORIZONS, load_run

KEYS = ["match_id", "period", "t_s"]
# metric, higher is better
METRICS = {"pr_auc": True, "roc_auc": True, "brier": False}
ALARM_METRICS = {"miss_rate": False, "false_per_match": False}


def cv_ids(preds: pl.DataFrame, gamestate_dir: Path, folds: dict) -> dict[str, int]:
    ids = sorted(preds["match_id"].unique().to_list())
    kept, _, _ = assign_matches(ids, match_sources(ids, gamestate_dir), folds, False)
    return {i: f for i, f in kept.items() if f is not None}


def wins(a: list[float], b: list[float], higher: bool) -> int:
    return sum((x > y) if higher else (x < y) for x, y in zip(a, b, strict=True))


def fold_rows(rows: pl.DataFrame, shots: pl.DataFrame, meta: dict, h: str) -> dict[int, dict]:
    """Per fold: threshold-free metrics plus alarms at the run's own tau, if it has one."""
    folds = sorted(rows["fold"].unique().to_list())
    out = {f: threshold_free(rows.filter(pl.col("fold") == f), h) for f in folds}
    taus = taus_for(meta, h, folds)
    if taus:
        _, _, per_fold = alarms_at_tau(rows, shots, taus)
        for f in folds:
            out[f] |= per_fold[f]
    return out


def compare(
    run_a: Path,
    run_b: Path,
    gamestate_dir: Path = GAMESTATE_DIR,
    processed_dir: Path = PROCESSED_DIR,
    folds_path: Path = FOLDS_PATH,
) -> str:
    folds = load(folds_path)
    meta_a, preds_a = load_run(run_a)
    meta_b, preds_b = load_run(run_b)
    horizons = [h for h in HORIZONS if h in meta_a["horizons"] and h in meta_b["horizons"]]
    if not horizons:
        raise ReportError("the runs share no horizon")
    fold_a = cv_ids(preds_a, gamestate_dir, folds)
    fold_b = cv_ids(preds_b, gamestate_dir, folds)
    if set(fold_a) != set(fold_b):
        raise ReportError(
            f"runs cover different CV matches: only A {sorted(set(fold_a) - set(fold_b))}, "
            f"only B {sorted(set(fold_b) - set(fold_a))}"
        )
    ids = sorted(fold_a)
    sources = match_sources(ids, gamestate_dir)
    info = pl.DataFrame(
        {"match_id": ids, "fold": [fold_a[i] for i in ids], "source": [sources[i] for i in ids]},
        schema={"match_id": pl.String, "fold": pl.Int64, "source": pl.String},
    )
    frames_a = join_predictions(preds_a, ids, horizons, processed_dir).sort(KEYS)
    frames_b = join_predictions(preds_b, ids, horizons, processed_dir).sort(KEYS)
    if not frames_a.select(KEYS).equals(frames_b.select(KEYS)):
        raise ReportError("the runs' grid rows don't line up")
    frames = frames_a.join(info, on="match_id", how="left", maintain_order="left")
    shots = load_shots(ids, gamestate_dir)
    name_a, name_b = meta_a["run_id"], meta_b["run_id"]

    out = [
        f"# Compare: A = {name_a}, B = {name_b}",
        "",
        f"- A: `{meta_a['model']}`, config `{meta_a.get('config', {})}`",
        f"- B: `{meta_b['model']}`, config `{meta_b.get('config', {})}`",
        "- Same scored rows for both (07). Alarms at each run's own per-fold τ.",
        "",
    ]
    for h in horizons:
        out += [f"## {h} ({h[1:]} s horizon)", ""]
        for src in CV_SOURCES:
            part = frames.with_columns(p_a=frames_a[f"p_{h}"], p_b=frames_b[f"p_{h}"]).filter(
                pl.col("source") == src
            )
            if not part.height:
                continue
            per = {}
            for tag, meta in (("a", meta_a), ("b", meta_b)):
                rows = part.rename({f"p_{tag}": "p"})
                bad = null_scored(rows, h)
                if bad:
                    raise ReportError(f"run {tag.upper()}: p_{h} null on scored rows in {bad}")
                per[tag] = (threshold_free(rows, h), fold_rows(rows, shots, meta, h))
            out += source_block(src, per, h)
    return "\n".join(out).rstrip() + "\n"


def source_block(src: str, per: dict, h: str) -> list[str]:
    (pool_a, fa), (pool_b, fb) = per["a"], per["b"]
    fold_ids = sorted(fa)
    has_alarms = all("miss_rate" in fa[f] and "miss_rate" in fb[f] for f in fold_ids)
    metrics = METRICS | (ALARM_METRICS if has_alarms else {})
    names = {
        "pr_auc": "PR-AUC",
        "roc_auc": "ROC-AUC",
        "brier": "Brier",
        "miss_rate": "miss rate",
        "false_per_match": "false / match",
    }
    header = ["fold"] + [f"{names[m]} {t}" for m in metrics for t in ("A", "B")]
    rows = [[f] + [d[f][m] for m in metrics for d in (fa, fb)] for f in fold_ids]
    rows.append(["pooled"] + [p.get(m) for m in metrics for p in (pool_a, pool_b)])
    out = [f"### {src.upper()}", "", *table(header, rows), ""]
    for m, higher in metrics.items():
        a = [fa[f][m] for f in fold_ids]
        b = [fb[f][m] for f in fold_ids]
        k = wins(a, b, higher)
        verdict = "A wins" if k > len(fold_ids) / 2 else "A doesn't win"
        out.append(
            f"- {names[m]} ({'higher' if higher else 'lower'} is better): A better on "
            f"**{k}/{len(fold_ids)}** folds, mean Δ (A − B) "
            f"{sum(a) / len(a) - sum(b) / len(b):+.4f}. {verdict}."
        )
    if not has_alarms:
        out.append("- Alarm metrics left out: a run has no per-fold τ.")
    return out + [""]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_a", type=Path)
    ap.add_argument("run_b", type=Path)
    ap.add_argument("--gamestate", type=Path, default=GAMESTATE_DIR)
    ap.add_argument("--processed", type=Path, default=PROCESSED_DIR)
    ap.add_argument("--folds", type=Path, default=FOLDS_PATH)
    ap.add_argument("--out", type=Path, help="default: print")
    args = ap.parse_args(argv)
    try:
        text = compare(args.run_a, args.run_b, args.gamestate, args.processed, args.folds)
    except ReportError as e:
        print(f"compare: {e}", file=sys.stderr)
        return 1
    if args.out:
        args.out.write_text(text, encoding="utf-8")  # τ, Δ: Windows defaults to cp1252
        print(f"wrote {args.out}")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
