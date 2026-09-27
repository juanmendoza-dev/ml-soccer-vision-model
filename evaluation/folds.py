"""Grouped CV folds by match -> data/splits/folds.json (07, "Splits").

    python -m evaluation.folds [--gamestate DIR] [--out FILE]

Append-only: matches already in the file never move. New sources are stratified
by open-play shot count (the shots 05's labels use); a new match from a source
that's already there goes to its smallest fold.
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

from prediction.labels import label_events

GAMESTATE_DIR = Path("data/gamestate")
FOLDS_PATH = Path("data/splits/folds.json")
N_FOLDS = 5
SEED = 20260926
CV_SOURCES = ("pff", "skillcorner")  # IDSSE is the external test set, Metrica is dev only (06)
INNER_SHARE = 0.15


def open_play_shots(match_dir: Path) -> int:
    """Open-play shots with a tracked frame: the positives 05's labels are built from."""
    frames = pl.read_parquet(
        match_dir / "frames.parquet", columns=["frame_id", "period", "timestamp_s"]
    )
    events = pl.read_parquet(match_dir / "events.parquet")
    lev = label_events(events, frames)
    return lev.filter(pl.col("kind") == "shot", pl.col("period").is_not_null()).height


def cv_matches(gamestate_dir: Path = GAMESTATE_DIR) -> pl.DataFrame:
    """match_id, source, open_play_shots for every match in a CV source."""
    rows = []
    for d in sorted(gamestate_dir.glob("*/match.parquet")):
        source = pl.read_parquet(d)["source"].item()
        if source in CV_SOURCES:
            rows.append((d.parent.name, source, open_play_shots(d.parent)))
    return pl.DataFrame(
        rows,
        schema={"match_id": pl.String, "source": pl.String, "open_play_shots": pl.Int64},
        orient="row",
    )


def by_shots(ids: list[str], shots: list[int], rng: np.random.Generator) -> list[int]:
    """Indices sorted by shot count, ties in random (seeded) order."""
    tiebreak = rng.permutation(len(ids))
    return sorted(range(len(ids)), key=lambda i: (shots[i], tiebreak[i], ids[i]))


def stratified_folds(
    ids: list[str], shots: list[int], seed: int, n_folds: int = N_FOLDS
) -> dict[str, int]:
    """Sort by shots, cut into blocks of n_folds, shuffle each block across the folds."""
    rng = np.random.default_rng(seed)
    order = by_shots(ids, shots, rng)
    out = {}
    for start in range(0, len(order), n_folds):
        block = order[start : start + n_folds]
        for i, fold in zip(block, rng.permutation(n_folds)[: len(block)], strict=True):
            out[ids[i]] = int(fold)
    return out


def assign(
    existing: list[dict], new: pl.DataFrame, seed: int = SEED, n_folds: int = N_FOLDS
) -> list[dict]:
    """Add fold rows for `new` matches not already in `existing`; existing rows are untouched."""
    have = {m["match_id"] for m in existing}
    new = new.filter(~pl.col("match_id").is_in(list(have))).sort("match_id")
    out = [dict(m) for m in existing]
    for (source,), g in new.group_by("source", maintain_order=True):
        ids, shots = g["match_id"].to_list(), g["open_play_shots"].to_list()
        known = [m for m in out if m["source"] == source]
        if not known:
            folds = stratified_folds(ids, shots, seed, n_folds)
            out += [
                {"match_id": i, "source": source, "fold": folds[i], "open_play_shots": s}
                for i, s in zip(ids, shots, strict=True)
            ]
            continue
        for i, s in zip(ids, shots, strict=True):
            load = {f: [0, 0] for f in range(n_folds)}
            for m in known:
                load[m["fold"]][0] += 1
                load[m["fold"]][1] += m["open_play_shots"]
            fold = min(load, key=lambda f: (load[f][0], load[f][1], f))
            row = {"match_id": i, "source": source, "fold": fold, "open_play_shots": s}
            out.append(row)
            known.append(row)
    return sorted(out, key=lambda m: (m["source"], m["match_id"]))


def refresh_shots(existing: list[dict], current: pl.DataFrame) -> tuple[list[dict], int]:
    """Update open_play_shots of matches already in the file after a label definition
    change. Folds never move; returns the rows and how many counts changed."""
    shots = dict(current.select("match_id", "open_play_shots").iter_rows())
    out, changed = [], 0
    for m in existing:
        new = shots.get(m["match_id"], m["open_play_shots"])
        changed += new != m["open_play_shots"]
        out.append(m | {"open_play_shots": new})
    return out, changed


def load(path: Path = FOLDS_PATH) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"n_folds": N_FOLDS, "seed": SEED, "frozen": {}, "matches": []}


def inner_split(fold: int, folds: dict | None = None, share: float = INNER_SHARE) -> list[str]:
    """Inner validation matches for an outer fold: ~share of each source's training matches,
    stratified by shots like the outer folds. Deterministic in (seed, fold)."""
    folds = folds or load()
    rng = np.random.default_rng([folds["seed"], fold])
    out = []
    train = [m for m in folds["matches"] if m["fold"] != fold]
    for source in sorted({m["source"] for m in train}):
        ms = sorted((m for m in train if m["source"] == source), key=lambda m: m["match_id"])
        ids = [m["match_id"] for m in ms]
        order = by_shots(ids, [m["open_play_shots"] for m in ms], rng)
        n_val = max(1, round(share * len(ids)))
        for block in np.array_split(np.array(order), n_val):
            out.append(ids[int(rng.choice(block))])
    return sorted(out)


def fold_table(folds: dict) -> pl.DataFrame:
    return (
        pl.DataFrame(folds["matches"])
        .group_by("source", "fold")
        .agg(matches=pl.len(), open_play_shots=pl.col("open_play_shots").sum())
        .sort("source", "fold")
    )


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gamestate", type=Path, default=GAMESTATE_DIR)
    ap.add_argument("--out", type=Path, default=FOLDS_PATH)
    ap.add_argument(
        "--refresh-shots",
        action="store_true",
        help="update stored shot counts after a label change; folds don't move",
    )
    args = ap.parse_args(argv)
    folds = load(args.out)
    before = {m["match_id"] for m in folds["matches"]}
    current = cv_matches(args.gamestate)
    if args.refresh_shots:
        folds["matches"], changed = refresh_shots(folds["matches"], current)
        print(f"{changed} shot counts refreshed")
    folds["matches"] = assign(folds["matches"], current, folds["seed"], folds["n_folds"])
    added = [m for m in folds["matches"] if m["match_id"] not in before]
    today = dt.datetime.now().astimezone().date().isoformat()
    for source in {m["source"] for m in added}:
        folds["frozen"].setdefault(source, today)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(folds, indent=2) + "\n")
    print(f"{len(added)} matches added, {len(before)} kept")
    print(fold_table(folds))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
