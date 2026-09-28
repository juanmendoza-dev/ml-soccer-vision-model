"""Model runs on disk: data/runs/<run_id>/{run.json, predictions.parquet} (07, "Run format")."""

import datetime as dt
import json
import subprocess
from pathlib import Path

import polars as pl

RUNS_DIR = Path("data/runs")
KEYS = ("match_id", "period", "t_s")
HORIZONS = ("h5", "h3")


def git_state() -> tuple[str | None, bool | None]:
    """(commit, dirty); (None, None) outside a git checkout."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit, bool(status.strip())


def check_predictions(preds: pl.DataFrame, horizons: list[str]) -> None:
    unknown = set(horizons) - set(HORIZONS)
    if not horizons or unknown:
        raise ValueError(f"horizons must be a non-empty subset of {HORIZONS}, got {horizons}")
    need = [*KEYS, *(f"p_{h}" for h in horizons)]
    missing = [c for c in need if c not in preds.columns]
    if missing:
        raise ValueError(f"predictions are missing columns {missing}")
    for h in horizons:
        p = preds[f"p_{h}"].drop_nulls()
        if len(p) and (p.is_nan().any() or p.min() < 0 or p.max() > 1):
            raise ValueError(f"p_{h} must be in [0, 1], null where the model doesn't predict")


def save_run(run_dir: Path, preds: pl.DataFrame, meta: dict) -> Path:
    """Write predictions.parquet and run.json; stamps git commit, dirty flag and time.

    `meta` needs `model` and `horizons`; `config` and per-fold `tau` are optional.
    """
    run_dir = Path(run_dir)
    for key in ("model", "horizons"):
        if key not in meta:
            raise ValueError(f"meta needs {key!r}")
    horizons = list(meta["horizons"])
    check_predictions(preds, horizons)
    commit, dirty = git_state()
    doc = {
        "run_id": run_dir.name,
        **meta,
        "horizons": horizons,
        "git_commit": commit,
        "git_dirty": dirty,
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    preds.select(*KEYS, *(f"p_{h}" for h in horizons)).write_parquet(
        run_dir / "predictions.parquet"
    )
    (run_dir / "run.json").write_text(json.dumps(doc, indent=2) + "\n")
    return run_dir


def load_run(run_dir: Path) -> tuple[dict, pl.DataFrame]:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run.json").read_text())
    preds = pl.read_parquet(run_dir / "predictions.parquet")
    check_predictions(preds, meta["horizons"])
    return meta, preds
