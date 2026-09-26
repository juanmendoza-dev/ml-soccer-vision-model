"""Shared pieces for dataset converters: velocities, the conversion report, writing."""

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import polars as pl

from gamestate.validate import validate_match

# Trailing smoothing window for velocities, in seconds.
VELOCITY_WINDOW_S = 0.2


def causal_velocities(objects: pl.DataFrame, frames: pl.DataFrame) -> pl.DataFrame:
    """Add vx, vy (m/s) from frames <= t of the same track only (02).

    A track segment breaks on a missing frame or a new period; the first frame
    of each segment gets null. Positions are smoothed with a trailing mean,
    then differenced backwards, so nothing after t leaks in.
    """
    fps = 1 / frames["timestamp_s"].diff().filter(frames["timestamp_s"].diff() > 0).median()
    window = max(1, round(VELOCITY_WINDOW_S * fps))
    track = ["object_id", "segment"]
    return (
        objects.drop("vx", "vy", strict=False)
        .join(frames.select("frame_id", "period", "timestamp_s"), on="frame_id")
        .sort("object_id", "frame_id")
        .with_columns(
            segment=((pl.col("frame_id").diff() != 1) | (pl.col("period").diff() != 0))
            .fill_null(True)
            .cum_sum()
            .over("object_id")
        )
        .with_columns(
            xs=pl.col("x").rolling_mean(window, min_samples=1).over(track),
            ys=pl.col("y").rolling_mean(window, min_samples=1).over(track),
            dt=pl.col("timestamp_s").diff().over(track),
        )
        .with_columns(
            vx=(pl.col("xs").diff().over(track) / pl.col("dt")),
            vy=(pl.col("ys").diff().over(track) / pl.col("dt")),
        )
        .drop("segment", "xs", "ys", "dt", "period", "timestamp_s")
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@dataclass
class ConversionReport:
    """conversion_report.json (06). Anything dropped without a line here is a bug."""

    match_id: str
    source: str
    source_files: dict[str, str] = field(default_factory=dict)  # name -> sha256
    converter_commit: str | None = field(default_factory=git_commit)
    rows: dict[str, dict[str, int]] = field(default_factory=dict)  # table -> {in, out}
    dropped: list[dict] = field(default_factory=list)
    changed: list[dict] = field(default_factory=list)
    unresolved: list[dict] = field(default_factory=list)
    checks: dict = field(default_factory=dict)
    validation_errors: list[str] = field(default_factory=list)

    def add_source(self, path: Path) -> None:
        self.source_files[path.name] = sha256(path)

    def drop(self, table: str, count: int, reason: str) -> None:
        if count:
            self.dropped.append({"table": table, "count": int(count), "reason": reason})

    def change(self, table: str, count: int, reason: str) -> None:
        if count:
            self.changed.append({"table": table, "count": int(count), "reason": reason})

    def unresolve(self, what: str, detail) -> None:
        self.unresolved.append({"what": what, "detail": detail})

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, default=str) + "\n")


def write_gamestate(
    tables: dict[str, pl.DataFrame], out_dir: Path, report: ConversionReport
) -> list[str]:
    """Write the five Parquet tables and the report, then validate. Returns errors."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.write_parquet(out_dir / f"{name}.parquet")
        report.rows.setdefault(name, {})["out"] = df.height
    report.validation_errors = validate_match(out_dir)
    report.write(out_dir / "conversion_report.json")
    return report.validation_errors
