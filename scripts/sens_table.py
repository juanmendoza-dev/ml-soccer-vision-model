"""Sensitivity runs vs the clean baseline, one row per run, from each run's compare.md.

    python scripts/sens_table.py [data/runs]

Needs `python -m evaluation.compare data/runs/<run> <clean> --out data/runs/<run>/compare.md`
first (scripts/sens_queue.sh does that).
"""

import json
import re
import sys
from pathlib import Path


def fold_rows(md: str) -> tuple[list[list[float]], list[float]]:
    rows, pooled = [], []
    for line in md.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0].isdigit():
            rows.append([float(c) for c in cells[1:]])
        elif cells and cells[0] == "pooled":
            pooled = [float(c) if c not in ("–", "-") else float("nan") for c in cells[1:]]
    return rows, pooled


def deltas(md: str) -> dict[str, tuple[int, float]]:
    """compare's own summary lines: metric -> (folds where A is better, mean delta A - B)."""
    out = {}
    for m in re.finditer(
        r"^- (.+?) \(.*?A better on \*\*(\d)/\d\*\* folds, mean Δ \(A − B\) ([-+.\d]+)",
        md,
        re.MULTILINE,
    ):
        out[m[1]] = (int(m[2]), float(m[3]))
    return out


def main(runs: Path) -> None:
    out = [
        (
            "| run | degradation | arm | realized | PR-AUC | ΔPR-AUC (mean over folds) "
            "| folds worse | Δmiss rate | Δfalse / match |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for d in sorted(runs.glob("sens-*")):
        cmp_ = d / "compare.md"
        if not cmp_.exists():
            continue
        cfg = json.loads((d / "run.json").read_text())["config"]
        md = cmp_.read_text()
        rows, pooled = fold_rows(md)
        dl = deltas(md)
        n = len(rows)
        worse = sum(r[0] < r[1] for r in rows)
        real = "; ".join(
            f"{a} {next(iter(v.values())):g}" for a, v in cfg.get("degrade_realized", {}).items()
        )
        specs = cfg.get("degrade", [])
        name = "target (all)" if len(specs) > 3 else ", ".join(specs)
        if cfg.get("degrade_seed") != 20260928:
            name += f" (seed {cfg.get('degrade_seed')})"
        out.append(
            f"| {d.name} | {name} | {cfg.get('degrade_arm')} | "
            f"{real if len(specs) <= 3 else 'see run.json'} | {pooled[0]:.3f} | "
            f"{dl['PR-AUC'][1]:+.4f} | {worse}/{n} | {dl['miss rate'][1]:+.3f} | "
            f"{dl['false / match'][1]:+.2f} |"
        )
    print("\n".join(out))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/runs"))
