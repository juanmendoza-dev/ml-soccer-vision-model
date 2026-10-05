"""Label tool logic (10-ball 1c): suggestions, auto-accept, spot check and flags, from the
PFF reference (vision.ball_truth) and the clip's ball candidates. scripts/ball_click.py
draws them and the person decides; nothing here writes a label file.

    python -m vision.ball_assist --status        # per clip: labeled, auto, spot check, flags
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import polars as pl

SUGGEST_PX = {"pff": 40.0, "estimated": 60.0}
AUTO_CONF, AUTO_PX, AUTO_ALONE_PX = 0.5, 15.0, 40.0
SPOT_SHARE, SPOT_MAX_CHANGED = 0.10, 0.03
FLAG_LABEL_PX, FLAG_NONE_PX = 30.0, 25.0
MANIFESTS = (Path("data/splits/vision_benchmark.json"), Path("data/splits/demo_clips.json"))

Cand = tuple[float, float, float]  # box center u, v (source pixels), confidence


def cands_by_src(balls: pl.DataFrame, skip: int) -> dict[int, list[Cand]]:
    """balls.parquet (or --candidates, same columns) keyed by source frame index."""
    out: dict[int, list[Cand]] = {}
    for f, x1, y1, x2, y2, c in balls.select(
        "frame_id", "x1", "y1", "x2", "y2", "det_confidence"
    ).iter_rows():
        out.setdefault(f + skip, []).append(((x1 + x2) / 2, (y1 + y2) / 2, c))
    return out


def _near(cands: list[Cand], proj) -> list[tuple[float, Cand]]:
    return sorted(
        ((float(np.hypot(u - proj[0], v - proj[1])), (u, v, c)) for u, v, c in cands),
        key=lambda dc: dc[0],
    )


def suggest(cands: list[Cand], proj, kind: str) -> Cand | None:
    """The candidate nearest PFF's projection, at any confidence, within 40 px on `pff`
    frames and 60 px on `estimated` ones."""
    r = SUGGEST_PX.get(kind)
    near = _near(cands, proj) if r is not None else []
    return near[0][1] if near and near[0][0] <= r else None


def auto_accept(cands: list[Cand], proj, kind: str) -> tuple[float, float] | None:
    """A `pff` frame's label without a look: the nearest candidate at >= 0.5, within 15 px
    of the projection, with no other candidate within 40 px of it."""
    if kind != "pff":
        return None
    near = _near(cands, proj)
    if not near or near[0][0] > AUTO_PX or near[0][1][2] < AUTO_CONF:
        return None
    if len(near) > 1 and near[1][0] <= AUTO_ALONE_PX:
        return None
    return near[0][1][:2]


def spot_sample(auto: list[int], share: float = SPOT_SHARE, seed: int = 0) -> list[int]:
    """The auto-accepted frames a person looks at anyway: a seeded tenth, at least one."""
    if not auto:
        return []
    k = max(1, math.ceil(share * len(auto)))
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.choice(sorted(auto), k, replace=False))


def spot_result(entry: dict) -> tuple[int, int, float]:
    """(checked, changed, share): a spot-checked frame no longer in `auto` was changed."""
    checked = entry.get("spot_checked", [])
    auto = set(entry.get("auto", []))
    changed = sum(i not in auto for i in checked)
    return len(checked), changed, changed / len(checked) if checked else float("nan")


def apply_spot(entry: dict, sample: list[int]) -> bool:
    """Once every sampled frame is checked: above 3% changed, auto-accept is off for the
    clip and the unchecked auto labels go back to the person. True if that happened."""
    checked = set(entry.get("spot_checked", []))
    if not set(sample) <= checked:
        return False
    _, _, share = spot_result(entry)
    if not share > SPOT_MAX_CHANGED:
        return False
    for i in entry.get("auto", []):
        if i not in checked:
            entry["labels"].pop(i, None)
    entry["auto"] = []
    entry["auto_off"] = True
    return True


def _proj(truth: pl.DataFrame) -> dict[int, tuple[str, float | None, float | None]]:
    return {s: (k, u, v) for s, k, u, v in truth.select("src", "kind", "u", "v").iter_rows()}


def flag_reason(labels: dict, truth: pl.DataFrame, cands: dict, src: int) -> str | None:
    """Why a label disagrees with PFF (10-ball 1c --flag), or None: an [x, y] more than
    30 px from a `pff` projection, or "none" with a candidate within 25 px of it."""
    kind, u, v = _proj(truth).get(src, (None, None, None))
    lab = labels.get(src)
    if kind != "pff" or u is None:
        return None
    if isinstance(lab, list):
        d = float(np.hypot(lab[0] - u, lab[1] - v))
        return f"label {d:.0f} px from PFF" if d > FLAG_LABEL_PX else None
    if lab == "none":
        near = [float(np.hypot(x - u, y - v)) for x, y, _ in cands.get(src, [])]
        if near and min(near) <= FLAG_NONE_PX:
            return f"none, candidate {min(near):.0f} px from PFF"
    return None


def flags(labels: dict, truth: pl.DataFrame, cands: dict) -> list[int]:
    return sorted(i for i in labels if flag_reason(labels, truth, cands, i) is not None)


def unresolved_flags(entry: dict, truth: pl.DataFrame, cands: dict) -> list[int]:
    seen = set(entry.get("flag_checked", []))
    return [i for i in flags(entry["labels"], truth, cands) if i not in seen]


def clip_inputs(clip: dict) -> tuple[pl.DataFrame | None, dict]:
    """The clip's PFF reference and candidates by source frame (both empty if missing)."""
    from vision import ball_truth

    truth = ball_truth.load(clip)
    cache = Path("data/vision_cache") / clip["clip_id"]
    side = ball_truth.TRUTH_DIR / f"{clip['clip_id']}.json"
    if truth is None or not (cache / "balls.parquet").exists():
        return truth, {}
    fps = json.loads(side.read_text())["fps"]
    start = json.loads((cache / "run.json").read_text())["video_start_s"]
    return truth, cands_by_src(pl.read_parquet(cache / "balls.parquet"), round(start * fps))


def status(labels_path: Path) -> list[str]:
    from vision import bench

    all_labels = bench.load_ball_labels(labels_path)
    out = []
    for manifest in MANIFESTS:
        for clip in bench.load_manifest(manifest):
            entry = all_labels.get(clip["clip_id"])
            if entry is None:
                continue
            truth, cands = clip_inputs(clip)
            total = "?" if truth is None else truth.height
            checked, changed, share = spot_result(entry)
            spot = (
                "auto-accept off"
                if entry.get("auto_off")
                else (
                    f"spot check {changed} of {checked} changed ({share:.0%})"
                    if checked
                    else "no spot check yet"
                )
            )
            flag = (
                "no PFF reference"
                if truth is None
                else (f"{len(unresolved_flags(entry, truth, cands))} unresolved flags")
            )
            out.append(
                f"{clip['clip_id']}: {len(entry['labels'])} of {total} frames labeled, "
                f"{len(entry.get('auto', []))} auto, {spot}, {flag}"
            )
    return out


def main(argv: list[str] | None = None) -> None:
    from vision import bench

    ap = argparse.ArgumentParser(prog="vision.ball_assist")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--labels", type=Path, default=bench.BALL_LABELS)
    args = ap.parse_args(argv)
    if args.status:
        for line in status(args.labels):
            print(line)


if __name__ == "__main__":
    main()
