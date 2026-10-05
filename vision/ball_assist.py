"""Label tool logic (10-ball 1c): suggestions, auto-accept, spot check and flags, from the
PFF reference (vision.ball_truth) and the clip's ball candidates. scripts/ball_click.py
draws them and the person decides; nothing here writes a label file.

    python -m vision.ball_assist --status        # per clip: labeled, auto, spot check, flags
"""

import numpy as np
import polars as pl

SUGGEST_PX = {"pff": 40.0, "estimated": 60.0}
AUTO_CONF, AUTO_PX, AUTO_ALONE_PX = 0.5, 15.0, 40.0

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
