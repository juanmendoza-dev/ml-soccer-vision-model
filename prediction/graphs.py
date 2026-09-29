"""Graphs on the 10 Hz grid for the frame GNN (05 model 2).

One graph per grid row: the VISIBLE players and goalkeepers on the row plus the held
VISIBLE ball, in the attacking frame. The inputs and the rotation are the hand
features' (prediction/features.py): PFF's ESTIMATED positions never reach a node,
velocities only come from two VISIBLE sightings 0.5 s apart, and everything is rotated
with the row's own flip.

A match's graphs are one padded float16 array (rows, MAX_NODES, len(NODE_FEATURES)),
nodes packed at the front with the ball first, plus the node count per row. Edge
features are left to the model, which computes them from the node features.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from prediction.features import (
    PLAYER_TYPES,
    VEL_ROWS,
    ball_track,
    fresh_velocity,
    goal_angle,
    goal_distance,
    grid_rows,
    tenths,
)

# cached graphs are keyed by version: add a new one when any node definition changes
GRAPHS_VERSION = "1"
MAX_PLAYERS = 22  # PFF never has more visible on a row; past it, the nearest the ball stay
MAX_NODES = MAX_PLAYERS + 1
NODE_FEATURES = (
    "x",
    "y",
    "vx",
    "vy",
    "has_vel",
    "is_ball",
    "is_att",
    "is_def",
    "is_gk",
    "goal_dist",
    "goal_angle",
    "ball_dist",
    "has_ball",
    "ball_age_s",
    "ball_z",
)
IDX = {f: i for i, f in enumerate(NODE_FEATURES)}
X_SCALE, Y_SCALE = 52.5, 34.0
VEL_SCALE = 10.0
DIST_SCALE = 52.5
Z_SCALE = 3.0


def _put(out: np.ndarray, **cols) -> np.ndarray:
    for name, a in cols.items():
        out[:, IDX[name]] = a
    return out


def match_graphs(
    frames: pl.DataFrame, objects: pl.LazyFrame, ball_source: str = "held"
) -> dict[str, np.ndarray]:
    """frames (period, t_s, possession_team, flipped) and objects_10hz for one match.
    Returns nodes (rows, MAX_NODES, F) float16, n (rows,) uint8, and the grid keys
    period / k the rows are in (sorted, as load_match sorts them), plus capped: rows
    with more than MAX_PLAYERS visible players."""
    g = grid_rows(frames)
    t = g.height
    sign, seg = g["sign"].to_numpy(), g["seg"].to_numpy()
    nodes = np.zeros((t, MAX_NODES, len(NODE_FEATURES)), np.float32)

    # the ball: held VISIBLE ball, velocity only between two fresh sightings (features.py)
    ball = ball_track(g, objects, ball_source)
    bx, by = sign * ball["x"], sign * ball["y"]
    bvx, bvy = fresh_velocity(ball, seg)
    bvx, bvy = sign * bvx, sign * bvy
    has_ball = ~np.isnan(bx)
    b = np.flatnonzero(has_ball)
    b_vel = ~np.isnan(bvx[b])
    nodes[b, 0] = _put(
        np.zeros((len(b), len(NODE_FEATURES)), np.float32),
        x=bx[b] / X_SCALE,
        y=by[b] / Y_SCALE,
        vx=np.where(b_vel, bvx[b], 0.0) / VEL_SCALE,
        vy=np.where(b_vel, bvy[b], 0.0) / VEL_SCALE,
        has_vel=b_vel,
        is_ball=1.0,
        goal_dist=goal_distance(bx[b], by[b]) / DIST_SCALE,
        goal_angle=goal_angle(bx[b], by[b]) / np.pi,
        has_ball=1.0,
        ball_age_s=ball["age_s"][b],
        ball_z=np.nan_to_num(ball["z"][b]) / Z_SCALE,
    )

    # players: VISIBLE only, velocity from the same object 0.5 s back in the same seg
    rows = g.select("period", "k", "r", "seg", "sign", "possession_team")
    p = (
        objects.filter(pl.col("object_type").is_in(PLAYER_TYPES), pl.col("visible"))
        .select("period", "object_id", "object_type", "team", "x", "y", k=tenths())
        .collect()
        .join(rows, on=["period", "k"])
    )
    earlier = p.select("object_id", "seg", "x", "y", r=pl.col("r") + VEL_ROWS)
    p = p.join(earlier, on=["object_id", "r", "seg"], how="left", suffix="_0").with_columns(
        att=pl.col("team").eq(pl.col("possession_team")).fill_null(False),
        dfn=pl.col("team").ne(pl.col("possession_team")).fill_null(False),
        gk=pl.col("object_type") == "goalkeeper",
    )
    r = p["r"].to_numpy()
    s = p["sign"].to_numpy()
    x, y = p["x"].to_numpy(), p["y"].to_numpy()
    x0 = p["x_0"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    y0 = p["y_0"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    xa, ya = s * x, s * y
    dt = VEL_ROWS / 10
    vxa, vya = s * (x - x0) / dt, s * (y - y0) / dt
    vel = ~np.isnan(vxa)
    gd = goal_distance(xa, ya)
    d = np.hypot(xa - bx[r], ya - by[r])  # NaN without a ball

    # order in a row: nearest the ball first (no ball: nearest the goal); ties by position
    order = np.lexsort((ya, xa, gd, np.where(np.isnan(d), np.inf, d), r))
    r_sorted = r[order]
    first = np.r_[True, r_sorted[1:] != r_sorted[:-1]]
    starts = np.flatnonzero(first)
    rank = np.arange(len(order)) - np.repeat(starts, np.diff(np.r_[starts, len(order)]))
    keep = rank < MAX_PLAYERS
    order, rank = order[keep], rank[keep]
    rr = r[order]
    feats = _put(
        np.zeros((len(order), len(NODE_FEATURES)), np.float32),
        x=xa[order] / X_SCALE,
        y=ya[order] / Y_SCALE,
        vx=np.where(vel, vxa, 0.0)[order] / VEL_SCALE,
        vy=np.where(vel, vya, 0.0)[order] / VEL_SCALE,
        has_vel=vel[order],
        is_att=p["att"].to_numpy()[order],
        is_def=p["dfn"].to_numpy()[order],
        is_gk=p["gk"].to_numpy()[order],
        goal_dist=gd[order] / DIST_SCALE,
        goal_angle=goal_angle(xa[order], ya[order]) / np.pi,
        ball_dist=np.nan_to_num(d[order]) / DIST_SCALE,
        has_ball=has_ball[rr],
    )
    nodes[rr, rank + has_ball[rr]] = feats
    count = np.bincount(r, minlength=t)
    return {
        "nodes": nodes.astype(np.float16),
        "n": (has_ball + np.minimum(count, MAX_PLAYERS)).astype(np.uint8),
        "period": g["period"].to_numpy(),
        "k": g["k"].to_numpy(),
        "capped": np.array(int((count > MAX_PLAYERS).sum())),
    }


def cache_path(match_dir: Path, ball_source: str = "held") -> Path:
    return match_dir / f"graphs_v{GRAPHS_VERSION}_{ball_source}.npz"


def load_graphs(
    match_id: str, processed_dir: Path, ball_source: str = "held", cache: bool = True
) -> tuple[dict[str, np.ndarray], bool]:
    """One match's graphs, aligned with its frames_10hz rows sorted by (period, t_s).
    Cached next to the feature caches under their own name, so features_v* are never
    read or written. Returns (graphs, built): built is False when the cache was used."""
    d = processed_dir / match_id
    path = cache_path(d, ball_source)
    frames = pl.read_parquet(
        d / "frames_10hz.parquet", columns=["period", "t_s", "possession_team", "flipped"]
    ).sort("period", "t_s")
    keys = frames.select("period", k=tenths())
    if cache and path.exists():
        with np.load(path) as z:
            out = {name: z[name] for name in z.files}
        if not (
            np.array_equal(out["period"], keys["period"].to_numpy())
            and np.array_equal(out["k"], keys["k"].to_numpy())
        ):
            raise ValueError(f"{match_id}: {path.name} doesn't match frames_10hz; delete it")
        return out, False
    out = match_graphs(frames, pl.scan_parquet(d / "objects_10hz.parquet"), ball_source)
    if cache:
        np.savez_compressed(path, **out)
    return out, True


@dataclass
class GraphStore:
    """Graphs of many matches in one array, row i matching row i of prediction.cv's data
    (load_data concatenates the matches in the same order). check() makes sure a frame
    of that data still lines up before anything is read."""

    ids: list[str]
    nodes: np.ndarray  # (rows, MAX_NODES, F) float16
    n: np.ndarray  # (rows,) uint8
    match: np.ndarray  # (rows,) index into ids
    period: np.ndarray
    k: np.ndarray
    built: int = 0  # matches whose cache had to be built
    capped: int = 0  # rows with more than MAX_PLAYERS visible players

    @classmethod
    def empty(cls, ids: list[str], total: int) -> "GraphStore":
        return cls(
            ids=list(ids),
            nodes=np.zeros((total, MAX_NODES, len(NODE_FEATURES)), np.float16),
            n=np.zeros(total, np.uint8),
            match=np.zeros(total, np.int32),
            period=np.zeros(total, np.int64),
            k=np.zeros(total, np.int64),
        )

    def put(self, j: int, at: int, g: dict) -> int:
        """Match j's graphs at row `at`; returns the next free row."""
        m = len(g["n"])
        self.nodes[at : at + m] = g["nodes"]
        self.n[at : at + m] = g["n"]
        self.match[at : at + m] = j
        self.period[at : at + m] = g["period"]
        self.k[at : at + m] = g["k"]
        self.capped += int(g["capped"])
        return at + m

    @classmethod
    def from_parts(cls, parts: list[tuple[str, dict]]) -> "GraphStore":
        store = cls.empty([i for i, _ in parts], sum(len(g["n"]) for _, g in parts))
        at = 0
        for j, (_, g) in enumerate(parts):
            at = store.put(j, at, g)
        return store

    @classmethod
    def load(
        cls, ids: list[str], processed_dir: Path, ball_source: str = "held", log=None
    ) -> "GraphStore":
        """Match by match into one preallocated array, so peak memory is the store plus
        one match."""
        total = sum(
            pl.scan_parquet(processed_dir / i / "frames_10hz.parquet")
            .select(pl.len())
            .collect()
            .item()
            for i in ids
        )
        store = cls.empty(ids, total)
        at = 0
        for j, i in enumerate(ids):
            g, built = load_graphs(i, processed_dir, ball_source)
            at = store.put(j, at, g)
            store.built += built
            if log and built:
                log(f"built graphs for {i} ({j + 1}/{len(ids)})")
        if at != total:
            raise ValueError(f"{at} graph rows for {total} grid rows")
        return store

    def check(self, df: pl.DataFrame) -> np.ndarray:
        """df's `row` column as store indices, after checking every row is the same
        (match_id, period, tenth) here."""
        rows = df["row"].to_numpy()
        if len(rows) and (rows.min() < 0 or rows.max() >= len(self.n)):
            raise ValueError("rows outside the graph store")
        match = (
            df["match_id"]
            .replace_strict(self.ids, list(range(len(self.ids))), default=-1, return_dtype=pl.Int32)
            .to_numpy()
        )
        k = df.select(tenths())["t_s"].to_numpy()
        if not (
            np.array_equal(self.match[rows], match)
            and np.array_equal(self.period[rows], df["period"].to_numpy())
            and np.array_equal(self.k[rows], k)
        ):
            raise ValueError("data rows don't line up with the graph store")
        return rows

    def take(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.nodes[rows], self.n[rows]
