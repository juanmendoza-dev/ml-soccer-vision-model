"""Vision-like degradations of objects_10hz, for the sensitivity test (05, "Vision
sensitivity test").

Applied in memory when a match is loaded; data/processed is never rewritten. Every
random draw is a hash of (seed, arm, purpose, match, period, grid tenth[, object]), and
episodes and AR(1) noise only run forward in time. So a degraded row depends only on
rows at or before it, and on nothing else in the match.

Where the ball is (drift center, "near the ball") comes from the clean held VISIBLE
ball, never an ESTIMATED one: those use later frames (05, Leakage).

Degraded objects drop vx/vy and the *_att columns, which would be stale. Features
don't read them.
"""

import math
import zlib
from dataclasses import dataclass

import numpy as np
import polars as pl

DEFAULT_SEED = 20260928
# camera_drift's scale term (sigma / 20 per m from the ball) adds to its offset: on 8 PFF
# games the RMS error per axis of visible objects came out 1.25-1.32 x the offset sigma.
# The offset is severity / DRIFT_SPREAD, so the severity is the realized error.
DRIFT_SPREAD = 1.3
NEAR_BALL_M = 10.0  # team flips are twice as likely this close to the ball
PLAYER_TYPES = ("player", "goalkeeper")
DROPPED = ("vx", "vy", "x_att", "y_att", "vx_att", "vy_att")

# arm -> (severity unit, level at the detection review's targets). Arms always apply in
# this order, whatever order they're given in.
ARMS = {
    "camera_drift": ("sigma_m", 0.93),
    "player_noise": ("sigma_m", 0.93),
    "ball_noise": ("sigma_m", 1.0),
    "ball_false": ("share", 0.05),
    "no_ball_z": (None, None),
    "ball_miss": ("share", 0.10),
    "geom_loss": ("share", 0.17),
    "player_miss": ("share", 0.10),
    "team_flip": ("share", 0.05),
    "team_unknown": ("share", 0.05),
    "id_fragment": ("track_life_s", 2.0),
}
# everything at its target; drift and anchor noise split 0.93 m between them
PRESETS = {
    "target": [
        f"{a}:{0.66 if a in ('camera_drift', 'player_noise') else lvl:g}" if lvl is not None else a
        for a, (_, lvl) in ARMS.items()
    ]
}


def parse(specs) -> list[tuple[str, float | None]]:
    """['ball_miss:0.1', 'target', ...] -> [(arm, severity)] in ARMS order."""
    out = {}
    for spec in specs:
        for s in PRESETS.get(spec, [spec]):
            name, _, sev = s.partition(":")
            if name not in ARMS:
                raise ValueError(f"unknown degradation {name!r}; arms: {', '.join(ARMS)}")
            if name in out:
                raise ValueError(f"{name} given twice")
            unit = ARMS[name][0]
            if unit is None:
                if sev:
                    raise ValueError(f"{name} takes no severity")
                out[name] = None
                continue
            try:
                v = float(sev)
            except ValueError:
                raise ValueError(f"{s!r}: severity must be a number") from None
            ok = {"share": 0 <= v < 1, "sigma_m": v >= 0, "track_life_s": v > 0}[unit]
            if not ok or math.isnan(v):
                raise ValueError(f"{s!r}: bad severity for a {unit}")
            out[name] = v
    return [(a, out[a]) for a in ARMS if a in out]


def canonical(specs) -> list[str]:
    return [a if v is None else f"{a}:{v:g}" for a, v in parse(specs)]


# counter-based random numbers: splitmix64 over the keys
_C = [np.uint64(c) for c in (0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB)]


def _mix(h: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        h = h + _C[0]
        h = (h ^ (h >> np.uint64(30))) * _C[1]
        h = (h ^ (h >> np.uint64(27))) * _C[2]
        return h ^ (h >> np.uint64(31))


def code(s: str) -> int:
    return zlib.crc32(s.encode())


def uniform(*keys) -> np.ndarray:
    """Uniform on [0, 1), one per broadcast element of the integer keys."""
    h = np.zeros(np.broadcast(*[np.asarray(k) for k in keys]).shape, np.uint64)
    for key in keys:
        h = _mix(h ^ np.asarray(key).astype(np.uint64))
    return (h >> np.uint64(11)).astype(np.float64) * 2.0**-53


@dataclass
class Line:
    """One dense timeline: a period (or a period of one object) from its first grid row
    to its last, with a cell per tenth whether or not a row is there."""

    rows: np.ndarray  # row indices into the objects table
    pos: np.ndarray  # their cells
    ks: np.ndarray  # tenth of each cell
    period: int
    obj: int  # object code, 0 for frame-level lines


def lines(period, k, obj=None, mask=None) -> list[Line]:
    n = len(period)
    t = pl.DataFrame(
        {"i": np.arange(n), "p": period, "k": k, "o": np.zeros(n, np.int64) if obj is None else obj}
    )
    if mask is not None:
        t = t.filter(pl.Series(mask))
    out = []
    for p, o, i, kk in t.group_by("p", "o").agg("i", "k").sort("p", "o").iter_rows():
        kk = np.asarray(kk)
        k0 = kk.min()
        out.append(Line(np.asarray(i), kk - k0, np.arange(k0, kk.max() + 1), p, o))
    return out


def episodes(u_start, u_len, share: float, mean_rows: float):
    """Episodes starting with a fixed chance per cell, geometric lengths with mean
    mean_rows; the start chance makes the expected covered share `share`. Returns
    (active per cell, index of the latest episode start per cell, -1 before any)."""
    n = len(u_start)
    start = u_start < -math.log1p(-share) / mean_rows
    if mean_rows > 1:
        length = 1 + np.floor(np.log(1 - u_len) / math.log(1 - 1 / mean_rows))
    else:
        length = np.ones(n)
    idx = np.flatnonzero(start)
    end = np.minimum(idx + length[idx].astype(np.int64), n)
    d = np.zeros(n + 1, np.int64)
    np.add.at(d, idx, 1)
    np.add.at(d, end, -1)
    last = np.maximum.accumulate(np.where(start, np.arange(n), -1))
    return np.cumsum(d[:n]) > 0, last


def normal(u1, u2) -> np.ndarray:
    return np.sqrt(-2 * np.log(1 - u1)) * np.cos(2 * np.pi * u2)


def ar1(z: np.ndarray, sigma: float, tau_rows: float) -> np.ndarray:
    """Stationary AR(1) with std sigma and correlation time tau_rows, driven by z ~ N(0,1),
    started at its stationary spread. Causal: cell i uses z[: i + 1]."""
    from scipy.signal import lfilter

    a = math.exp(-1 / tau_rows)
    b = math.sqrt(1 - a * a)
    z = z.astype(np.float64).copy()
    z[:1] /= b
    return lfilter([sigma * b], [1.0, -a], z)


class Degrader:
    """Holds one match's objects as arrays while the arms run, plus per-arm stats
    (numerator, denominator) measured on the scored rows."""

    def __init__(self, objects: pl.DataFrame, match_id: str, seed: int, scored):
        self.df = objects.with_columns(k=(pl.col("t_s") * 10).round().cast(pl.Int64))
        g = self.df
        self.period = g["period"].to_numpy()
        self.k = g["k"].to_numpy()
        ids = g["object_id"].unique().to_list()
        self.obj = (
            g["object_id"]
            .replace_strict({i: code(i) for i in ids}, return_dtype=pl.Int64)
            .to_numpy()
        )
        self.x = g["x"].cast(pl.Float64).fill_null(np.nan).to_numpy().copy()
        self.y = g["y"].cast(pl.Float64).fill_null(np.nan).to_numpy().copy()
        self.z = g["z"].cast(pl.Float64).fill_null(np.nan).to_numpy().copy()
        self.visible = g["visible"].fill_null(False).to_numpy().copy()
        self.team = g["team"].to_numpy().astype(object)
        self.ball = (g["object_type"] == "ball").to_numpy()
        self.player = g["object_type"].is_in(PLAYER_TYPES).to_numpy()
        self.x0, self.y0 = self.x.copy(), self.y.copy()
        self.seed, self.match = seed, code(match_id)
        self.scored = (
            np.ones(g.height, bool)
            if scored is None
            else g.select("period", "k")
            .join(
                scored.select("period", k=(pl.col("t_s") * 10).round().cast(pl.Int64))
                .unique()
                .with_columns(s=pl.lit(True)),
                on=["period", "k"],
                how="left",
                maintain_order="left",
            )["s"]
            .fill_null(False)
            .to_numpy()
        )
        self.bx, self.by = self._held_ball()
        self.frame_lines = lines(self.period, self.k)
        self.player_lines = lines(self.period, self.k, self.obj, self.player)
        self.segment = np.zeros(g.height, np.int64)
        self.stats: dict[str, list] = {}

    def _held_ball(self):
        """The clean VISIBLE ball, carried forward within the period, on every row."""
        b = self.df.filter(pl.col("object_type") == "ball", pl.col("visible")).select(
            "period", "k", bx="x", by="y"
        )
        grid = (
            self.df.select("period", "k")
            .unique()
            .join(b, on=["period", "k"], how="left")
            .sort("period", "k")
            .with_columns(pl.col("bx", "by").forward_fill().over("period"))
        )
        h = self.df.select("period", "k").join(
            grid, on=["period", "k"], how="left", maintain_order="left"
        )
        return (h[c].cast(pl.Float64).fill_null(np.nan).to_numpy() for c in ("bx", "by"))

    def u(self, arm: str, purpose: str, line: Line) -> np.ndarray:
        return uniform(
            self.seed, code(f"{arm}/{purpose}"), self.match, line.period, line.ks, line.obj
        )

    def n(self, arm: str, purpose: str, line: Line) -> np.ndarray:
        return normal(self.u(arm, purpose + "1", line), self.u(arm, purpose + "2", line))

    def stat(self, arm: str, num: float, den: float):
        s = self.stats.setdefault(arm, [0.0, 0.0])
        s[0] += float(num)
        s[1] += float(den)

    def hide(self, arm: str, mask: np.ndarray, among: np.ndarray):
        """Set `mask` rows not visible; stat = share of visible scored `among` rows hit."""
        seen = self.visible & among & self.scored
        self.stat(arm, (seen & mask).sum(), seen.sum())
        self.visible &= ~mask

    def per_frame(self, fn) -> np.ndarray:
        """fn(line) -> value per cell, spread to every row of that frame."""
        out = np.zeros(len(self.k))
        for ln in self.frame_lines:
            out[ln.rows] = fn(ln)[ln.pos]
        return out

    def per_player(self, fn) -> np.ndarray:
        out = np.zeros(len(self.k))
        for ln in self.player_lines:
            out[ln.rows] = fn(ln)[ln.pos]
        return out

    def moved(self, arm: str, x_before, y_before, among):
        """stat = RMS displacement per axis over visible scored `among` rows."""
        m = self.visible & among & self.scored & ~np.isnan(self.x)
        d2 = (self.x - x_before)[m] ** 2 + (self.y - y_before)[m] ** 2
        self.stat(arm, d2.sum() / 2, m.sum())

    # the arms, in ARMS order

    def camera_drift(self, sigma):
        arm, tau = "camera_drift", 30
        sigma = sigma / DRIFT_SPREAD
        dx = self.per_frame(lambda ln: ar1(self.n(arm, "dx", ln), sigma, tau))
        dy = self.per_frame(lambda ln: ar1(self.n(arm, "dy", ln), sigma, tau))
        s = self.per_frame(lambda ln: ar1(self.n(arm, "s", ln), sigma / 20, tau))
        cx, cy = np.nan_to_num(self.bx), np.nan_to_num(self.by)
        x, y = self.x.copy(), self.y.copy()
        self.x = cx + (1 + s) * (x - cx) + dx
        self.y = cy + (1 + s) * (y - cy) + dy
        self.moved(arm, x, y, self.ball | self.player)

    def player_noise(self, sigma):
        arm = "player_noise"
        x, y = self.x.copy(), self.y.copy()
        m = self.player
        self.x[m] += self.per_player(lambda ln: ar1(self.n(arm, "x", ln), sigma, 5))[m]
        self.y[m] += self.per_player(lambda ln: ar1(self.n(arm, "y", ln), sigma, 5))[m]
        self.moved(arm, x, y, m)

    def ball_noise(self, sigma):
        arm = "ball_noise"
        x, y = self.x.copy(), self.y.copy()
        m = self.ball
        self.x[m] += self.per_frame(lambda ln: ar1(self.n(arm, "x", ln), sigma, 10))[m]
        self.y[m] += self.per_frame(lambda ln: ar1(self.n(arm, "y", ln), sigma, 10))[m]
        self.moved(arm, x, y, m)

    def ball_false(self, share):
        arm = "ball_false"
        ox, oy = np.zeros(len(self.k)), np.zeros(len(self.k))
        for ln in self.frame_lines:
            active, last = episodes(self.u(arm, "start", ln), self.u(arm, "len", ln), share, 5)
            # one offset per episode, keyed on its start
            src = Line(ln.rows, ln.pos, ln.ks[np.maximum(last, 0)], ln.period, 0)
            mag = 5 + 25 * self.u(arm, "mag", src)
            ang = 2 * np.pi * self.u(arm, "ang", src)
            ox[ln.rows] = np.where(active, mag * np.cos(ang), 0)[ln.pos]
            oy[ln.rows] = np.where(active, mag * np.sin(ang), 0)[ln.pos]
        m = self.ball & self.visible & (ox != 0)
        seen = self.ball & self.visible & self.scored
        self.stat(arm, (m & self.scored).sum(), seen.sum())
        self.x[m] += ox[m]
        self.y[m] += oy[m]

    def no_ball_z(self, _):
        m = self.ball & self.scored
        self.stat("no_ball_z", (m & ~np.isnan(self.z)).sum(), m.sum())
        self.z[self.ball] = np.nan

    def _frame_episodes(self, arm, share, mean_rows) -> np.ndarray:
        return self.per_frame(
            lambda ln: episodes(self.u(arm, "start", ln), self.u(arm, "len", ln), share, mean_rows)[
                0
            ]
        ).astype(bool)

    def _player_episodes(self, arm, purpose, share, mean_rows) -> np.ndarray:
        return self.per_player(
            lambda ln: episodes(
                self.u(arm, purpose + "start", ln),
                self.u(arm, purpose + "len", ln),
                share,
                mean_rows,
            )[0]
        ).astype(bool)

    def ball_miss(self, share):
        self.hide("ball_miss", self.ball & self._frame_episodes("ball_miss", share, 20), self.ball)

    def geom_loss(self, share):
        self.hide(
            "geom_loss", self._frame_episodes("geom_loss", share, 10), self.ball | self.player
        )

    def player_miss(self, share):
        arm = "player_miss"
        self.hide(arm, self.player & self._player_episodes(arm, "", share, 10), self.player)

    def _team_errors(self, arm, share) -> np.ndarray:
        a = self._player_episodes(arm, "a", share, 30)
        b = self._player_episodes(arm, "b", share, 30)
        near = np.hypot(self.x0 - self.bx, self.y0 - self.by) <= NEAR_BALL_M  # NaN: False
        known = np.array([t is not None for t in self.team])
        m = self.player & self.visible & known & (a | (b & near))
        seen = self.player & self.visible & self.scored & known
        self.stat(arm, (m & self.scored).sum(), seen.sum())
        return m

    def team_flip(self, share):
        m = self._team_errors("team_flip", share)
        self.team[m] = np.where(self.team[m] == "home", "away", "home")

    def team_unknown(self, share):
        self.team[self._team_errors("team_unknown", share)] = None

    def id_fragment(self, life_s):
        arm = "id_fragment"
        rate = 1 / (10 * life_s)  # breaks per grid row
        self.segment = self.per_player(
            lambda ln: np.cumsum(self.u(arm, "break", ln) < rate)
        ).astype(np.int64)
        m = self.player
        tracks = (
            pl.DataFrame({"p": self.period[m], "o": self.obj[m], "s": self.segment[m]})
            .unique()
            .height
        )
        self.stat(arm, m.sum() / 10, tracks)

    def result(self) -> pl.DataFrame:
        ids = self.df["object_id"]
        seg = pl.Series(self.segment)
        return self.df.drop("k", *[c for c in DROPPED if c in self.df.columns]).with_columns(
            x=pl.Series(self.x).fill_nan(None),
            y=pl.Series(self.y).fill_nan(None),
            z=pl.Series(self.z).fill_nan(None),
            visible=pl.Series(self.visible),
            interpolated=pl.col("interpolated") | ~pl.Series(self.visible),
            team=pl.Series(self.team.tolist(), dtype=pl.String),
            object_id=pl.when(seg > 0).then(ids + "~" + seg.cast(pl.String)).otherwise(ids),
        )


def apply(
    objects: pl.DataFrame, match_id: str, specs, seed: int = DEFAULT_SEED, scored=None
) -> tuple[pl.DataFrame, dict]:
    """Degrade one match's objects_10hz rows. `scored` (period, t_s rows, optional)
    limits the stats to those frames. Returns (objects, {arm: [num, den]})."""
    d = Degrader(objects, match_id, seed, scored)
    for arm, sev in parse(specs):
        getattr(d, arm)(sev)
    return d.result(), d.stats


def summarize(stats: dict) -> dict:
    """Realized severity per arm, in the arm's unit, from summed (num, den)."""
    out = {}
    for arm, (num, den) in stats.items():
        unit = ARMS[arm][0] or "share"
        v = num / den if den else float("nan")
        out[arm] = {unit: round(math.sqrt(v) if unit == "sigma_m" else v, 4)}
    return out
