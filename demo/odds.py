"""08 market odds panel: each team's Polymarket win price at a frame's wall-clock time.

    python -m demo.odds --fetch 10517            # prices-history -> data/odds/10517.json
    python -m demo.odds --check 10517            # each PFF goal's mapped UTC, price either side
    python -m demo.odds --preview 10517 --at 2:2158.6 --out odds.png [--image frame.png]

Markets, colors and period starts come from data/splits/odds_markets.json (08 "Market odds
panel"). The points are the market price about once a minute, not trades. Display only:
prediction never reads any of this.
"""

import argparse
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import polars as pl

from demo import overlay as ov

MANIFEST = Path("data/splits/odds_markets.json")
CACHE = Path("data/odds")
HISTORY_URL = (
    "https://clob.polymarket.com/prices-history?market={token}&startTs={a}&endTs={b}&fidelity=1"
)
BEFORE_S = 2 * 3600  # fetched ahead of kickoff, so the pre-match price is there
AFTER_S = 3600
LEAD_S = 300  # the chart starts this long before kickoff
COUNT_S = 0.5  # a moved price counts to its new value over this long
PULSE_S = 1.5
CHIP_S = 3.0
CARD_W, CARD_H = 600, 300
MARGIN = 16


def utc_s(s: str) -> float:
    return datetime.fromisoformat(s).timestamp()


def hhmmss(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%H:%M:%S")


def hex_bgr(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


def load_manifest(path: Path = MANIFEST) -> dict[str, dict]:
    return {m["match_id"]: m for m in json.loads(path.read_text())["matches"]}


def fetch(entry: dict) -> dict:
    """Both tokens' prices-history over the match, BEFORE_S ahead to AFTER_S past it."""
    first, last = entry["periods"][0], entry["periods"][-1]
    a = int(utc_s(first["start_utc"]) - BEFORE_S)
    b = int(utc_s(last["start_utc"]) + last["length_s"] + AFTER_S)
    out = {"fetched_utc": datetime.now(UTC).isoformat(), "start_ts": a, "end_ts": b}
    for team, token in entry["tokens"].items():
        url = HISTORY_URL.format(token=token, a=a, b=b)
        # Cloudflare turns urllib's default user agent away with a 403
        req = urllib.request.Request(url, headers={"user-agent": "Mozilla/5.0"})
        out[team] = json.load(urllib.request.urlopen(req, timeout=30))["history"]
        if not out[team]:
            raise SystemExit(f"{entry['match_id']}: no price history for {team}")
    return out


def load_history(match_id: str, cache: Path = CACHE) -> dict:
    path = cache / f"{match_id}.json"
    if not path.exists():
        raise SystemExit(f"no {path}: run python -m demo.odds --fetch {match_id}")
    return json.loads(path.read_text())


def goals_utc(gamestate: Path, odds: "Odds") -> list[tuple[float, str, str | None]]:
    """The match's PFF goals as (UTC, team, set_piece), in periods the manifest has."""
    times = pl.read_parquet(
        gamestate / "frames.parquet", columns=["frame_id", "period", "timestamp_s"]
    )
    goals = (
        pl.read_parquet(gamestate / "events.parquet")
        .filter(pl.col("event_type") == "goal")
        .join(times, on="frame_id")
        .sort("frame_id")
    )
    return [
        (odds.utc(g["period"], g["timestamp_s"]), g["team"], g["set_piece"])
        for g in goals.iter_rows(named=True)
        if g["period"] in odds.starts
    ]


def ease(f: float) -> float:
    """Ease-out cubic on 0-1."""
    f = min(max(f, 0.0), 1.0)
    return 1 - (1 - f) ** 3


class Odds:
    """One match's two price series, looked up causally by wall-clock time: a moment at
    UTC u sees each team's latest point at or before u, never a later one (08)."""

    def __init__(self, entry: dict, history: dict):
        self.entry = entry
        self.starts = {p["period"]: utc_s(p["start_utc"]) for p in entry["periods"]}
        self.ends = {p["period"]: utc_s(p["start_utc"]) + p["length_s"] for p in entry["periods"]}
        self.series = {}
        for team in ("home", "away"):
            h = sorted(history[team], key=lambda r: r["t"])
            t = np.array([r["t"] for r in h], dtype=float)
            p = np.array([r["p"] for r in h], dtype=float)
            moves = (
                np.flatnonzero(np.diff(p) != 0) + 1
            )  # indices whose price differs from the one before
            self.series[team] = (t, p, moves)

    def utc(self, period: int, t: float) -> float:
        return self.starts[period] + t

    def index(self, team: str, u: float) -> int:
        """Index of the team's latest point at or before u, -1 before the first."""
        return int(np.searchsorted(self.series[team][0], u, side="right")) - 1

    def price(self, team: str, u: float) -> float | None:
        i = self.index(team, u)
        return None if i < 0 else float(self.series[team][1][i])

    def last_move(self, team: str, u: float) -> tuple[float, float, float] | None:
        """(time, old, new) of the team's latest price change at or before u."""
        t, p, moves = self.series[team]
        k = int(np.searchsorted(t[moves], u, side="right")) - 1
        if k < 0:
            return None
        i = moves[k]
        return float(t[i]), float(p[i - 1]), float(p[i])

    def shown(self, team: str, u: float) -> tuple[float | None, int | None, float | None]:
        """The number on screen (counting toward a just-moved price), the delta chip in
        points while it's up, and the pulse's progress 0-1 while it runs."""
        p = self.price(team, u)
        m = self.last_move(team, u)
        if p is None or m is None:
            return p, None, None
        age, old, new = u - m[0], m[1], m[2]
        value = old + (new - old) * ease(age / COUNT_S) if age < COUNT_S else p
        delta = round(100 * new) - round(100 * old)
        chip = delta if age < CHIP_S and delta != 0 else None
        pulse = age / PULSE_S if age < PULSE_S else None
        return value, chip, pulse

    def view(self, period: int, t: float, goals: list[tuple[float, str, str | None]] = ()) -> dict:
        """Everything overlay.odds_card draws for the moment (period, t). x positions are
        fractions of the chart's width, from kickoff - LEAD_S to the end of the current
        period (a later period is never on the axis before it starts)."""
        e = self.entry
        now = self.utc(period, t)
        x0 = self.starts[1] - LEAD_S
        x1 = self.ends.get(period, max(self.ends.values()))
        fx = lambda u: (u - x0) / (x1 - x0)
        colors = {k: hex_bgr(v) for k, v in e["colors"].items()}
        teams, lines = [], []
        for team in ("home", "away"):
            ts, ps, _ = self.series[team]
            i0 = max(self.index(team, x0), 0)
            i1 = self.index(team, now)
            pts = [(fx(max(ts[i], x0)), float(ps[i])) for i in range(i0, i1 + 1)]
            if pts:
                pts.append((fx(now), pts[-1][1]))
            lines.append((pts, colors[team]))
            value, chip, pulse = self.shown(team, now)
            teams.append(
                {
                    "name": e["short"][team],
                    "color": colors[team],
                    "price": value,
                    "delta": chip,
                    "pulse": pulse,
                }
            )
        ticks = [(fx(self.starts[1]), "0'")]
        ht = None
        if period >= 2 and 2 in self.starts:
            ht = (fx(self.ends[1]), fx(self.starts[2]))
            ticks.append(((ht[0] + ht[1]) / 2, "HT"))
            ticks.append((1.0, "90'"))
        else:
            ticks.append((1.0, "45'"))
        return {
            "title": "POLYMARKET",
            "subtitle": f"{e['label']} - win price",
            "teams": teams,
            "lines": lines,
            "now": fx(now),
            "ht": ht,
            "ticks": ticks,
            "goals": [(fx(g), colors.get(team, ov.GOLD)) for g, team, _ in goals if g <= now],
            "clock": hhmmss(now) + " UTC",
        }


def card_box(w: int, h: int, bottom: int) -> tuple[int, int, int, int]:
    """The card's (x, y, w, h): bottom right, `bottom` px above the frame's bottom edge."""
    return w - MARGIN - CARD_W, h - bottom - MARGIN - CARD_H, CARD_W, CARD_H


def check(odds: Odds, goals: list[tuple[float, str, str | None]]) -> None:
    """Each goal's mapped UTC with the scoring team's last price before it and the first
    after (08 "Check")."""
    for u, team, set_piece in goals:
        ts, ps, _ = odds.series[team]
        i = odds.index(team, u)
        before = f"{hhmmss(ts[i])} {ps[i]:.2f}" if i >= 0 else "--"
        after = f"{hhmmss(ts[i + 1])} {ps[i + 1]:.2f}" if i + 1 < len(ts) else "--"
        moved = next((k for k in range(i + 1, len(ts)) if ps[k] != ps[max(i, 0)]), None)
        first = (
            f"{hhmmss(ts[moved])} {ps[moved]:.2f} (+{ts[moved] - u:.0f} s)"
            if moved is not None
            else "--"
        )
        print(
            f"{odds.entry['short'][team]} goal ({set_piece}) at {hhmmss(u)}: "
            f"before {before}, next point {after}, first move {first}"
        )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="demo.odds")
    ap.add_argument("match_id")
    do = ap.add_mutually_exclusive_group(required=True)
    do.add_argument("--fetch", action="store_true", help="download the price history into --cache")
    do.add_argument(
        "--check", action="store_true", help="each goal's mapped time and price either side"
    )
    do.add_argument("--preview", action="store_true", help="the card at --at as a still")
    ap.add_argument("--at", help="period:timestamp_s for --preview, e.g. 2:2158.6")
    ap.add_argument(
        "--image", type=Path, help="background for --preview (default a dark 1920x1080)"
    )
    ap.add_argument("--out", type=Path, help="--preview output png")
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--gamestate", type=Path, default=Path("data/gamestate"))
    args = ap.parse_args(argv)

    entry = load_manifest(args.manifest)[args.match_id]
    if args.fetch:
        hist = fetch(entry)
        args.cache.mkdir(parents=True, exist_ok=True)
        path = args.cache / f"{args.match_id}.json"
        path.write_text(json.dumps(hist))
        print(f"wrote {path} ({len(hist['home'])} home, {len(hist['away'])} away points)")
        return
    odds = Odds(entry, load_history(args.match_id, args.cache))
    goals = goals_utc(args.gamestate / args.match_id, odds)
    if args.check:
        check(odds, goals)
        return
    if not (args.at and args.out):
        raise SystemExit("--preview needs --at and --out")
    period, t = args.at.split(":")
    img = cv2.imread(str(args.image)) if args.image else np.full((1080, 1920, 3), 40, np.uint8)
    h, w = img.shape[:2]
    ov.odds_card(img, card_box(w, h, 50), odds.view(int(period), float(t), goals))
    cv2.imwrite(str(args.out), img)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
