"""Stage 8 (vision.state) on dataset tracking against the provider's values (03, 07 #6).

    python -m vision.state_check [--games 10502 ...] [--gamestate DIR] [--out FILE]

Per match and pooled over the provider's frames:
- possession: agreement where the provider has a team, on all frames and on alive ones
  (an inferred null counts as a disagreement), and on alive frames with a confirmed
  carrier (does the carrier's team match, when there is one)
- provider dead frames labeled dead or null (03 target >= 90%), and labeled dead
- provider alive frames labeled alive (no target, but it keeps "everything is dead" honest)
- possession changes per match, inferred vs provider, and the null shares

Matches whose provider has no possession or ball state (Metrica CSV) are skipped.
"""

import argparse
import sys
from pathlib import Path

import polars as pl

from vision.state import StateConfig, infer, rule_only

GAMESTATE_DIR = Path("data/gamestate")
OBJECT_COLS = ["frame_id", "object_id", "object_type", "team", "x", "y", "z", "visible"]
COUNTS = (
    "frames",
    "poss_n",
    "poss_agree",
    "alive_n",
    "alive_poss_agree",
    "carrier_n",
    "carrier_agree",
    "dead_n",
    "dead_ok",
    "dead_dead",
    "alive_ok",
    "poss_null",
    "state_null",
    "changes_provider",
    "changes_inferred",
)


def changes(s: pl.Series) -> int:
    """Switches from one team to the other (nulls in between don't count as a switch)."""
    v = s.drop_nulls()
    return int((v != v.shift(1)).sum()) if len(v) else 0


def check_match(match_dir: Path, config: StateConfig | None = None) -> dict | None:
    rule_only(config or StateConfig())
    frames = pl.read_parquet(match_dir / "frames.parquet")
    if frames["possession_team"].is_null().all() and frames["ball_state"].is_null().all():
        return None
    objects = pl.scan_parquet(match_dir / "objects.parquet").select(OBJECT_COLS)
    inferred = infer(frames, objects, config)
    teams = objects.select("object_id", carrier_team="team").unique("object_id").collect()
    j = (
        frames.select(
            "frame_id", "period", "timestamp_s", p_state="ball_state", p_poss="possession_team"
        )
        .join(inferred, on="frame_id")
        .join(teams, left_on="ball_carrier_id", right_on="object_id", how="left")
        .sort("period", "timestamp_s")
    )
    agree = (j["p_poss"] == j["possession_team"]).fill_null(False)
    has = j["p_poss"].is_not_null()
    alive, dead = (
        (j["p_state"] == "alive").fill_null(False),
        (j["p_state"] == "dead").fill_null(False),
    )
    carrier = alive & j["carrier_team"].is_not_null() & has
    changes_p = sum(changes(g["p_poss"]) for _, g in j.group_by("period", maintain_order=True))
    changes_i = sum(
        changes(g["possession_team"]) for _, g in j.group_by("period", maintain_order=True)
    )
    return {
        "match_id": match_dir.name,
        "frames": j.height,
        "poss_n": int(has.sum()),
        "poss_agree": int((agree & has).sum()),
        "alive_n": int((alive & has).sum()),
        "alive_poss_agree": int((agree & alive & has).sum()),
        "carrier_n": int(carrier.sum()),
        "carrier_agree": int((carrier & (j["carrier_team"] == j["p_poss"]).fill_null(False)).sum()),
        "dead_n": int(dead.sum()),
        "dead_ok": int((dead & (j["ball_state"] != "alive").fill_null(True)).sum()),
        "dead_dead": int((dead & (j["ball_state"] == "dead").fill_null(False)).sum()),
        "alive_ok": int((alive & (j["ball_state"] == "alive").fill_null(False)).sum()),
        "poss_null": int(j["possession_team"].is_null().sum()),
        "state_null": int(j["ball_state"].is_null().sum()),
        "changes_provider": changes_p,
        "changes_inferred": changes_i,
    }


def shares(c: dict) -> dict:
    div = lambda a, b: round(c[a] / c[b], 4) if c[b] else None
    return {
        "possession agrees": div("poss_agree", "poss_n"),
        "on alive frames": div("alive_poss_agree", "alive_n"),
        "carrier's team agrees (alive)": div("carrier_agree", "carrier_n"),
        "alive frames with a carrier": div("carrier_n", "alive_n"),
        "dead -> dead or null": div("dead_ok", "dead_n"),
        "dead -> dead": div("dead_dead", "dead_n"),
        "alive -> alive": div("alive_ok", "alive_n"),
        "possession null": div("poss_null", "frames"),
        "ball state null": div("state_null", "frames"),
    }


def report(rows: list[dict], config: StateConfig) -> str:
    pooled = {k: sum(r[k] for r in rows) for k in COUNTS}
    lines = [
        f"# Stage 8 vs provider ({len(rows)} matches)",
        "",
        f"Config: `{config.to_dict()}`",
        "",
        "| pooled | share |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in shares(pooled).items()],
        (
            f"| possession changes (provider / inferred) | {pooled['changes_provider']} / "
            f"{pooled['changes_inferred']} |"
        ),
        "",
        "| match | possession | alive | carrier | dead ok | alive ok | changes p / i |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        s = shares(r)
        carrier = s["carrier's team agrees (alive)"]
        lines.append(
            f"| {r['match_id']} | {s['possession agrees']} | {s['on alive frames']} | "
            f"{carrier} | {s['dead -> dead or null']} | "
            f"{s['alive -> alive']} | {r['changes_provider']} / {r['changes_inferred']} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", nargs="+")
    ap.add_argument("--gamestate", type=Path, default=GAMESTATE_DIR)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    ids = args.games or sorted(p.parent.name for p in args.gamestate.glob("*/frames.parquet"))
    config = StateConfig()
    rows = []
    for i in ids:
        r = check_match(args.gamestate / i, config)
        if r is None:
            print(f"{i}: no provider possession or ball state, skipped", file=sys.stderr)
            continue
        rows.append(r)
    if not rows:
        print("no matches with provider values", file=sys.stderr)
        return 1
    text = report(rows, config)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
