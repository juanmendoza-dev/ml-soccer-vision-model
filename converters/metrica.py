"""Metrica sample games 1-2 (CSV format) -> game state (02).

    python -m converters.metrica [--games 1 2] [--raw DIR] [--out DIR]

Tracking is read with kloppy; the event CSV has no kloppy loader and is parsed
here. Metrica has no ball status, possession or carrier, so those are null (02).
"""

import argparse
import re
import sys
from pathlib import Path

import polars as pl
from kloppy import metrica

from converters.common import ConversionReport, causal_velocities, write_gamestate
from gamestate.schema import PITCH_LENGTH, PITCH_WIDTH, SCHEMA_VERSION

RAW_DIR = Path("data/raw/metrica/repo/data")
OUT_DIR = Path("data/gamestate")
FPS = 25.0
MAX_PLAYER_SPEED = 12.0  # m/s, only used for the report check

SET_PIECES = {
    "FREE KICK": "free_kick",
    "CORNER KICK": "corner",
    "PENALTY": "penalty",
    "THROW IN": "throw_in",
    "KICK OFF": "kickoff",
    "GOAL KICK": "goal_kick",
}


def shot_outcome(subtype: str | None) -> str:
    s = subtype or ""
    if "GOAL" in s:
        return "goal"
    if "BLOCKED" in s:
        return "blocked"
    if "SAVED" in s:
        return "saved"
    if "WOODWORK" in s:
        return "woodwork"
    if "OFF TARGET" in s or "OUT" in s:
        return "off_target"
    return "unknown"


def player_ref(team: str, name: str | None) -> str | None:
    """'Player 26' / 'Player26' on team Away -> kloppy's 'away_26'."""
    if name is None:
        return None
    m = re.search(r"\d+", name)
    return f"{team.lower()}_{m.group()}" if m else None


def to_meters(x: pl.Expr, y: pl.Expr, y_down: bool) -> tuple[pl.Expr, pl.Expr]:
    """Metrica 0-1 coordinates -> meters from the center spot, +y up (02)."""
    ym = (0.5 - y) if y_down else (y - 0.5)
    return (x - 0.5) * PITCH_LENGTH, ym * PITCH_WIDTH


def load_tracking(home_csv: Path, away_csv: Path) -> tuple[pl.DataFrame, list]:
    ds = metrica.load_tracking_csv(str(home_csv), str(away_csv))
    df = ds.to_df(engine="polars").with_columns(
        timestamp_s=pl.col("timestamp").dt.total_microseconds() / 1e6
    )
    return df, ds.metadata.teams


def long_objects(df: pl.DataFrame, teams: list, report: ConversionReport) -> pl.DataFrame:
    """Wide kloppy frame table -> one row per object per frame (kloppy y is already up)."""
    parts = []
    refs = [("ball", "ball", None)] + [
        (p.player_id, "player", t.ground.value) for t in teams for p in t.players
    ]
    for ref, kind, team in refs:
        if f"{ref}_x" not in df.columns:
            continue
        x, y = to_meters(pl.col(f"{ref}_x"), pl.col(f"{ref}_y"), y_down=False)
        if kind == "ball":
            missing = df.filter(pl.col("ball_x").is_null() | pl.col("ball_y").is_null()).height
            report.drop("objects", missing, "frames where the ball has no position (not tracked)")
        parts.append(
            df.select(
                "frame_id",
                object_id=pl.lit(ref),
                object_type=pl.lit(kind),
                team=pl.lit(team, pl.String),
                x=x,
                y=y,
            ).filter(pl.col("x").is_not_null() & pl.col("y").is_not_null())
        )
    # Null player positions are players not on the pitch (before coming on / after
    # going off); those rows don't exist rather than being dropped.
    return pl.concat(parts)


def mark_goalkeepers(objects: pl.DataFrame, frames: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Metrica doesn't flag keepers. Per team, the first-frame starter who stays
    deepest toward one end over period 1 is the keeper."""
    first = frames["frame_id"].min()
    starters = objects.filter((pl.col("frame_id") == first) & (pl.col("object_type") == "player"))
    p1 = objects.join(frames.filter(pl.col("period") == 1).select("frame_id"), on="frame_id")
    mean_x = (
        p1.join(starters.select("object_id"), on="object_id")
        .group_by("object_id", "team")
        .agg(pl.col("x").mean())
    )
    keepers = {}
    for team in ("home", "away"):
        t = mean_x.filter(pl.col("team") == team)
        keepers[team] = t.sort(pl.col("x").abs(), descending=True)["object_id"][0]
    objects = objects.with_columns(
        object_type=pl.when(pl.col("object_id").is_in(list(keepers.values())))
        .then(pl.lit("goalkeeper"))
        .otherwise("object_type")
    )
    return objects, keepers


def attack_direction(objects: pl.DataFrame, frames: pl.DataFrame, home_gk: str) -> dict:
    """Mean x of the home keeper per period; home attacks +x where it's negative."""
    gk = (
        objects.filter(pl.col("object_id") == home_gk)
        .join(frames.select("frame_id", "period"), on="frame_id")
        .group_by("period")
        .agg(pl.col("x").mean())
    )
    return dict(zip(gk["period"], gk["x"]))


def load_events(path: Path) -> pl.DataFrame:
    ev = pl.read_csv(path, null_values=["NaN", ""])
    return ev.rename(
        {c: re.sub(r"\W+", "_", c.split(" [")[0]).strip("_").lower() for c in ev.columns}
    )


def build_events(ev: pl.DataFrame, match_id: str, flip: bool, report: ConversionReport):
    """Shots and goals. Scoring team = the team that didn't take the next kickoff."""
    sx, sy = to_meters(pl.col("start_x"), pl.col("start_y"), y_down=True)
    ev = ev.with_row_index("row").with_columns(x=sx, y=sy)
    if flip:
        ev = ev.with_columns(x=-pl.col("x"), y=-pl.col("y"))

    set_pieces = ev.filter(pl.col("type") == "SET PIECE").select(
        "team",
        frame="start_frame",
        set_piece=pl.col("subtype")
        .str.replace("-RETAKEN", "")
        .replace_strict(SET_PIECES, default=None),
    )
    shots = (
        ev.filter(pl.col("type") == "SHOT")
        .join(set_pieces, left_on=["team", "start_frame"], right_on=["team", "frame"], how="left")
        .with_columns(set_piece=pl.col("set_piece").fill_null("open_play"))
    )
    rows = []
    for s in shots.iter_rows(named=True):
        rows.append(
            {
                "frame_id": s["start_frame"],
                "event_type": "shot",
                "team": s["team"].lower(),
                "player_id": player_ref(s["team"], s["from"]),
                "x": s["x"],
                "y": s["y"],
                "outcome": shot_outcome(s["subtype"]),
                "set_piece": s["set_piece"],
            }
        )
    unknown = sum(r["outcome"] == "unknown" for r in rows)
    if unknown:
        report.unresolve("shot outcome", f"{unknown} shots with an unmapped subtype")

    kickoffs = ev.filter(
        (pl.col("type") == "SET PIECE") & pl.col("subtype").str.starts_with("KICK OFF")
    )
    goal_coded = ev.filter(
        pl.col("subtype").str.contains("GOAL") & ~pl.col("subtype").str.contains("GOAL KICK")
    )
    own_goals = 0
    for g in goal_coded.iter_rows(named=True):
        nxt = kickoffs.filter(
            (pl.col("period") == g["period"]) & (pl.col("start_frame") > g["start_frame"])
        ).head(1)
        if nxt.height:
            scorer = "home" if nxt["team"][0] == "Away" else "away"
        else:
            scorer = g["team"].lower()
            report.unresolve("goal", f"no kickoff after the goal at frame {g['start_frame']}")
        own_goal = scorer != g["team"].lower()
        own_goals += own_goal
        rows.append(
            {
                "frame_id": g["start_frame"],
                "event_type": "goal",
                "team": scorer,
                "player_id": None if own_goal else player_ref(g["team"], g["from"]),
                "x": g["x"],
                "y": g["y"],
                "outcome": "own_goal" if own_goal else "goal",
                "set_piece": None,
            }
        )
    if own_goals:
        report.change(
            "events", own_goals, "goal credited to the other team (own goal, from next kickoff)"
        )
    report.rows["events"] = {"in": ev.height}
    kept = ev.filter((pl.col("type") == "SHOT") | pl.col("row").is_in(goal_coded["row"].implode()))
    report.drop("events", ev.height - kept.height, "not a shot or goal")
    return pl.DataFrame(
        rows,
        schema={
            "frame_id": pl.Int64,
            "event_type": pl.String,
            "team": pl.String,
            "player_id": pl.String,
            "x": pl.Float64,
            "y": pl.Float64,
            "outcome": pl.String,
            "set_piece": pl.String,
        },
    ).with_columns(
        match_id=pl.lit(match_id),
        set_play_phase=pl.lit(None, pl.Boolean),
        player_id=pl.concat_str(pl.lit(f"{match_id}_"), "player_id"),
    )


def convert_game(game: int, raw_dir: Path = RAW_DIR, out_dir: Path = OUT_DIR) -> list[str]:
    match_id = f"metrica-{game}"
    src = raw_dir / f"Sample_Game_{game}"
    home_csv = src / f"Sample_Game_{game}_RawTrackingData_Home_Team.csv"
    away_csv = src / f"Sample_Game_{game}_RawTrackingData_Away_Team.csv"
    events_csv = src / f"Sample_Game_{game}_RawEventsData.csv"
    report = ConversionReport(match_id=match_id, source="metrica")
    for p in (home_csv, away_csv, events_csv):
        report.add_source(p)

    df, teams = load_tracking(home_csv, away_csv)
    report.rows["frames"] = {"in": df.height}
    frames = df.select("frame_id", period="period_id", timestamp_s="timestamp_s")

    objects = long_objects(df, teams, report)
    objects, keepers = mark_goalkeepers(objects, frames)
    report.change("objects", 2, "goalkeepers inferred: deepest starter per team in period 1")
    gk_x = attack_direction(objects, frames, keepers["home"])
    # 02 fixes +x to where home attacks in period 1. Rotate if the raw data has it the other way.
    flip = gk_x[1] > 0
    if flip:
        objects = objects.with_columns(x=-pl.col("x"), y=-pl.col("y"))
        gk_x = {p: -v for p, v in gk_x.items()}
        report.change("objects", objects.height, "rotated 180 deg so home attacks +x in period 1")
    report.checks["home_gk_mean_x_by_period"] = {str(k): round(v, 1) for k, v in gk_x.items()}

    frames = frames.with_columns(
        match_id=pl.lit(match_id),
        period=pl.col("period").cast(pl.Int64),
        home_attacks_positive_x=pl.col("period").replace_strict(
            {p: v < 0 for p, v in gk_x.items()}, return_dtype=pl.Boolean
        ),
        ball_state=pl.lit(None, pl.String),
        possession_team=pl.lit(None, pl.String),
        ball_carrier_id=pl.lit(None, pl.String),
        view_polygon=pl.lit(None, pl.List(pl.Float64)),
        set_play_phase=pl.lit(None, pl.Boolean),  # no possession to hang a phase on (02)
    )

    objects = causal_velocities(objects, frames).with_columns(
        match_id=pl.lit(match_id),
        player_id=pl.when(pl.col("object_type") != "ball").then(
            pl.concat_str(pl.lit(f"{match_id}_"), "object_id")
        ),
        z=pl.lit(None, pl.Float64),
        visible=pl.lit(True),
        interpolated=pl.lit(False),
        confidence=pl.lit(1.0),
    )

    events = build_events(load_events(events_csv), match_id, flip, report)

    players = pl.DataFrame(
        [
            {
                "player_id": f"{match_id}_{p.player_id}",
                "team": t.ground.value,
                "jersey_number": int(p.jersey_no),
                "position": "GK" if p.player_id in keepers.values() else None,
                "name": None,
            }
            for t in teams
            for p in t.players
        ],
        schema_overrides={"position": pl.String, "name": pl.String},
    ).with_columns(match_id=pl.lit(match_id))

    match = pl.DataFrame(
        {
            "match_id": [match_id],
            "schema_version": [SCHEMA_VERSION],
            "source": ["metrica"],
            "competition": [None],
            "season": [None],
            "date": [None],
            "home_team": ["home"],
            "away_team": ["away"],
            "native_fps": [FPS],
        },
        schema_overrides={"competition": pl.String, "season": pl.String, "date": pl.Date},
    )

    report.checks.update(sanity_checks(objects, events))
    tables = {
        "match": match,
        "objects": objects,
        "frames": frames,
        "events": events,
        "players": players,
    }
    return write_gamestate(tables, out_dir / match_id, report)


def sanity_checks(objects: pl.DataFrame, events: pl.DataFrame) -> dict:
    """Numbers that show whether coordinates and events line up. Stored in the report."""
    ball = objects.filter(pl.col("object_type") == "ball").select("frame_id", bx="x", by="y")
    shots = events.filter(pl.col("event_type") == "shot").join(ball, on="frame_id", how="left")
    dist = ((pl.col("x") - pl.col("bx")) ** 2 + (pl.col("y") - pl.col("by")) ** 2).sqrt()
    d = shots.select(dist.alias("d"))["d"].drop_nulls()
    goals = events.filter(pl.col("event_type") == "goal")
    speed = (pl.col("vx") ** 2 + pl.col("vy") ** 2).sqrt()
    fast = objects.filter((pl.col("object_type") != "ball") & (speed > MAX_PLAYER_SPEED)).height
    return {
        # Source tracking jumps, kept as-is; a human tops out around 10-11 m/s.
        f"player_rows_over_{MAX_PLAYER_SPEED:g}_mps": fast,
        "shots": shots.height,
        "open_play_shots": shots.filter(pl.col("set_piece") == "open_play").height,
        "shot_to_ball_m": {
            "n": d.len(),
            "median": round(d.median(), 2) if d.len() else None,
            "p90": round(d.quantile(0.9), 2) if d.len() else None,
        },
        "score": {t: goals.filter(pl.col("team") == t).height for t in ("home", "away")},
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--raw", type=Path, default=RAW_DIR)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    failed = 0
    for g in args.games:
        errors = convert_game(g, args.raw, args.out)
        print(f"{'FAIL' if errors else 'ok  '} metrica-{g}")
        for e in errors:
            print(f"  {e}")
        failed += bool(errors)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
