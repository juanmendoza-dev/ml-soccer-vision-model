"""PFF World Cup 2022 event JSON -> events table (02).

    python -m converters.pff_events [--games ID ...] [--raw DIR]

Needs only Event Data + Metadata, so it runs on all 64 games (with or without
tracking) and prints the counts 06 pins. The tracking converter reuses it.

Rules (02, 06):
- shot = possessionEventType SH; set_piece from gameEvents.setpieceType
- goal = shotOutcomeType G whose next restart is a kickoff or the period end;
  a G followed by a free kick is a disallowed goal (shot outcome, no goal event)
- goals not coded as shots (CR, RE) still get a goal event, unless an SH goal
  comes before the same restart (Sabiri: CR + SH, count the SH)
- scoring team = the team not taking the next kickoff; shooter's team at a period end
- shootout = a P shot in period 4 with no kickoff after it; not a shot event
"""

import argparse
import json
import sys
from pathlib import Path

import polars as pl

from converters.common import ConversionReport

RAW_DIR = Path("data/raw/pff")
FPS = 29.97
SET_PLAY_WINDOW_S = 10.0

SET_PIECES = {
    "O": "open_play",
    "C": "corner",
    "F": "free_kick",
    "P": "penalty",
    "T": "throw_in",
    "G": "goal_kick",
    "K": "kickoff",
    "D": "drop_ball",
}
# shotOutcomeType. G/S/B/O are clear from the counts; C/F/L follow PFF's naming
# (off-target block, frame of the goal, goal-line clearance), not checked on video.
SHOT_OUTCOMES = {
    "G": "goal",
    "S": "saved",
    "B": "blocked",
    "C": "blocked",
    "O": "off_target",
    "F": "woodwork",
    "L": "cleared_off_line",
}
KICKOFFS = {"FIRSTKICKOFF", "SECONDKICKOFF", "THIRDKICKOFF", "FOURTHKICKOFF"}
PERIODS = {1, 2, 3, 4}

EVENT_SCHEMA = {
    "match_id": pl.String,
    "frame_id": pl.Int64,
    "event_type": pl.String,
    "team": pl.String,
    "player_id": pl.String,
    "x": pl.Float64,
    "y": pl.Float64,
    "outcome": pl.String,
    "set_piece": pl.String,
    "set_play_phase": pl.Boolean,
}


def load_metadata(game_id: str, raw_dir: Path = RAW_DIR) -> dict:
    return json.loads((raw_dir / "Metadata" / f"{game_id}.json").read_text())[0]


def frame_of(seconds: float) -> int:
    """PFF frameNum for a video time (06: frameNum = round(videoTime_s x 29.97))."""
    return round(seconds * FPS)


def restart_of(row: dict) -> str | None:
    """What a game event restarts play with: a set-piece code, 'K', 'END', or None."""
    g = row["gameEvents"]
    if g["gameEventType"] in KICKOFFS or g["setpieceType"] == "K":
        return "K"
    if g["gameEventType"] == "END":
        return "END"
    if g["setpieceType"] not in (None, "O"):
        return g["setpieceType"]
    return None


def location(row: dict, player_id, flip: bool) -> tuple[float, float, bool] | None:
    """(x, y, from_player): ball position of the event row, else the player's
    snapshot position (from_player=True). None if neither exists."""
    point, from_player = (row["ball"][0] if row.get("ball") else None), False
    if point is None or point.get("x") is None:
        side = (row.get("homePlayers") or []) + (row.get("awayPlayers") or [])
        point = next((p for p in side if p.get("playerId") == player_id), None)
        from_player = True
    if point is None or point.get("x") is None:
        return None
    s = -1 if flip else 1
    return s * point["x"], s * point["y"], from_player


def set_play_restarts(rows: list[dict]) -> list[tuple[int, bool, float]]:
    """(period, home team, startTime) of every corner and free kick: the restarts that
    start a set-play phase (02). Shared by the events proxy and frames.set_play_phase."""
    return [
        (r["gameEvents"]["period"], r["gameEvents"]["homeTeam"], r["startTime"])
        for r in rows
        if r["gameEvents"]["period"] in PERIODS and r["gameEvents"]["setpieceType"] in ("C", "F")
    ]


def load_set_play_restarts(game_id: str, raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    """set_play_restarts as a table: period, team (home/away), start_s (video time)."""
    raw = json.loads((raw_dir / "Event Data" / f"{game_id}.json").read_text())
    return pl.DataFrame(
        [(p, "home" if home else "away", t) for p, home, t in set_play_restarts(raw)],
        schema={"period": pl.Int64, "team": pl.String, "start_s": pl.Float64},
        orient="row",
    )


def parse_events(
    game_id: str, raw_dir: Path = RAW_DIR, report: ConversionReport | None = None
) -> pl.DataFrame:
    """Shots and goals for one game in 02's frame (+x = home's period-1 attack)."""
    report = report or ConversionReport(match_id=str(game_id), source="pff")
    path = raw_dir / "Event Data" / f"{game_id}.json"
    report.add_source(path)
    raw = json.loads(path.read_text())
    report.rows["events"] = {"in": len(raw)}

    rows = [r for r in raw if r["gameEvents"]["period"] in PERIODS]
    report.drop("events", len(raw) - len(rows), "period not in 1-4 (placeholder rows)")
    # 02's +x is where home attacks in period 1. Raw coordinates are a fixed pitch
    # frame with home starting on the left (-x) when homeTeamStartLeft is true.
    flip = not load_metadata(game_id, raw_dir)["homeTeamStartLeft"]

    # File order is the event sequence; ~100 rows (mostly FOULs) are out of time order by
    # up to a few seconds, so restarts go by position, time windows by startTime.
    restarts = [restart_of(r) for r in rows]
    kickoff_after = [False] * len(rows)
    seen = False
    for i in range(len(rows) - 1, -1, -1):
        kickoff_after[i] = seen
        seen = seen or restarts[i] == "K"

    def next_restart(i: int) -> int:
        """Index of the next restart after row i's game event; len(rows) at the end."""
        geid = rows[i]["gameEventId"]
        for j in range(i + 1, len(rows)):
            if rows[j]["gameEventId"] != geid and restarts[j]:
                return j
        return len(rows)

    def is_shot_goal(r: dict) -> bool:
        pe = r["possessionEvents"]
        return pe["possessionEventType"] == "SH" and pe["shotOutcomeType"] == "G"

    # Same-team corners and free kicks, for the set-play-phase proxy. Times are the
    # game event's startTime; that reproduces 06 (eventTime lets 5 more shots through).
    set_plays = set_play_restarts(rows)

    def in_set_play(row: dict) -> bool:
        g = row["gameEvents"]
        return any(
            p == g["period"]
            and home == g["homeTeam"]
            and 0 <= row["startTime"] - t <= SET_PLAY_WINDOW_S
            for p, home, t in set_plays
        )

    out, no_location, player_location = [], [], 0
    shootout_kicks = shootout_goals = disallowed_other = duplicate_goals = 0
    shootout_start = None
    for i, r in enumerate(rows):
        g, pe = r["gameEvents"], r["possessionEvents"]
        kind, result = pe["possessionEventType"], pe["shotOutcomeType"]
        is_shot = kind == "SH"
        if not is_shot and result != "G":
            continue
        if is_shot and g["setpieceType"] == "P" and g["period"] == 4 and not kickoff_after[i]:
            if shootout_start is None:
                # From the period-4 END before the first kick, else the kick itself.
                ends = [x["startTime"] for x in rows[:i] if restart_of(x) == "END"]
                shootout_start = ends[-1] if ends and g["period"] == 4 else r["startTime"]
            shootout_kicks += 1
            shootout_goals += result == "G"
            continue

        team = "home" if g["homeTeam"] else "away"
        player = pe["shooterPlayerId"] if is_shot else g["playerId"]
        loc = location(r, player, flip)
        if loc is None:
            no_location.append(f"row {i} ({kind})")
            continue
        player_location += loc[2]
        base = {
            "frame_id": frame_of(r["eventTime"]),
            "x": loc[0],
            "y": loc[1],
            "set_piece": SET_PIECES.get(g["setpieceType"]),
            "set_play_phase": in_set_play(r),
        }

        outcome = SHOT_OUTCOMES.get(result) if is_shot else None
        if result == "G":
            j = next_restart(i)
            restart = restarts[j] if j < len(rows) else "END"
            if restart == "F":
                if is_shot:
                    outcome = "disallowed"
                else:
                    disallowed_other += 1
            elif restart in ("K", "END"):
                # A goal coded on a cross/rebound as well as on a later shot: keep the shot.
                if not is_shot and any(is_shot_goal(rows[k]) for k in range(i + 1, j)):
                    duplicate_goals += 1
                else:
                    scorer = team
                    if restart == "K":
                        scorer = "away" if rows[j]["gameEvents"]["homeTeam"] else "home"
                    own = scorer != team
                    if own and is_shot:
                        report.unresolve(
                            "goal", f"row {i}: shot goal but the scorer kicks off next"
                        )
                    out.append(
                        base
                        | {
                            "event_type": "goal",
                            "team": scorer,
                            "player_id": None if own else _id(player),
                            "outcome": "own_goal" if own else "goal",
                        }
                    )
            else:
                report.unresolve("goal", f"row {i}: G followed by restart {restart!r}")
        if is_shot:
            if outcome is None:
                report.unresolve("shot outcome", f"row {i}: {result!r}")
            out.append(
                base
                | {"event_type": "shot", "team": team, "player_id": _id(player), "outcome": outcome}
            )

    if no_location:
        report.unresolve("event location", no_location)
    report.drop("events", len(no_location), "shot/goal with no ball or player position")
    report.change("events", player_location, "no ball in the event row: shooter's position used")
    report.drop("events", shootout_kicks, "shootout kicks (period 5, 02): not shot events")
    report.drop(
        "events", disallowed_other, "disallowed goal coded on a non-shot (G then free kick)"
    )
    report.drop("events", duplicate_goals, "goal coded twice (non-shot + shot); kept the shot")
    kept = sum(
        r["possessionEvents"]["possessionEventType"] == "SH"
        or r["possessionEvents"]["shotOutcomeType"] == "G"
        for r in rows
    )
    report.drop("events", len(rows) - kept, "not a shot or goal")
    report.checks["shootout"] = {
        "kicks": shootout_kicks,
        "goals": shootout_goals,
        "start_s": shootout_start,  # video time; tracking frames from here are period 5
    }
    report.checks["disallowed_non_shot_goals"] = disallowed_other
    return (
        pl.DataFrame(out, schema={k: v for k, v in EVENT_SCHEMA.items() if k != "match_id"})
        .with_columns(match_id=pl.lit(str(game_id)))
        .select(list(EVENT_SCHEMA))
    )


def _id(v) -> str | None:
    return None if v is None else str(v)


def game_ids(raw_dir: Path = RAW_DIR) -> list[str]:
    return sorted(p.stem for p in (raw_dir / "Event Data").glob("*.json"))


def summary(events: pl.DataFrame) -> dict:
    shots = events.filter(pl.col("event_type") == "shot")
    open_play = shots.filter(pl.col("set_piece") == "open_play")
    goals = events.filter(pl.col("event_type") == "goal")
    return {
        "shots": shots.height,
        "open_play_strict": open_play.height,
        "open_play_proxy": open_play.filter(~pl.col("set_play_phase")).height,
        "goals": goals.height,
        "disallowed": shots.filter(pl.col("outcome") == "disallowed").height,
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", nargs="+")
    ap.add_argument("--raw", type=Path, default=RAW_DIR)
    args = ap.parse_args(argv)
    games = args.games or game_ids(args.raw)
    events = pl.concat([parse_events(g, args.raw) for g in games])
    print(f"{len(games)} games: {summary(events)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
