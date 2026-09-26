"""PFF World Cup 2022 tracking + events -> game state (02).

    python -m converters.pff [--games ID ...] [--raw DIR] [--out DIR]

Tracking is read from the raw JSONL (kloppy drops visibility and confidence and
only gives the smoothed positions), streamed from the bz2 file in chunks. Events
come from converters.pff_events. Without --games, converts every game that has
a tracking file.
"""

import argparse
import bz2
import json
import sys
from pathlib import Path

import polars as pl

from converters.common import ConversionReport, causal_velocities, write_gamestate
from converters.pff_events import RAW_DIR, load_metadata, parse_events
from gamestate.schema import (
    PFF_CONFIDENCE,
    PITCH_LENGTH,
    PITCH_WIDTH,
    SCHEMA_VERSION,
    SHOOTOUT_PERIOD,
)
from gamestate.validate import POSITION_MARGIN_M

OUT_DIR = Path("data/gamestate")
CHUNK_FRAMES = 10_000
MAX_PLAYER_SPEED = 12.0  # m/s, only used for the report check
KICKOFFS = {"FIRSTKICKOFF", "SECONDKICKOFF", "THIRDKICKOFF", "FOURTHKICKOFF"}
LEFT = {"LB", "LWB", "LW", "LM", "LCB"}
RIGHT = {"RB", "RWB", "RW", "RM", "RCB"}

OBJECT_SCHEMA = {
    "frame_id": pl.Int64,
    "object_id": pl.String,
    "object_type": pl.String,
    "team": pl.String,
    "player_id": pl.String,
    "x": pl.Float64,
    "y": pl.Float64,
    "z": pl.Float64,
    "visible": pl.Boolean,
    "confidence": pl.Float64,
}
FRAME_SCHEMA = {
    "frame_id": pl.Int64,
    "period": pl.Int64,
    "timestamp_s": pl.Float64,
    "video_time_s": pl.Float64,
    "ball_state": pl.String,
    "possession_team": pl.String,
}


def load_roster(game_id: str, raw_dir: Path, meta: dict) -> pl.DataFrame:
    rows = json.loads((raw_dir / "Rosters" / f"{game_id}.json").read_text())
    home_id = meta["homeTeam"]["id"]
    return pl.DataFrame(
        [
            {
                "player_id": r["player"]["id"],
                "team": "home" if r["team"]["id"] == home_id else "away",
                "jersey_number": int(r["shirtNumber"]),
                "position": r["positionGroupType"],
                "name": r["player"]["nickname"],
                "started": r["started"],
            }
            for r in rows
        ]
    )


def read_tracking(
    path: Path, roster: pl.DataFrame, report: ConversionReport
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Raw JSONL -> (frames, objects) in PFF's own pitch frame.

    Uses the raw homePlayers/awayPlayers/balls, not the smoothed copies. Ball
    state and possession follow the inline game_event the way kloppy does it:
    OUT/END -> dead, OTB/kickoffs -> alive, home_ball -> team, carried forward.
    """
    lookup = {
        (r["team"], str(r["jersey_number"])): (r["player_id"], r["position"] == "GK")
        for r in roster.iter_rows(named=True)
    }
    ids: dict[tuple[str, str], str] = {}  # reuse one string per object
    unmatched: set[tuple[str, str]] = set()

    frames: list[pl.DataFrame] = []
    objects: list[pl.DataFrame] = []
    f_cols = {k: [] for k in FRAME_SCHEMA}
    o_cols = {k: [] for k in OBJECT_SCHEMA}
    ball_state, possession = "dead", None
    lines = duplicates = no_period = no_ball = no_xy = dup_jersey = 0
    last_frame = None

    def flush():
        frames.append(pl.DataFrame(f_cols, schema=FRAME_SCHEMA))
        objects.append(pl.DataFrame(o_cols, schema=OBJECT_SCHEMA))
        for cols in (f_cols, o_cols):
            for v in cols.values():
                v.clear()

    def add(frame_id, object_id, kind, team, player_id, p, z, confidence):
        o_cols["frame_id"].append(frame_id)
        o_cols["object_id"].append(object_id)
        o_cols["object_type"].append(kind)
        o_cols["team"].append(team)
        o_cols["player_id"].append(player_id)
        o_cols["x"].append(p["x"])
        o_cols["y"].append(p["y"])
        o_cols["z"].append(z)
        o_cols["visible"].append(p["visibility"] == "VISIBLE")
        o_cols["confidence"].append(confidence)

    with bz2.open(path, "rt") as fh:
        for line in fh:
            lines += 1
            d = json.loads(line)
            event = d.get("game_event")
            if event:
                kind = event.get("game_event_type")
                if kind in ("OUT", "END"):
                    ball_state = "dead"
                elif kind in KICKOFFS or kind == "OTB":
                    ball_state = "alive"
                if event.get("home_ball") is not None:
                    possession = "home" if event["home_ball"] else "away"
            if d["period"] is None:
                no_period += 1
                continue
            frame_id = d["frameNum"]
            if frame_id == last_frame:
                duplicates += 1  # checked: identical copies of the previous frame
                continue
            last_frame = frame_id

            f_cols["frame_id"].append(frame_id)
            f_cols["period"].append(d["period"])
            f_cols["timestamp_s"].append(d["periodElapsedTime"])
            f_cols["video_time_s"].append(d["videoTimeMs"] / 1000)
            f_cols["ball_state"].append(ball_state)
            f_cols["possession_team"].append(possession)

            for side, key in (("home", "homePlayers"), ("away", "awayPlayers")):
                seen = set()
                for p in d[key] or []:
                    jersey = str(p["jerseyNum"])
                    if p["x"] is None or p["y"] is None:
                        no_xy += 1
                        continue
                    if jersey in seen:
                        dup_jersey += 1
                        continue
                    seen.add(jersey)
                    k = (side, jersey)
                    if k not in ids:
                        ids[k] = f"{side}_{jersey}"
                    player_id, is_gk = lookup.get(k, (None, False))
                    if player_id is None:
                        unmatched.add(k)
                    add(
                        frame_id,
                        ids[k],
                        "goalkeeper" if is_gk else "player",
                        side,
                        player_id,
                        p,
                        None,
                        PFF_CONFIDENCE[p["confidence"]],
                    )
            balls = [b for b in d["balls"] or [] if b.get("x") is not None]
            if balls:
                # Checked on the dev games: never more than one ball per frame.
                add(frame_id, "ball", "ball", None, None, balls[0], balls[0]["z"], 1.0)
            else:
                no_ball += 1

            if len(f_cols["frame_id"]) >= CHUNK_FRAMES:
                flush()
    flush()

    report.rows["frames"] = {"in": lines}
    report.drop("frames", duplicates, "duplicate frameNum (identical copy of the previous frame)")
    report.drop("frames", no_period, "frame with no period")
    report.drop("objects", no_ball, "frames with an empty balls list (no ball tracked)")
    report.drop("objects", no_xy, "player with a null x/y")
    report.drop("objects", dup_jersey, "same jersey twice on one team in a frame")
    if unmatched:
        report.unresolve("jersey", sorted(f"{s}_{j}" for s, j in unmatched))
    return pl.concat(frames), pl.concat(objects)


def move_shootout(frames: pl.DataFrame, start_s: float | None, report: ConversionReport):
    """Period-4 frames from the shootout start (video time) -> period 5, dead ball,
    no possession, timestamp from the shootout start (02)."""
    if start_s is None:
        return frames
    shootout = (pl.col("period") == 4) & (pl.col("video_time_s") >= start_s)
    report.change("frames", frames.filter(shootout).height, "shootout frames moved to period 5")
    return frames.with_columns(
        period=pl.when(shootout).then(SHOOTOUT_PERIOD).otherwise("period"),
        timestamp_s=pl.when(shootout)
        .then(pl.col("video_time_s") - start_s)
        .otherwise("timestamp_s"),
        ball_state=pl.when(shootout).then(pl.lit("dead")).otherwise("ball_state"),
        possession_team=pl.when(shootout).then(None).otherwise("possession_team"),
    )


def keeper_x(objects: pl.DataFrame, frames: pl.DataFrame, object_id: str) -> dict[int, float]:
    gk = (
        objects.filter(pl.col("object_id") == object_id)
        .join(frames.select("frame_id", "period"), on="frame_id")
        .group_by("period")
        .agg(pl.col("x").mean())
    )
    return {int(p): round(x, 1) for p, x in gk.iter_rows()}


def side_check(objects: pl.DataFrame, frames: pl.DataFrame, roster: pl.DataFrame) -> dict:
    """+y check (02): facing the +x goal, +y is on your left. So for a team
    attacking +x, its left-sided players should have the larger mean y."""
    side = roster.select(
        "player_id",
        side=pl.when(pl.col("position").is_in(list(LEFT)))
        .then(1)
        .when(pl.col("position").is_in(list(RIGHT)))
        .then(-1),
    ).drop_nulls()
    rows = (
        objects.join(side, on="player_id")
        .join(frames.select("frame_id", "period", "home_attacks_positive_x"), on="frame_id")
        .group_by("team", "period", "side")
        .agg(y=pl.col("y").mean(), attacks_pos=pl.col("home_attacks_positive_x").first())
    )
    out = {}
    for (team, period), g in rows.group_by("team", "period"):
        y = dict(zip(g["side"], g["y"]))
        if 1 not in y or -1 not in y:
            continue
        attacks_pos = g["attacks_pos"][0] == (team == "home")
        # left minus right, signed so positive = consistent with 02
        out[f"{team}_p{period}"] = round((y[1] - y[-1]) * (1 if attacks_pos else -1), 1)
    return dict(sorted(out.items()))


def link_events(events: pl.DataFrame, frames: pl.DataFrame, report: ConversionReport):
    """Event frame = round(eventTime x fps), the same rule as frameNum. Events that
    land on a missing frame snap to the nearest tracked frame (logged)."""
    known = events.join(frames.select("frame_id"), on="frame_id", how="semi")
    missing = events.join(frames.select("frame_id"), on="frame_id", how="anti")
    if missing.height:
        snapped = (
            missing.with_columns(t=pl.col("frame_id").cast(pl.Float64))
            .sort("t")
            .join_asof(
                frames.select(pl.col("frame_id").alias("nearest"))
                .with_columns(t=pl.col("nearest").cast(pl.Float64))
                .sort("t"),
                on="t",
                strategy="nearest",
            )
        )
        report.change(
            "events",
            missing.height,
            "event frame not in tracking; moved to the nearest frame "
            f"(max {snapped.select((pl.col('nearest') - pl.col('frame_id')).abs().max()).item()} frames)",
        )
        missing = snapped.with_columns(frame_id=pl.col("nearest")).drop("t", "nearest")
    return pl.concat([known, missing.select(known.columns)]).sort("frame_id")


def convert_game(game_id: str, raw_dir: Path = RAW_DIR, out_dir: Path = OUT_DIR) -> list[str]:
    game_id = str(game_id)
    report = ConversionReport(match_id=game_id, source="pff")
    meta = load_metadata(game_id, raw_dir)
    track_path = raw_dir / "Tracking Data" / f"{game_id}.jsonl.bz2"
    for p in (
        raw_dir / "Metadata" / f"{game_id}.json",
        raw_dir / "Rosters" / f"{game_id}.json",
        track_path,
    ):
        report.add_source(p)

    events = parse_events(game_id, raw_dir, report)
    roster = load_roster(game_id, raw_dir, meta)
    frames, objects = read_tracking(track_path, roster, report)

    # Raw tracking is one fixed pitch frame for the whole game, home on the left
    # (-x) in period 1 when homeTeamStartLeft. 02 wants home attacking +x in
    # period 1, so a single 180 deg rotation covers every period.
    if not meta["homeTeamStartLeft"]:
        objects = objects.with_columns(x=-pl.col("x"), y=-pl.col("y"))
        report.change("objects", objects.height, "rotated 180 deg so home attacks +x in period 1")
    # Estimated ball positions in the stands after a clearance or a shot over the bar.
    far = (pl.col("x").abs() > PITCH_LENGTH / 2 + POSITION_MARGIN_M) | (
        pl.col("y").abs() > PITCH_WIDTH / 2 + POSITION_MARGIN_M
    )
    for kind, n in objects.filter(far).group_by("object_type").len().iter_rows():
        report.drop("objects", n, f"{kind} more than {POSITION_MARGIN_M:g} m off the pitch")
    objects = objects.filter(~far)

    frames = move_shootout(frames, report.checks["shootout"]["start_s"], report)

    home_gk = roster.filter(
        (pl.col("team") == "home") & pl.col("started") & (pl.col("position") == "GK")
    )
    gk_x = keeper_x(objects, frames, f"home_{home_gk['jersey_number'][0]}")
    # Expected from metadata; the keeper's side is the check.
    expected = {1: True, 2: False}
    if meta.get("homeTeamStartLeftExtraTime") is not None:
        same = meta["homeTeamStartLeftExtraTime"] == meta["homeTeamStartLeft"]
        expected |= {3: same, 4: not same}
    attacks_pos = {p: x < 0 for p, x in gk_x.items() if p != SHOOTOUT_PERIOD}
    mismatch = {p: v for p, v in attacks_pos.items() if expected.get(p, v) != v}
    if mismatch:
        report.unresolve(
            "direction", f"home keeper side disagrees with metadata in periods {sorted(mismatch)}"
        )
    report.checks["home_gk_mean_x_by_period"] = {str(k): v for k, v in gk_x.items()}

    frames = frames.with_columns(
        match_id=pl.lit(game_id),
        home_attacks_positive_x=pl.col("period").replace_strict(
            attacks_pos | {SHOOTOUT_PERIOD: attacks_pos.get(4, True)}, return_dtype=pl.Boolean
        ),
        ball_carrier_id=pl.lit(
            None, pl.String
        ),  # PFF tracking doesn't give one (02: null, not a guess)
        view_polygon=pl.lit(None, pl.List(pl.Float64)),  # no camera footprint in PFF
    )

    objects = causal_velocities(objects, frames).with_columns(
        match_id=pl.lit(game_id),
        interpolated=~pl.col("visible"),  # ESTIMATED -> visible=False, interpolated=True
    )
    report.checks["plus_y_left_minus_right_m"] = side_check(objects, frames, roster)

    events = link_events(events, frames, report)
    players = roster.select(
        pl.lit(game_id).alias("match_id"),
        pl.col("player_id").cast(pl.String),
        "team",
        pl.col("jersey_number").cast(pl.Int64),
        "position",
        "name",
    )
    match = pl.DataFrame(
        {
            "match_id": [game_id],
            "schema_version": [SCHEMA_VERSION],
            "source": ["pff"],
            "competition": [meta["competition"]["name"]],
            "season": [meta["season"]],
            "date": [meta["date"][:10]],
            "home_team": [meta["homeTeam"]["name"]],
            "away_team": [meta["awayTeam"]["name"]],
            "native_fps": [float(meta["fps"])],
        }
    ).with_columns(pl.col("date").str.to_date())

    report.checks.update(sanity_checks(objects, frames, events))
    tables = {
        "match": match,
        "objects": objects.select(
            "match_id",
            "frame_id",
            "object_id",
            "object_type",
            "team",
            "player_id",
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "visible",
            "interpolated",
            "confidence",
        ),
        "frames": frames.drop("video_time_s"),
        "events": events,
        "players": players,
    }
    return write_gamestate(tables, out_dir / game_id, report)


def sanity_checks(objects: pl.DataFrame, frames: pl.DataFrame, events: pl.DataFrame) -> dict:
    """Numbers that show whether coordinates and events line up. Stored in the report."""
    ball = objects.filter(pl.col("object_type") == "ball").select("frame_id", bx="x", by="y")
    shots = events.filter(pl.col("event_type") == "shot").join(ball, on="frame_id", how="left")
    dist = ((pl.col("x") - pl.col("bx")) ** 2 + (pl.col("y") - pl.col("by")) ** 2).sqrt()
    d = shots.select(dist.alias("d"))["d"].drop_nulls()
    goals = events.filter(pl.col("event_type") == "goal")
    people = objects.filter(pl.col("object_type") != "ball")
    speed = people.select((pl.col("vx") ** 2 + pl.col("vy") ** 2).sqrt().alias("s"))[
        "s"
    ].drop_nulls()
    all_estimated = people.group_by("frame_id").agg(pl.col("visible").any())["visible"]
    return {
        "shots": shots.height,
        "shots_without_ball": shots.height - d.len(),
        "shot_to_ball_m": {
            "n": d.len(),
            "median": round(d.median(), 2) if d.len() else None,
            "p90": round(d.quantile(0.9), 2) if d.len() else None,
        },
        "score": {t: goals.filter(pl.col("team") == t).height for t in ("home", "away")},
        # Source tracking jumps, kept as-is; a human tops out around 10-11 m/s.
        "player_speed_p99_mps": round(speed.quantile(0.99), 2),
        f"player_rows_over_{MAX_PLAYER_SPEED:g}_mps": int((speed > MAX_PLAYER_SPEED).sum()),
        "frames_all_players_estimated": round(1 - all_estimated.mean(), 3),
        "frames_ball_alive": round(frames["ball_state"].eq("alive").mean(), 3),
    }


def tracked_games(raw_dir: Path = RAW_DIR) -> list[str]:
    return sorted(p.name.split(".")[0] for p in (raw_dir / "Tracking Data").glob("*.jsonl.bz2"))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", nargs="+")
    ap.add_argument("--raw", type=Path, default=RAW_DIR)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    failed = 0
    for g in args.games or tracked_games(args.raw):
        try:
            errors = convert_game(g, args.raw, args.out)
        except Exception as e:  # noqa: BLE001 -- one bad file shouldn't stop a batch
            errors = [f"{type(e).__name__}: {e}"]
        print(f"{'FAIL' if errors else 'ok  '} {g}", flush=True)
        for e in errors:
            print(f"  {e}")
        failed += bool(errors)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
