"""Game state schema (Docs/Specs/02-game-state-schema.md), as code.

Keep this in sync with 02. Change the spec first, then this file.
"""

from dataclasses import dataclass

SCHEMA_VERSION = "0.6"
# 0.6 only relaxed a rule (players may be empty), so 0.5 files are still valid
SUPPORTED_VERSIONS = frozenset({"0.5", "0.6"})

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

TEAMS = frozenset({"home", "away"})
SOURCES = frozenset({"skillcorner", "pff", "idsse", "metrica", "vision"})
OBJECT_TYPES = frozenset({"player", "goalkeeper", "referee", "ball"})
BALL_STATES = frozenset({"alive", "dead"})
PERIODS = frozenset({1, 2, 3, 4, 5})
SHOOTOUT_PERIOD = 5
SET_PIECES = frozenset(
    {
        "open_play",
        "corner",
        "free_kick",
        "penalty",
        "throw_in",
        "goal_kick",
        "kickoff",
        "drop_ball",
    }
)

# PFF confidence strings -> objects.confidence (ordinal, not a probability)
PFF_CONFIDENCE = {"HIGH": 1.0, "MEDIUM": 0.67, "LOW": 0.33}


@dataclass(frozen=True)
class Column:
    name: str
    kind: str  # str, int, float, bool, date, list_float
    nullable: bool = False
    values: frozenset | None = None  # allowed values, for enums


TABLES: dict[str, list[Column]] = {
    "match": [
        Column("match_id", "str"),
        Column("schema_version", "str", values=SUPPORTED_VERSIONS),
        Column("source", "str", values=SOURCES),
        Column("competition", "str", nullable=True),
        Column("season", "str", nullable=True),
        Column("date", "date", nullable=True),
        Column("home_team", "str"),
        Column("away_team", "str"),
        Column("native_fps", "float"),
    ],
    "objects": [
        Column("match_id", "str"),
        Column("frame_id", "int"),
        Column("object_id", "str"),
        Column("object_type", "str", values=OBJECT_TYPES),
        Column("team", "str", nullable=True, values=TEAMS),
        Column("player_id", "str", nullable=True),
        Column("x", "float"),
        Column("y", "float"),
        Column("z", "float", nullable=True),
        Column("vx", "float", nullable=True),
        Column("vy", "float", nullable=True),
        Column("visible", "bool"),
        Column("interpolated", "bool"),
        Column("confidence", "float"),
    ],
    "frames": [
        Column("match_id", "str"),
        Column("frame_id", "int"),
        Column("period", "int", values=PERIODS),
        Column("timestamp_s", "float"),
        Column("home_attacks_positive_x", "bool"),
        Column("ball_state", "str", nullable=True, values=BALL_STATES),
        Column("possession_team", "str", nullable=True, values=TEAMS),
        Column("ball_carrier_id", "str", nullable=True),
        Column("view_polygon", "list_float", nullable=True),
        Column("set_play_phase", "bool", nullable=True),  # label-side only (02)
    ],
    "events": [
        Column("match_id", "str"),
        Column("frame_id", "int"),
        Column("event_type", "str"),
        Column("team", "str", values=TEAMS),
        Column("player_id", "str", nullable=True),
        Column("x", "float"),
        Column("y", "float"),
        Column("outcome", "str", nullable=True),
        Column("set_piece", "str", nullable=True, values=SET_PIECES),
        Column("set_play_phase", "bool", nullable=True),
    ],
    "players": [
        Column("match_id", "str"),
        Column("player_id", "str"),
        Column("team", "str", values=TEAMS),
        Column("jersey_number", "int"),
        Column("position", "str", nullable=True),
        Column("name", "str", nullable=True),
    ],
}

# Tables that may legitimately have zero rows: vision output has no events, and
# no players until jersey OCR (0.6).
MAY_BE_EMPTY = frozenset({"events", "players"})
