# 02 — Game State Schema

The contract between every component. Vision output and dataset converters must both produce exactly this.

## Coordinate system
- Pitch normalized to **105 m × 68 m**.
- Origin **(0, 0) at the center spot**. x ∈ [-52.5, 52.5], y ∈ [-34, 34].
- +x points toward the goal the **home team attacks in period 1**. Direction flips are recorded, not applied (see `frames` table).
- Units: meters, seconds, meters/second.

## Storage
Parquet files under `data/gamestate/<match_id>/`: `objects.parquet`, `frames.parquet`, `events.parquet`, `players.parquet`.

## `objects.parquet` (one row per object per frame)
| Column | Type | Notes |
|---|---|---|
| match_id | str | |
| frame_id | int | Monotonic within match |
| object_id | str | Stable track ID within match |
| object_type | enum | player, goalkeeper, referee, ball |
| team | enum/null | home, away, null |
| player_id | str/null | Links to `players.parquet` once identified |
| x, y | float | Meters |
| vx, vy | float | m/s, smoothed |
| visible | bool | False if outside camera view |
| interpolated | bool | True if filled in, not detected |
| confidence | float | 0–1; 1.0 for dataset tracking |

## `frames.parquet` (one row per frame)
| Column | Type | Notes |
|---|---|---|
| match_id, frame_id | | |
| period | int | 1, 2 (3, 4 for extra time) |
| timestamp_s | float | Seconds since period start |
| home_attacks_positive_x | bool | Attacking direction this period |
| possession_team | enum/null | home, away, null |
| ball_carrier_id | str/null | object_id of carrier, if any |

## `events.parquet`
| Column | Type | Notes |
|---|---|---|
| match_id, frame_id | | Frame the event starts |
| event_type | enum | shot, goal, pass, ... (shot and goal required) |
| team, player_id | | |
| x, y | float | Location |
| outcome | str/null | e.g. goal, saved, blocked, off_target |

## `players.parquet`
| Column | Type | Notes |
|---|---|---|
| match_id, player_id | | |
| team, jersey_number, position | | |
| name | str/null | |

## Rules
- Frame rate: store native rate; prediction resamples to **10 Hz**.
- Missing players (off camera) stay missing rows or `visible=False`; never guessed positions without `interpolated=True`.
- Any schema change is made here first, with a version bump.

**Schema version:** 0.1
