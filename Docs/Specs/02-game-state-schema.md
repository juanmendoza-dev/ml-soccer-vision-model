# 02 — Game State Schema

The contract between every component. Vision output and dataset converters must both produce exactly this.

## Coordinate system
- Pitch normalized to **105 m × 68 m**.
- Origin **(0, 0) at the center spot**. x ∈ [-52.5, 52.5], y ∈ [-34, 34].
- +x points toward the goal the **home team attacks in period 1**. Direction flips are recorded, not applied (see `frames` table).
- +y is 90° counter-clockwise from +x: the left touchline when facing the +x goal. Seen as a TV picture with +x to the right, +y is up.
- Units: meters, seconds, meters/second.

## Storage
Parquet files under `data/gamestate/<match_id>/`: `match.parquet`, `objects.parquet`, `frames.parquet`, `events.parquet`, `players.parquet`.

## `match.parquet` (one row per match)
| Column | Type | Notes |
|---|---|---|
| match_id | str | |
| schema_version | str | e.g. `0.3`; the validator rejects versions it doesn't know |
| source | enum | skillcorner, pff, idsse, metrica, vision |
| competition, season | str/null | Season as `YYYY/YY` or `YYYY`; used for the previous-season rule in 04. null for anonymized sources (Metrica) and unknown vision footage |
| date | date/null | Kickoff date; null when unknown |
| home_team, away_team | str | |
| native_fps | float | Frame rate as stored; prediction resamples to 10 Hz |

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
| z | float/null | Ball height in meters where the source has it (PFF). Always null for non-ball objects |
| vx, vy | float/null | m/s, smoothed. **Causal:** computed from frames `<= t` of the same track only (backward differences, trailing smoothing), never across a period boundary or a gap in the track. null on a track's first frame after a start or gap |
| visible | bool | False if outside camera view. `visible=False` implies `interpolated=True` (the position is a guess) |
| interpolated | bool | True if filled in, not detected |
| confidence | float | 0–1. 1.0 for dataset tracking without a confidence field. PFF maps HIGH / MEDIUM / LOW → 1.0 / 0.67 / 0.33 (ordinal, not a probability) |

## `frames.parquet` (one row per frame)
| Column | Type | Notes |
|---|---|---|
| match_id, frame_id | | |
| period | int | 1, 2; 3, 4 for extra time; 5 for a penalty shootout |
| timestamp_s | float | Seconds since period start |
| home_attacks_positive_x | bool | Attacking direction this period |
| ball_state | enum/null | alive, dead, null = unknown |
| possession_team | enum/null | home, away, null = none or unknown |
| ball_carrier_id | str/null | object_id of the player in control of the ball; null when the ball is loose or in flight |
| view_polygon | list[float]/null | Camera footprint on the pitch in meters (x1, y1, ... x4, y4); null for full-pitch tracking |

## `events.parquet`
| Column | Type | Notes |
|---|---|---|
| match_id, frame_id | | Frame the event starts |
| event_type | enum | shot, goal, pass, ... (shot and goal required) |
| team, player_id | | |
| x, y | float | Location |
| outcome | str/null | e.g. goal, saved, blocked, off_target, disallowed |
| set_piece | enum/null | Restart the event itself was taken from: open_play, corner, free_kick, penalty, throw_in, goal_kick, kickoff, drop_ball. null = unknown |
| set_play_phase | bool/null | True if the event happens during a set-play phase (the attack that follows a restart, see below). null = producer can't tell |

## `players.parquet`
| Column | Type | Notes |
|---|---|---|
| match_id, player_id | | |
| team, jersey_number, position | | |
| name | str/null | |

## Possession and ball state
The predictor needs `ball_state`, `possession_team` and `ball_carrier_id` at inference time, so every producer must fill them.

| Producer | ball_state | possession_team / ball_carrier_id |
|---|---|---|
| IDSSE, Metrica, PFF | Provider ball status via kloppy, where present (Metrica CSV games 1–2 have none: null) | Provider fields via kloppy, where present (Metrica CSV: null) |
| SkillCorner | Derived: dead between a possession ending in a game interruption (`game_interruption_after`) and the next restart; dead where `period` is null | Frame `possession.group` / `possession.player_id` |
| Vision | Inferred (03, stage 8) | Inferred (03, stage 8) |

- A producer that can't decide writes null, never a guess.
- The inference rules in 03 stage 8 work on pitch coordinates only, so they can also be run on dataset tracking and checked against provider values.

## Shots, goals and set plays
05 trains on open play only, and each source marks it differently, so producers normalize it here.

| Producer | set_piece | set_play_phase |
|---|---|---|
| PFF | `gameEvents.setpieceType`: O → open_play, C → corner, F → free_kick, P → penalty, T → throw_in, G → goal_kick, K → kickoff, D → drop_ball | Proxy: true if ≤ 10 s after a same-team corner or free kick in the same period (06; window unverified) |
| SkillCorner | From the possession's start type where available, else null | `team_in_possession_phase_type = set_play` |
| IDSSE, Metrica | Provider event qualifiers via kloppy, where present. Metrica CSV: the `SET PIECE` row at the shot's frame (FREE KICK, CORNER KICK, PENALTY, ...), else open_play | null unless the provider marks it |
| Vision | null | null |

- Open-play labels (05) use `set_piece = open_play` and `set_play_phase` not true.
- `outcome = goal` only for goals that stand. A goal that is ruled out (e.g. PFF `shotOutcomeType = G` followed by a free-kick restart) gets `outcome = disallowed` and no `goal` event.
- Own goals and goals not coded as shots still get a `goal` event, with the scoring team in `team`.
- Penalty shootouts are period 5 (PFF puts them in period 4; the converter moves them using the shootout rule in 06). All period-5 frames have `ball_state = dead`, and shootout kicks are not `shot` events.

## Rules
- Frame rate: store native rate; prediction resamples to **10 Hz**.
- Missing players (off camera) stay missing rows or `visible=False`; never guessed positions without `interpolated=True`. PFF `visibility = ESTIMATED` and SkillCorner `is_detected = False` rows are written as `visible=False, interpolated=True`.
- Any schema change is made here first, with a version bump.

**Schema version:** 0.4
- 0.4: `competition`, `season`, `date` nullable; +y direction defined; `vx`/`vy` causal and nullable
- 0.3: added `match.schema_version`, `objects.z`, `events.set_piece`, `events.set_play_phase`, period 5 for shootouts, `disallowed` outcome, PFF confidence mapping, `visible=False` ⇒ `interpolated=True`
- 0.2: added `match.parquet`, `ball_state`, `view_polygon`; defined who fills possession fields
