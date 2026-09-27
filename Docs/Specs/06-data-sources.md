# 06 — Data Sources

Check each license before use and keep attribution where required.

## Tracking + events (shot labels)
The shot predictor needs continuous tracking **and** shot events on the same frames. Only these sources have both (researched 2026-09-25).

| Dataset | Matches | Tracking | Shot events | Players | License / access | Loader |
|---|---|---|---|---|---|---|
| SkillCorner open data (A-League 2024/25) | 20 | 10 Hz, broadcast-derived, off-camera players extrapolated | Yes: `dynamic_events.csv`, `player_possession` rows with `end_type = shot` and `frame_start`/`frame_end`. **526 shots** counted, 409 outside set plays; 61 followed by `game_interruption_after = goal_for` | Named, with IDs and jersey numbers | MIT, on GitHub (tracking via git-lfs, ~90 MB/match; all 20 confirmed downloadable) | kloppy `skillcorner` (verify it reads dynamic events) |
| PFF FC World Cup 2022 | 64 (tracking for all 64; extra time missing or unusable in 4, see below) | 29.97 Hz, broadcast-derived, all 22 players every frame with a per-player `visibility` flag (VISIBLE / ESTIMATED), 3D ball | Yes: per-game event JSON, `possessionEventType = SH`. **1,518 shots** counted (shootouts excluded), 1,450 not from a penalty/free kick/corner; 172 goals | Rosters with names and IDs; tracking uses jersey numbers | Free, by request form. **Access granted 2026-09-25**; terms of use still to confirm | kloppy `pff.load_tracking` only (drops visibility, see below); no event loader |
| IDSSE (Bassek et al. 2025, Bundesliga 1 + 2, 2022/23) | 7 | 25 Hz, TRACAB optical, full pitch | Yes: official DFL events | Named | CC-BY 4.0, attribute DFL + cite paper | kloppy `sportec.load_open_tracking_data` |
| Metrica Sports sample data | 3 | 25 Hz, full pitch | Yes: synchronized events | Anonymized | No formal license; acknowledge source | kloppy `metrica` |

### Shot volume
| Scenario | Matches | Shots |
|---|---|---|
| SkillCorner only | 20 | 526 counted (409 open play) |
| + IDSSE + Metrica | 30 | ~800 (estimated at ~25/match) |
| PFF, games with tracking | 51 | 1,179 counted (1,125 open play strict, ~890 set-play-phase proxy) |
| PFF, all games | 64 | 1,518 counted (1,450 open play strict, ~1,153 set-play-phase proxy) |

#### PFF counts (counted 2026-09-25)
| | All 64 games | First 51 with tracking |
|---|---|---|
| Shots (`SH`, shootout kicks excluded) | 1,518 | 1,179 |
| Open play, strict (`setpieceType = O`) | 1,450 | 1,125 |
| Open play, set-play-phase proxy (also drops shots ≤ 10 s after a same-team corner/free kick) | 1,153 | 890 |
| Goals | 172 | 129 |
| of which open-play shots | 149 | 113 |
| Shots per match, min / median / max | 9 / 23 / 41 | 9 / 22 / 41 |

How they were identified (event files, one row per possession event):
- **Shot:** `possessionEvents.possessionEventType = "SH"`.
- **Set piece:** `gameEvents.setpieceType` of the game event the shot belongs to: `O` open, `P` penalty, `F` free kick, `C` corner, `T` throw-in, `G` goal kick, `K` kickoff, `D` drop ball. Shots by type: 1,450 `O`, 44 `F`, 24 `P` (in-match), 0 `C`. A header from a corner cross is its own game event coded `O`, so "strict" keeps it. SkillCorner's 409 drops the whole set-play phase; the proxy row is the closer comparison (unverified: 10 s window is arbitrary, throw-ins not excluded). The window is measured between the game events' `startTime`s; using the shot row's `eventTime` lets 5 more shots through (1,158 / 894), so the counts above depend on it.
- **Shootout:** a `P` shot in period 4 with no kickoff after it. 41 kicks, 26 scored; matches the five real shootouts. The shootout starts at the period-4 `END` event. PFF doesn't track shootouts: tracking ends at that `END` (at most one frame after it), so period 5 is empty or a single dead frame.
- **Goal:** `shotOutcomeType = "G"` on an `SH` row whose next restart is a kickoff (or period end). 168 shot goals (149 open play, 2 free kick, 17 penalties) + 4 goals not coded as shots (1 `CR`, Bruno Fernandes vs Uruguay; 3 `RE`, the 2 own goals plus Costa Rica's 2nd vs Germany) = **172**, matching the official total. Rebuilt scores match the real results for all 64 games.
- `shotOutcomeType = "G"` followed by a free-kick restart = **disallowed goal** (22, e.g. 3 in Argentina–Saudi Arabia). Don't label these as goals.
- Sabiri's goal (Belgium–Morocco) is coded twice, as `CR` + `SH`; count the `SH` only. PFF credits that `SH` to Romain Saïss, so the goal event carries his `player_id`. The parser's rule is general: a non-shot `G` is skipped when an `SH` `G` comes before the same restart.
- One more disallowed goal is coded on a cross, not a shot: Ziyech's free kick in the same game (3837, `CR` `G` then a free-kick restart). It's not a shot event, so the 22 above stand; the converter logs it (`disallowed_non_shot_goals`).
- Scoring team = the team *not* taking the next kickoff (the shooter's team when the period ends instead). The 3 `RE` goals are coded on a player from the conceding team, so the converter writes them as `outcome = own_goal`, `player_id` null. That includes Costa Rica's 2nd vs Germany (3855, coded on Neuer's touch), which officially isn't an own goal: 3 `own_goal` rows, not 2. The team and the score are right either way.
- Every other `G` is followed by a kickoff, a free kick or a period end; nothing is left unresolved.
- `shotOutcomeType` → `outcome`: `G` goal, `S` saved, `B` blocked, `O` off_target. `C` (53) → blocked, `F` (17) → woodwork, `L` (11) → cleared_off_line follow PFF's naming and haven't been checked on video.
- 3 in-match shots have an empty `ball` in the event row (3840 ×1, 3845 ×2); their location is the shooter's position in the row's player snapshot (logged).
- Event rows are in sequence order, not strictly time order: ~100 rows (mostly `FOUL`) sit up to a few seconds early. Restarts are found by row order.

### PFF tracking format (checked 2026-09-25 on 10502, 10504, 3814, 3850)
The v2.2 spec PDF in `docs/` is corrupt (binary bytes replaced by `EF BF BD`, pages render blank); re-download it in binary. Everything below comes from the files and kloppy's source, not the spec.

| Item | Finding |
|---|---|
| Frame rate | 29.97 Hz (`fps` in Metadata; frames 33.4 ms apart). `frameNum = round(videoTime_s × 29.97)`. A few hundred duplicate frames per game (same `frameNum`, dt = 0), identical copies; dedupe. |
| Players per frame | Always 11 per side (12–13 briefly around subs), incl. off-camera players. Keys: `jerseyNum, x, y, visibility, confidence`. No speed. |
| Visibility | `visibility` = `VISIBLE` or `ESTIMATED`; `confidence` = HIGH / MEDIUM / LOW. ~31–36% of player rows are VISIBLE overall. On in-play frames that show anything, median 10–18 of 22 players visible. |
| ESTIMATED quality | Poor. At 76 shots, the shooter's tracked position is 0.0 m (median) from the event's position when VISIBLE, but 12 m from the ball (median, p90 24 m) when ESTIMATED. The shooter was ESTIMATED in 32% of shot frames. |
| Cutaway frames | 26–35% of all frames have an empty `balls` list and every player ESTIMATED (replays, close-ups). On in-play frames, 6–14% have zero visible players. |
| Raw vs smoothed | Each frame has raw (`homePlayers`, `balls`) and smoothed (`homePlayersSmoothed`, `ballsSmoothed`) copies. Smoothed ball has x/y null on ~8–14% more frames than raw. |
| Camera footprint | None. No field like SkillCorner's `image_corners_projection`. |
| Ball | 3D: `x, y, z` (z always present when the ball is), plus `visibility`. No ball detected = empty `balls` list. |
| Coordinates | Meters, origin at the center spot, pitch 105 × 68. Directions are **not** normalized: teams swap ends each period. Home attacks +x in period 1 when `homeTeamStartLeft = true` (home GK at x ≈ −41 in P1, +42 in P2). kloppy defines +y as bottom-to-top; confirmed from left/right-sided players on all 51 games (see converter results). |
| Event sync | Tracking frames carry `game_event_id` / `possession_event_id` and an inline `game_event` (with `start_frame`, `end_frame`, `home_ball`) on the frames it spans. Event `eventTime` (s) × 1000 = frame `videoTimeMs`. Checked on 76 shots: tracking ball is 0.18 m from the event's ball (median, p90 1.4 m). |
| Ball state / possession | No per-frame flag. Derivable from the inline `game_event`: `OUT`/`END` → dead, `OTB`/kickoffs → alive; possession team from `home_ball`. This is what kloppy does (10502: 65% alive). Fouls without an `OUT` event don't set dead. The OTB spans are *not* a ball carrier (see converter results). |
| Player IDs | Tracking uses jersey number only. (team, `shirtNumber`) → `player.id` via Rosters is unique in every game checked, and all roster IDs are in `players.csv`. Event rows carry `playerId` directly. |
| Event snapshots | Every event row also has its own 22-player + ball snapshot (`homePlayers` with `playerId`, `visibility`, `speed`). Useful for shot-time xG features. |

### PFF converter results (all 51 tracked games, 2026-09-26)
`converters/pff.py` + `converters/pff_events.py`; every game passes the validator with nothing unresolved.
- **Cost:** ~17 s and 2.4 GB peak per game on the M1; 173k–198k frames and 3.9M–4.5M object rows per game (objects.parquet ~44 MB).
- **Duplicates:** 33,336 duplicate `frameNum`s over 51 games, every one an identical copy of the previous frame. No `frameNum` gaps after dedupe, `periodElapsedTime` never goes backwards, no null positions, never more than one ball.
- **Direction:** 8 games have `homeTeamStartLeft = false` and are rotated 180°. The home keeper is on the expected side in periods 1 and 2 in all 51. **+y confirmed:** in every team-period, left-sided players (LB/LWB/LW/LM/LCB) have a larger mean y than right-sided ones when the team attacks +x (margin 1.7–31 m), so kloppy's claim holds.
- **Shot vs tracked ball:** 1,176 of 1,179 shots have a tracked ball; median 0.0 m, p90 0.44 m, per-game median ≤ 0.67 m. 80% are exactly 0 m, so the event's ball position is mostly copied from tracking. This check catches frame or direction mistakes, not independent disagreement. **Doesn't reproduce the 0.18 m above:** on the same four games (10502, 10504, 3814, 3850), all 104 shots give median 0.0 m, p90 1.0 m. Which 76 shots and which ball copy (raw or smoothed) the earlier check used isn't recorded.
- **Ball far off the pitch:** 845 ball rows in 25 games are more than 15 m off the pitch (ESTIMATED balls in the stands). They're dropped and logged, because the validator treats them as pixel errors.
- **Speed:** raw positions jitter. Player speed p99 is 6.9–12.4 m/s (median 8.3); 5k–42k rows per game are over 12 m/s, from VISIBLE as well as ESTIMATED rows. They're reported and kept as-is; smoothing is left to feature code.
- **Visibility:** two definitions. Every player ESTIMATED (the converter's check): 29–54% of frames, median 41%. Cutaway as in the table above (no ball *and* no visible player): 21–47%, median 32%. That's wider than the 26–35% seen on the first 4 games. Ball alive on 48–72% of frames.
- **Ball carrier: not in the data; `ball_carrier_id` stays null.** Each frame of an `OTB` span carries that event's `shirt_number` + `home_ball`, and that player looks like a carrier at first: 0.9–1.4 m from the ball (per-game median, VISIBLE player) over 51 games, always on the `possession_team` side. But the span (its own `start_frame`..`end_frame`) runs until PFF codes the next event: through the pass flight and through touches PFF didn't code. In 13 games the p90 is 5–24 m. In 10512, center backs Pepe and Dias stay "on the ball" for 5–7 s while another Portugal player is 0.5 m from it. It's "last player PFF coded on the ball", not control, so 02's null-not-a-guess rule applies. It can still help score 03 stage 8's inferred carrier.
- `view_polygon` is null.

### Extra time and the last 13 files (added 2026-09-26)
All 64 tracking files are now on disk; the second 13 all convert and validate. Five of them went to extra time, and only one has usable extra-time tracking:
| Game | Extra-time tracking | What the converter does |
|---|---|---|
| 10508 Morocco–Spain | Fine: coded player 1.1–1.2 m from the ball, keeper on the side the metadata says | Kept |
| 10510 Croatia–Brazil, 10511 Netherlands–Argentina | **None**: the tracking file ends at the end of period 2 | Extra-time events dropped as "outside tracked time" (17 shots, incl. 10510's 2 goals, so its events score 0–0) |
| 10506 Japan–Croatia, 10517 Argentina–France | Present but **doesn't line up with the events**: the player PFF codes on the ball is 14–19 m from the tracked ball (1.1–1.4 m in regulation). The home keeper stands at the end his own team is shooting at, and the frame of Messi's 108' goal shows a France attack. No constant time offset within ±2 min fixes it | Periods 3–4 dropped whole (frames, objects, events), so they can't become unlabeled stretches. Events score 1–1 and 2–2 |
- **The alignment check** is independent of the event's own ball position (PFF samples that from tracking, hence the 0.0 m shot-to-ball). For each event row, it measures the coded `playerId`'s tracked position against the tracked ball, both VISIBLE, at `round(eventTime × 29.97)`. The converter runs it on every period of every game and drops any period with median > 5 m (`event_player_to_ball_m_by_period` in the report). Regulation is 1.1–1.5 m everywhere.
- `homeTeamStartLeftExtraTime` agreed with the keeper in 10508. In 10506/10517 the keeper contradicts it, but those periods are dropped anyway.
- **Events snap to a tracked frame only within 0.5 s.** Before this, 10510/10511's extra-time events were snapped onto the last regulation frames.
- Net: extra-time tracking with labels exists for one game (10508). The 172 goals and 64 scores from the event parser are unaffected; only the tracked `events.parquet` loses the dropped stretches.

### 10 Hz resample and labels (all 64 games, 2026-09-26)
`prediction/resample.py` (05, "Resampled frames and labels"). ~25 s for all 66 games on the M1, ~1.4 GB peak per game; `data/processed/summary.json` has the totals.
- **Rows:** 3,912,924 grid rows over 64 games (~61k per game), 88.7M object rows (~1.39M per game). No grid points skipped for gaps (there are none, see converter results), and every shot/goal event has a tracked frame.
- **Shots reconcile with the counts above.** The tracked `events.parquet` has 1,479 shots: 1,412 strict open play and **1,127 proxy open play**. The extra-time drops (10506, 10510, 10511, 10517) account for the rest exactly: 39 shots, 38 strict and 26 proxy, so 1,450 − 38 = 1,412 and 1,153 − 26 = 1,127. The labels use the proxy definition, so 1,127 is the shot count. 391 shot/goal rows aren't open play and only mask windows. 129 open-play goal rows (incl. the 3 own goals).
- **Eligible** (ball alive and possession set): 58.8% of rows, 47.8–72.1% per game (median 58.7%). 12.2% of eligible rows are all-ESTIMATED. Set-play windows mask only 7,808 eligible rows at H = 5 (6,171 at H = 3), ~20 per set-play event: most of a direct free kick's or penalty's window is dead ball already. Trainable rows at H = 5 (unmasked, not all-ESTIMATED): 2,014,791, 51% of all rows.
- **Positives:**

| H | Unmasked rows | Shot positives | Rate | Goal positives | Rate | Open-play shots with a positive frame |
|---|---|---|---|---|---|---|
| 5 s | 2,294,619 | 51,482 | 2.24% | 6,137 | 0.27% | 1,127 / 1,127 |
| 3 s | 2,296,256 | 31,820 | 1.39% | 3,745 | 0.16% | 1,121 / 1,127 |

  Of the H = 5 shot positives, 2,301 (4.5%) are all-ESTIMATED frames, which training drops. Counting only trainable positives (not all-ESTIMATED), 3 shots have none at H = 5 (3840 ×1, 3845 ×2: the whole lead-up is a cutaway) and 9 at H = 3 (those 3 plus the 6 below).
- **Shots with no positive frame:** none at H = 5. At H = 3, six (10505, 10506, 10510, 3829, 3851, and 3853's away goal at 208 s): PFF's `possession_team` only switches to the shooting team at the shot, so the whole 3 s window belongs to the other side. Listed per game in `resample_report.json`.
- **Flip check:** at the grid row covering each shot, with possession agreeing with the shooter, the ball's `x_att` is > 0 for every shot in 10502, 10504, 10505 and 3855. The one negative found (10508) is a real shot from the center circle 0.3 s into period 4.
- **Metrica** gets no eligible rows (null `ball_state` and `possession_team`), so no labels, as expected. Its goal rows have a null `set_piece`, so they'd count as not open play; that only matters once Metrica has possession.

### Loading PFF with kloppy (3.19.0, latest release)
- `kloppy.pff.load_tracking(meta_data, roster_meta_data, raw_data)` loads the June 2025 files without errors (~34 s per game, 176,818 frames for 10502). It fills `ball_state` and `ball_owning_team` as described above.
- It reads **smoothed** positions only and **drops `visibility` and `confidence`**. A converter that only uses kloppy loses the broadcast-view signal, so read `visibility` and the raw ball from the JSONL directly.
- No PFF event loader in any release. An open PR (PySport/kloppy#467, last updated 2025-05-25) predates the June 2025 format. Parse the event JSON ourselves; it's plain JSON.
- Drop events whose period isn't 1–4 before assigning shootouts to period 5. Only one exists: the first row of 3833, a `G` placeholder at time 0 with no team or player, before the real kickoff at 140 s. Counted in the converter report.

### Decisions
- **Primary development set: PFF (decided 2026-09-26).** With tracking for 51 games it has ~890 open-play shots on the comparable (set-play-phase) definition vs. SkillCorner's 409, about 2.2×; ~1,150 once all 64 are downloaded. That makes the temporal GNN a real target, not a stretch goal. It is broadcast-derived with a per-player visibility flag, so the broadcast-view training in 05 works directly. Costs: no camera footprint, ESTIMATED positions are poor (below), no kloppy event loader, and the World Cup is one short tournament of national teams.
- **Second grouped-CV pool: SkillCorner.** Different league, has a real `view_polygon`, and its extrapolated off-camera positions make it the place to run the full vs. broadcast-view comparison (07 #5); PFF's full view is mostly ESTIMATED guesses. Named players, and 04 uses its season aggregates under a same-season exception.
- **IDSSE is a clean held-out check.** Optical full-pitch tracking from a different league tests whether the model generalizes beyond one competition and one tracking method.
- **Metrica is for converter development only.** It has only 3 anonymized matches, which is too few to matter for training.
- Even with PFF, a fixed test split would waste matches. Use match-grouped cross-validation instead (see 07).

### Caveats
- SkillCorner dynamic events include SkillCorner's own model outputs (`xshot_*`, `xthreat`, `lead_to_shot`, ...). **Never use these as features.** They come from another model and some look into the future. `xshot_player_possession_*` can serve as an external benchmark to compare against.
- SkillCorner and PFF positions for off-camera players are extrapolated or estimated. Both mark them (`is_detected`, `visibility = ESTIMATED`); carry the flag into `visible`.
- Frame rates differ (10 / 25 / 29.97 Hz). Converters store the native rate; prediction resamples to 10 Hz (see 02).
- Don't build PFF player profiles from the 2022 World Cup's own event data. That's the same tournament, which breaks the previous-season rule in 04.
- Set-play shots are identifiable via `team_in_possession_phase_type = set_play`. 05 trains on open play, so usable positives are ~409, not 526.

### PFF release layout (as shared on Google Drive)
- `Event Data/{game_id}.json`, `Metadata/{game_id}.json`, `Rosters/{game_id}.json`: one file per game, small.
- `Tracking Data/{game_id}.jsonl.bz2`: 64 files, the bulk of the download (~40–65 MB each, keep compressed). All 64 on disk; 13 came from the second Drive part (added 2026-09-26).
- `players.csv`, `competitions.csv`.
- `PFF FC Change Log` (Google Doc): format changes. Latest noted: June 2025, which added `teamAttackingDirection`. Older dated subfolders in `Event Data` are earlier versions; use the top-level files.
- Store under `data/raw/pff/` keeping PFF's folder names. Don't put the Drive links in the repo; `data/README.md` points to the request form instead.

## Other sources
| Dataset | Type | Use in project |
|---|---|---|
| StatsBomb open data (incl. 360) | Events + freeze frames at shots, no continuous tracking. 360 for 300 men's matches (9 competitions, incl. World Cup 2022 = same matches as PFF) | xG training (has defender and GK positions at the shot), player profiles |
| Wyscout open events (Pappalardo et al. 2019) | Events only | Large shot sample for a location-only xG (via defcon CSV) |
| SoccerNet (GSR, tracking, action spotting) | Broadcast video + labels | Vision development and evaluation; video access requires signing their NDA |
| Roboflow Universe soccer datasets | Labeled images | Player, ball, pitch keypoint detection fine-tuning |
| USSF counterattack graphs | Pre-built graphs | GNN warm-up / sanity check |

## Storage
- Raw downloads in `data/raw/<source>/`, never modified.
- Converted game state in `data/gamestate/`.
- `data/` is gitignored; a `data/README.md` records download steps.

### Converter report
Every converter writes `data/gamestate/<match_id>/conversion_report.json` next to the Parquet files (not part of the schema; the validator ignores it):
- source file names and hashes, converter git commit
- rows in vs. rows out per table
- everything dropped or changed, with a count and a reason (e.g. duplicate frames, events outside periods 1–4, shootout frames moved to period 5, disallowed goals)
- anything unresolved (e.g. jersey numbers with no roster match)

Anything dropped without a line in the report is a bug.

## Open questions
- PFF terms of use: allowed for a public demo/write-up? Answer when access is granted.
- ~~PFF primary vs. SkillCorner primary?~~ PFF primary, SkillCorner second pool (Decisions).
- ~~Does PFF tracking include off-camera players, and how are they marked?~~ Yes, all 22 every frame; off-camera ones have `visibility = ESTIMATED` (see PFF tracking format).
- Exact shot counts for IDSSE (count once the converter exists).
- ~~Metrica shot counts?~~ Games 1–2: 24 + 24 shots (24 + 21 open play), goals 3–1 and 3–2. Game 1's away goal is an own goal coded as `BALL OUT, WOODWORK-GOAL`, not a shot; the converter takes the scoring team from who kicks off next.

## Sources
- SkillCorner: https://github.com/SkillCorner/opendata (dynamic events spec PDF linked from README)
- PFF: https://www.blog.fc.pff.com/blog/pff-fc-release-2022-world-cup-data, https://kloppy.pysport.org/user-guide/loading-data/pff/
- IDSSE: https://github.com/spoho-datascience/idsse-data, https://doi.org/10.1038/s41597-025-04505-y
- Metrica: https://github.com/metrica-sports/sample-data
