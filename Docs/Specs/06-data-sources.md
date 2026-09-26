# 06 — Data Sources

Check each license before use and keep attribution where required.

## Tracking + events (shot labels)
The shot predictor needs continuous tracking **and** shot events on the same frames. Only these sources have both (researched 2026-09-25).

| Dataset | Matches | Tracking | Shot events | Players | License / access | Loader |
|---|---|---|---|---|---|---|
| SkillCorner open data (A-League 2024/25) | 20 | 10 Hz, broadcast-derived, off-camera players extrapolated | Yes: `dynamic_events.csv`, `player_possession` rows with `end_type = shot` and `frame_start`/`frame_end`. **526 shots** counted, 409 outside set plays; 61 followed by `game_interruption_after = goal_for` | Named, with IDs and jersey numbers | MIT, on GitHub (tracking via git-lfs, ~90 MB/match; all 20 confirmed downloadable) | kloppy `skillcorner` (verify it reads dynamic events) |
| PFF FC World Cup 2022 | 64 (tracking on disk for 51) | 29.97 Hz, broadcast-derived, all 22 players every frame with a per-player `visibility` flag (VISIBLE / ESTIMATED), 3D ball | Yes: per-game event JSON, `possessionEventType = SH`. **1,518 shots** counted (shootouts excluded), 1,450 not from a penalty/free kick/corner; 172 goals | Rosters with names and IDs; tracking uses jersey numbers | Free, by request form. **Access granted 2026-09-25**; terms of use still to confirm | kloppy `pff.load_tracking` only (drops visibility, see below); no event loader |
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
| | All 64 games | 51 with tracking |
|---|---|---|
| Shots (`SH`, shootout kicks excluded) | 1,518 | 1,179 |
| Open play, strict (`setpieceType = O`) | 1,450 | 1,125 |
| Open play, set-play-phase proxy (also drops shots ≤ 10 s after a same-team corner/free kick) | 1,153 | 890 |
| Goals | 172 | 129 |
| of which open-play shots | 149 | 113 |
| Shots per match, min / median / max | 9 / 23 / 41 | 9 / 22 / 41 |

How they were identified (event files, one row per possession event):
- **Shot:** `possessionEvents.possessionEventType = "SH"`.
- **Set piece:** `gameEvents.setpieceType` of the game event the shot belongs to: `O` open, `P` penalty, `F` free kick, `C` corner, `T` throw-in, `G` goal kick, `K` kickoff, `D` drop ball. Shots by type: 1,450 `O`, 44 `F`, 24 `P` (in-match), 0 `C`. A header from a corner cross is its own game event coded `O`, so "strict" keeps it. SkillCorner's 409 drops the whole set-play phase; the proxy row is the closer comparison (unverified: 10 s window is arbitrary, throw-ins not excluded).
- **Shootout:** a `P` shot in period 4 with no kickoff after it. 41 kicks, 26 scored; matches the five real shootouts.
- **Goal:** `shotOutcomeType = "G"` on an `SH` row whose next restart is a kickoff (or period end). 168 shot goals (149 open play, 2 free kick, 17 penalties) + 4 goals not coded as shots (1 `CR`, Bruno Fernandes vs Uruguay; 3 `RE`, the 2 own goals plus Costa Rica's 2nd vs Germany) = **172**, matching the official total. Rebuilt scores match the real results for all 64 games.
- `shotOutcomeType = "G"` followed by a free-kick restart = **disallowed goal** (22, e.g. 3 in Argentina–Saudi Arabia). Don't label these as goals.
- Sabiri's goal (Belgium–Morocco) is coded twice, as `CR` + `SH`; count the `SH` only.

### PFF tracking format (checked 2026-09-25 on 10502, 10504, 3814, 3850)
The v2.2 spec PDF in `docs/` is corrupt (binary bytes replaced by `EF BF BD`, pages render blank); re-download it in binary. Everything below comes from the files and kloppy's source, not the spec.

| Item | Finding |
|---|---|
| Frame rate | 29.97 Hz (`fps` in Metadata; frames 33.4 ms apart). `frameNum = round(videoTime_s × 29.97)`. A few hundred duplicate frames per game (same `frameNum`, dt = 0); dedupe. |
| Players per frame | Always 11 per side (12–13 briefly around subs), incl. off-camera players. Keys: `jerseyNum, x, y, visibility, confidence`. No speed. |
| Visibility | `visibility` = `VISIBLE` or `ESTIMATED`; `confidence` = HIGH / MEDIUM / LOW. ~31–36% of player rows are VISIBLE overall. On in-play frames that show anything, median 10–18 of 22 players visible. |
| ESTIMATED quality | Poor. At 76 shots, the shooter's tracked position is 0.0 m (median) from the event's position when VISIBLE, but 12 m from the ball (median, p90 24 m) when ESTIMATED. The shooter was ESTIMATED in 32% of shot frames. |
| Cutaway frames | 26–35% of all frames have an empty `balls` list and every player ESTIMATED (replays, close-ups). On in-play frames, 6–14% have zero visible players. |
| Raw vs smoothed | Each frame has raw (`homePlayers`, `balls`) and smoothed (`homePlayersSmoothed`, `ballsSmoothed`) copies. Smoothed ball has x/y null on ~8–14% more frames than raw. |
| Camera footprint | None. No field like SkillCorner's `image_corners_projection`. |
| Ball | 3D: `x, y, z` (z always present when the ball is), plus `visibility`. No ball detected = empty `balls` list. |
| Coordinates | Meters, origin at the center spot, pitch 105 × 68. Directions are **not** normalized: teams swap ends each period. Home attacks +x in period 1 when `homeTeamStartLeft = true` (home GK at x ≈ −41 in P1, +42 in P2). kloppy defines +y as bottom-to-top (not checked against the spec). |
| Event sync | Tracking frames carry `game_event_id` / `possession_event_id` and an inline `game_event` (with `start_frame`, `end_frame`, `home_ball`) on the frames it spans. Event `eventTime` (s) × 1000 = frame `videoTimeMs`. Checked on 76 shots: tracking ball is 0.18 m from the event's ball (median, p90 1.4 m). |
| Ball state / possession | No per-frame flag. Derivable from the inline `game_event`: `OUT`/`END` → dead, `OTB`/kickoffs → alive; possession team from `home_ball`. This is what kloppy does (10502: 65% alive). Fouls without an `OUT` event don't set dead. |
| Player IDs | Tracking uses jersey number only. (team, `shirtNumber`) → `player.id` via Rosters is unique in every game checked, and all roster IDs are in `players.csv`. Event rows carry `playerId` directly. |
| Event snapshots | Every event row also has its own 22-player + ball snapshot (`homePlayers` with `playerId`, `visibility`, `speed`). Useful for shot-time xG features. |

### Loading PFF with kloppy (3.19.0, latest release)
- `kloppy.pff.load_tracking(meta_data, roster_meta_data, raw_data)` loads the June 2025 files without errors (~34 s per game, 176,818 frames for 10502). It fills `ball_state` and `ball_owning_team` as described above.
- It reads **smoothed** positions only and **drops `visibility` and `confidence`**. A converter that only uses kloppy loses the broadcast-view signal, so read `visibility` and the raw ball from the JSONL directly.
- No PFF event loader in any release. An open PR (PySport/kloppy#467, last updated 2025-05-25) predates the June 2025 format. Parse the event JSON ourselves; it's plain JSON.

### Decisions
- **Primary development set: SkillCorner.** It has the most usable open shots available now. It is broadcast-derived like our vision output, so the domain gap is smaller. It has named players, so identity is solved, but a stats source for A-League players is still open: StatsBomb open data doesn't cover them. SkillCorner's season aggregates (physical, off-ball runs, passing) cover all 406 players for the full season; 04 uses them under a same-season exception with guardrails.
- **Request PFF access now.** It is the only way to get into the thousands of shots. Without it, treat the temporal GNN as a stretch goal and the LightGBM baseline as the main deliverable.
- **IDSSE is a clean held-out check.** Optical full-pitch tracking from a different league tests whether the model generalizes beyond one competition and one tracking method.
- **Metrica is for converter development only.** It has only 3 anonymized matches, which is too few to matter for training.
- ~500 shots over 20 matches means a fixed test split would be ~3 matches. Use match-grouped cross-validation instead (see 07).
- **Proposed (not confirmed): make PFF the primary development set, SkillCorner second.** With tracking for 51 games PFF has ~890 open-play shots on the comparable (set-play-phase) definition vs. SkillCorner's 409, about 2.2×; ~1,150 once all 64 are downloaded. It is broadcast-derived with a real per-player visibility flag, so the broadcast-view training in 05 works directly. Costs: no camera footprint, ESTIMATED positions are much worse than SkillCorner's extrapolation looks (worth checking), no kloppy event loader, and World Cup games are one short tournament of national teams. SkillCorner stays useful as a second grouped-CV pool (different league, has `view_polygon`); IDSSE stays the external test.

### Caveats
- SkillCorner dynamic events include SkillCorner's own model outputs (`xshot_*`, `xthreat`, `lead_to_shot`, ...). **Never use these as features.** They come from another model and some look into the future. `xshot_player_possession_*` can serve as an external benchmark to compare against.
- SkillCorner and PFF positions for off-camera players are extrapolated or estimated. Both mark them (`is_detected`, `visibility = ESTIMATED`); carry the flag into `visible`.
- Frame rates differ (10 / 25 / 29.97 Hz). Converters store the native rate; prediction resamples to 10 Hz (see 02).
- Don't build PFF player profiles from the 2022 World Cup's own event data. That's the same tournament, which breaks the previous-season rule in 04.
- Set-play shots are identifiable via `team_in_possession_phase_type = set_play`. 05 trains on open play, so usable positives are ~409, not 526.

### PFF release layout (as shared on Google Drive)
- `Event Data/{game_id}.json`, `Metadata/{game_id}.json`, `Rosters/{game_id}.json`: one file per game, small.
- `Tracking Data/{game_id}.jsonl.bz2`: 64 files, the bulk of the download (~40 MB each, keep compressed). 51 on disk; missing 10503 10506 10507 10508 10510 10511 10517 3812 3813 3819 3827 3834 3848 (second Drive part).
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

## Open questions
- PFF terms of use: allowed for a public demo/write-up? Answer when access is granted.
- ~~Does PFF tracking include off-camera players, and how are they marked?~~ Yes, all 22 every frame; off-camera ones have `visibility = ESTIMATED` (see PFF tracking format).
- PFF primary vs. SkillCorner primary: see the Proposed note under Decisions.
- Exact shot counts for IDSSE and Metrica (count once converters exist).

## Sources
- SkillCorner: https://github.com/SkillCorner/opendata (dynamic events spec PDF linked from README)
- PFF: https://www.blog.fc.pff.com/blog/pff-fc-release-2022-world-cup-data, https://kloppy.pysport.org/user-guide/loading-data/pff/
- IDSSE: https://github.com/spoho-datascience/idsse-data, https://doi.org/10.1038/s41597-025-04505-y
- Metrica: https://github.com/metrica-sports/sample-data
