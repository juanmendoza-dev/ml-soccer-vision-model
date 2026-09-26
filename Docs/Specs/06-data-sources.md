# 06 — Data Sources

Check each license before use and keep attribution where required.

## Tracking + events (shot labels)
The shot predictor needs continuous tracking **and** shot events on the same frames. Only these sources have both (researched 2026-09-25).

| Dataset | Matches | Tracking | Shot events | Players | License / access | Loader |
|---|---|---|---|---|---|---|
| SkillCorner open data (A-League 2024/25) | 20 | 10 Hz, broadcast-derived, off-camera players extrapolated | Yes: `dynamic_events.csv`, `player_possession` rows with `end_type = shot` and `frame_start`/`frame_end`. **526 shots** counted | Named, with IDs and jersey numbers | MIT, on GitHub (tracking via git-lfs) | kloppy `skillcorner` (verify it reads dynamic events) |
| PFF FC World Cup 2022 | 64 | 29.97 Hz, broadcast-derived and manually refined, 3D ball | Yes: synchronized event data | Rosters with names | Free, by request form; terms unknown until access is granted | kloppy `pff` |
| IDSSE (Bassek et al. 2025, Bundesliga 1 + 2, 2022/23) | 7 | 25 Hz, TRACAB optical, full pitch | Yes: official DFL events | Named | CC-BY 4.0, attribute DFL + cite paper | kloppy `sportec.load_open_tracking_data` |
| Metrica Sports sample data | 3 | 25 Hz, full pitch | Yes: synchronized events | Anonymized | No formal license; acknowledge source | kloppy `metrica` |

### Shot volume
| Scenario | Matches | Shots |
|---|---|---|
| SkillCorner only | 20 | 526 (counted) |
| + IDSSE + Metrica | 30 | ~800 (estimated at ~25/match) |
| + PFF | 94 | ~2,400 (estimated) |

### Decisions
- **Primary development set: SkillCorner.** It has the most usable open shots available now. It is broadcast-derived like our vision output, so the domain gap is smaller. It also has real player IDs, which unblocks player profiles (04).
- **Request PFF access now.** It is the only way to get into the thousands of shots. Without it, treat the temporal GNN as a stretch goal and the LightGBM baseline as the main deliverable.
- **IDSSE is a clean held-out check.** Optical full-pitch tracking from a different league tests whether the model generalizes beyond one competition and one tracking method.
- **Metrica is for converter development only.** It has only 3 anonymized matches, which is too few to matter for training.
- ~500 shots over 20 matches means a fixed test split would be ~3 matches. Use match-grouped cross-validation instead (see 07).

### Caveats
- SkillCorner dynamic events include SkillCorner's own model outputs (`xshot_*`, `xthreat`, `lead_to_shot`, ...). **Never use these as features.** They come from another model and some look into the future. `xshot_player_possession_*` can serve as an external benchmark to compare against.
- SkillCorner and PFF positions for off-camera players are extrapolated or estimated. Track which rows are like this if the source marks them.
- Frame rates differ (10 / 25 / 29.97 Hz). Every converter resamples to 10 Hz (see 02).

## Other sources
| Dataset | Type | Use in project |
|---|---|---|
| StatsBomb open data (incl. 360) | Events + freeze frames at shots, no continuous tracking | xG training (has defender and GK positions at the shot), player profiles |
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
- Does PFF tracking include off-camera players, and how are they marked?
- Exact shot counts for IDSSE and Metrica (count once converters exist).

## Sources
- SkillCorner: https://github.com/SkillCorner/opendata (dynamic events spec PDF linked from README)
- PFF: https://www.blog.fc.pff.com/blog/pff-fc-release-2022-world-cup-data, https://kloppy.pysport.org/user-guide/loading-data/pff/
- IDSSE: https://github.com/spoho-datascience/idsse-data, https://doi.org/10.1038/s41597-025-04505-y
- Metrica: https://github.com/metrica-sports/sample-data
