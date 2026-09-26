# 04 — Player Profiles

## Goal
Give each player a feature vector that tells the model how dangerous they are, so a striker in space counts for more than a center back in the same spot.

## Leakage rule
A profile used for a match may only contain data from **before that match**. Previous seasons are always fine. Earlier matches in the same season are fine too. The one exception is below, and it must be tested.

## Static features: sources per dataset
No single open source covers every player in our data (researched 2026-09-25):

| Training data | Profile source | Status |
|---|---|---|
| SkillCorner (A-League 2024/25) | SkillCorner season aggregates (`data/aggregates/*.csv` in their repo): 406 players, all 13 teams, full season | **Same-season exception**, see below |
| PFF (World Cup 2022) | StatsBomb open data from earlier seasons (Euro 2020, La Liga 2020/21, Ligue 1 2021/22) | Partial coverage; measure it once PFF access arrives. Never the World Cup 2022 itself |
| IDSSE (Bundesliga 2022/23) | None open: StatsBomb has Bundesliga 2023/24 only, which is later | Position-average profiles only |
| Metrica | None: anonymized | Position-average profiles only |

### SkillCorner same-season exception
The aggregates cover the whole 2024/25 season, **including the match being predicted** and later matches. They're allowed only under these guardrails:
- **Trait columns only**, i.e. how a player moves and plays, not what happened:
  - peak speed (`psv99`), sprint and high-speed-running counts per 90
  - off-ball run rates by run type (e.g. `behindrun_count_p30tip`) and runs into the penalty area
  - passes attempted, line-breaking passes attempted and average pass distance per 30 min in possession
  - position group, age at match date
- **Banned columns:** anything outcome-based or model-based. That means every `*_shotwithin10s_*`, `*_goalwithin10s_*`, `*_dangerous_*`, `*_targeted_*`, `*_received_*` and `*xpass*` column. These are derived from shots/goals (our labels) or from SkillCorner's own models.
- **Minimum sample:** use a player's aggregate only if `count_match` ≥ 5, so a single match is at most ~20% of it. Players below that get the position-average profile with `profile_known=False`.
- **Leakage check (required):** in the ablation, split the gain from profiles by the player's `count_match`. If the gain mostly comes from low-count players, where the evaluated match makes up more of the aggregate, treat it as leakage and drop the exception.

### Not available anywhere yet
- Finishing quality (shots/90, xG/90, goals minus xG) for SkillCorner players: the aggregates have no shooting stats. It can only come from our own earlier matches, which is too sparse (teams appear in 1–7 of the 20 matches).
- Preferred foot: not in any open source for these players.

Players without a usable profile get a **position-average profile** plus `profile_known=False`.

## Dynamic features (per player, per frame)
From game state:
- Speed, acceleration
- Distance covered so far this match (fatigue proxy); sprint count
- Distance to nearest opponent
- Distance and angle to the goal they attack

## Identity linking
`player_id` from jersey OCR (03) or directly from tracking datasets. Without a `player_id`, use only dynamic features and the position-average profile.

## Outputs
`profiles.parquet` keyed by `player_id` + season, with a `profile_source` column (skillcorner_agg, statsbomb, position_avg); dynamic features computed on the fly in the graph builder (05).

## Acceptance criteria
- Ablation: model with profiles vs. without, on the same folds (see 07). Profiles are kept only if they help.
- The `count_match` leakage check above passes.

## Open questions
- How many PFF World Cup players have StatsBomb data from earlier seasons? If it's low, PFF runs use position-average profiles too.
- Is a profile ablation on SkillCorner alone (~400 shots) enough to show a real effect, or does it have to wait for PFF?
