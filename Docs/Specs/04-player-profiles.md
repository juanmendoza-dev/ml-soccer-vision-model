# 04 — Player Profiles

## Goal
Give each player a feature vector that tells the model how dangerous they are, so a striker in space counts for more than a center back in the same spot.

## Static features (per player, per season)
From event data (StatsBomb open data preferred so features are reproducible):
- Shots per 90, xG per 90, goals minus xG (finishing)
- Dribble success rate, progressive carries per 90
- Key passes per 90
- Position (one-hot), preferred foot
- Top speed (from tracking data, where available)

Unknown players get a **position-average profile** plus a flag `profile_known=False`.

## Dynamic features (per player, per frame)
From game state:
- Speed, acceleration
- Distance covered so far this match (fatigue proxy); sprint count
- Distance to nearest opponent
- Distance and angle to the goal they attack

## Identity linking
`player_id` from jersey OCR (03) or directly from tracking datasets. Without a `player_id`, use only dynamic features and the position-average profile.

## Outputs
`profiles.parquet` keyed by `player_id` + season; dynamic features computed on the fly in the graph builder (05).

## Acceptance criteria
- Ablation: model with profiles vs. without, on the same split (see 07). Profiles are kept only if they help.

## Open questions
- Which season's stats to use for a given match (previous season only, to avoid leakage from future matches)?
