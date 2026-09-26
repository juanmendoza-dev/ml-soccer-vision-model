# 05 — Prediction Model

## Goal
At each frame t, output P(shot in (t, t+H]) and P(goal in (t, t+H]).

## Labels
- `shot_within_H`: 1 if the attacking team shoots within H seconds after t.
- Default **H = 5 s**; also evaluate H = 3 s.
- Only frames in open play with a team in possession. Exclude dead-ball frames.
- Attacking team = `possession_team`; flip coordinates so it always attacks +x.

## Leakage rules
- Inputs use frames `<= t` only.
- Split train/val/test **by match**.
- Report performance separately for lead times (see 07); a model that only fires 0.2 s before the shot is not useful.

## Models (build in order)
1. **Baseline:** gradient boosting (LightGBM) on hand features — ball distance/angle to goal, defenders in shooting cone, carrier speed, pitch control near the box.
2. **Frame GNN:** one graph per frame. Nodes = players + ball (+ goals); node features = position, velocity, team, dynamic + profile features (04); edges = all pairs or k-nearest, edge features = distance, relative velocity. Built with `unravelsports` SoccerGraphConverter.
3. **Temporal GNN:** last 2–3 s of frames (at 10 Hz) through a GNN backbone, then a GRU/T-GCN over time. Follows the SoccerAI approach.

## xG model
- Trained on Wyscout open shot data (preprocessed CSV from `defcon`) and/or StatsBomb 360 shots.
- Features: distance, angle, body part, defenders between ball and goal, goalkeeper position.
- P(goal) = P(shot) × xG(current ball carrier's position and context).

## Class imbalance
- Weighted loss or focal loss; evaluate with PR-AUC, not accuracy.

## Acceptance criteria
- Temporal GNN beats baseline on PR-AUC on the test matches.
- Calibration error acceptable after (optional) temperature scaling.

## Open questions
- Predict "which player will shoot" as a node-level head too?
- Add uncertainty (MC dropout / Bayesian head, as in Goka et al. 2023)?
