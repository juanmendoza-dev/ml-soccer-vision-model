# 00 — Overview

## Goal
Predict, a few seconds in advance, when a goal is likely in a soccer match, using video of the match plus a profile of each player on the pitch.

## Core framing
Goals are too rare (~2–3 per match) to predict directly. Instead:

**P(goal in next H seconds) = P(shot in next H seconds) × xG of that shot**

- Shots happen ~25 times per match, giving far more training signal.
- xG (expected goals) is a well-studied problem with open training data.
- The output is a continuous "danger meter" that rises as an attack develops, not a yes/no.

## Scope
**In:** broadcast video → player/ball positions → per-frame goal probability → video overlay.
**Out (for now):** set-piece-specific models, betting/odds use, multi-camera setups.

## Success criteria
1. Shot predictor beats a distance + angle floor and a LightGBM hand-feature baseline in grouped cross-validation by match (see 05, 07).
2. Predictions are calibrated (a 30% prediction is right ~30% of the time).
3. Median lead time of at least 2 seconds before a shot at a useful threshold.
4. End-to-end demo: a video clip with a live probability bar overlaid.

## Spec index
| File | Covers |
|---|---|
| 01-architecture.md | Pipeline and hardware split |
| 02-game-state-schema.md | Data contract between vision and prediction |
| 03-vision-pipeline.md | Video → positions |
| 04-player-profiles.md | Player features |
| 05-prediction-model.md | Positions → probability |
| 06-data-sources.md | Datasets |
| 07-evaluation.md | Metrics and testing |
| 08-demo-overlay.md | Visual output |
| 09-hardware.md | Workstation specs, live feasibility |
| roadmap.md | Phases and tasks |
| references.md | Inspiration projects and papers |
