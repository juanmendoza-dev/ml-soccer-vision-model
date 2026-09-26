# Roadmap

## Phase 1 — Predictor on tracking data (M1)
- [ ] Repo skeleton per 01; schema validator in `tests/`
- [ ] Metrica → game state converter (via kloppy)
- [ ] Label generation: `shot_within_H`
- [ ] Baseline model + evaluation report
- [ ] xG model on Wyscout/StatsBomb shots
- [ ] SkillCorner converter; expand training data
- [ ] Frame GNN with unravelsports
- [ ] Temporal GNN
- [ ] Player profiles + ablation

## Phase 2 — Vision pipeline (RTX 2060)
- [ ] Run roboflow/sports end to end on a sample clip
- [ ] Homography → pitch meters → game state writer
- [ ] Ball tracking improvements + interpolation
- [ ] Jersey OCR → player_id
- [ ] Evaluate on SoccerNet-GSR clips

## Phase 3 — End to end + demo
- [ ] Feed vision game state into trained predictor
- [ ] Compare predictor accuracy on vision vs. dataset tracking
- [ ] Offline overlay renderer
- [ ] Live mode on workstation
- [ ] Demo clip + write-up
