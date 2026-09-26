# Roadmap

## Now (no code needed)
- [ ] Request PFF World Cup 2022 access (form in 06); read their terms when granted

## Phase 1 — Predictor on tracking data (M1)
Main dataset is SkillCorner (06). Build in this order.

**Foundation**
- [ ] Repo skeleton per 01; schema v0.2 validator in `tests/`
- [ ] Metrica → game state converter via kloppy (small, for getting the converter pattern right)
- [ ] SkillCorner → game state converter: tracking, possession, `visible` from `is_detected`, `view_polygon`, `ball_state` from dynamic events (02)
- [ ] IDSSE → game state converter (external test set; don't look at results until the final check)
- [ ] 10 Hz resampling + label generation: `shot_within_H`, open play only (set plays excluded, 05)
- [ ] Fixed grouped folds → `data/splits/skillcorner_folds.json` (07)

**Evaluation harness**
- [ ] Metrics: PR-AUC, ROC-AUC, Brier + calibration, alarms / lead time / misses / false alarms (07)
- [ ] `evaluation/report.md` generator: pooled out-of-fold, per-fold, IDSSE section

**First models**
- [ ] Distance + angle floor (logistic regression)
- [ ] LightGBM baseline on hand features
- [ ] Full vs. broadcast-view training comparison (05, 07 #5)
- [ ] xG model on StatsBomb 360 (features known before the shot only); Wyscout location-only xG as a check
- [ ] xG calibration check on SkillCorner shots (61 goals)
- [ ] P(goal) v1: P(shot) × xG at carrier position

**Possession inference (03 stage 8, runs on dataset tracking)**
- [ ] Ball carrier / possession / ball state rules on pitch coordinates
- [ ] Check against SkillCorner and IDSSE provider values (≥ 90% targets)
- [ ] Provider vs. inferred possession comparison (07 #6)

**GNNs**
- [ ] Frame GNN with unravelsports
- [ ] Temporal GNN (stretch goal unless PFF data arrives)
- [ ] Node-level "who will shoot" head → P(goal) v2

**Player profiles**
- [ ] Profiles from SkillCorner season aggregates (trait columns only, `count_match` ≥ 5, 04)
- [ ] Three-arm ablation: none vs. position only vs. full, plus the `count_match` leakage check

**Wrap-up**
- [ ] Final models on IDSSE, once

**If PFF access comes through**
- [ ] PFF → game state converter; add to the grouped-CV pool
- [ ] Drop StatsBomb World Cup 2022 from xG training
- [ ] Profiles from earlier-season StatsBomb data; measure coverage
- [ ] Re-run the profile ablation with clean profiles (turns provisional results into final ones)

## Phase 2 — Vision pipeline (RTX 2060)
- [ ] Run roboflow/sports end to end on a SoccerNet sample clip
- [ ] Homography → pitch meters → game state writer (incl. `match.parquet`, `view_polygon`)
- [ ] Ball tracking improvements + interpolation
- [ ] Plug in stage 8 (possession / ball state) from Phase 1
- [ ] Jersey OCR → player_id
- [ ] Evaluate on SoccerNet-GSR clips

## Phase 3 — End to end + demo
- [ ] Feed vision game state into trained predictor
- [ ] Compare predictor accuracy on vision vs. dataset tracking
- [ ] Offline overlay renderer
- [ ] Live mode on workstation
- [ ] Record a local match for the public demo (see 08)
- [ ] Fine-tune detection/keypoints on self-recorded footage
- [ ] Demo clip + write-up
