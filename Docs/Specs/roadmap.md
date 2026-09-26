# Roadmap

## Now (no code needed)
- [x] Request PFF World Cup 2022 access (granted 2026-09-25)
- [ ] Read PFF terms of use (blocks the public demo only, not development)
- [ ] Download the 13 missing PFF tracking files (second Drive part, list in 06)
- [ ] Re-download the PFF tracking spec PDF in binary (current copy is corrupt)
- [ ] Download SkillCorner, IDSSE, Metrica into `data/raw/` and add steps to `data/README.md`

## Phase 1 — Predictor on tracking data (M1)
Main dataset is PFF, SkillCorner is the second CV pool (06). Build in this order.

**Foundation**
- [x] Repo skeleton per 01; schema v0.3 validator (`gamestate/validate.py`, tests in `tests/`)
- [ ] Metrica → game state converter via kloppy (small, for getting the converter pattern right)
- [ ] PFF → game state converter: tracking from the raw JSONL (raw ball incl. `z`, `ESTIMATED` → `visible=False, interpolated=True`, `confidence` mapping), ball state + possession from the inline game events, jersey → `player_id` via Rosters, direction per period from `homeTeamStartLeft`, shootouts → period 5, frame dedupe, `conversion_report.json` (02, 06)
- [ ] PFF event parser → `events.parquet`: shots, goals (incl. own goals and non-shot goals, disallowed goals marked), `set_piece`, `set_play_phase`; check it reproduces the 06 counts and all 64 scores
- [ ] SkillCorner → game state converter: tracking, possession, `visible` / `interpolated` from `is_detected`, `view_polygon`, `ball_state` from dynamic events (02)
- [ ] IDSSE → game state converter (external test set; don't look at results until the final check)
- [ ] 10 Hz resampling + label generation: `shot_within_H`, open play only (set plays excluded, 05)
- [ ] Grouped folds over PFF + SkillCorner → `data/splits/folds.json` (07); provisional until all 64 PFF games are in

**Evaluation harness**
- [ ] Metrics: PR-AUC, ROC-AUC, Brier + calibration, alarms / lead time / misses / false alarms (07)
- [ ] `evaluation/report.md` generator: pooled out-of-fold (PFF and SkillCorner rows), per-fold, IDSSE section

**First models**
- [ ] Distance + angle floor (logistic regression)
- [ ] LightGBM baseline on hand features
- [ ] Full vs. broadcast-view training comparison on SkillCorner folds (05, 07 #5)
- [ ] xG model on StatsBomb 360 without World Cup 2022 (features known before the shot only); Wyscout location-only xG as a check
- [ ] xG calibration check on PFF shots (129 goals in tracked games) and SkillCorner shots (61 goals)
- [ ] P(goal) v1: P(shot) × xG at carrier position

**Possession inference (03 stage 8, runs on dataset tracking)**
- [ ] Ball carrier / possession / ball state rules on pitch coordinates
- [ ] Check against PFF, SkillCorner and IDSSE provider values (≥ 90% targets)
- [ ] Provider vs. inferred possession comparison (07 #6)

**GNNs**
- [ ] Frame GNN with unravelsports
- [ ] Temporal GNN
- [ ] Node-level "who will shoot" head → P(goal) v2

**Player profiles**
- [ ] PFF: profiles from earlier-season StatsBomb data (never World Cup 2022); measure coverage (04)
- [ ] SkillCorner: profiles from season aggregates (trait columns only, `count_match` ≥ 5, 04)
- [ ] Three-arm ablation: none vs. position only vs. full, plus the `count_match` leakage check on SkillCorner. PFF results are the clean ones; SkillCorner results stay provisional

**Wrap-up**
- [ ] Final models on IDSSE, once

## Phase 2 — Vision pipeline (RTX 2060)
- [ ] Run roboflow/sports end to end on a SoccerNet sample clip
- [ ] Homography → pitch meters → game state writer (incl. `match.parquet`, `view_polygon`)
- [ ] Detections cache + `run.json` (03 Diagnostics)
- [ ] Ball tracking improvements + interpolation
- [ ] Plug in stage 8 (possession / ball state) from Phase 1
- [ ] Jersey OCR → player_id
- [ ] Evaluate on SoccerNet-GSR clips

## Phase 3 — End to end + demo
- [ ] Feed vision game state into trained predictor
- [ ] Compare predictor accuracy on vision vs. dataset tracking
- [ ] Offline overlay renderer, with debug mode (08)
- [ ] Live mode on workstation
- [ ] Record a local match for the public demo (see 08)
- [ ] Fine-tune detection/keypoints on self-recorded footage
- [ ] Demo clip + write-up
