# Roadmap

## Now (no code needed)
- [x] Request PFF World Cup 2022 access (granted 2026-09-25)
- [ ] Read PFF terms of use (blocks the public demo only, not development)
- [x] Download the 13 missing PFF tracking files (second Drive part; all 64 on disk)
- [ ] Re-download the PFF tracking spec PDF in binary (current copy is corrupt)
- [x] Download Metrica into `data/raw/` and add steps to `data/README.md`
- [ ] Same for SkillCorner, IDSSE

## Phase 1 — Predictor on tracking data (M1)
Main dataset is PFF, SkillCorner is the second CV pool (06). Build in this order.

**Foundation**
- [x] Repo skeleton per 01; schema v0.4 validator (`gamestate/validate.py`, tests in `tests/`)
- [x] Metrica → game state converter, games 1–2 (`converters/metrica.py`; tracking via kloppy, events CSV parsed directly). Game 3 not converted
- [x] PFF → game state converter (`converters/pff.py`, all 64 games; extra time usable only in 10508, see 06): tracking from the raw JSONL (raw ball incl. `z`, `ESTIMATED` → `visible=False, interpolated=True`, `confidence` mapping), ball state + possession from the inline game events, jersey → `player_id` via Rosters, direction per period from `homeTeamStartLeft`, shootouts → period 5, frame dedupe, `conversion_report.json` (02, 06)
- [x] PFF event parser → `events.parquet` (`converters/pff_events.py`, all 64 games): shots, goals (incl. own goals and non-shot goals, disallowed goals marked), `set_piece`, `set_play_phase`; reproduces the 06 counts and all 64 scores (tests). PFF doesn't track shootouts, so period 5 stays empty
- [ ] SkillCorner → game state converter: tracking, possession, `visible` / `interpolated` from `is_detected`, `view_polygon`, `ball_state` from dynamic events (02)
- [ ] IDSSE → game state converter (external test set; don't look at results until the final check)
- [x] 10 Hz resampling + label generation (`prediction/resample.py`, `prediction/labels.py`, all 64 PFF + Metrica 1–2): causal grid, flip to the attacking team, `shot`/`goal` within 3 s and 5 s, set-play windows masked (05). H = 5: 2.24% shot positives on 2.29M unmasked rows, eligible 58.8%, every one of the 1,127 open-play shots has a positive frame (6 miss at H = 3); numbers in 06
- [x] Set-play phases masked whether or not they end in a shot: schema 0.5 `frames.set_play_phase`, all 66 games reconverted (events unchanged), 7.6% of eligible rows masked (06)
- [x] Decide whether own-half free kicks start a set-play phase: no, only final-third free kicks and corners (02). Open-play shots 1,127 → 1,154, fold counts refreshed (06, 07)
- [x] Rerun the resampler under the final-third rule: 2.1% of eligible rows masked (was 7.6%), H = 5 shot rate 2.32%, all 1,154 shots have a positive frame (6 miss at H = 3); numbers in 06
- [x] Grouped folds → `data/splits/folds.json` (`evaluation/folds.py`, 07): PFF frozen 2026-09-26 (64 games, 206–244 open-play shots per fold), append-only, inner split helper. SkillCorner gets appended when its converter lands

**Evaluation harness**
- [x] Metrics: PR-AUC, ROC-AUC, Brier + calibration, alarms / lead time / misses / false alarms, τ sweep (`evaluation/metrics.py`, 07). Label floor: an oracle still misses 2.4% of shots with 0.95 false alarms per match, from PFF possession flips before shots
- [ ] Decide the τ selection rule and whether a short opposing possession ends an alarm (07, open)
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
In priority order from the 2026-09-27 review (`Docs/reviews/vision-review-2026-09-27.md`; F-numbers refer to it).

**Done**
- [x] Streaming `VisionPipeline` (03): stage 0 gate, detection + ByteTrack with frame skip, kit-color teams after warmup, homography from roboflow's 32 keypoints in 02 coords, ball extrapolation. Tested with fake stages on a synthetic match (output passes the 02 validator)
- [x] `vision.run` (video → game state + detections cache) and `demo.debug` renderer (08 debug mode)
- [x] Detections cache + `view.parquet` + `run.json` (03 Diagnostics)

**1. First real run (2060)**
- [x] **First workstation run:** `vision.run` + validator + `demo.debug` on a short demo clip with roboflow's weights. Ran end to end, validator fails on off-pitch rows (`Docs/reviews/smoke-test-2026-09-27.md`)
- [x] Check keypoint orientation on that clip (center spot, penalty spots land right; 03 Pitch template)
- [x] Pick `home_cluster` from the debug video; later a warmup prompt in live mode

**2. Bugs that give wrong output (M1)**
- [x] F2: view gate stays `match` on green close-ups (unsampled keypoint frames count as passing)
- [x] Tracker: lost-track buffer is double-scaled (1 s → 20 updates), and the 0.3 conf filter removes ByteTrack's low-conf second pass
- [x] F3: filled boxes between detections move too slow (motion measured from the last filled box)
- [x] F4: home/away can flip after a team refit (match new clusters to old ones)
- [x] F1: velocity window picked from the whole run's timestamps; use known fps so it's strictly causal

**3. Crashes and guardrails (M1)**
- [x] F5: one-frame run crashes in velocities; empty detections cache has no columns (debug renderer crashes)
- [x] F9: reject bad config (`detect_every=0`, bad fps/period)
- [x] Run the validator at the end of `close()`; an empty run is a failure, not a schema change
- [x] F7: clip `box_frac` to 0–1

**4. Quality (after real-clip results)**
- [x] Homography acceptance: inlier count, error threshold, no matrix averaging across camera motion, null positions > 10 m off the pitch (03 Homography acceptance). Code done, thresholds untuned
- [x] Run `vision.offpitch` on smoke01 to see what the 36 off-pitch rows were, then a second smoke run with the new acceptance: 36 off-pitch rows → 0, `homography_ok` 375/375 → 310/375 match frames, 75 projections nulled (`Docs/reviews/smoke-test-2026-09-27-followup.md`). Thresholds still untuned
- [ ] Ball: reset motion after long gaps and cuts, temporal candidate association, null when homography is invalid (F8)
- [ ] Per-stage timings + effective model settings in `run.json` (03)
- [ ] Run roboflow/sports end to end on a SoccerNet sample clip as a reference
- [ ] Tune the stage 0 thresholds on broadcast clips with ads and studio cuts (`view.parquet`, 03)

**5. Small cleanups**
- [ ] 03: header still says v0.4 and omits `events.parquet`; points 31/32 aren't on the halfway line
- [ ] `--period-start-s` flag so `timestamp_s` is the period clock, one period per run (F6)
- [ ] `demo.debug` renders only the processed frame range (a hand-typed `--frames` gave smoke04 a 2 s debug video of a 25 s run, see the follow-up review)

**6. Later**
- [ ] Move off `sv.ByteTrack` before supervision 0.31 (pinned below it)
- [ ] Bounded-memory writer before full-match runs
- [ ] Replay detection
- [ ] Plug in stage 8 (possession / ball state) from Phase 1
- [ ] Jersey OCR → player_id
- [ ] Evaluate on SoccerNet-GSR clips

Skipped for now: schema 0.7 for empty runs, shootout (period 5) handling in vision.

## Phase 3 — End to end + demo
- [ ] Feed vision game state into trained predictor
- [ ] Compare predictor accuracy on vision vs. dataset tracking
- [ ] Offline overlay renderer, with debug mode (08)
- [ ] Benchmark each vision stage on the 2060 (FP16 / TensorRT) and pick the live config (09)
- [ ] Live mode on workstation (in `soccer-live-overlay`, 09)
- [ ] Record a local match for the public demo (see 08)
- [ ] Fine-tune detection/keypoints on self-recorded footage
- [ ] Demo clip + write-up
