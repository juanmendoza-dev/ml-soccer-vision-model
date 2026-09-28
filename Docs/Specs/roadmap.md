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
- [x] Metrics: PR-AUC, ROC-AUC, Brier + calibration, alarms / lead time / misses / false alarms, τ sweep (`evaluation/metrics.py`, 07). Label floor: an oracle still misses 2.4% of shots with 1.17 false alarms per match, from PFF possession flips before shots
- [x] τ selection rule: lowest miss rate at ≤ 3 false alarms per match on the inner split (07)
- [ ] Decide whether a short opposing possession ends an alarm (07, open)
- [x] Report generator (`evaluation/report.py`, run format `evaluation/runs.py`, 07): pooled out-of-fold per source, per fold with mean ± std, alarms at each fold's τ, calibration and τ sweep as tables, IDSSE only with `--final`. Checked on the oracle run
- [x] Paired comparison of two runs by fold (`python -m evaluation.compare A B`, 07: A beats B only if it wins on most folds)
- [x] Alarms end after 2 s with no prediction, so a long cutaway can't hold one (07; detection review D10). Oracle floor 61 → 75 false alarms at H = 5, floor results unchanged
- [x] Resampler staleness from the declared `native_fps`, not a whole-match median (05; detection review D8). Identical output on all 66 games

**First models**
- [x] Distance + angle floor (logistic regression, `prediction/floor.py` + CV driver `prediction/cv.py`): PR-AUC 0.167 at H = 5 (base rate 0.025), calibrated, no useful alarms at 3 false alarms per match (`Docs/reviews/floor-2026-09-27.md`)
- [x] Check whether PFF's ESTIMATED ball positions are interpolated with later frames (05, leakage; 25.6% of scored rows): **they are**. After ≥ 1 s gaps where the ball moved > 5 m, the estimate lands 0.5 m (median) from the next detection. Models use a causal held ball (`ball_source=held`) and grid velocities from visible positions only. The floor gets rerun on it, since its 0.167 was measured with the leaky ball
- [x] LightGBM baseline on hand features (`prediction/lgbm.py`, `Docs/reviews/lgbm-2026-09-27.md`): PR-AUC **0.296** at H = 5 vs 0.177 for the floor on the same held ball, better on 5/5 folds, calibrated. It alarms (174 of 1,154 shots at 2.5 false alarms per match), but the median lead is 0.7 s, short of 00's 2 s. Gain: ball position 0.54, carrier 0.19, defenders 0.13
- [ ] Lead time: features that see an attack building over 3–5 s, then the temporal GNN. Most misses never reach τ (941 of 980), so ranking early in the attack is the gap, not the alarm rules
- [x] Vision sensitivity test (`prediction/degrade.py`, `Docs/reviews/sensitivity-2026-09-28.md`, 34 runs): every arm at the detection review targets costs −0.070 PR-AUC (0.296 → 0.226), −0.086 if trained clean. Biggest losses: homography loss (17%: −0.018, 50%: −0.082) and ball misses (10%: −0.017). Unknown team costs less than a wrong one. ID fragmentation and ball height don't matter to this baseline (rerun on the temporal models)
- [ ] Train the model that runs on vision output with the degradations on (recovers ~0.016 of the combined loss, sensitivity review)
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
In priority order from the 2026-09-27 review (`Docs/reviews/vision-review-2026-09-27.md`; F-numbers refer to it). The sensitivity test (`Docs/reviews/sensitivity-2026-09-28.md`) sets the order of the quality work: homography availability and accuracy, then the ball, then an unknown team option, then player detection.

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
- [ ] **Tune the homography acceptance thresholds** (top priority from the sensitivity test): keep frames without geometry at or under smoke04's 17% (30% doubles the loss), and don't reject frames whose error is under ~2 m, since 17% rejected costs the same as 2 m drift everywhere
- [x] Run `vision.offpitch` on smoke01 to see what the 36 off-pitch rows were, then a second smoke run with the new acceptance: 36 off-pitch rows → 0, `homography_ok` 375/375 → 310/375 match frames, 75 projections nulled (`Docs/reviews/smoke-test-2026-09-27-followup.md`). Thresholds still untuned
- [x] Ball history: expire before a new detection uses it, reset when the ball has no pitch position, no extrapolation without valid geometry (03; detection review D1)
- [x] `visible=False` for filled boxes that drift fully off screen, instead of everything visible (03; detection review D6)
- [ ] Ball: reset on camera cuts, temporal candidate association instead of max confidence (F8, detection review D2/W2). The sensitivity test puts the ball second after geometry (misses −0.014 per 10% of time), so this is on. Targets: recall ≥ 90%, precision ≥ 95%
- [ ] Per-stage timings + effective model settings in `run.json` (03)
- [ ] Run roboflow/sports end to end on a SoccerNet sample clip as a reference
- [ ] Tune the stage 0 thresholds on broadcast clips with ads and studio cuts (`view.parquet`, 03)

**5. Small cleanups**
- [ ] 03: header still says v0.4 and omits `events.parquet`; points 31/32 aren't on the halfway line
- [ ] `--period-start-s` flag so `timestamp_s` is the period clock, one period per run (F6), plus explicit per-period attacking direction instead of odd/even periods, which is wrong in extra time (detection review D7)
- [ ] `demo.debug` renders only the processed frame range (a hand-typed `--frames` gave smoke04 a 2 s debug video of a 25 s run, see the follow-up review)

**Detection review follow-ups** (`Docs/reviews/detection-improvement-spec-2026-09-27.md`, taken in reduced form)
- [ ] Small vision benchmark: 5–10 labelled clips from different matches (ball, player positions in meters, teams, live vs. replay), scored per clip. Not the review's 30–50 clip set with double annotation; grow it only if results are borderline (W0)
- [ ] Camera cuts and replays detected separately from the grass/keypoint gate; reset tracks, teams and ball on a confirmed cut, emit nothing prediction-eligible during a replay (W3)
- [ ] Record why frames and projections were rejected, plus model settings, in the vision cache (W1, the parts that help debugging; not the full replayable cache yet)
- [ ] From the sensitivity test, in order: geometry accuracy where attacks happen (W4), teams/keepers with an "unknown" option (W5, cheaper than a wrong team at every level), then detector fine-tuning (W8), ball first. Player misses are the smallest measured loss (40% missed: −0.023). ID/tracking quality waits for a rerun on the temporal models
- Not adopted: goal/market (Polymarket) framing in the specs, six parallel lanes, bootstrap intervals and locked check sets before any labels exist. Before any trading design, measure broadcast delay against market data latency: a 2–5 s warning from a stream that runs 5–30 s behind live may arrive after the market moves

**6. Later**
- [ ] Move off `sv.ByteTrack` before supervision 0.31 (pinned below it)
- [ ] Bounded-memory writer before full-match runs
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
