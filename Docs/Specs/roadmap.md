# Roadmap

## Now (no code needed)
- [x] Request PFF World Cup 2022 access (granted 2026-09-25)
- [ ] Read PFF terms of use (blocks the public demo only, not development)
- [x] Download the 13 missing PFF tracking files (second Drive part; all 64 on disk)
- [ ] Re-download the PFF tracking spec PDF in binary (current copy is corrupt)
- [x] Download Metrica into `data/raw/` and add steps to `data/README.md`
- [ ] Same for SkillCorner, IDSSE

## Next session: what's left from 2026-09-29
Code for all of this is written and pushed. What's left is running and checking it. Tick items here and in their sections below.

**M1 (verification, no GPU)**
- [x] Full suite on the M1 (2026-09-29): 490 passed in 2 min 58 s, 500 with the inferred possession tests (same time). The workstation's 390 passed / 110 skipped is still a count until it runs there
- [ ] Optional: delete the unused `data/processed/*/graphs_v1_held.npz` (the cache moved to v2)

**RTX 2060 (training), in this order.** Every command is in 09, "GNN runs (workstation runbook)"
- [x] Setup: copy `data/workstation-data-2026-09-29.tar` and its `.sha256`, check, unpack, `git pull`, `uv sync` with the `prediction` extra on CUDA, CUDA check. The tar doesn't need repacking: it holds no caches, and graphs v2 get built on first use. **Done 2026-09-29:** hash matches, 66 folders in `data/processed`, `2.11.0+cu128 12.8 True NVIDIA GeForce RTX 2060 True`
- [x] Tests on the workstation: expect 390 passed, 110 skipped. Then `scripts\gnn_smoke.py --device cuda`
  - 2026-09-29: 400 passed, 110 skipped in 2 min 6 s (510 tests now; before the unpack it was 388 / 122, the resampler's real-data tests skipping). Smoke fit on CUDA: 9,939 training rows/s (M1 MPS: about 2,800), peak GPU memory 753 MB, 10 epochs in 16 s, loss 0.126 → 0.052, held-out 10505 PR-AUC 0.341 at a 0.020 base rate (M1: 0.34). Graphs for 4 matches built in the same run
- [x] Frame GNN: `--model gnn --one-fold 0` timing, then full H = 5 and H = 3. Compare with `lgbm-held-2026-09-27` (`evaluation.compare`, `scripts/lead_time.py`)
  - Timing done 2026-09-29 (`gnn-frame-timing-h5`): fold 0 took 7.8 min (inner fit + τ 3.5, outer fit 3.8, predict 0.5), so a full H = 5 run is about 42 min. First run also built the graph caches, 76 s for 64 matches. Peak GPU memory 810 MB. Fold 0 PR-AUC 0.305 vs LightGBM's 0.330 (one fold, not a result). Both fits early-stopped at epoch 1 (6 epochs each)
- [x] Temporal GNN: `--model tgnn --one-fold 0` timing first and watch GPU memory. It has never been fitted on real data, so this is also its first real check. If it's too slow or runs out of memory, use `--gnn-param steps=4`, `stride=8` or `batch_size=64`. Then full H = 5 and H = 3. Compare with the frame GNN first (does history help?), then with LightGBM
  - Timing done 2026-09-29 (`gnn-temporal-timing-h5`, default params): fold 0 took 58.6 min (inner fit + τ 28.7, outer fit 26.2, predict 3.7), so a full run is about 5.4 h per horizon. About 195–250 s an epoch, ~6.5× the frame GNN. Peak GPU memory 1,181 MB, so no need for `steps`/`stride`/`batch_size` tweaks. Fold 0 PR-AUC 0.305, same as the frame GNN's fold 0 (one fold, not a result). Frame GNN full runs: H = 5 0.288 vs LightGBM 0.296 (1/5 folds), H = 3 0.294 vs 0.298 (2/5)
- [x] Sensitivity on the temporal GNN (H = 5): `--degrade id_fragment:2` and `--degrade target`, each against the clean temporal run. id_fragment +0.001 (2/5), target 0.284 → 0.225 (0/5), about where LightGBM lands (0.226)
- [x] Write the results up in `Docs/reviews/` (frame GNN, temporal GNN, degraded runs) and tick the GNN items below. **Done 2026-09-30 (`Docs/reviews/gnn-2026-09-30.md`): neither GNN beats LightGBM, history doesn't help (temporal vs frame 2/5 and 1/5), median lead still 0.7–0.8 s. LightGBM stays the baseline, GNN work is parked**

**After the GNN runs: order of work (proposed 2026-09-30, from the GNN review)**
1. Stage 8 carry-forward fix, then rerun 07 #6 (in progress; faster nearest-player rules failed on scored rows 2026-09-30; the learned rule is specced and reviewed, the residual diagnostic, extractor and touch-vs-turnover check are done; the model build and pilot are next). The biggest cheap loss (−0.038), and it's what live mode feeds the model
2. Train the model that runs on vision output with the degradations on (M1, small; Phase 1)
3. xG + P(goal) v1 = P(shot) × xG (M1; Phase 1). The overlay's headline number, nothing built yet
4. Vision quality on the 2060 in the sensitivity order: benchmark clips (W0), homography thresholds, ball association (Phase 2 group 4)
5. Stage benchmark → live config → `soccer-live-overlay` (Phase 3)
- Parked: more model architectures, SkillCorner, player profiles. Worth doing, but they don't change the demo

**M1 (code, next build)**
- [x] Provider vs inferred possession (07 #6): the same LightGBM trained on stage 8's possession instead of PFF's. It decides whether stage 8's 77.9% is good enough (`Docs/reviews/stage8-2026-09-29.md`). **Done 2026-09-29: −0.038 PR-AUC at H = 5, 5/5 folds, so stage 8 needs work** (`Docs/reviews/possession-inferred-2026-09-29.md`)
  - [x] Spec in 05/07 first. Possession sets the attacking team, the flip, which rows are eligible and the labels. Proposal: the model's inputs use inferred possession, while labels and scored rows stay on PFF's (the shots really happened, so what gets scored shouldn't change)
  - [x] Code: a flag like `--possession inferred` that runs stage 8 (`vision.state.infer`) per match and feeds it through the resampler and features, with its own caches, plus tests (past only, provider run unchanged)
  - [x] Run: `--model lgbm` with the flag, H = 5 first (about 5–8 min), H = 3 if there's time (about 10–15 min for both; estimated from the earlier LightGBM runs, not measured). Stage 8 itself is about 15 s for 64 games; the resampler part is untimed. Measured: both horizons in one run, 7 min 22 s (stage 8 12 s, features 7 s, CV 400 s)
  - [x] Compare with `lgbm-held-2026-09-27` (`evaluation.compare`, `scripts/lead_time.py`) and write a short review. A small drop: stage 8 is good enough, move on. A big one: stage 8 needs work (see the review's "what it means")
- [ ] Parked (no GNN win, see the GNN review): spec the node-level "who will shoot" head in 05 (labels from the shot's `player_id`, per-node team is already cached, the shooter is off camera in about a third of shot frames)

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
- [x] Lead time, hand features: 21 attack-building features over 2–5 s (`features_version = "2"`, `Docs/reviews/lead-time-2026-09-28.md`). **Negative:** PR-AUC 0.287 vs 0.296 (0/5 folds), median p before shots unchanged, fewer shots caught at matched false alarms. v1 stays the baseline
- [x] Lead time, temporal GNN: tried 2026-09-30, **negative** (`Docs/reviews/gnn-2026-09-30.md`): 0.284 vs 0.296, median lead 0.77 s vs 0.71 s, worse than the frame GNN 2–5 s out. Three models and the v2 features all stop at the same place, so the inputs are the limit, not the model. Was: Most misses never reach τ (941 of 980 for v1), and a median lead of ~0.7 s holds at every budget from 2 to 12 false alarms per match, so ranking the rows 2–5 s before a shot is the gap, not the alarm rules
- [x] Vision sensitivity test (`prediction/degrade.py`, `Docs/reviews/sensitivity-2026-09-28.md`, 34 runs): every arm at the detection review targets costs −0.070 PR-AUC (0.296 → 0.226), −0.086 if trained clean. Biggest losses: homography loss (17%: −0.018, 50%: −0.082) and ball misses (10%: −0.017). Unknown team costs less than a wrong one. ID fragmentation and ball height don't matter to this baseline (rerun on the temporal models)
- [ ] Train the model that runs on vision output with the degradations on (recovers ~0.016 of the combined loss, sensitivity review)
- [ ] Full vs. broadcast-view training comparison on SkillCorner folds (05, 07 #5)
- [ ] xG model on StatsBomb 360 without World Cup 2022 (features known before the shot only); Wyscout location-only xG as a check
- [ ] xG calibration check on PFF shots (129 goals in tracked games) and SkillCorner shots (61 goals)
- [ ] P(goal) v1: P(shot) × xG at carrier position

**Possession inference (03 stage 8, runs on dataset tracking)**
- [x] Ball carrier / possession / ball state rules on pitch coordinates (`vision/state.py`, 2026-09-29): one causal state machine on visible objects, thresholds in seconds
- [x] Check against PFF (`python -m vision.state_check`, `Docs/reviews/stage8-2026-09-29.md`): possession 77.9% (82.9% alive), under 90%. A confirmed carrier matches PFF 97.9%, but only 26% of alive frames have one. Dead → dead or null 89.9%, nearly all null
- [ ] Same check on SkillCorner and IDSSE when their converters land (SkillCorner also checks the carrier)
- [x] Provider vs. inferred possession comparison (07 #6, `Docs/reviews/possession-inferred-2026-09-29.md`): PR-AUC 0.296 → 0.258 at H = 5 (5/5 folds worse), past 07's "big" line. On the 14% of scored rows where stage 8 names the other team, the model can't rank at all (PR-AUC 0.026)
- [x] Give the model stage 8's staleness first (07 #6 follow-up, `Docs/reviews/possession-stale-2026-09-29.md`): carrier age as a feature (arm S, −0.037) and possession unknown past 10 s (arm U, −0.040) both stay past "big", 0/5 folds. The feature does nothing with the provider's possession either (−0.0006, 3/5). The model can't fix a wrong possession
- [ ] Stage 8, carry-forward fix: possession changes with no confirmed carrier (stage 8 review "What it means"). 85% of disagreeing rows are PFF changing and stage 8 not following, worst in the first 1–3 s after the change and with no visible ball. Needs a new `StateConfig` field so the caches get a new key (05 Caches). Rerun 07 #6 after, about 8 min on the M1
  - 2026-09-30, first try didn't help: "same team nearest for 0.05–0.2 s moves possession" gives +0.3 pt on alive frames and −0.8 to −3.6 pt overall on the 8 dev games, same as the old 0.1 s carrier row (stage 8 review table). Reverted. Next: before another rule, split the remaining disagreements by visible vs unseen ball and by seconds since PFF's change, then go after the unseen-ball-after-change bucket (a rule on the ball reappearing after a gap, or the learned rule from the review)
  - Split done 2026-09-30 (`scripts/possession_split.py`, stage 8 review "Where the disagreement is"): 65% of the disagreeing frames have had no visible ball for over 2 s (39% over 10 s), and 57% are more than 10 s after PFF's change. Ball-based rules reach about a third at most, so the next try is possession from player positions (learned). First: the same split on 07 #6's scored rows only
  - Scored-row split done 2026-09-30 (`possession_split.py --scored h5`, stage 8 review "On 07 #6's scored rows"): **it flips.** 55% of scored disagreement is in the first 3 s after PFF's change and 45% has the ball visible; gaps over 2 s are 33%, changes over 10 s old 19%. The long no-ball stretches were mostly dead time. Next: recheck the two rules already tried (team-nearest, 0.1 s carrier) on scored rows, since they target the change lag; the learned rule comes after
  - Rechecked on scored rows 2026-09-30 (stage 8 review "The change-lag rules on scored rows"). The team rule is back as `StateConfig.team_near_s`, off by default (default key unchanged, tests), and `possession_split.py --state KEY=VALUE` runs any config. None passed the bar set before the runs: the first-3-s disagreement drops 15%, but the positives double (7.8% → 15%) on defenders' brief touches during attacks. 0.2 s is the only net gain overall (−0.9%). **07 #6 not rerun.** Hand rules on proximity stop here
- [ ] Stage 8, learned possession (full spec in 03 stage 8, 2026-09-30; plumbing in 05, gate in 07 #6): LightGBM P(home has the ball), 159 causal VISIBLE-only inputs (`pfeat-v1`: shape, contact, heading, last contact, pressure, view extent, two lag blocks), mirrored rows and symmetrized output. **Strict nested training is decided:** 25 logical fits (5 outer on four folds, 20 inner-OOF on three) backed by 15 trainings, since each pair of inner fits shares its training set. Final τ reuses outer-test possession OOF for every match; the all-64 live fit is separate. No schema change
  - [x] Spec reviewed (`Docs/reviews/learned-possession-spec-review-2026-09-30.md`) and revised 2026-09-30 with every blocking, should-fix and nice-to-have item and the user's answers
  - [x] Diagnostic first: split the unexplained 30% ball-visible first-second residual (height, ESTIMATED player on the ball, no visible player in reach, timing), with the action per bucket fixed in 03. **Done 2026-09-30 (`Docs/reviews/possession-learned-diagnostic-2026-09-30.md`): 77% has no visible player within 1.5 m, height 0.3%, ESTIMATED 1.4%; reachable 81% (positives 71%), but 42% of that comes before the new team's first touch (PFF changes a median 0.33 s early). v1 contract unchanged**
  - [x] Build the `pfeat-v1` feature extractor in `vision/` (the 159 columns, 03 Exact input columns), enough to run the check below. No model code yet. **Done 2026-09-30: `vision/possession_features.py` (`python -m vision.possession_features`), 28 tests (causal, VISIBLE only, mirror, lags, segments, cache hashes). All 64 matches cached in 99 s on the workstation, grid identical to the resampler on all 64.** Still to come with the model: batch/streaming parity (no streaming path yet) and `all_estimated` parity
  - [x] Touch-vs-turnover check: rows where `team_near_s = 0.05` switches and the default doesn't, turnover vs touch by PFF over the next 3 s. At least one feature at separation AUC ≥ 0.70, or stop before any fit. Freeze the feature contract only after it passes. **Passed 2026-09-30 (`Docs/reviews/possession-touch-check-2026-09-30.md`): `nearest_away_1_r15` (the old team's share of being on the ball over the last second) at 0.755 on 12,949 trigger rows; on positives, shape separates best (~0.81). `pfeat-v1` frozen**
  - [x] Build the rest in `vision/`, with only `prediction/possession.py` bridging its output. Add the off-by-default `StateConfig.possession_model` (default key stays `e737054b5d`), fix the call sites that hardcode the config (B1), the per-context assembly in `prediction.cv` (contexts outside horizons), test/inner-OOF caches with a hash check on every read, and the resample globs. Provider and old inferred paths stay byte-identical. Tests per 03, including refit determinism and refused learned configs **Done 2026-09-30:** `vision/possession_model.py` (labels, ES draw, mirrored design, probe + refit, symmetrized p, manifest checks, 2D-rule grid, `predict_match`), `vision/possession_train.py` (nested driver, determinism twin, sealed manifest, `--pilot`), `StateConfig.possession_model` (default key still `e737054b5d`) with every rule-only path refusing it, B1, the `prediction/possession.py` bridge (per-context state and feature caches, hash checks on every read), the `prediction.cv` fold data provider (6 assemblies) and `--possession-model`, and the resample globs. The model reads `diff_*` recomputed in Float32 so M is exact (03 clarification). Tests in `tests/test_possession_model.py` and `tests/test_possession_learned.py`, full suite 478 passed. Still open: the gate script (`possession_split.py --possession-model`), and the live wrapper with batch/streaming parity (comes with live)
  - [x] Pilot on the workstation CPU (`num_threads = 6`): one match's extraction and `outer_0` probe + refit, with best rounds and peak RSS. Project the whole possession stage against the overnight 10 h budget; stride 1, else 2, else 4, else stop (03, 09 runbook). Decide on time and memory only **Done 2026-09-30 (`Docs/reviews/possession-pilot-2026-09-30.md`): stride 1.** `outer_0` 801 s on 3.32M mirrored rows, ran to the 2,000-round cap, peak 6.8 GiB. Projection 3.2 h (4.8 h if the determinism check fails) against 10 h
  - [ ] Overnight: `python -m vision.possession_train --model-id lgbm-v1-nested5x4-s1` for the 15 trainings (outer_0 is reused from the pilot), then the 25 contexts' predictions via the gate script
  - [ ] `possession_split.py --scored h5 --possession-model <id>` on concatenated outer-test output. Five bars fixed: overall ≤ 253,445; first 3 s ≤ 131,585; positives ≤ 3,862; late-change bucket rate ≤ default + 0.010; churn ≤ 18,182 changes. A failure stops before shot-model CV. At most v1 plus one revision go through the gate
  - [ ] Only if the gate passes, 07 #6 once, H = 5 and H = 3, against `lgbm-held-2026-09-27` and `lgbm-pinf-2026-09-29` via `evaluation.compare` and `scripts/lead_time.py`. Same decision lines (small above −0.010; big ≤ −0.018 and worse on ≥ 4/5 folds), no PR-AUC tuning. Expect "in between" or "big" even after a pass
  - [ ] After the decision, prepare the all-64 live possession fit if proceeding to deployment; verify batch/streaming parity and warmed per-tick latency on its own worker (p95 < 10 ms, p99 < 33 ms). Needs the attacking direction from vision first. Team-assignment errors (`team_flip`/`team_unknown`) and a fully live alarm evaluation remain separate checks

**GNNs** (code and tests on the M1, full CV runs on the workstation's RTX 2060, decided 2026-09-28)
- [x] Frame GNN spec in 05 (2026-09-28): a graph per 10 Hz row (visible players + held ball, attacking frame), fully connected with edges in meters, v1 hand features as a global vector, plain log loss, early stopping on whole matches without a refit, stride-4 epochs. Built in-house, not with unravelsports (its velocities use later frames). Ablation arms `--gnn-layers 0` and `--gnn-globals none` say which claim a win supports
- [x] Frame GNN code (`prediction/graphs.py`, `prediction/gnn.py`, `prediction/gnn_net.py`, `--model gnn` in `prediction.cv`, `--one-fold` timing run), tests (causal, ESTIMATED-free, mirror, batch-independent, out of fold, learns through `run_cv`), and a smoke fit on 3 matches on the M1 (`scripts/gnn_smoke.py`; loss goes down, held-out PR-AUC 0.34 at a 0.020 base rate). No full CV on the M1
- [x] Move the data to the workstation (`data/` is gitignored). Packed on the M1 (`scripts/pack_workstation_data.sh`: `data/workstation-data-2026-09-29.tar`, 2.5 GB, 402 files) and the runbook is in 09. Done 2026-09-29: unpacked, tests 400 / 110 skipped, smoke fit on CUDA at ~9,900 rows/s
- [x] Frame GNN full CV on the 2060 (2026-09-29, `Docs/reviews/gnn-2026-09-30.md`: H = 5 0.288 vs 0.296, H = 3 0.294 vs 0.298, no win) (H = 5 first, then H = 3): time one fold first, then compare with `lgbm-held-2026-09-27` using `evaluation.compare` and `scripts/lead_time.py`
- [x] Temporal GNN spec and code (2026-09-29, 05 model 3): 6 steps 0.5 s apart (last 2.5 s) turned into the anchor row's frame, the frame GNN's message passing on each, a GRU over them, batch 128. Graphs cache v2 keeps node teams (frame GNN inputs unchanged). `--model tgnn`, `--gnn-param` for timing tweaks. Tests on the M1 (window edges, turnover, mirror, causal, out of fold; learns a history-only toy at PR-AUC 0.64 vs 0.37 for the frame GNN). No real-data fit yet
- [x] Temporal GNN runs on the 2060 after the frame GNN (2026-09-30, `Docs/reviews/gnn-2026-09-30.md`: H = 5 0.284, H = 3 0.290, no better than the frame GNN): one-fold timing, then H = 5 and H = 3. Compare with the frame GNN first (09 runbook)
- [x] `--degrade` wired into the GNN graphs (2026-09-29): built from the same degraded objects as the features, never cached; the test arm swaps in a degraded store for the held-out folds. Checked end to end on 10 matches (tgnn, `ball_miss`, test arm, one fold)
- [x] Rerun `id_fragment` (and the combined arm) on the temporal GNN (2026-09-30, `Docs/reviews/gnn-2026-09-30.md`: +0.001 and −0.059) on the 2060, against the clean temporal run. Tracks aren't linked across steps, so it only reaches the model through `has_vel` (05)
- [ ] Parked: node-level "who will shoot" head → P(goal) v2 (needs a GNN worth using first)

**Player profiles**
- [ ] PFF: profiles from earlier-season StatsBomb data (never World Cup 2022); measure coverage (04)
- [ ] SkillCorner: profiles from season aggregates (trait columns only, `count_match` ≥ 5, 04)
- [ ] Three-arm ablation: none vs. position only vs. full, plus the `count_match` leakage check on SkillCorner. PFF results are the clean ones; SkillCorner results stay provisional

**Wrap-up**
- [ ] Final models on IDSSE, once

## Phase 2 — Vision pipeline (RTX 2060)
In priority order from the 2026-09-27 review (`Docs/reviews/vision-review-2026-09-27.md`; F-numbers refer to it). The sensitivity test (`Docs/reviews/sensitivity-2026-09-28.md`) sets the order of the quality work: homography availability, then the ball, then player position accuracy, then an unknown team option (if the benchmark supports it), then player recall. **Stage 8's possession goes ahead of all of them** (07 #6, 2026-09-29: −0.038, twice the worst single arm; `Docs/reviews/possession-inferred-2026-09-29.md`).

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
- [ ] **Tune the homography acceptance thresholds** (top priority from the sensitivity test): keep frames without geometry at or under smoke04's 17% (30% more than doubles the loss). Lean permissive: per frame, a rejection costs more than keeping 2–4 m of error, so set the actual cutoff on the vision benchmark (W0)
- [x] Run `vision.offpitch` on smoke01 to see what the 36 off-pitch rows were, then a second smoke run with the new acceptance: 36 off-pitch rows → 0, `homography_ok` 375/375 → 310/375 match frames, 75 projections nulled (`Docs/reviews/smoke-test-2026-09-27-followup.md`). Thresholds still untuned
- [x] Ball history: expire before a new detection uses it, reset when the ball has no pitch position, no extrapolation without valid geometry (03; detection review D1)
- [x] `visible=False` for filled boxes that drift fully off screen, instead of everything visible (03; detection review D6)
- [ ] Ball: reset on camera cuts, temporal candidate association instead of max confidence (F8, detection review D2/W2). The sensitivity test puts the ball second after geometry (misses −0.014 per 10% of time), so this is on. Targets: recall ≥ 90%, precision ≥ 95%
- [ ] Per-stage timings + effective model settings in `run.json` (03)
- [ ] Run roboflow/sports end to end on a SoccerNet sample clip as a reference
- [ ] Tune the stage 0 thresholds on broadcast clips with ads and studio cuts (`view.parquet`, 03)

**5. Small cleanups**
- [ ] 03: header omits `events.parquet` (the v0.4 → 0.6 part fixed 2026-09-30); points 31/32 aren't on the halfway line
- [ ] `--period-start-s` flag so `timestamp_s` is the period clock, one period per run (F6), plus explicit per-period attacking direction instead of odd/even periods, which is wrong in extra time (detection review D7)
- [ ] `demo.debug` renders only the processed frame range (a hand-typed `--frames` gave smoke04 a 2 s debug video of a 25 s run, see the follow-up review)

**Detection review follow-ups** (`Docs/reviews/detection-improvement-spec-2026-09-27.md`, taken in reduced form)
- [ ] Small vision benchmark: 5–10 labelled clips from different matches (ball, player positions in meters, teams, live vs. replay), scored per clip. Not the review's 30–50 clip set with double annotation; grow it only if results are borderline (W0)
- [ ] Camera cuts and replays detected separately from the grass/keypoint gate; reset tracks, teams and ball on a confirmed cut, emit nothing prediction-eligible during a replay (W3)
- [ ] Record why frames and projections were rejected, plus model settings, in the vision cache (W1, the parts that help debugging; not the full replayable cache yet)
- [ ] From the sensitivity test, in order: geometry accuracy where attacks happen (W4), detector fine-tuning (W8) for the ball first and then player foot position (0.93 m noise: −0.013, third largest), teams/keepers with an "unknown" option (W5; cheaper than a wrong team at the same share, but only a win if abstentions land on would-be flips, check on the benchmark). Player misses are the smallest measured loss (40% missed: −0.023). ID/tracking quality waits for a rerun on the temporal GNN (the v2 hand features weren't adopted, so no rerun there)
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
  - [ ] Ball highlight ring + velocity arrow (08; uses existing ball position/vx,vy, no new inference)
  - [ ] Possession % and territorial/attacking-third % panels (08; aggregates possession_team + pitch_x over time, no new inference)
  - [ ] Sprint highlight, ball trail, confidence/uncertainty tint, event ticker, shooting-lane cone (08; all rendering over existing fields, no new inference)
  - [ ] Offside line (08; geometry off team + x-positions, needs solid homography accuracy on the defensive line)
  - [ ] Pitch control / space heatmap (08; Voronoi over player positions, CPU-only, no new inference)
- [ ] Benchmark each vision stage on the 2060 (FP16 / TensorRT) and pick the live config (09)
- [ ] Live mode on workstation (in `soccer-live-overlay`, 09)
- [ ] Record a local match for the public demo (see 08)
- [ ] Fine-tune detection/keypoints on self-recorded footage
- [ ] Demo clip + write-up
