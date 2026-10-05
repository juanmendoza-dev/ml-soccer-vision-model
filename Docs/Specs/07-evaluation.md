# 07 — Evaluation

## Splits
A fixed 70/15/15 split wastes matches: even PFF's 51 tracked games would give a ~8-match test set. Use grouped cross-validation instead.

### Development: grouped 5-fold on PFF + SkillCorner
- Pool: **PFF (primary)** and **SkillCorner (second pool)**, see 06.
- **5 folds grouped by match**, assigned separately within each source (~1/5 of PFF matches and 4 of 20 SkillCorner matches per fold) so every fold has both. A match's frames are never split across folds.
- Fold assignment is fixed once, saved to `data/splits/folds.json` (`match_id`, `source`, `fold`), and reused by every model so comparisons are paired.
- Balance folds roughly by shot count within each source (e.g. `StratifiedGroupKFold` on match-level shot totals), fixed seed.
- **Freeze folds only once all 64 PFF games are on disk.** Until then `folds.json` is provisional and results from it aren't reported as final. Adding matches later must not move existing ones.
- **Built by `python -m evaluation.folds`.** It only assigns matches that aren't in the file yet, so a match never moves:
  - **Per source:** matches are sorted by open-play shot count, the count 05's labels use (ties shuffled with the fixed seed). They're cut into blocks of 5, and each block is shuffled across the 5 folds. Fold sizes differ by at most one match, and shot totals come out close.
  - **New matches from a source already in the file** each go to that source's fold with the fewest matches, then the fewest shots.
  - **`--refresh-shots`** updates the stored shot counts after a label definition change (e.g. the final-third free-kick rule, 02). Folds never move.
- **Format and storage:** `folds.json` has `n_folds`, `seed`, `frozen` (source → date its assignment was fixed) and `matches` (`match_id`, `source`, `fold`, `open_play_shots`). It's committed to git (an exception to `data/` being ignored), so every machine uses the same folds.
- **Status:** PFF was frozen on 2026-09-26 with all 64 games. SkillCorner gets appended once its converter exists, without moving any PFF match. PFF folds: 12/13/13/13/13 matches with 206/232/238/244/234 open-play shots (1,154 total, refreshed 2026-09-27 after the final-third free kick rule in 02; 1,127 before). The 16 knockout games fall 2/3/5/4/2 across folds 0–4; stage isn't stratified (not in 02), so a per-stage breakdown in reports is worth having.
- **Inner split:** `evaluation.folds.inner_split(fold)` returns the inner validation matches for an outer fold: about 15% of the training matches from each source, chosen with the same shot stratification. It's deterministic, so every model tunes on the same matches.
- Each outer fold: train on the other 4 folds (both sources), evaluate on the held-out fold.
- **Anything tuned is tuned inside the training matches only:** hyperparameters, early stopping, temperature scaling, and threshold τ. Use an inner grouped split (~15% of training matches from each source as inner validation). Never tune on the evaluation fold.
- PFF-only training runs (e.g. when a feature only exists in PFF) use the same folds, restricted to PFF matches.
- Report:
  - **Pooled out-of-fold:** concatenate all 5 folds' predictions, then compute every metric once. **PFF pooled OOF is the headline number**; SkillCorner pooled OOF is reported next to it, never merged into one figure (different leagues, tracking and frame rates).
  - **Per-fold spread:** mean ± std across folds, to show how much the result depends on which matches were held out.
- Model comparisons are **paired by fold**: model A beats model B only if it wins on most folds, not just on the pooled number.

### Final check: external test set
- **IDSSE (7 Bundesliga matches)** is the untouched test set. Different league, optical tracking. Evaluate once per final model, trained on all PFF + SkillCorner matches with settings chosen during CV.
- No tuning, threshold picking or model selection on IDSSE. If a result there changes a decision, note it in the report.

### Rules
- Splits are by match, never by frame or by possession.
- Player profile features (04) come from seasons before the match, so the same player appearing in train and eval folds is not leakage. Per-match stats from the eval match are.

## Metrics
| Metric | Why |
|---|---|
| PR-AUC | Main metric; positives are rare. Average precision (step sum over thresholds, ties grouped, same as sklearn's `average_precision_score`), not a trapezoid |
| ROC-AUC | Comparable to papers |
| Brier score + calibration curve | Probabilities must mean what they say. Calibration uses quantile bins: at a ~2% base rate equal-width bins put almost every row in the first one |
| Lead time | Seconds from the start of the alarm covering the shot to the shot (see below); report median and distribution |
| Missed shots | Share of shots with no covering alarm (see below) |
| False alarms per match | Alarms with no shot (see below); keeps lead time honest |

Code: `evaluation/metrics.py`, plain numpy/polars.

**Scored rows** for PR-AUC, ROC-AUC, Brier and calibration: `label_mask_H` true and not `all_estimated` (05). Picked from the labels, never from p, so every model is scored on the same rows and comparisons stay paired. A null p on a scored row is an error, not a skipped row.

### Alarms and lead time
The probability rises and falls, so "first crossing τ" is ambiguous. Use alarms with hysteresis:
- **Alarms run over every row** of the match, masked or not: the model predicts through set plays live. Each (match, period) is processed separately; an alarm never crosses a period.
- **Alarm starts** when P(shot) rises above τ, with a team in possession and `ball_state` not dead. The alarm belongs to that team.
- **Alarm ends** when P(shot) drops below 0.8·τ, a *different* team is in possession, or `ball_state` is dead. Dips between 0.8·τ and τ don't end it. Its end time is the row that ended it (or the period's last row).
- **Nulls hold, for a while:** a null P(shot) (all-ESTIMATED cutaway, ineligible row), a null `possession_team` (loose ball) or a null `ball_state` (vision gap, 05) doesn't start an alarm or end one by itself. But an alarm ends once it has gone more than **2 s without a non-null P(shot)** (`MAX_NULL_HOLD_S`, ended_by `stale`), so a long cutaway or replay can't keep a warning alive (2026-09-27, detection review D10).
- **An alarm covers a shot** by its team if it started strictly before the shot and the shot is no later than 1 s after the alarm ends (grace for the last-frame drop).
- **True alarm:** it covers a shot, open play or set play. Otherwise it's a **false alarm**. A warning before a corner header isn't false, but set-play shots don't count toward lead time or misses.
- **Lead time** = open-play shot time − start of the covering alarm. If two alarms cover a shot, the one active at the shot wins over one in its grace second. A shot with no covering alarm is a **miss**, not a lead time of 0. The same grace applies to misses and to true/false, so a shot is never both covered and missed.
- One alarm can cover several shots (rebounds); each shot gets its own lead time from the same start.
- **False alarms per match** divide by every match evaluated, including matches with no alarm.
- τ is chosen on training matches only (see Splits). Also report the trade-off across τ values: median lead time and miss rate vs. false alarms per match (`tau_sweep`).
- **Choosing τ** (decided 2026-09-27, `evaluation.metrics.choose_tau`): **the lowest miss rate with at most 3 false alarms per match**, on the inner validation matches.
  - Per outer fold: fit on the fold's training matches minus `inner_split(fold)`, predict the inner matches, pick τ there, then refit on all the fold's training matches and apply that τ to the held-out fold.
  - Candidates are high quantiles of p over the rows with a non-null p (the rows alarms run on), deep into the tail, since a calibrated model at a ~2% base rate rarely goes high.
  - Ties go to the higher τ. If no candidate meets the budget, take the one with the fewest false alarms and flag it (`tau_met` in `run.json`, shown in the report).
  - Why miss rate and not lead time: median lead is only over caught shots, so a τ that catches only the easy shots early can look good. Lead time is reported at the chosen τ, and 00's ≥ 2 s is checked there.
  - Why 3: even the oracle has 1.17 false alarms per match (below), so 1 is out of reach. With ~18 open-play shots per match, 3 keeps most alarms real. One fixed budget compares models at the same false-alarm level. It's one constant (`MAX_FALSE_PER_MATCH`); the τ sweep shows what other budgets would give.
  - The `"final"` τ (IDSSE) uses `inner_split(None)`: the same stratified inner split over all CV matches.
- 00's success criterion (median lead time ≥ 2 s) is measured at the τ chosen on training matches.

#### Floor from the labels (PFF, H = 5, 2026-09-27)
An oracle that outputs 0.9 on every positive row and < 0.3 elsewhere (null where 05 scores null) still gets, at τ = 0.5: **28 of 1,154 shots missed (2.4%)**, **75 false alarms (1.17 per match)**, median lead time 4.9 s. That's the best any model can do under these rules (61 / 0.95 before the 2 s null-hold cap; the cap splits ~15 alarms held through long cutaways, and the part before the cutaway counts as false):
- **25 misses:** the shooting team had the ball, then PFF credits the other team for the last 1–4 s before the shot. This is the same PFF possession lag as 06's H = 3 misses (all six of those are among the 25). The possession change ends the shooting team's alarm more than 1 s before the shot, so it's a miss, and that alarm is false. This is most of the false alarms.
- **3 misses:** the whole lead-up is an all-ESTIMATED cutaway (3840 ×1, 3845 ×2, see 06), so there's no prediction to alarm on.

**Open:** whether a short opposing possession (< 1–2 s) should end an alarm. Holding through it would remove most of that floor, but a real turnover should still end the alarm. Decide with the first model results, not before.
- **First evidence (LightGBM, 2026-09-27, H = 5):** of its 980 misses at the chosen τ, 941 never had p above τ in the 5 s before the shot, and only 39 had a crossing that didn't become a covering alarm. So this rule isn't what limits the baseline. Revisit once a model alarms seconds ahead, when the oracle floor's possession-lag misses start to matter.

## Required comparisons
1. Distance + angle floor vs. LightGBM baseline vs. frame GNN vs. temporal GNN.
2. Player profiles: none vs. position only vs. full (see 04).
3. H = 3 s vs. H = 5 s.
4. Dataset tracking vs. vision-pipeline tracking on the same matches, if available (measures how much vision errors hurt). Before paired footage exists, the vision sensitivity test (05) degrades PFF tracking the way vision fails and measures the loss.
5. Full tracking vs. broadcast view (off-camera players dropped, see 05) on the same folds. **SkillCorner folds only** (plus IDSSE at the final check): PFF's off-camera positions are ESTIMATED and ~12 m off at shots (06), so its full view isn't a meaningful arm.
6. Provider vs. inferred possession/ball state (03 stage 8) on dataset tracking: same model, same folds. Measures how much the inference rules alone cost before vision errors are added.
   - **What's swapped (05, Inferred possession):** the model's inputs take stage 8's possession (the flip, attackers vs defenders, `possession_s`). Labels, `eligible`, `all_estimated` and the scored rows stay on PFF's, so both arms are scored on the same rows. Ball state isn't swapped: no feature reads it.
   - **The alarm rule and τ selection stay on PFF's `possession_team` and `ball_state`** in this comparison, so the only difference between the arms is what the model sees. A fully live evaluation, with alarms starting and ending on stage 8's possession and ball state as well, is a separate, later question: it changes which alarms exist, not how well the model ranks rows.
   - **Caveat:** PFF's possession is itself laggy (Floor from the labels: 25 of the oracle's 28 misses). Where stage 8 is right and PFF late, the inferred arm is penalized for disagreeing with PFF, not with the game.
   - **Decision rule, set before the run (2026-09-29).** On the paired per-fold ΔPR-AUC at H = 5 (inferred − provider, `evaluation.compare`). The yardsticks: the provider run's fold-to-fold spread is ±0.023, a second degrade seed moved a result by 0.0014, and the vision sensitivity arms at their targets cost −0.010 to −0.018 each (05).
     - **Small, stage 8 is good enough, move on:** mean Δ above −0.010 (under half the fold spread, and less than any vision arm that was worth acting on).
     - **Big, stage 8 needs work before vision's own quality work:** mean Δ at or below −0.018 (the size of the worst single vision arm, `geom_loss`) and worse on at least 4 of 5 folds.
     - **In between:** stage 8 goes on the vision quality list, ranked by its Δ next to the sensitivity arms, not ahead of them.
   - **Result (2026-09-29, `lgbm-pinf-2026-09-29`, `Docs/reviews/possession-inferred-2026-09-29.md`):** H = 5 PR-AUC 0.296 → 0.258, mean Δ −0.038 (per fold −0.033 to −0.043), worse on 5/5 folds; H = 3 −0.036, 5/5. **Big:** stage 8 needs work. The model collapses on the 14.2% of scored rows where stage 8 names the other team (PR-AUC there 0.026 vs 0.237), and is worse where they agree too (−0.014). On the disagreeing rows PFF's team shoots about 3× as often as stage 8's, so PFF's lag is the smaller part.
   - **Stale possession follow-up (set 2026-09-29, before the runs; 05 "Stale possession").** Stage 8's disagreement with PFF climbs with the seconds since it last confirmed a carrier (2% at 0 s, 51.5% past 10 s). Three runs, each H = 5 and H = 3: inferred + `poss_carrier_age_s` (v3, arm S), inferred with possession unknown past S = 10 s (arm U), and provider + v3.
     - **The verdict uses the same rule as above,** against `lgbm-held-2026-09-27`. An arm that lands above −0.010 makes stage 8 plus that arm good enough.
     - **The feature might help on its own.** If provider + v3 beats provider v1 on at least 4 of 5 folds (ΔPR-AUC at H = 5), a v3 arm's cost is read against provider + v3 instead: inferred + v3 − provider + v3. That Δ decides whether it's "small". Otherwise a feature gain would be credited to stage 8.
     - Each arm is also compared with `lgbm-pinf-2026-09-29`, to show how much of the −0.038 it recovers.
     - If no arm reaches "small", stage 8's rules change next (03 stage 8), and 07 #6 is rerun on the new rules.
   - **Result (2026-09-29, `Docs/reviews/possession-stale-2026-09-29.md`):** no arm reaches "small". H = 5 mean Δ vs `lgbm-held-2026-09-27`: arm S −0.037 (0/5 folds), arm U −0.040 (0/5). Provider + v3 beats provider v1 on only 3/5 folds (−0.0006), so arm S is read against provider v1. Against `lgbm-pinf-2026-09-29` they recover +0.001 (S) and −0.002 (U) of the −0.038. H = 3 agrees (S −0.034, U −0.037, 0/5 each). **Stage 8's rules change next.**

   - **Learned possession follow-up (spec 2026-09-30, not built).** Faster proximity rules failed the scored-row gate: about 15% fewer first-3-s errors, but positive errors doubled (stage 8 review, “The change-lag rules on scored rows”). The next arm is 03 stage 8's LightGBM possession model. Strict nested training from the start: 5 outer test fits on four folds and 20 inner-OOF fits on three folds, 25 logical fits backed by 15 trainings (each inner pair of folds shares one model, 03). Outer k's training inputs never come from a possession fit that saw its test fold. Final τ reuses the five outer-test possession fits to give every CV match OOF possession; it never uses the all-64 live fit (03/05). Same PFF scored rows, labels, alarms and τ rule as above.
     - **Before any learned fit:** diagnose the remaining ball-visible first-second disagreement under `team_near_s = 0.05`, separating height/flight evidence, ESTIMATED nearby players, missing contact and event/tracking timing (03's exact protocol). Then build the extractor and check that its features separate a defender's touch from a real turnover where the fast rule switches (03, hypothesis check, AUC ≥ 0.70); if not, stop before any fit. Only after the check passes, freeze the feature contract, without looking at PR-AUC, and pass 03's implementation tests. Pilot one match's extraction and `outer_0` on the workstation CPU, project the overnight 10 h budget, and apply only 03's predeclared stride order (1, 2, 4) if needed.
     - **Before goal-model CV:** `scripts/possession_split.py --scored h5 --possession-model <id>` reads only the five outer `test` roles, 1,977,379 rows including 49,812 positives. Reproduce the historical default counts, then require **all five** bars: overall disagreement ≤ **253,445** (−10% from 281,606); first-3-s disagreement ≤ **131,585** (−15% from 154,806); positive disagreement ≤ **3,862**; and the rate in the bucket labeled `> 10 s` after PFF's change no more than **0.010** above the default's unrounded rate. Preserve the existing bucket's `since >= 10` boundary; it is not ball age. Fifth, churn: at most **18,182** possession changes on the concatenated outer-test output (1.25 × PFF's 14,546; the default rule makes 10,074). Null counts as disagreement. A failed bar stops before 07 #6, with no parameter/threshold search on that result.
     - **At most two versions through the gate:** v1 and one revision, which needs a spec change and a new model ID and may only use the pooled diagnostic and gate tables. Every attempt is reported beside any 07 #6 result. If both fail, learned possession stops and stage 8's "big" verdict stands. Disagreement on the 25 oracle-floor windows (Floor from the labels) is reported too, as description only.
     - **Only after it passes:** run `prediction.cv --model lgbm --possession inferred --possession-model <id> --features-version 1 --ball-source held --horizons h5 h3` once, using 05's per-fold data assembly. Compare against both `lgbm-held-2026-09-27` and `lgbm-pinf-2026-09-29` with `evaluation.compare` and `scripts/lead_time.py`. Report calibration, alarms/false alarms at each fold's own τ and at matched budgets, and lead-time/5–2–1 s probability checks alongside paired fold metrics. No tuning on PR-AUC.
     - **Decision lines unchanged:** H = 5 mean paired ΔPR-AUC against **provider** above −0.010 is small; at or below −0.018 and worse on at least 4/5 folds is big; otherwise use the in-between action above. Improvement over old inferred possession describes recovery, not acceptance. H = 3 does not pick a different winner. PFF's lag, clean tracking and provider-based alarms remain the same limitations. Expect "in between" or "big" even after a pass: a straight-line read says "small" needs about 3.7% disagreement. The run measures how much better possession recovers. The τ-validation overlap in 03 (inner fits train on some of a fold's τ-validation matches) is accepted; PR-AUC doesn't depend on τ.

## Vision benchmark (W0)
The reduced W0 from the detection review (roadmap Phase 2): about 8 World Cup 2022 matches, one or two 30–60 s clips of broadcast each, scored per clip and pooled with counts. It's what the homography thresholds and later the ball work get tuned on. No locked check set and no bootstrap intervals (not adopted, roadmap); if a result is borderline, the set grows.

**Truth for people comes from PFF.** PFF's tracking was made from the broadcast: a `VISIBLE` player is on camera, in 02 meters, with a known team, and sits 0.0 m (median) from the event position at shots (06). Cutaway frames have every player `ESTIMATED`. So no person gets hand-labelled. PFF's own error is the floor of what can be measured, and it's reported as is. Humans mark only what PFF can't know: which parts of the clip aren't live wide play, and (later) where the ball is in the image.

**Fold overlap.** All 64 matches are in `folds.json`. The predictor doesn't train on vision output yet, so tuning vision on them leaks nothing today. A later predictor result on vision output from these matches names the matches vision was tuned on (detection review §4).

**Manifest:** `data/splits/vision_benchmark.json`, committed like `folds.json`. Footage never enters the repo or a synced folder (08). No video paths or file names in the manifest: a gitignored `data/vision_bench/videos.json` maps `clip_id` → local path, and the manifest's hash checks it's the right file.
- `version`, and `clips`, each with:
  - `clip_id`, `match_id` (PFF), `period`, `video_sha256`
  - `match_id` may be null, for footage PFF doesn't cover (other tournaments). Such a clip needs no sync, and scores only geometry missing, the gate and false live; people and team numbers come from PFF clips only. It still counts toward the pooled homography-rejected share that the sweep keeps under 17%, which is why it's worth having: different stadiums and lighting.
  - `video_start_s`, `video_end_s`: the clip's range in the source video
  - `home_attacks_tv_right_p1`, `home_cluster` (null until picked from the debug video, as in the smoke runbook)
  - `sync`: one or more `{video_s, timestamp_s}` pairs tying source-video time to PFF's `frames.timestamp_s` (seconds since period start). PFF time = `video_s + offset`.
  - `marks`: `{start_s, end_s, label}` in source-video seconds, label `replay`, `closeup` or `other` (ads, studio, crowd, graphics over the pitch). Anything unmarked is live wide play.
- **Sync, in this order.** The scoreboard clock gives the first pair to about ±1 s (period clock; the second half's starts at 45:00). Then refine it without vision's homography, which is what's being scored:
  1. a shot in the clip: its strike frame on video against the shot's `events.parquet` frame;
  2. else the edges of hand-marked cutaways against PFF's switches to and from all-`ESTIMATED` frames (when the footage is the world feed, they match to a frame);
  3. else kicks: the frame on video where a foot meets the ball, read by eye from frame strips, against the jump in PFF's ball speed (ball rows `visible` and not `interpolated`; long passes and clearances, where the jump is clean). At least 3 kicks spread over the clip, and the spread of their offsets is reported as the sync's precision. Needed for the stitched 2022 videos, where every cutaway falls between pieces (2026-10-02). PFF's ball timing may lag the video the way its players seem to (`--offset-check`, vision bench review), so this rung is only used once it agrees with the cut edges on a clip that has both (vb02). If it doesn't, the clip is scored with the difference corrected and says so in `sync_method`. **Checked 2026-10-02:** on vb02 kicks read 0.11 s early (3 kicks, spread 0.12 s), so kick-synced clips add 0.11 s (vision bench review, clip 3);
  4. else the clip gets `sync_coarse: true` and is reported apart.

  A second pair checks drift: if the offsets differ by more than 0.1 s, the source video's frame rate is wrong and the scorer refuses the clip. `--offset-check` reports the offset, within ±1 s in PFF-frame steps, that minimizes the median player error on frames with geometry. That's a check only, never used to score.

**Scored frames:** every frame inside `[video_start_s, video_end_s)` and outside the marks that lands within half a PFF frame (16.7 ms) of a PFF frame. The denominator never comes from vision's view gate. Vision runs with up to 30 s of pre-roll (`vision.run --start-s` before `video_start_s`), so the gate's first second and the team warmup aren't scored.

**Checks before scoring:**
- the run's video hash and attacking direction must match the manifest;
- the run config's replay must reproduce the run's detections cache (every score is a replay).

A clip that fails any of these is refused.

**Scorecard per clip** (`python -m vision.bench`; it reads the detections cache, or a replay of it, and PFF's game state, so it lives in `vision/` like `vision.state_check`):
- **Geometry missing:** the share of scored frames where vision has no homography, split in two:
  - view `other` (the gate's lag after cuts lands here, and no homography setting moves it);
  - **homography rejected:** match view with no accepted fit, over match-view frames. That's smoke04's 17% (65 of 375 match-view frames) and the roadmap's target.
- **People in meters:**
  - Truth is PFF's `VISIBLE` players and keepers on the matched PFF frame. Vision is its player and goalkeeper rows that have pitch x/y. Referees are left out, since PFF has none.
  - Each frame gets a one-to-one match (Hungarian on distance) with a 5 m gate.
  - **Within 2 m** = matched pairs at ≤ 2 m over all truth rows, so misses and frames without geometry count as failures (detection review target ≥ 90%).
  - Next to it: the median and p90 error of matched pairs, the same three numbers on frames with geometry only (so accuracy and availability stay separate), and unmatched vision rows per scored frame.
- **Teams**, on matched pairs, outfield and keepers apart:
  - accuracy where vision's team isn't null;
  - coverage (the share that isn't null).

  Teams are read from `team_cluster` and `home_cluster`, so a replay keeps them; keepers get their cluster from position, as the pipeline does. If `home_cluster` is null, the scorer uses whichever mapping agrees more, and says so.
- **False live:** seconds inside the marks where vision has view `match` and a homography, so the predictor would get positions from a replay or close-up. Per label.
- **Ball** (2026-10-02), on verified labels. Checked 2026-10-02 (`Docs/reviews/ball-research-2026-10-02.md`):
  - PFF's VISIBLE ball is a grounded ball, within 25 px of the real one on 84% of frames, with ~1 m biases lasting 5–15 s;
  - airborne and lost balls are ESTIMATED and 44 px off (median), on 31–39% of live frames;
  - PFF can't mark a hidden ball.

  So PFF pre-places and checks the labels and gives a loose second score, never the targets' truth (10-ball). See Ball labels and Ball score below.
- **Not scored yet:**
  - tracking IDs;
  - possession.

**Ball labels:** `data/splits/vision_ball_labels.json`, committed next to the manifest (`data/vision_bench/` is gitignored). Made with `scripts/ball_click.py`.
- **With `--assist`** (10-ball 1c), each frame shows a suggestion: the ball candidate nearest PFF's projected ball. Enter accepts it. A click, space or `u` overrides it, as before.
  - **Auto-accept:** a confident, isolated candidate within 15 px of a VISIBLE projection is saved without being shown and listed under the clip's `auto`. On vb02, 81 of 82 such frames match the click within 15 px. A random 10% of them is shown for checking.
  - Every other label is what the person chose.
- **`--flag`** reopens labels that disagree with PFF's projection (> 30 px from a VISIBLE one, or `"none"` with a candidate near it) for a second look. PFF has its own biased stretches, so a flag never rejects a label by itself.
- vb02's first pass (2026-10-02) has known bad clicks (specks at 240–280, trailing clicks at 575–610, guesses where the ball wasn't visible); they're redone with `--flag` before vb02's ball score counts.
- `version`, and `clips`: `clip_id` → `video_sha256`, `every` (N), `labels`.
- **Which frames:** source-video frames whose index is a multiple of N, inside `[video_start_s, video_end_s)` and outside the marks (frame index / fps, as the scored frames). N = 5, so 6 labels a second, about 930 on the three clips:
  - neighbouring frames at 30 fps are near duplicates (the ball moves a few pixels), so N = 1 or 3 mostly re-measures the same frames for 5× or 1.7× the clicking;
  - 6 a second still puts 6 labels inside the longest gap the ball may be extrapolated over (`ball_max_gap_s` 1 s), so gaps and drift show up;
  - ~900 labels put a 90% recall at about ±1 pt (binomial SE), enough to see a fix worth 2–3 pt.
- **Keyed by the integer source-video frame index,** never `video_s` or a run's `frame_id`, so labels survive reruns with another pre-roll. A run's `frame_id` is the index minus `round(video_start_s × fps)` of its `run.json`.
- **Each label:** `[x, y]`, the ball's center in source pixels (1080p); `"none"`: not visible (off screen, or hidden behind a player); `"unsure"`: can't tell (heavy blur, a ball-like blob). `"unsure"` frames are left out of recall and precision.
- Pixels, allowed inside `vision/` under the detections-cache exception (03 Diagnostics). They never leave vision.

**Ball score** (`vision.bench`; needs the run's `balls.parquet`, so stage 5 is replayed like everything else, 03 Diagnostics). On labeled frames:
- **Where vision's ball is in the image:** a detected row's box center; an extrapolated row's pitch x/y projected back through the frame's homography, since its box is the last detection's and doesn't move.
- **Hit:** a vision ball row with pitch x/y within **R = 15 px** of the click. 15 px is about one ball diameter on the bench (median detected box 15–18 px wide at 1080p, 5th percentile 10.5): a click is good to ~2–3 px and a blurred ball's box center can sit half a diameter off, while wrong picks (heads, boots, line marks, spare balls) land tens to hundreds of pixels away. Recall at 10 and 25 px is printed next to it, so the choice can be seen not to matter. A row without pitch x/y never hits: it doesn't reach game state.
- **Recall:** hits / labeled-visible frames, all frames (end to end, the gate included) and on match-view frames with geometry (the ball stage alone). Detected and extrapolated hits apart.
- **Precision:** hits / vision ball rows with pitch x/y on labeled frames (`[x, y]` or `"none"`). A row on a `"none"` frame is a false ball. Detected and extrapolated apart.
- **Misses** (labeled-visible, no hit), each in the first bucket that fits:
  1. view `other`: the gate, not the ball stage;
  2. no geometry: match view, but the frame has no homography or the ball projects off the pitch;
  3. wrong pick: a candidate ≥ `min_det_conf` within R existed, another was picked;
  4. low confidence: a candidate within R existed only under `min_det_conf`;
  5. drift: no candidate within R, and an extrapolated row more than R away;
  6. not detected: no candidate within R (a far detection picked instead, if any, also counts against precision).
- **Error in meters:** median and p90 of the click and vision's ball both projected through the frame's homography, on hits. It's the image error in meters at that spot; the homography's own error is in the people score.
- **Targets** (roadmap, detection review): recall ≥ 90%, precision ≥ 95% on usable live frames (match view with geometry). On verified labels only.
- **PFF score** (10-ball 1b, 1d), printed under the ball score on every clip with a `match_id`, labeled or not, never against the targets.
  - **Truth:** PFF's ball projected through a PnLCalib camera solved on each label frame, at the ball's center, on frames where PFF says VISIBLE (`kind = pff`).
  - **Hit:** a vision ball row with pitch x/y within **40 px**.
  - **Printed with it:** recall, precision, recall at 25 px, the candidate ceiling, and vision rows on ESTIMATED frames (counted, not scored).
  - **Why 40 px:** on vb02 it gave the same verdict as the clicks on 92.5% of frames, recall 3.4 pt lower. At 25 px the verdicts agreed on 87.8% of frames and recall was 10.9 pt lower, because PFF's ball drifts ~1 m for seconds at a time.
  - **What it's for:** paired comparisons on the same frames (sweeps, A vs B) and clips without labels.
  - **Agreement check per labeled clip:** good `[x, y]` labels within 25 px of the projection (vb02 84.4%). A clip far below that has a sync or camera problem, and its PFF score is withheld. Withheld below **70%** once the clip has at least 20 such labels (`AGREE_MIN`, `vision/bench.py`). All labels count, the first-pass bad clicks included, so vb02 reads 76.5% until its flags are fixed (84.4% on its good clicks).

**Homography threshold sweep** (roadmap Phase 2, group 4). The sweep runs offline on the stage 4 cache (03 Diagnostics), with no detector rerun. What's swept depends on the run's `calib_backend` (03 Pitch calibration):
- **`pnlcalib`:** `max_calib_err_px`, the camera checks, `max_homography_jump_m`, `homography_max_age_s` and `homography_window` from the cached cameras; `pnl_kp_threshold` and `pnl_line_threshold` with `--revote` (voting redone on CPU from the cached peaks, ~30 s a clip). The checks also feed the gate, which the replay doesn't simulate, so a pick that tightens them is confirmed with a fresh run before it's adopted.
- **`roboflow`** (old runs): `ransac_m`, `min_inliers`, `max_homography_err_m`, `max_homography_jump_m`, `homography_max_age_s` and `homography_window`. The gate's `min_keypoint_conf` and `min_keypoints` aren't swept, because they change the view gate.
- A `keypoints_every` that's a multiple of the run's replays with every k-th call: the cost of calibrating less often, for the live budget (09). Reported next to the pick, never picked by it.
- **Pick:** the highest pooled within-2 m with homography rejected ≤ 17%.
- **Tie-break:** within 0.5 pt, the setting with the least homography rejected wins (the permissive lean: 05 says a rejection costs more than 2–4 m of error). `python -m vision.bench --grid FIELD=V1,V2 ...` sweeps the cartesian grid and prints the pick.
- Per-clip numbers for the pick and for the current defaults go in the review.

## Outputs

### Run format
Every model run writes one directory, `data/runs/<run_id>/` (gitignored), via `evaluation.runs.save_run`:
- **`run.json`**: `run_id`, `model`, `horizons` (e.g. `["h5", "h3"]`), `config` (anything the model needs to be rerun), `tau` (per horizon, per outer fold: `{"h5": {"0": 0.41, ...}}`, chosen on that fold's training matches only, plus `"final"` for the model trained on all CV matches, used for IDSSE; optional), plus `git_commit`, `git_dirty` and `created` stamped by `save_run`.
- **`predictions.parquet`**: `match_id`, `period`, `t_s` (copied from `frames_10hz`), and `p_h5` / `p_h3` for the horizons in `run.json`. Out-of-fold for CV matches: each match's p comes from the model that didn't train on its fold. Include every grid row the model sees, not just scored rows, since alarms run over all rows. p is null where the model doesn't predict (05).
- Fold and source are **not** stored in the predictions. The report looks them up (`folds.json`, `match.parquet`) so a run can't mislabel them.

### Report
`python -m evaluation.report data/runs/<run_id>` writes `data/runs/<run_id>/report.md`. Reports worth keeping get copied into `Docs/reviews/`.
- **Matches by source** (`match.parquet`): `pff` and `skillcorner` are CV and must be in `folds.json`. `idsse` is external. Anything else (Metrica) is dropped, and the report says how many.
- **Coverage is checked:** for each CV source in the run, every match `folds.json` lists must have predictions, or the report fails. Otherwise two runs on different matches would look comparable. `--allow-partial` renders anyway under a "PARTIAL RUN" header, for debugging only.
- **Strict join** on (`match_id`, `period`, `t_s` in whole tenths): duplicate prediction keys, or prediction rows that don't land on a `frames_10hz` row, are errors. A null p on a scored row names the match.
- **Per horizon, per CV source:** pooled out-of-fold metrics (rows, positives, base rate, PR-AUC, ROC-AUC, Brier), a per-fold table with mean ± std, a calibration table (10 quantile bins), and a τ sweep over pooled OOF. The sweep is descriptive only: it never picks τ.
- **Alarms at the chosen τ** come only from `run.json`'s per-fold τ: each fold is scored with its own τ on its own matches, then pooled (lead-time median and quartiles, a lead-time table in 1 s bins, misses, false alarms per match). Without τ in `run.json` the report says "τ not chosen" and shows no alarm numbers at a single τ.
- **IDSSE** is only rendered with `--final` (07: evaluate once per final model). Without it, the report just counts the IDSSE matches present.
- Calibration and lead time are tables, not plots, until a plotting dependency is added.
- The report records its own git commit next to the run's.
- Not yet: a per-stage (knockout) breakdown (stage isn't in 02).

### Paired comparison
`python -m evaluation.compare data/runs/<A> data/runs/<B> [--out FILE]` puts two runs side by side per fold: PR-AUC, ROC-AUC, Brier, and miss rate and false alarms per match at each run's own per-fold τ (left out if either run has no τ). Then, per metric, it counts the folds where A is better and gives the mean Δ. **"A wins" means better on most folds** (the rule above). Both runs must cover the same CV matches, or it fails. They're scored on the same rows, since scored rows come from the labels. CV sources only; IDSSE is never used to compare models.
- **Checked end to end** (2026-09-27) with the oracle from "Floor from the labels" written as a run with τ = 0.5 on every fold: the report gives the same 28 / 1,154 missed and 61 false alarms at H = 5 (75 with the null-hold cap) (H = 3: 28 missed, 28 false alarms, median lead 2.9 s). About 20 s for 64 games and both horizons on the M1.
