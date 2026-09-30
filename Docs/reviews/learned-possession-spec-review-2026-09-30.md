# Review: learned possession spec (03 stage 8), 2026-09-30

Spec under review: 03 stage 8 "Learned possession" (lines 42–143), 05 "Learned possession", 07 #6 "Learned possession follow-up", and the roadmap item, from commits 3fc5690, fef7904, 23cf1a9 and 9ede7eb. I checked the claims against the code and data on the M1 using read-only scripts. Nothing was trained and no caches were written.

## Verdict
**Ready after fixes.** The design is leak-free: I checked all 25 fits against `folds.json` and traced every feature group. One blocking gap remains. The spec never names the existing call sites that hardcode or ignore the stage 8 config: `cv.py:472`, `load_data`, `infer`/`inferred_state` and `possession_staleness.py:158,169`. Built as written, a learned run would record the wrong config, or silently write an unsuffixed rule cache under the learned key. Several should-fix items also need an answer before coding: the timing plan has no basis (and 10 of the 25 fits are duplicates), churn isn't gated, the gate can be retried without limit, and some definitions are ambiguous.

## Blocking

### B1. The spec doesn't name the call sites that hardcode or ignore the config, so a learned run would record the wrong config and could write an unsuffixed rule cache
- **Spec:** 03:114 ("`--possession-model` sets the config field"), 03:120 ("retain the existing `possession`, `state_config`…"), 05 Plumbing. These assume `state_config` in run.json is the config that actually ran, and that no path treats a learned config as rule-only.
- **Code today:**
  - `prediction/cv.py:472` writes `possession.StateConfig().to_dict()`, the hardcoded default, whatever ran.
  - `load_data` (`cv.py:114-146`) has no `state_config` parameter, so `load_match` always gets `None` and uses the default (`features.py:660`).
  - `vision.state.infer` and `prediction.possession.inferred_state` accept any `StateConfig`. Neither would read `possession_model`. `inferred_state` writes `state_inferred_age_<config_key>.parquet` (`possession.py:72`), which is unsuffixed.
  - `scripts/possession_staleness.py:158` rebuilds `StateConfig(**run.json state_config)` and calls `inferred_state(...)` at line 169.
- **Failure:** someone points `possession_staleness.py` (or a future split script) at the learned run.
  - If run.json still carries the hardcoded default, the script analyzes the default rule and labels it as the learned run.
  - If run.json carries the real config, `inferred_state` runs the rule-only `infer` under the learned key. It writes `data/processed/<m>/state_inferred_age_<learnedkey>.parquet` holding rule output. That breaks the spec's own "no learned run reads or writes an unsuffixed state/feature cache" (03:118).
  - Either way the provenance is wrong, and nothing fails.
- **Fix:** spell these out in 03/05:
  - `load_data` takes and forwards `state_config`, and `run()` records the config that ran.
  - `infer`, `StateMachine`, `inferred_state` and `vision.state_check` raise when `possession_model` is set. Only the learned wrapper accepts it.
  - `possession_staleness.py` and `possession_split.py --state` refuse learned configs. `--state possession_model=…` would crash anyway, since it parses values as float (`possession_split.py:39`).
  - Add tests for all of it (S9).

## Should fix

### S1. The timing prior has no basis, the fallback probably dead-ends, and 10 of the 25 fits are duplicates
- **Where the 35 s comes from:** it's an M1 number. 09:166 says "LightGBM runs stay on the M1" and measured 35 s per fold per horizon there. The spec times on the workstation's i5-11400F with `num_threads = 4`. The baseline sets no thread count, so it used every core.
- **Sizes:** the baseline outer fit (probe + refit) took 13–20 s on 1.57–1.63M rows × 30 columns, with `best_iter` 113–216 (`lgbm-pinf-2026-09-29/run.json`). That's about 35 ms per round over roughly 400–500 rounds. A possession fit has 3.21–3.32M mirrored rows × 117 columns (Verified), about 8× the work per round, so about 0.28 s per round at M1 speed.
- **Rounds decide the rest:** if the probe stops near the baseline's round counts, a fit takes about 2–3 min and the prior holds. Possession is an easy, high-signal target, and at learning rate 0.05 log loss can keep dropping for 1,000–2,000 rounds. The probe plus the refit would then be about 2,000–4,000 rounds: **about 10–20 min per fit at M1 speed.** It's slower still if 4 threads on the i5 are slower than the M1's cores, which is unmeasured.
- **The duplicates:** only 15 of the 25 fits are distinct trainings.
  - `outer_k_inner_j` and `outer_j_inner_k` both train on `U − (F_j ∪ F_k)`, and the early-stopping draw depends only on that set (03:57). With `deterministic: true`, fixed threads and sorted IDs, they're the same booster. The spec requires keeping all 20 even so (03:100).
  - Keep the 25 logical IDs and roles, but train 15 models (the table under Verified). Each pair ID points at one stored model and training-set hash.
  - Nothing about the nesting changes. The pair model predicts F_j for outer k and F_k for outer j, and neither side saw its own test fold.
  - This is the cheapest fix: it saves 40% of possession training. Add a determinism test: train one pair twice and assert the model SHA256s match. If that fails, the pair has to be trained twice, and the test says so.
- **So the fallback dead-ends:** the 5-min bar would trip, stride 2 would most likely still exceed 5 min, and the plan stops for a new timing plan (03:132). That's exactly the case the fallback was meant to settle in advance.
- **Feature extraction isn't timed:** the 2D rule plus the 117-column extractor over about 11.7M native frames (64 matches) sits outside the pilot's pass/fail. A per-frame Python loop could take longer than all the fits together.
- **Fix:**
  - Set the limit as a total wall-clock budget for the 15 trainings (see Q1), not 5 min per fit with no reason given.
  - Record the probe's round count in the pilot.
  - Time the extraction on one match and extrapolate to 64.
  - Fix the fallback order now, for example stride 2, then stride 4. Stride is already predeclared, and neither step looks at accuracy.
  - Say which machine runs it. If it's the workstation, update 09.
  - Use 6 threads (physical cores) or say why 4.

### S2. Churn isn't gated, and flicker hurts v1 even on rows it gets right
- v1 has no hysteresis or dwell, and a hard 0.5 threshold (03:87-88). Rows near p = 0.5 will flip back and forth at 10 Hz.
- Each flip resets `possession_s` (`features.py:318-329`, a v1 feature), even when possession is right on both sides of the flip. It also rotates the frame for one row.
- A model can pass all four row-count bars while flickering, and still lose PR-AUC in 07 #6 for a reason the gate never saw. The earlier review already flagged this pattern: "the all-frames runs paid for fewer lag errors with extra flip-flops."
- **Fix:** add a churn bar before any result. For example, learned possession changes per match ≤ the default rule's on the same matches. Or tell the user it's descriptive only (Q3). Churn is saved (03:127) but decides nothing.

### S3. Nothing limits how many times the gate can be retried
- 03:127 says a failure means "return to spec review". Nothing stops v2, v3 and so on from being gated on the same 1,977,379 pooled rows until one passes, followed by 07 #6 "once".
- That run is chosen on the test rows. It's chosen on disagreement, not PR-AUC, but the two are correlated.
- **Fix:** cap it now. For example, v1 plus at most one revision, with every attempt and its counts listed in the review next to the eventual 07 #6 result.

### S4. No action is fixed in advance for each diagnostic result, and the diagnostic skips the failure that killed the fast rules
- 03:51 asks only to "record whether the diagnostic supports" v1. It should say now what each bucket result leads to:
  - **Height gate:** the 2D rule has no height. Report how many residual rows the 2D rule, rather than the height-gated rule, already calls right. The diagnostic uses the historical rule (03:46), but the model's features come from the 2D rule.
  - **ESTIMATED player on the ball:** no VISIBLE-only model can reach these. Subtract them from what's reachable and report it.
  - **No visible player in reach:** only anticipation cues can help (where the ball is heading, relative to each team). If this bucket is large, add S5's receiver features before freezing.
  - **Timing mismatch:** unreachable. Report the size.
- The diagnostic targets the ball-visible lag rows only. The fast rules failed on something else: extra errors on positives, more than 10 s after PFF's change, with the ball visible (stage 8 review, lines 94-99).
- Add one descriptive table on the `team_near_s = 0.05` rule's extra positive errors against real turnovers. Show how the v1 shape features differ between the two groups (for example, the attacking team's max X and past-halfway counts). If they don't separate, the model's main hypothesis (03:44) is in trouble before any fit.

### S5. The features probably miss the cues that separate a turnover from a touch
The v1 set is mostly a snapshot of shape and contact, plus two lags. Missing:
- **Where the ball is heading relative to each team:** the nearest player of each team to the ball's extrapolated position at +0.3/+0.5 s, or the angle between the ball's velocity and the nearest home/away player. This is the pass-in-flight and interception cue, and it's causal (extrapolated from past sightings).
- **Last contact of any length:** the last team within 1.5 m and the seconds since, with no 0.3 s minimum. Right now that only shows up blurred in the 0.5/1 s shares.
- **Pressure:** counts of each team within 5 m and 10 m of the ball.
- **What the camera covers:** the X extent of `view_polygon`. Shape comes from visible players only. On scored rows (12 matches checked), each team averages 6.8 visible outfield players, and 27% of rows have fewer than 5 of one team. So `highest_x`/`deepest_x`/`past_halfway`/`visible_n` partly measure the camera, and the model can't tell "deepest visible" from "deepest".
- **A note, not a bug:** under M, `diff_centroid_x`, `diff_highest_x` and `diff_deepest_x` don't change, so on their own they carry no direction. The direction is in home + away. The table reads as if the differences carry the signal.

### S6. Some definitions two engineers would implement differently
- **`<team>_highest_x` / `_deepest_x` (03:78):** "Max / min X … away's highest attacking position is its minimum X" reads two ways: plain max/min for both teams, or "highest" meaning most advanced (min X for away). The mirror rule at 03:86 only works with plain max/min. Rename them `max_x`/`min_x` and drop the "attacking" wording.
- **`rule_team` when the rule has no possession** (null before the period's first carrier): is it coded 0 or NaN? Not stated.
- **`rule_candidate_team`:** it uses 0 both for "no ball/candidate" and "candidate with no team". Unknown team never happens on PFF but is common live. Use NaN for absent and 0 for unknown team.
- **Contact-share window start (03:82):** say that the native frame before `t − W` contributes the part of its hold interval inside the window. The formula implies it, but the "integrate over" wording doesn't say which frame covers the start.
- **"Usable" and `interpolated`:**
  - On all 64 PFF matches, 0 rows are both `visible` and `interpolated`, so training never sees them.
  - Vision does produce them: tracker fills (`vision/pipeline.py:203`, `interpolated=tr.tracked_only`), and offline stage 5 ball interpolation (03:24), which uses the detection after the gap and so isn't causal.
  - Define usable as `visible & ~interpolated`, or say which interpolated rows count (live extrapolation, which is causal) and which are refused (offline stage 5 output).

### S7. Plumbing gaps
- **Who writes the learned grid state?** The model lives in `vision/`, but the cache sits in `data/processed/<m>/state_inferred_age_<key>_<context>.parquet` (03:117), which is prediction's. Name the function. For example, `vision.possession_model.predict_match(model_dir, fit_id, match) -> grid table`, called by `prediction/possession.py`, which writes the cache and its provenance.
- **Loop order:** `run_cv` loops horizons outside folds (`cv.py:216-218`). "One outer context at a time" then means 12 assemblies (5 folds + final, × 2 horizons), about 4 min at the 21 s load time seen in `lgbm-pinf`. Say whether the learned branch reorders the loops or reloads. Either is fine. Keep the old branch untouched.
- **Stats counting:** `add_stats` accumulates on every `load_match` call (`features.py:677`), so repeated contexts would multiply the counts. Say that only `test`-role loads add stats. The `training_rows(test_data, …)` meta block (`cv.py:268`) needs the canonical table.
- **Redundant names:** in `outer<k>_inner-oof<j>`, j is always the match's own fold, and `outerall_final<j>` duplicates `outer<j>_test` byte for byte. That's fine if it's intended. Otherwise alias `final` through provenance instead of copying.

### S8. Cache invalidation relies on a rule the code doesn't enforce
- 03:119 says "Reconvert always implies resample." Nothing in `converters/` touches `data/processed`, and `process_game` only deletes the three globs at `resample.py:223`, none of them `.json`.
- `possession_inputs_*` are built from `data/gamestate`, not the resampled tables. So a resample is the wrong trigger. A reconvert without a resample would leave stale inputs.
- **Fix:** make the native-input hash check on every read the real guard. Deleting them on resample is a harmless extra. List the exact new globs: `state_inferred_*.json`, `features_v*.json`, `data/vision_cache/<m>/possession_inputs_*`.

### S9. Missing tests
- Refit determinism: same training set, same model SHA256. It backs the dedupe in S1 and the manifest's immutability.
- Learned config refused by `infer`, `inferred_state`, `state_check`, `possession_staleness.py` and `possession_split.py --state` (B1).
- run.json `state_config` equals the config that ran, for the default, `team_near_s` and learned runs.
- `all_estimated` parity between vision and the resampler. It's required in prose (03:53) but isn't in the test list.
- Training-row selection: dead, null-possession and all-ESTIMATED rows are excluded, set plays are kept, and the early-stopping IDs reproduce `default_rng(20260927).choice` on sorted IDs.
- The gap and segment rules can only be tested with synthetic fixtures. PFF has no native gaps over `1.5 / native_fps` (0 of 11,727,099 intervals, all 64 matches), so a run on real data would never exercise them.

### S10. Say what agreement can't prove
03:55 mentions PFF's lag in general terms only. Name the evidence: the oracle floor loses 25 of its 28 misses to PFF crediting the other team 1–4 s before a shot (07:72). Wherever PFF is wrong like that, a model trained on PFF is rewarded for copying the mistake. Report disagreement on those 25 windows separately, as a description only.

## Nice to have
- **Live latency:** p99 < 100 ms (03:143) is a whole tick. If stage 8 runs in the frame loop, that drops about 3 frames at 30 fps. Run it off the frame thread, or use p99 < 33 ms. Stage 8 isn't wired into the pipeline at all yet: `vision/writer.py:90` writes null.
- **History buffer:** state it explicitly for live. That's the last 11 grid vectors for the lag blocks, plus native ball, contact and per-track windows of at most 1 s.
- **Attack direction live:** all 117 features depend on `home_attacks_positive_x`. PFF gets it from metadata (03:185), and no vision stage produces it yet. It has to exist before live.
- **Vision cache key:** keying `possession_inputs` by model ID duplicates identical inputs for `s1` and `s2`. Key them by feature-contract version.
- **Early stopping:** it watches raw q, while the output is the symmetrized p. That's fine, but say so.
- **Provider smoothing:** PFF's VISIBLE positions may be smoothed by PFF (this can't be checked here). The 0.2 s velocity features would be the most exposed. The project already makes the same assumption elsewhere.
- **Stale header:** 03:8 says schema v0.4. 02 is at 0.6.
- **09 needs updating too:** its stage table (09:148) still calls stage 8 "Negligible". With 117 features and two LightGBM predictions per tick, that's no longer true. Update it along with the M1-vs-workstation point in S1.

## Questions for you
1. What wall-clock budget is acceptable for possession training on the chosen machine, for example overnight (≤ 6 h for 15 trainings)? S1's fallback depends on it.
2. OK to train 15 models behind the 25 logical fit IDs (S1)? The nesting is unchanged.
3. Should churn be a fifth gate bar (for example, changes per match ≤ the default rule's) or descriptive only (S2)?
4. How many learned versions may be gated on the pooled rows before the plan is dropped (S3)?
5. The gate asks for a 10% cut in disagreement. By a rough linear read (−0.038 at 14.2% disagreement), "small" (−0.010) would need about 3.7% disagreement, roughly a 74% cut. That's crude: the −0.014 on agreeing rows breaks linearity. Do you still want the one 07 #6 run on a model that clears only the 10% bar, or should the gate be tighter?
6. Which machine runs the possession fits: the M1 (09:166) or the workstation (03:130)?
7. Do you accept the τ-validation overlap the spec concedes (03:109)? In every outer fold, all 4 inner fits train on some of that fold's 8 τ-validation matches. My read is yes. PR-AUC is threshold-free, so only the own-τ alarm counts are touched.

## Verified
- **Default config key:** `config_key(StateConfig())` = `e737054b5d` (run 2026-09-30). `to_dict()` drops None values (`state.py:45-48`), so a new `possession_model: None` keeps the key. Old run.json `state_config` dicts lack the field and still load through `StateConfig(**…)` because of the dataclass default.
- **Folds:** `folds.json` has 64 PFF matches in folds 12/13/13/13/13, frozen `pff: 2026-09-26`.
- **The nested scheme holds for every fit.** Checked from the real fold lists (full table below):
  - Every fit used when scoring outer k (`outer_k` and its four inner fits) trains on none of F_k.
  - No fit predicts a match it trained on.
  - Early-stopping holdouts are drawn inside each training set.
- **Every feature uses frames ≤ t only.** Traced group by group:
  - The `rule_*` columns come from the 2D state machine at native frame u ≤ t.
  - Ball columns use sightings at or before t within the segment, held at most 1 s.
  - Ball velocities use visible endpoints in [t−W, t], with no provider vx/vy.
  - Near-distances and shape use frame u only. Per-track VX uses [u−0.2, u].
  - Contact shares integrate over (t−W, t]. A frame's hold reaches the next frame only up to t.
  - Lags read stored past vectors, and the symmetrization uses Mx at the same t.
  - PFF ball_state, possession, events, z, player_id and provider vx/vy never enter the matrix. ball_state and possession only pick training rows.
  - The only exposures left are in the inputs themselves: interpolated rows on vision output (S6) and possible provider smoothing (Nice to have).
- **Final τ can't move 07 #6:** it's read only by `external_section` (`evaluation/report.py:289`, IDSSE). It uses `outer_j` for F_j, so every match's possession is out of fold.
- **LightGBM settings:** the listed parameters match `prediction/lgbm.PARAMS`. The early-stopping and refit procedure matches `LGBMModel.fit` (`lgbm.py:71-82`).
- **`all_estimated`:** the definition matches `resample.add_frame_flags` (`resample.py:97-108`). The grid rule and integer microseconds match `build_grid` (`resample.py:49-84`).
- **Bars and buckets:** 1,977,379 scored rows, 49,812 positives, and the four bar numbers match the stage 8 review (lines 77-87). The bucket labeled `> 10 s` does include exactly 10 s (`possession_split.py:98-107` uses `< hi`). 07 #6's decision lines are unchanged (07:88-91).
- **Columns and mirror:** 39 columns × 3 = 117. The mirror rules for team codes, distances, shares, counts, `mean_vx_own` and `past_halfway` check out algebraically.
- **PFF data, all 64 matches:**
  - 0 rows are both `visible` and `interpolated`.
  - All 4,890,788 visible ball rows have `z`, so the historical rule's height gate is always active. The 2D rule really does differ from the comparator.
  - There are no native gaps.
- **Memory:** there are 2,022,220 training rows (alive, PFF team, not all-ESTIMATED). The largest fit is 1.66M rows, 3.32M mirrored: 3.32M × 117 × 4 B = 1.55 GB raw, plus about 0.4 GB binned. One outer context of shot features is about 3.9M rows × ~43 columns, around 1.5 GB. All of this fits under 24 GiB if contexts load one at a time.
- **Schema:** no change is needed. All inputs are existing 02 fields.

### Fits against training matches
Early-stopping holdout = `round(0.15 × matches)`. Rows are PFF-alive, team known, not all-ESTIMATED, before mirroring.

| training ID | logical IDs | trains on folds | matches | ES holdout | rows | predicts |
|---|---|---|---|---|---|---|
| outer_0 | outer_0 (test) | 1,2,3,4 | 52 | 8 | 1,660,373 | F0 (12); also `final` for F0 |
| outer_1 | outer_1 (test) | 0,2,3,4 | 51 | 8 | 1,604,276 | F1 (13); `final` F1 |
| outer_2 | outer_2 (test) | 0,1,3,4 | 51 | 8 | 1,604,571 | F2 (13); `final` F2 |
| outer_3 | outer_3 (test) | 0,1,2,4 | 51 | 8 | 1,611,293 | F3 (13); `final` F3 |
| outer_4 | outer_4 (test) | 0,1,2,3 | 51 | 8 | 1,608,367 | F4 (13); `final` F4 |
| pair_01 | outer_0_inner_1, outer_1_inner_0 | 2,3,4 | 39 | 6 | 1,242,429 | F1 for outer 0; F0 for outer 1 |
| pair_02 | outer_0_inner_2, outer_2_inner_0 | 1,3,4 | 39 | 6 | 1,242,724 | F2 / F0 |
| pair_03 | outer_0_inner_3, outer_3_inner_0 | 1,2,4 | 39 | 6 | 1,249,446 | F3 / F0 |
| pair_04 | outer_0_inner_4, outer_4_inner_0 | 1,2,3 | 39 | 6 | 1,246,520 | F4 / F0 |
| pair_12 | outer_1_inner_2, outer_2_inner_1 | 0,3,4 | 38 | 6 | 1,186,627 | F2 / F1 |
| pair_13 | outer_1_inner_3, outer_3_inner_1 | 0,2,4 | 38 | 6 | 1,193,349 | F3 / F1 |
| pair_14 | outer_1_inner_4, outer_4_inner_1 | 0,2,3 | 38 | 6 | 1,190,423 | F4 / F1 |
| pair_23 | outer_2_inner_3, outer_3_inner_2 | 0,1,4 | 38 | 6 | 1,193,644 | F3 / F2 |
| pair_24 | outer_2_inner_4, outer_4_inner_2 | 0,1,3 | 38 | 6 | 1,190,718 | F4 / F2 |
| pair_34 | outer_3_inner_4, outer_4_inner_3 | 0,1,2 | 38 | 6 | 1,197,440 | F4 / F3 |

Goal-model τ-validation sets (`inner_split`), with the fold of each match:
- outer 0: 8 matches from folds 2,2,1,1,2,4,4,2
- outer 1: 2,0,4,3,2,4,2,0
- outer 2: 1,1,3,4,3,3,3,3
- outer 3: 2,0,2,2,2,4,1,0
- outer 4: 3,2,0,1,0,2,3,2
- final: 10 matches, folds 4,3,0,3,0,0,0,3,3,0

None of them falls in its own outer test fold. Each outer fold's τ-validation set covers every other fold except fold 3 in outer 0 and fold 0 in outer 2, so every inner fit or pair model trains on at least one τ-validation match. That's the overlap 03:109 concedes (Q7).
