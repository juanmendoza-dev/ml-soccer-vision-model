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
