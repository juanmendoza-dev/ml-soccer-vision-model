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
- **Status:** PFF was frozen on 2026-09-26 with all 64 games. SkillCorner gets appended once its converter exists, without moving any PFF match. PFF folds: 12/13/13/13/13 matches with 202/227/230/238/230 open-play shots (1,127 total). The 16 knockout games fall 2/3/5/4/2 across folds 0–4; stage isn't stratified (not in 02), so a per-stage breakdown in reports is worth having.
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
| PR-AUC | Main metric; positives are rare |
| ROC-AUC | Comparable to papers |
| Brier score + calibration curve | Probabilities must mean what they say |
| Lead time | Seconds from the start of the alarm active at the shot to the shot (see below); report median and distribution |
| Missed shots | Share of shots with no alarm active at the shot |
| False alarms per match | Alarms with no shot (see below); keeps lead time honest |

### Alarms and lead time
The probability rises and falls, so "first crossing τ" is ambiguous. Use alarms with hysteresis:
- **Alarm starts** when P(shot) rises above τ.
- **Alarm ends** when P(shot) drops below 0.8·τ, possession changes, or `ball_state` becomes dead. Dips between 0.8·τ and τ don't end it.
- **True alarm:** the possessing team shoots while it's active, or within 1 s after it ends (grace for the last-frame drop). Otherwise it's a **false alarm**.
- **Lead time** = shot time − start of the alarm active at the shot. A shot with no active alarm is a **miss**, not a lead time of 0.
- One alarm can cover several shots (rebounds); each shot gets its own lead time from the same start.
- τ is chosen on training matches only (see Splits). Also report the trade-off across τ values: median lead time and miss rate vs. false alarms per match.
- 00's success criterion (median lead time ≥ 2 s) is measured at the τ chosen on training matches.

## Required comparisons
1. Distance + angle floor vs. LightGBM baseline vs. frame GNN vs. temporal GNN.
2. Player profiles: none vs. position only vs. full (see 04).
3. H = 3 s vs. H = 5 s.
4. Dataset tracking vs. vision-pipeline tracking on the same matches, if available (measures how much vision errors hurt).
5. Full tracking vs. broadcast view (off-camera players dropped, see 05) on the same folds. **SkillCorner folds only** (plus IDSSE at the final check): PFF's off-camera positions are ESTIMATED and ~12 m off at shots (06), so its full view isn't a meaningful arm.
6. Provider vs. inferred possession/ball state (03 stage 8) on dataset tracking: same model, same folds. Measures how much the inference rules alone cost before vision errors are added.

## Outputs
- `evaluation/report.md` generated per run: pooled out-of-fold metrics table (PFF and SkillCorner rows), per-fold table, calibration plot, lead-time histogram. IDSSE results in a separate section.
- Every run logs config + git commit.
