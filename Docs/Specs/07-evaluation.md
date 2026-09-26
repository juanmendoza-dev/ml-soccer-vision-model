# 07 — Evaluation

## Splits
~409 open-play shots over 20 SkillCorner matches (see 06) is too few for a fixed 70/15/15 split: the test set would be ~3 matches and ~60 shots. Use grouped cross-validation instead.

### Development: grouped k-fold on SkillCorner
- **5 folds grouped by match** (4 matches per fold). A match's frames are never split across folds.
- Fold assignment is fixed once, saved to `data/splits/skillcorner_folds.json`, and reused by every model so comparisons are paired.
- Balance folds roughly by shot count (e.g. `StratifiedGroupKFold` on match-level shot totals), fixed seed.
- Each outer fold: train on 16 matches, evaluate on 4.
- **Anything tuned is tuned inside the training matches only:** hyperparameters, early stopping, temperature scaling, and threshold τ. Use an inner grouped split (e.g. 3 of the 16 matches as inner validation). Never tune on the evaluation fold.
- Report:
  - **Pooled out-of-fold:** concatenate all 5 folds' predictions, then compute every metric once over all 20 matches. This is the headline number.
  - **Per-fold spread:** mean ± std across folds, to show how much the result depends on which matches were held out.
- Model comparisons are **paired by fold**: model A beats model B only if it wins on most folds, not just on the pooled number.

### Final check: external test set
- **IDSSE (7 Bundesliga matches)** is the untouched test set. Different league, optical tracking. Evaluate once per final model, trained on all 20 SkillCorner matches with settings chosen during CV.
- No tuning, threshold picking or model selection on IDSSE. If a result there changes a decision, note it in the report.
- If PFF access comes through: add it to the grouped-CV pool (group by match), and keep IDSSE as the external test.

### Rules
- Splits are by match, never by frame or by possession.
- Player profile features (04) come from seasons before the match, so the same player appearing in train and eval folds is not leakage. Per-match stats from the eval match are.

## Metrics
| Metric | Why |
|---|---|
| PR-AUC | Main metric; positives are rare |
| ROC-AUC | Comparable to papers |
| Brier score + calibration curve | Probabilities must mean what they say |
| Lead time | For each shot, seconds between probability first crossing threshold τ and the shot; report median and distribution |
| False alarms per match | At the same τ; keeps lead time honest |

## Required comparisons
1. Baseline vs. frame GNN vs. temporal GNN.
2. With vs. without player profiles.
3. H = 3 s vs. H = 5 s.
4. Dataset tracking vs. vision-pipeline tracking on the same matches, if available (measures how much vision errors hurt).
5. Provider vs. inferred possession/ball state (03 stage 8) on dataset tracking: same model, same folds. Measures how much the inference rules alone cost before vision errors are added.

## Outputs
- `evaluation/report.md` generated per run: pooled out-of-fold metrics table, per-fold table, calibration plot, lead-time histogram. IDSSE results in a separate section.
- Every run logs config + git commit.
