# Learned possession v1h on the shot model (07 #6, 2026-10-01)

03 stage 8, Checks step 3. The LightGBM baseline retrained with its input possession from learned possession `lgbm-v1h-nested5x4-s1` (nested: inner-OOF possession for training matches, outer-test for the held-out fold, final from outer test), instead of PFF's. Same v1 features, held ball, seed, folds and τ rule as `lgbm-held-2026-09-27`; labels, scored rows and alarms stay on PFF. Run `lgbm-plearn-v1h-2026-10-01` at `7c80b7e`, both horizons, once, no tuning. Comparisons by `evaluation.compare` and `scripts/lead_time.py`; outputs in the run folder (`compare.md`, `compare_pinf.md`, `lead_time_h5.md`, `lead_time_h3.md`, `report.md`).

## Attempts through the gate

| model | overall | first 3 s | positives | late rate | churn | gate |
|---|---|---|---|---|---|---|
| `lgbm-v1-nested5x4-s1` | 150,748 | 90,441 | 1,450 | 0.0278 | 69,357 | FAIL (churn) |
| `lgbm-v1h-nested5x4-s1` | 160,257 | 116,492 | 1,081 | 0.0169 | 15,829 | PASS |

Limits 253,445 / 131,585 / 3,862 / 0.0551 / 18,182. v1h is v1's boosters with hysteresis at 0.75 / 0.25 and the model's team held through fallback rows (`Docs/reviews/possession-smoothing-2026-10-01.md`). On outer-test scored rows it differs from PFF on 8.1% (the default rule: 14.2%).

## PR-AUC, paired by fold

| | H = 5 mean Δ | folds better | H = 3 mean Δ | folds better |
|---|---|---|---|---|
| v1h vs provider (`lgbm-held-2026-09-27`) | **−0.0277** | 0/5 | −0.0283 | 0/5 |
| default rule vs provider (`lgbm-pinf-2026-09-29`) | −0.0380 | 0/5 | −0.0363 | 0/5 |
| v1h vs default rule | +0.0103 | 5/5 | +0.0081 | 5/5 |

H = 5 per fold, v1h / provider: 0.303 / 0.330, 0.254 / 0.276, 0.254 / 0.284, 0.251 / 0.285, 0.291 / 0.316; pooled 0.268 / 0.296. ROC-AUC −0.016 (0/5), Brier +0.0005 (0/5).

**Decision (07 #6's fixed lines): big.** −0.0277 is past −0.018, worse on 5/5 folds. The spec expected "in between" or "big" even after a pass. Learned possession recovers 0.010 of the default rule's 0.038 loss (27%), on every fold at both horizons; H = 3 is a check and agrees.

## Alarms, lead time, calibration (H = 5, descriptive)

- **Own τ:** miss rate 0.003 lower than provider (better on 4/5 folds), but 1.09 more false alarms per match (better on 1/5). Against the default rule: 1.49 fewer false alarms per match (5/5), miss rate 0.048 higher.
- **Matched budgets, pooled:** at ≤ 3 false alarms per match v1h catches 145 shots vs provider's 172; at ≤ 12, 398 vs 401.
- **Lead:** median 0.63 s vs 0.71 s at own τ; shots caught with ≥ 2 s lead: 2 vs 3. Median p before open-play shots runs about 10% lower at 5 / 2 / 1 / 0.2 s (0.046 / 0.139 / 0.204 / 0.266 vs 0.051 / 0.158 / 0.225 / 0.296).
- **By time to shot:** lower PR-AUC in every bucket on 0/5 folds, worst in the last second (0.221 vs 0.257).
- **Calibration:** fine, as for provider: the top decile's mean p 0.174 vs positive rate 0.172.

## What it means

- Better possession helps: 8.1% disagreement instead of 14.2% buys back 0.010 PR-AUC, consistently. It doesn't buy back enough. A straight line through the two points (14.2% → −0.038, 8.1% → −0.028) would need disagreement near 0% for "small", which matches 03's warning that the loss isn't linear in disagreement.
- The remaining disagreement sits where the shot model looks hardest: first 3 s after PFF's change (116,492 of 334,949 rows) and the last second before a shot. v1h trades first-3-s accuracy for churn by design.
- Per 03 there's no further possession revision through the gate (v1 plus one). The open choices are outside learned possession v1: run the goal model on learned possession and accept the gap, train the goal model so it tolerates possession noise (roadmap Phase 1, "the model that runs on vision output with the degradations on"), or reduce how much the features lean on possession (05 spec work).
