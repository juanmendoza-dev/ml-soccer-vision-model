# Stale possession (07 #6 follow-up, 2026-09-29)

The inferred-possession run (`possession-inferred-2026-09-29.md`) cost −0.038 PR-AUC, nearly all of it on rows where stage 8 carries a stale team forward. This review measures when stage 8 is wrong (`scripts/possession_staleness.py`) and then runs the three model-side arms set in 05 "Stale possession" and 07 #6 before any run. All at `0b769be`, clean tree, same folds, seed, held ball and τ rule as `lgbm-held-2026-09-27`:
- **Arm S** `lgbm-pinf-age-2026-09-29`: inferred possession + `poss_carrier_age_s` (features v3)
- **Arm U** `lgbm-pinf-unknown-2026-09-29`: inferred possession, set to unknown when the carrier age is over S = 10 s
- **Control** `lgbm-held-v3-2026-09-29`: provider possession + v3, to check whether the feature helps on its own

## Bottom line
- **Neither arm fixes it.** H = 5: arm S −0.037 (0/5 folds), arm U −0.040 (0/5 folds), against the provider baseline. Both are well past the "big" line (≤ −0.018, worse on ≥ 4 folds), and nowhere near "small" (above −0.010).
- **Against plain inferred possession (−0.038) they recover close to nothing.** S gets +0.001 back (4/5 folds, too small to matter). U is −0.002 worse (1/5).
- **The feature doesn't help on its own either.** Provider + v3 vs provider v1: −0.0006, better on 3/5 folds, below the 4/5 bar set before the run. So arm S's cost is read against provider v1, as the rule says.
- **Verdict (07 #6, set before the runs): stage 8's rules change next.** S+U isn't run, since neither helped alone.
- **Why:** 85% of the disagreeing rows are PFF changing possession while stage 8 hasn't followed (median carrier age 8.6 s). Telling the model the row is stale can't fix the frame: the pitch is still rotated toward the wrong goal. And "unknown" isn't neutral, because with no rotation the ball features point at the +x goal. What's missing is detecting the change of possession, which has to happen in stage 8.

## When stage 8 is wrong (64 matches, H = 5 scored rows)
Carrier age is the seconds since stage 8 last confirmed a carrier. It's causal and reads only 02's `ball_carrier_id` (05).

| carrier age (s) | scored rows | positives | disagree with PFF | PR-AUC provider | PR-AUC inferred |
|---|---|---|---|---|---|
| null (no carrier yet) | 0.1% | 0.0% | 100% | 0.020 | 0.003 |
| 0 | 29.4% | 30.8% | 2.1% | 0.358 | 0.338 |
| (0, 0.5] | 12.2% | 13.0% | 4.6% | 0.299 | 0.261 |
| (0.5, 1] | 11.3% | 11.5% | 4.4% | 0.297 | 0.259 |
| (1, 2] | 14.1% | 14.3% | 7.0% | 0.268 | 0.229 |
| (2, 3] | 6.8% | 8.1% | 15.0% | 0.245 | 0.193 |
| (3, 5] | 7.5% | 8.8% | 25.1% | 0.271 | 0.211 |
| (5, 10] | 7.2% | 7.9% | 37.1% | 0.249 | 0.186 |
| > 10 | 11.3% | 5.5% | 51.5% | 0.159 | 0.083 |

- **Disagreement climbs steeply with carrier age:** 2% at 0 s, 51.5% past 10 s. That's where S = 10 s came from, chosen from the disagreement column only, before any run.
- **A missing ball makes it worse:** 30.5% disagreement with no VISIBLE ball on the row vs 8.7% with one. Within each age bin it's still higher without a ball (e.g. 1–2 s: 12.1% vs 5.4%).
- **Stage 8 lags PFF's changes.** Within 1 s of PFF's own possession change they disagree on 55.6% of rows, 1–2 s 46%, 2–3 s 35%, and past 10 s only 4.5%.
- **Who changed last, on the disagreeing rows:** PFF changed more recently on 85.3% (median carrier age 8.6 s). There PFF's team shoots next 2× as often as stage 8's (0.99% vs 0.47%). Stage 8 changed more recently on 14.7% (median age 0.9 s), and PFF's team shoots next 18× as often (3.7% vs 0.2%). So when stage 8 does switch against PFF it's usually wrong too, but those rows are the smaller part.
- Carrier age quartiles on disagreeing rows: 2.8 / 7.2 / 24.1 s. On agreeing rows: 0.0 / 0.6 / 2.0 s.
- The PR-AUC and shot columns are descriptive: the bins come from stage 8's output, and nothing here was used to pick S.

## Headline (PFF pooled out of fold)
| | provider v1 (`lgbm-held-2026-09-27`) | provider v3 | inferred v1 (`lgbm-pinf-2026-09-29`) | arm S | arm U |
|---|---|---|---|---|---|
| PR-AUC, H = 5 | 0.296 | 0.296 | 0.258 | 0.259 | 0.255 |
| Δ vs provider v1, mean over folds | | −0.0006 | −0.038 | −0.037 | −0.040 |
| folds better than provider v1 | | 3/5 | 0/5 | 0/5 | 0/5 |
| Δ vs inferred v1 | | | | +0.0013 (4/5) | −0.0022 (1/5) |
| PR-AUC, H = 3 | 0.298 | 0.295 | 0.260 | 0.262 | 0.260 |
| Δ vs provider v1, H = 3 | | −0.0023 (2/5) | −0.036 (0/5) | −0.034 (0/5) | −0.037 (0/5) |
| ROC-AUC, H = 5 | 0.931 | 0.932 | 0.906 | 0.907 | 0.902 |
| shots caught at ≤ 3 false / match (pooled τ, H = 5) | 172 | 166 | 149 | 156 | 145 |
| median p 2 s before a shot | 0.158 | 0.157 | 0.131 | 0.136 | 0.133 |

Per-fold PR-AUC, H = 5:

| fold | provider v1 | provider v3 | inferred v1 | arm S | arm U |
|---|---|---|---|---|---|
| 0 | 0.330 | 0.330 | 0.292 | 0.292 | 0.283 |
| 1 | 0.276 | 0.276 | 0.241 | 0.243 | 0.241 |
| 2 | 0.284 | 0.282 | 0.243 | 0.240 | 0.242 |
| 3 | 0.285 | 0.286 | 0.242 | 0.244 | 0.239 |
| 4 | 0.316 | 0.314 | 0.283 | 0.288 | 0.284 |

Per-fold τ alarms repeat the inferred run's problem: the arms miss about as often or less, but pay +1.3 (U) to +2.4 (S) false alarms per match over the provider, because τ transfers badly from the inner folds. At matched false-alarm budgets (above, and every budget from 2 to 12 per match) every inferred arm catches fewer shots than the provider. So rank on PR-AUC, as before.

## Why arm S doesn't help (`scripts/possession_split.py`)
| H = 5 scored rows | n | PR-AUC provider | PR-AUC inferred v1 | PR-AUC arm S |
|---|---|---|---|---|
| stage 8 agrees | 1,695,773 | 0.302 | 0.288 | 0.288 |
| stage 8 names the other team | 278,679 | 0.237 | 0.026 | 0.028 |

- **The model uses the feature** (4.1% of gain, about as much as `possession_s`), but it only moves ranking where stage 8 already agrees with PFF, and there by nothing.
- **Knowing a row is stale can't turn the frame around.** On a disagreeing row, ball position, the carrier and the defenders are all measured toward the wrong goal and for the wrong team. Carrier age can at most tell the model to trust the row less. That lowers p on stale rows everywhere, including the ones stage 8 got right, and 49% of rows past 10 s are right.
- **In the control it's close to noise:** 1.2% of gain, PR-AUC −0.0006. With PFF's possession it only tells the model how long the ball has been loose, which the ball features mostly cover already.

## Why arm U doesn't help
| H = 5 scored rows | n | PR-AUC provider | PR-AUC arm U |
|---|---|---|---|
| stage 8 agrees | 1,587,032 | 0.306 | 0.293 |
| stage 8 names the other team | 163,049 | 0.272 | 0.035 |
| unknown (made null past 10 s) | 227,298 | 0.159 | 0.024 |

- **The unknown rows are ranked as badly as the wrong ones** (0.024 vs 0.159 for the provider). 05 flagged the reason before the run: null possession means no rotation, so `ball_x`, `ball_dist`, `ball_angle` and `ball_vgoal` are measured toward the +x goal. On a stale row that's a coin flip about direction, and those features are about half the gain. It pushed the model onto the carrier features (38.5% of gain, up from 24%), which don't depend on the frame as much.
- It removed 115k disagreeing rows (41% of them), but made 11.3% of all rows unknown, including the 49% of stale rows that stage 8 had right. That's the −0.002 against plain inferred.
- The "agrees" rows aren't the same set as in the arm S table (U's are the non-stale agreeing rows), so the 0.293 isn't a gain over 0.288.
- A neutral "unknown" would need the ball features to be direction-free too, for example distance to the nearer goal. That's a new feature set, not this arm, and it still can't rank a row where the model doesn't know who's attacking.

## What it means
- **Stage 8's rules are next (03 stage 8), as 07 #6 set before the runs.** The model can't make up for a wrong possession, so the fix is to detect the changes stage 8 misses. The staleness table says where they are: after PFF's changes (55.6% disagreement in the first second), on stretches with no confirmed carrier, and when the ball isn't visible.
- **The stage 8 review's candidate still stands:** a possession change with no confirmed carrier (one-touch passes, headers, the ball arriving off camera), from the last second of positions. A cheap first rule to try: switch to the team whose nearest player is closest to a moving ball once no carrier has been confirmed for a while. That needs a new `StateConfig` field so the caches get a new key (05 Caches).
- **Carrier age stays available** (v3, cached) to rerun on a new stage 8, but it isn't worth adding to the baseline: provider + v3 didn't beat v1.
- **Rerun 07 #6 after the stage 8 change**, same decision line. About 8 min on the M1.

## Caveats
- Same as the inferred review: the alarm rule and τ run on PFF's possession, stage 8 runs on clean PFF tracking (VISIBLE objects only), and PFF's own possession lags. The "stage 8 changed more recently" rows are where PFF lag would show up, and there PFF's team still shoots next 18× as often.
- One model (v1 hand features). A temporal model might learn to switch sides from the positions itself, but it isn't wired for inferred possession (05 Scope).
- S = 10 s was fixed before the runs and not tuned after. A shorter S makes more rows unknown (5 s: 18.6% of rows, of which 54% are right), so with the ball features as they are it would lose more on right rows than it saves on wrong ones.

## Timing (M1)
- CV, both horizons: arm S 402 s, arm U 476 s, control 334 s. About 7 min each with features cached.
- Staleness script: 18 s for all 64 matches from stage 8's cache.
- The first try at arm U was cut off with the session. The rerun at the same commit was paused by the Mac sleeping, so its wall time isn't a measure.
