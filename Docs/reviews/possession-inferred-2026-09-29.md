# Provider vs inferred possession (07 #6, 2026-09-29)

The LightGBM baseline retrained with its inputs built from stage 8's possession (`vision/state.py`, default `StateConfig`) instead of PFF's. Same features (v1), held ball, seed, folds and τ rule as `lgbm-held-2026-09-27`. Labels, scored rows, τ selection and the alarm rule stay on PFF's possession and ball state (05 "Inferred possession", 07 #6). Run `lgbm-pinf-2026-09-29` at `285edeb`, both horizons, compared by `evaluation.compare` and `scripts/lead_time.py`. Per-fold `train_rows` equal the baseline's on every fold at both horizons, and the provider features recompute byte for byte to the cache the baseline read, so the only thing that moved is possession.

## Bottom line
- **Stage 8 costs −0.038 PR-AUC at H = 5 (0.296 → 0.258), worse on 5/5 folds.** H = 3: −0.036, 5/5. That's past the "big" line set in 07 before the run (≤ −0.018 and worse on ≥ 4 folds). **Verdict: stage 8 needs work.**
- For scale, it's twice the worst single vision arm at its target (`geom_loss`, −0.018) and more than half of every vision arm at target together (−0.070). The per-fold Δ is very steady (H = 5: −0.033 to −0.043, std 0.004), so this isn't fold noise. The provider run's fold-to-fold spread is ±0.023.
- **The model collapses where stage 8 names the other team.** That's 14.2% of scored rows. There the inferred model's PR-AUC is 0.026, against 0.237 for the provider model on the same rows: it's close to useless (base rate 0.014). It's also worse where stage 8 agrees with PFF: −0.014 (0.302 → 0.288), close to the "big" line on its own, because it trained on 14% of rows with the frame reversed. Pooled PR-AUC doesn't split into parts, so neither piece is "the share" of the −0.038.
- **Stage 8 is mostly the one that's wrong there, not PFF.** On those rows, PFF's team takes an open-play shot within 5 s 3.2× as often as the team stage 8 names (1.38% vs 0.43% of rows).
- Stage 8's null possession doesn't matter: 0.15% of scored rows.

## Headline (PFF pooled out of fold)
| | provider (`lgbm-held-2026-09-27`) | inferred (`lgbm-pinf-2026-09-29`) | Δ | folds inferred better |
|---|---|---|---|---|
| PR-AUC, H = 5 | 0.296 | 0.258 | −0.038 | 0/5 |
| ROC-AUC, H = 5 | 0.931 | 0.906 | −0.025 | 0/5 |
| Brier, H = 5 | 0.020 | 0.021 | +0.001 | 0/5 |
| PR-AUC, H = 3 | 0.298 | 0.260 | −0.036 | 0/5 |
| ROC-AUC, H = 3 | 0.958 | 0.939 | −0.018 | 0/5 |
| shots caught at ≤ 3 false / match (pooled τ, H = 5) | 172 | 149 | −23 | |
| median lead there (s) | 0.67 | 0.62 | | |
| median p 2 s before a shot | 0.158 | 0.131 | | |

Per-fold τ alarms (H = 5): the inferred run misses less (−0.051, 4/5 folds) but pays +2.6 false alarms per match (0/5). Its τ transferred badly: inner validation came in at 2.75–2.88 false alarms per match, while the held-out folds got 3.7–6.7. At matched false-alarm budgets (lead-time table), it catches fewer shots at every budget from 2 to 12 per match. So the miss-rate "win" is bought with false alarms. Rank on PR-AUC, as in the sensitivity review.

## Where it's lost (`scripts/possession_split.py`)
| H = 5 scored rows | n | base rate | PR-AUC provider | PR-AUC inferred |
|---|---|---|---|---|
| all | 1,977,379 | 0.0252 | 0.296 | 0.258 |
| stage 8 agrees with PFF | 1,695,773 | 0.0271 | 0.302 | 0.288 |
| stage 8 disagrees | 281,606 | 0.0137 | 0.237 | 0.026 |

- **On a disagreeing row the model is looking the wrong way.** The pitch is rotated toward the other goal and attackers and defenders are swapped, while the label asks about PFF's team. The features describe the other team's attack, so the model can't rank those rows at all.
- **The disagreeing rows hold 7.8% of the positives.** Those are rows where PFF's team really does shoot within 5 s, yet stage 8 says the other team has the ball.
- **Who's right on those rows:** the team that goes on to shoot. Over the 278,679 disagreeing rows where stage 8 names a team, PFF's team shoots (open play, within 5 s) on 1.38% and stage 8's team on 0.43%. H = 3: 0.87% vs 0.25%. Shots follow PFF's team about 3× as often, so on most of these rows stage 8 is carrying a stale team forward. Stage 8's team isn't at zero, though, so some of the disagreement is PFF being late (next section).
- **Agreeing rows lose a little too (−0.014).** The model trained on 14% of rows with the frame the wrong way round, which is label noise for everything it learns, so it's slightly worse even where its input is right.
- This split is descriptive: it's chosen from stage 8's output. The paired fold comparison above is the test.

## What it means
- **Stage 8 isn't good enough to feed this model yet.** The stage 8 review said a confirmed carrier's team matches PFF 97.9% of the time, and that the misses come from carrying the last carrier's team forward through the 74% of alive frames with no confirmed carrier. The model collapses on the rows where they differ, and shots there follow PFF's team about 3× as often, so that carry-forward rule is what to fix. The review's next step still stands: a learned rule on the last second of positions for changes of possession with no confirmed carrier (one-touch passes, headers, the ball arriving off camera).
- **A cheaper thing to try first:** let the model see how sure stage 8 is. For example, seconds since stage 8 last confirmed a carrier as an input, or leave possession unknown (no flip, no attacker/defender split) once it's stale. The inferred-null rows here are too few to show whether "unknown" costs less than "wrong", but the sensitivity test found that for teams (`team_unknown` < `team_flip`). That would be a 05 change and its own run, compared with this one.
- **Rerun this comparison after any stage 8 change.** It takes 7.5 min on the M1, and the decision line is in 07 #6.
- **Vision priorities:** the possession error ranks ahead of every single vision arm at target, so it goes at the top of the vision quality list, ahead of homography acceptance and the ball detector.

## Caveats
- **The alarm rule and τ still run on PFF's possession and ball state.** Live, alarms would start and end on stage 8's too. That's the later, fully live question, and it changes which alarms exist, not how rows rank. The PR-AUC Δ doesn't depend on it.
- **PFF's possession is itself laggy.** The oracle floor loses 25 shots to PFF crediting the other team 1–4 s before a shot (07 "Floor from the labels"). Where stage 8 is right and PFF is late, the inferred arm is penalized for disagreeing with PFF, not with the game. Labels come from PFF's possession, so this can't be removed from the Δ, and the −0.038 overstates stage 8's cost by some amount. The shot count above bounds how much: on rows where they disagree, the team that goes on to shoot is PFF's about 3× as often as stage 8's. So PFF's lag is the smaller part.
- **Stage 8 runs on clean PFF tracking** (VISIBLE objects only). On vision output it would see less, so this is the cost of the rules alone, before any vision error (07 #6). The two costs aren't simply additive (the sensitivity review found losses don't add up).
- One baseline model (v1 hand features). A temporal GNN could learn to discount a stale possession, but it isn't wired for inferred possession yet (05 Scope).

## Timing (M1)
- Stage 8 on all 64 games: 11.9 s (cached per match afterwards, `state_inferred_<key>.parquet`).
- Features from the inferred possession: 7.3 s for 64 games. Loading everything: 20.7 s.
- CV, both horizons: 400 s. Total wall time: 7 min 22 s. About 35 s per fold per horizon, like the provider run.
