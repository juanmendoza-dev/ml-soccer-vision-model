# Learned possession: residual diagnostic (2026-09-30)

03 stage 8, "First, diagnose the residual". No model, no PR-AUC. The question: of the ball-visible rows in the first second after PFF's change where the fast rule (`team_near_s = 0.05`) still disagrees, why does it disagree, and how many could a VISIBLE-only learned model reach?

```
PYTHONPATH=. python scripts/possession_split.py --scored h5 --state team_near_s=0.05 --residual
```

Full output: `data/runs/lgbm-pinf-2026-09-29/residual_scored_h5_team_near_s_0.05.md` (not in git).

## Bottom line

- **The residual reproduces:** 16,412 of 54,639 ball-visible first-second H = 5 scored rows disagree (30.0%). That's the same cell as the earlier run (stage 8 review, "The change-lag rules on scored rows"), and the summary counts match it too (297,421 overall, 130,977 first 3 s, 7,691 positives).
- **It's mostly "nobody visible within 1.5 m":** 77% of the residual. Height is almost nothing (0.3%), and so are ESTIMATED players on the ball (1.4%).
- **Reachable residual: 13,271 of 16,412 (81%)**; on positives 209 of 294 (71%). That's well over half, so the spec's warning ("first-3-s bar likely out of reach") doesn't apply.
- **PFF changes possession before the new team touches the ball:** the median first visible touch by the new team comes 0.33 s after PFF's change. 5,598 of the reachable rows (42%) come before that touch, so only anticipation can get them: where the ball is heading, and the pressure around it. Those features are already in v1.
- **Action, per 03's fixed table:** no feature change. The v1 contract stays as written. Next is the extractor and the touch-vs-turnover check.

## Definitions

- **Rows:** H = 5 scored rows (07 #6: label mask on, not all ESTIMATED) where the ball is VISIBLE and less than 1 s has passed since PFF's last possession change. Change bookkeeping is on native frames, as in the original script.
- **Residual:** those rows where `team_near_s = 0.05` (with the height gate) disagrees with PFF. Null counts as disagreement.
- **The rule's state** comes from stepping `StateMachine` exactly as `infer()` does, keeping its candidate, team streak and out-of-pitch flag per frame. The script asserts the replay matches `infer()` on every row.
- **Reach:** 1.5 m in 2D (`carrier_radius_m`).
- **Timing window:** from 1 s before to 2 s after PFF's change. "Contact" is the first frame in it where a VISIBLE player of the new team is within 1.5 m of the VISIBLE ball (2D, no height).
- **Timing-mismatch candidate:** no new-team contact in the window, with the ball visible on at least 50% of the window's frames. If the ball is mostly unseen, missing contact proves nothing: those rows are reported but stay reachable.
- **Events:** converted `events.parquet` holds only shots and goals. PFF's possession comes from the inline `game_event` / `home_ball` on each tracking line (`converters/pff.py`), so PFF's change frame *is* its event frame and the new team is the event's team. There is no separate event time to compare.

## Disjoint buckets (03's order)

Share is of the residual; rate is of all 54,639 ball-visible first-second rows.

| bucket | rows | share | rate | positives (of 294) |
|---|---|---|---|---|
| 1 height gate (ball z ≥ 1 m) | 51 | 0.3% | 0.09% | 0 |
| 2 ESTIMATED player in reach, no VISIBLE one | 224 | 1.4% | 0.41% | 8 |
| 3 no visible player in reach | 12,664 | 77.2% | 23.2% | 208 |
| 4 rule's candidate is the other team | 2,388 | 14.6% | 4.4% | 52 |
| 5 PFF's team is the candidate, streak < 0.05 s | 1,069 | 6.5% | 2.0% | 26 |
| 6 unexplained | 16 | 0.1% | 0.03% | 0 |

- All 16 unexplained rows have the ball out of play (the rule drops the candidate). No exact ties.
- Bucket 4 has no unknown-team candidates, since PFF's teams are always known. In every row it's the old team's player who's still nearest the ball.
- **Height:** PFF's visible ball always has a z. Unknown height never occurs on these rows, and 51 have z ≥ 1 m. The 2D rule (default, z null) gets 12 of those 51 right; `team_near_s = 0.05` without height gets 5.

**Overlapping flags** (the precedence hides little):

| flag | rows | share |
|---|---|---|
| no VISIBLE player within 1.5 m | 12,928 | 78.8% |
| rule's candidate is the other team | 2,388 | 14.6% |
| PFF's team candidate, streak < 0.05 s | 1,069 | 6.5% |
| ESTIMATED player within 1.5 m (any team) | 330 | 2.0% |
| ESTIMATED player of PFF's team within 1.5 m | 238 | 1.5% |
| ball out | 115 | 0.7% |
| ball z ≥ 1 m | 51 | 0.3% |

The only sizeable combination is ESTIMATED plus an other-team candidate (85 rows).

**Bucket 3, how far away is the nearest visible player?** Median 2.98 m (q10 1.75, q90 6.53). 20% are within 2 m and half within 3 m, so many are near misses rather than open space. On positives the median is 2.43 m, with 70% within 3 m.

## Timing

3,532 distinct PFF changes lie behind the residual:

| timing class | changes |
|---|---|
| new-team visible contact within [−1, +2] s | 2,571 |
| no contact, ball seen ≥ 50% (timing-mismatch candidate) | 386 |
| no contact, ball mostly unseen (can't tell) | 575 |

- **Contact minus change, s** (changes with contact): q10 −0.8, q25 −0.17, **median +0.33**, q75 +1.0, q90 +1.5. PFF usually stamps the change *before* the new team's first visible touch. That fits an event convention such as the pass or challenge that leads to the turnover, rather than the receiving touch. It isn't proof that PFF is wrong.
- **The rule's first switch to the new team, minus change, s** (2,181 changes switch inside the window; 373 already had the new team at its start): q10 −0.1, median +0.8, q90 +1.63. That's about half a second behind the median contact: the new team often touches the ball once, and the rule only follows when a new-team player is nearest again for 0.05 s.
- The first 20 changes per bucket, with contact, switch, ball-visible share and nearest visible distance, are in the full output. Example: 10502 period 1 at 222.99 s. PFF says home; the nearest visible player is 1.65 m away; home's first touch comes at +0.87 s and the rule follows at +0.93 s.

## Reachable residual

Per 03's fixed actions:

| bucket | action | rows |
|---|---|---|
| height gate | reachable (the learned path's 2D rule has no height) | counted |
| ESTIMATED player on the ball | unreachable for a VISIBLE-only model; subtract | 224 |
| no visible player in reach | reachable only by anticipation; v1 already has heading, last-contact and pressure features; no change | counted |
| timing mismatch (candidates, all buckets) | unreachable; subtract | 2,978 rows |
| overlap of the two subtracted groups | | 61 |

**Reachable: 16,412 − 3,141 = 13,271 (80.9%).** On positives: 294 − 85 = 209 (71.1%).

Reachable rows by when they fall relative to the new team's first visible contact:

| | all | positives |
|---|---|---|
| at/after first contact | 6,151 | 131 |
| before first contact (anticipation only) | 5,598 | 74 |
| no contact, ball mostly unseen | 1,522 | 4 |

"Reachable" means that nothing in the data rules those rows out. It doesn't mean a model will get them. Nearly half come before any new-team touch, so the model has to predict the change from the ball's direction and the players around it. That's harder than following a touch sooner.

## What it means

- **The v1 feature contract stays as written.** None of 03's bucket actions call for a change. Height is negligible. ESTIMATED players and timing are small and unreachable. The large bucket is the one v1's heading, last-contact and pressure features were added for.
- **The first-3-s bar isn't the hard one.** `team_near_s = 0.05` already reaches 130,977 against the bar's 131,585. What failed was the positives (7,691 against ≤ 3,862) and the late bucket. That's the touch-vs-turnover problem, and it's the next check.
- The residual is a first-second view, and it doesn't speak to the positives regression. That regression is mostly more than 10 s after PFF's change.

## Next

- Build the `pfeat-v1` extractor in `vision/` (roadmap), then run the touch-vs-turnover check (03, "Hypothesis check": at least one shape, contact or heading feature at separation AUC ≥ 0.70, or stop before any fit).
