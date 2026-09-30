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

On positives (294 rows): no visible player in reach 216 (73.5%), other-team candidate 52 (17.7%), short streak 26 (8.8%), ESTIMATED in reach 17 (5.8%; of PFF's team 5), height 0, ball out 0.

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

That's a lower bound on reachable. The contact test uses the same strict 1.5 m, and some "no contact, ball seen" changes have a visible player just outside it (10502 at 761.26 s and 868.57 s: 1.52–1.53 m). So part of the 2,978 are near-miss touches, not evidence that PFF's timing is off. Don't quote it as "18% timing mismatch".

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

## Appendix: first 20 distinct changes per bucket

From the run output, sorted by match, period and change time. Each change is summarized as offsets from PFF's change rather than frame-by-frame tracking: `rows` = residual rows it covers, `pos` = any positive, `d1` = nearest VISIBLE player to the ball (m, min over its rows), `contact` = first new-team visible contact, `switch` = first switch of the rule to the new team (s, null = none in [−1, +2] s), `held` = rule already had the new team at −1 s, `seen_share` = share of the window's frames with the ball visible.


### Bucket 1 height gate

| match | period | chg     | new  | rows | pos   | d1    | contact | switch | held  | seen_share | timing                         |
|-------|--------|---------|------|------|-------|-------|---------|--------|-------|------------|--------------------------------|
| 10502 | 1      | 1584.68 | home | 2    | false | 4.68  | null    | null   | false | 0.07       | no contact, ball mostly unseen |
| 10503 | 1      | 2297.83 | away | 3    | false | 1.25  | -0.07   | null   | false | 0.13       | contact                        |
| 10503 | 2      | 682.75  | away | 2    | false | 1.7   | null    | null   | false | 0.16       | no contact, ball mostly unseen |
| 10503 | 2      | 1858.83 | home | 2    | false | 0.84  | 0.07    | null   | false | 0.34       | contact                        |
| 10505 | 1      | 2474.04 | home | 1    | false | 0.92  | 0.43    | null   | false | 0.09       | contact                        |
| 10506 | 1      | 564.2   | away | 1    | false | 2.23  | 1.4     | 1.47   | false | 0.19       | contact                        |
| 10506 | 2      | 224.93  | away | 2    | false | 2.5   | null    | null   | false | 0.24       | no contact, ball mostly unseen |
| 10510 | 1      | 696.8   | home | 1    | false | 12.57 | 1.67    | 1.74   | false | 0.51       | contact                        |
| 10513 | 1      | 1243.51 | home | 1    | false | 5.17  | 1.23    | 1.3    | false | 0.87       | contact                        |
| 10515 | 1      | 2291.82 | away | 1    | false | 1.81  | 1.1     | 1.17   | false | 0.37       | contact                        |
| 10515 | 2      | 928.53  | home | 1    | false | 2.23  | 0.73    | 0.8    | false | 0.13       | contact                        |
| 10516 | 1      | 2496.03 | away | 1    | false | 1.21  | 0.63    | 0.77   | false | 0.16       | contact                        |
| 10516 | 2      | 2529.1  | home | 1    | false | 3.36  | null    | null   | false | 0.47       | no contact, ball mostly unseen |
| 3812  | 1      | 573.11  | away | 1    | false | 4.61  | null    | null   | false | 0.02       | no contact, ball mostly unseen |
| 3812  | 2      | 2980.88 | home | 1    | false | 1.01  | null    | null   | false | 0.04       | no contact, ball mostly unseen |
| 3813  | 1      | 1972.37 | away | 2    | false | 2.26  | 1.77    | 1.9    | false | 0.44       | contact                        |
| 3813  | 2      | 433.7   | away | 2    | false | 3.01  | null    | null   | false | 0.49       | no contact, ball mostly unseen |
| 3814  | 2      | 592.89  | away | 1    | false | 1.25  | 0.0     | 0.13   | false | 0.08       | contact                        |
| 3815  | 1      | 5.71    | away | 1    | false | 3.91  | null    | null   | false | 0.16       | no contact, ball mostly unseen |
| 3821  | 1      | 1686.62 | home | 1    | false | 8.38  | null    | null   | false | 0.13       | no contact, ball mostly unseen | 

### Bucket 2 ESTIMATED in reach, no VISIBLE

| match | period | chg     | new  | rows | pos   | d1    | contact | switch | held  | seen_share | timing                         |
|-------|--------|---------|------|------|-------|-------|---------|--------|-------|------------|--------------------------------|
| 10502 | 1      | 492.36  | home | 3    | false | 1.58  | 1.03    | 1.1    | false | 1.0        | contact                        |
| 10502 | 1      | 761.26  | away | 1    | false | 3.04  | null    | null   | false | 1.0        | no contact, ball seen          |
| 10502 | 2      | 934.5   | home | 2    | true  | 1.87  | -0.97   | 1.37   | true  | 1.0        | contact                        |
| 10502 | 2      | 1038.47 | home | 1    | false | 6.13  | null    | null   | false | 1.0        | no contact, ball seen          |
| 10502 | 2      | 2318.55 | home | 1    | false | 6.29  | null    | null   | false | 1.0        | no contact, ball seen          |
| 10502 | 2      | 2335.0  | home | 2    | false | 7.72  | null    | null   | false | 0.69       | no contact, ball seen          |
| 10502 | 2      | 2597.8  | home | 2    | false | 4.99  | -0.97   | null   | false | 0.66       | contact                        |
| 10503 | 2      | 1951.48 | away | 2    | false | 5.63  | null    | null   | false | 0.43       | no contact, ball mostly unseen |
| 10504 | 1      | 1191.83 | away | 1    | false | 2.03  | null    | null   | false | 0.27       | no contact, ball mostly unseen |
| 10504 | 2      | 608.71  | away | 2    | false | 3.63  | null    | null   | false | 0.98       | no contact, ball seen          |
| 10504 | 2      | 910.34  | home | 1    | false | 3.87  | 0.83    | 0.9    | false | 0.44       | contact                        |
| 10506 | 1      | 55.09   | home | 1    | false | 2.0   | 0.37    | 0.9    | false | 0.61       | contact                        |
| 10507 | 1      | 1897.13 | home | 1    | false | 1.73  | 0.13    | 0.2    | false | 1.0        | contact                        |
| 10508 | 2      | 1583.35 | away | 2    | false | 1.79  | -0.3    | -0.23  | false | 1.0        | contact                        |
| 10510 | 1      | 696.8   | home | 1    | false | 13.29 | 1.67    | 1.74   | false | 0.51       | contact                        |
| 10510 | 1      | 2420.59 | away | 2    | false | 1.57  | -0.97   | null   | true  | 1.0        | contact                        |
| 10511 | 1      | 2786.72 | away | 1    | false | 5.19  | 1.0     | 1.07   | false | 0.42       | contact                        |
| 10512 | 1      | 1146.18 | home | 1    | false | 4.63  | 1.94    | null   | false | 0.21       | contact                        |
| 10512 | 1      | 2278.15 | home | 2    | false | 1.6   | 0.17    | 0.83   | false | 1.0        | contact                        |
| 10513 | 1      | 405.51  | home | 1    | false | 1.66  | 0.13    | 0.2    | false | 0.42       | contact                        | 

### Bucket 3 no visible player in reach

| match | period | chg     | new  | rows | pos   | d1   | contact | switch | held  | seen_share | timing                         |
|-------|--------|---------|------|------|-------|------|---------|--------|-------|------------|--------------------------------|
| 10502 | 1      | 222.99  | home | 5    | false | 1.65 | 0.87    | 0.93   | false | 0.91       | contact                        |
| 10502 | 1      | 271.17  | away | 7    | false | 1.59 | 0.17    | null   | false | 0.58       | contact                        |
| 10502 | 1      | 342.18  | away | 5    | false | 1.62 | 0.83    | 0.9    | false | 0.48       | contact                        |
| 10502 | 1      | 449.98  | away | 1    | false | 2.91 | 1.5     | null   | false | 0.31       | contact                        |
| 10502 | 1      | 492.36  | home | 5    | false | 1.55 | 1.03    | 1.1    | false | 1.0        | contact                        |
| 10502 | 1      | 757.92  | away | 5    | false | 2.61 | 1.8     | 1.87   | false | 0.49       | contact                        |
| 10502 | 1      | 761.26  | away | 9    | false | 1.52 | null    | null   | false | 1.0        | no contact, ball seen          |
| 10502 | 1      | 771.34  | home | 3    | false | 1.52 | -0.9    | -0.77  | false | 1.0        | contact                        |
| 10502 | 1      | 868.57  | home | 9    | false | 1.53 | null    | null   | false | 0.98       | no contact, ball seen          |
| 10502 | 1      | 890.36  | home | 3    | false | 3.95 | null    | null   | false | 0.75       | no contact, ball seen          |
| 10502 | 1      | 914.91  | away | 2    | false | 3.93 | null    | null   | false | 0.09       | no contact, ball mostly unseen |
| 10502 | 1      | 973.91  | home | 4    | false | 1.68 | -0.23   | 1.1    | false | 0.78       | contact                        |
| 10502 | 1      | 1017.58 | home | 2    | false | 1.54 | -0.23   | 0.47   | true  | 0.89       | contact                        |
| 10502 | 1      | 1363.06 | away | 2    | false | 1.64 | 0.47    | 0.53   | false | 0.56       | contact                        |
| 10502 | 1      | 1516.65 | home | 7    | false | 1.6  | 0.87    | 0.93   | true  | 0.66       | contact                        |
| 10502 | 1      | 1796.13 | home | 4    | false | 2.24 | 0.47    | 0.53   | false | 0.67       | contact                        |
| 10502 | 1      | 2155.05 | away | 6    | false | 2.1  | 0.73    | 0.8    | false | 0.93       | contact                        |
| 10502 | 1      | 2302.44 | home | 2    | false | 1.79 | null    | null   | false | 0.42       | no contact, ball mostly unseen |
| 10502 | 1      | 2422.02 | home | 1    | false | 1.56 | -0.03   | null   | false | 0.18       | contact                        |
| 10502 | 1      | 2464.5  | home | 10   | false | 3.43 | null    | null   | false | 0.92       | no contact, ball seen          | 

### Bucket 4 candidate unknown/other team

| match | period | chg     | new  | rows | pos   | d1   | contact | switch | held  | seen_share | timing                         |
|-------|--------|---------|------|------|-------|------|---------|--------|-------|------------|--------------------------------|
| 10502 | 1      | 222.99  | home | 4    | false | 1.15 | 0.87    | 0.93   | false | 0.91       | contact                        |
| 10502 | 1      | 492.36  | home | 2    | false | 0.1  | 1.03    | 1.1    | false | 1.0        | contact                        |
| 10502 | 1      | 759.79  | home | 2    | false | 1.18 | 0.13    | 0.2    | true  | 1.0        | contact                        |
| 10502 | 1      | 771.34  | home | 2    | false | 1.05 | -0.9    | -0.77  | false | 1.0        | contact                        |
| 10502 | 1      | 868.57  | home | 1    | false | 1.5  | null    | null   | false | 0.98       | no contact, ball seen          |
| 10502 | 1      | 1017.58 | home | 1    | false | 1.36 | -0.23   | 0.47   | true  | 0.89       | contact                        |
| 10502 | 1      | 1171.3  | home | 3    | false | 1.21 | null    | null   | false | 0.11       | no contact, ball mostly unseen |
| 10502 | 1      | 1516.65 | home | 1    | false | 0.8  | 0.87    | 0.93   | true  | 0.66       | contact                        |
| 10502 | 1      | 2437.97 | away | 3    | false | 0.84 | 0.03    | 0.5    | true  | 0.51       | contact                        |
| 10502 | 1      | 2608.27 | home | 1    | false | 1.45 | null    | null   | false | 1.0        | no contact, ball seen          |
| 10502 | 2      | 332.53  | home | 3    | false | 0.28 | null    | null   | false | 0.62       | no contact, ball seen          |
| 10502 | 2      | 460.76  | home | 1    | false | 1.48 | null    | null   | false | 0.93       | no contact, ball seen          |
| 10502 | 2      | 934.5   | home | 3    | true  | 0.54 | -0.97   | 1.37   | true  | 1.0        | contact                        |
| 10502 | 2      | 940.87  | home | 7    | true  | 1.23 | null    | null   | false | 1.0        | no contact, ball seen          |
| 10502 | 2      | 1038.47 | home | 3    | false | 0.44 | null    | null   | false | 1.0        | no contact, ball seen          |
| 10502 | 2      | 1372.77 | home | 3    | false | 1.24 | 0.03    | 0.17   | false | 1.0        | contact                        |
| 10502 | 2      | 1467.33 | home | 1    | false | 1.45 | null    | null   | false | 1.0        | no contact, ball seen          |
| 10502 | 2      | 1747.18 | home | 3    | false | 0.47 | 1.94    | null   | false | 0.87       | contact                        |
| 10502 | 2      | 2342.24 | home | 10   | false | 0.32 | null    | null   | false | 0.56       | no contact, ball seen          |
| 10502 | 2      | 2486.52 | away | 6    | false | 0.09 | -0.2    | null   | true  | 1.0        | contact                        | 

### Bucket 5 PFF's team, streak < 0.05 s

| match | period | chg     | new  | rows | pos   | d1   | contact | switch | held  | seen_share | timing  |
|-------|--------|---------|------|------|-------|------|---------|--------|-------|------------|---------|
| 10502 | 1      | 139.47  | away | 1    | false | 1.11 | 0.0     | 0.07   | false | 0.67       | contact |
| 10502 | 1      | 222.99  | home | 1    | false | 1.48 | 0.87    | 0.93   | false | 0.91       | contact |
| 10502 | 1      | 271.17  | away | 1    | false | 1.32 | 0.17    | null   | false | 0.58       | contact |
| 10502 | 1      | 771.34  | home | 1    | false | 1.44 | -0.9    | -0.77  | false | 1.0        | contact |
| 10502 | 1      | 1017.58 | home | 1    | false | 1.45 | -0.23   | 0.47   | true  | 0.89       | contact |
| 10502 | 1      | 1796.13 | home | 1    | false | 1.43 | 0.47    | 0.53   | false | 0.67       | contact |
| 10502 | 1      | 2155.05 | away | 1    | false | 1.33 | 0.73    | 0.8    | false | 0.93       | contact |
| 10502 | 1      | 2267.83 | home | 1    | false | 0.75 | 0.1     | 0.17   | false | 0.64       | contact |
| 10502 | 1      | 2703.07 | home | 1    | false | 1.23 | 0.2     | 0.27   | false | 0.84       | contact |
| 10502 | 2      | 1372.77 | home | 1    | false | 1.24 | 0.03    | 0.17   | false | 1.0        | contact |
| 10503 | 1      | 454.39  | home | 2    | false | 1.33 | 0.07    | 1.0    | false | 0.78       | contact |
| 10503 | 1      | 798.0   | home | 1    | false | 1.38 | 0.0     | 0.07   | false | 0.6        | contact |
| 10503 | 1      | 1608.91 | home | 1    | false | 1.3  | 0.2     | 1.63   | false | 0.69       | contact |
| 10503 | 1      | 1744.41 | away | 1    | false | 1.48 | 0.67    | 0.73   | false | 0.96       | contact |
| 10503 | 1      | 1765.23 | home | 1    | false | 1.26 | 0.73    | 0.8    | false | 0.45       | contact |
| 10503 | 1      | 1970.24 | away | 1    | false | 1.38 | 0.07    | 0.4    | false | 0.74       | contact |
| 10503 | 2      | 1858.83 | home | 1    | false | 1.36 | 0.07    | null   | false | 0.34       | contact |
| 10503 | 2      | 2496.4  | away | 1    | false | 1.4  | -0.07   | 0.63   | false | 0.89       | contact |
| 10503 | 2      | 2581.55 | home | 1    | false | 1.29 | 0.3     | 0.37   | false | 1.0        | contact |
| 10503 | 2      | 2738.04 | home | 1    | false | 0.79 | 0.2     | 0.27   | false | 0.74       | contact | 

### Bucket 6 unexplained

| match | period | chg     | new  | rows | pos   | d1   | contact | switch | held  | seen_share | timing                |
|-------|--------|---------|------|------|-------|------|---------|--------|-------|------------|-----------------------|
| 3814  | 1      | 2930.66 | home | 3    | false | 0.93 | 0.23    | null   | false | 0.58       | contact               |
| 3828  | 2      | 1945.68 | away | 1    | false | 1.25 | 0.87    | 1.07   | false | 0.66       | contact               |
| 3853  | 1      | 379.35  | away | 7    | false | 1.0  | null    | null   | true  | 0.64       | no contact, ball seen |
| 3853  | 1      | 1500.8  | home | 1    | false | 0.88 | null    | null   | false | 0.84       | no contact, ball seen |
| 3858  | 1      | 547.15  | away | 4    | false | 0.99 | 0.07    | 0.53   | false | 0.58       | contact               | 

