# Learned possession: touch vs turnover check (2026-09-30)

03 stage 8, "Hypothesis check, before any fit". No model, no PR-AUC, no shot metrics. The question: when the fast rule (`team_near_s = 0.05`) switches team and the default rule doesn't, do the `pfeat-v1` inputs at that moment separate a real turnover from a brief touch?

## Definitions, fixed before the run

Fixed before any AUC was computed. None of them change after the result.

- **Matches and rows:** the 64 matches in `data/splits/folds.json`, and their H = 5 scored grid rows (`label_mask_h5` on, not `all_estimated`) from `data/processed/<m>/frames_10hz.parquet`. Features come through `vision.possession_features.load`, so every hash check runs. The join is on `(period, k)` with `k = round(10 * t_s)`, and the native `frame_id` is asserted equal.
- **Rules:** the historical stage 8 rule (`vision.state.infer`, with its height gate) under the default `StateConfig()` and under `team_near_s = 0.05`. Both are read at each grid row's native frame.
- **Trigger:** a scored grid row t whose previous grid row (k − 1) exists in the same period, where:
  - the fast rule's team at k − 1 (old) and at t (new) are both non-null and differ
  - the default rule has the old team at both k − 1 and t

  So the fast rule moves away from the team both rules had, and the default keeps it.
- **Groups:** PFF possession on native frames with `t < timestamp_s ≤ t + 3` in the same period.
  - **turnover** if PFF names the new team on any of those frames
  - **touch** if PFF names the old team on every one of them (and there is at least one)
  - every other row is left out (nulls, a period end)

  PFF's later labels only sort the groups and are never inputs.
- **Orientation:** rows where the new team is away go through `pf.mirror`, so `home_*` always means the team the fast rule picked, and team codes read +1 = that team.
- **Separation:** AUC of turnover (positive) vs touch, computed with Mann–Whitney ranks and averaged ties, on the rows where the feature is finite. Separation is `max(AUC, 1 − AUC)`. Each feature also gets its finite share in each group, the group medians and the IQRs.
- **Eligible for the bar:** only shape, contact and heading features:
  - shape: `<team>_centroid_x`, `_mean_vx_own`, `_past_halfway`, `_max_x`, `_min_x`, `_visible_n` and their six `diff_*`
  - contact: `near_<team>_m`, the eight `nearest_*` shares, `last_contact_team`
  - heading: `heads_*`

  Rule, ball, view, age and pressure (`within5/10`) columns are reported but can't pass the bar. A feature is eligible only if it's finite on at least 50% of each group, so a feature present on a handful of rows can't pass by chance.
- **Bar:** at least one eligible feature at separation ≥ **0.70** over all trigger rows. Otherwise stop and return to spec review before any fit. Positives only (`label_shot_h5` on the trigger row) is descriptive.

```
PYTHONPATH=. python scripts/possession_touch_check.py
```

## Result: PASS

Full output: `data/runs/lgbm-pinf-2026-09-29/touch_check_h5.md` (not in git). The first run crashed after the tables, while printing the verdict: the `eligible` flag came out as a float, so filtering on it failed. The fix (`bool(...)`) changed no definition, and the rerun's tables are identical to the first run's apart from how that flag prints.

- **Trigger rows:** 12,949 from all 64 matches: 8,419 turnovers, 4,530 touches. Both team directions are balanced (turnover 4,204 home / 4,215 away; touch 2,224 / 2,306).
- **Bar:** the best eligible feature is `nearest_away_1_r15` at separation **0.755** (AUC 0.245), over the 0.70 bar. **The model's premise holds, so the work goes on to the pilot.**

Top features over all trigger rows. After orientation, `home` = the team the fast rule picked (new) and `away` = the team it left (old):

| feature | kind | sep | turnover median (IQR) | touch median (IQR) |
|---|---|---|---|---|
| `nearest_away_1_r15` | contact | **0.755** | 0.00 (0.00–0.32) | 0.54 (0.13–0.82) |
| `nearest_away_1_any` | contact | 0.753 | 0.10 (0.00–0.54) | 0.73 (0.40–0.86) |
| `rule_carrier_age_s` | not eligible | 0.746 | 3.1 s (0.7–10.4) | 0.27 s (0.13–1.4) |
| `nearest_away_05_any` | contact | 0.736 | 0.00 (0.00–0.46) | 0.67 (0.20–0.75) |
| `nearest_away_05_r15` | contact | 0.729 | 0.00 (0.00–0.18) | 0.50 (0.00–0.73) |
| `heads_home_03_m` | heading | 0.692 | 1.7 m (1.1–2.5) | 2.6 m (1.7–3.8) |
| `nearest_home_1_any` | contact | 0.666 | 0.32 (0.14–0.67) | 0.16 (0.12–0.27) |
| `heads_home_05_m` | heading | 0.666 | 2.5 m (1.6–3.6) | 3.6 m (2.3–5.2) |
| `heads_home_angle` | heading | 0.637 | 1.59 rad | 2.15 rad |
| `near_away_m` | contact | 0.634 | 3.6 m (2.0–7.2) | 2.4 m (1.7–3.8) |
| `diff_mean_vx_own` | shape | 0.632 | 0.58 m/s | 2.16 m/s |

Every eligible feature is finite on at least 90% of both groups. The full table has all 53.

**What separates them:**
- **On a touch, the old team was still on the ball.** Over the last second it was nearest the ball, within 1.5 m, about half the time (median 0.54). Before a real turnover it almost never was (median 0). In plain terms, a defender's touch happens in the middle of the other team's spell on the ball, while a turnover comes after the old team has already lost it.
- **Heading adds to that.** On a turnover the ball is heading to the new team (1.7 m from its nearest player 0.3 s ahead, against 2.6 m on a touch).
- **Shape is weak over all trigger rows** (best 0.63, the own-goal velocity: on a touch the new team is still retreating).
- **Not eligible, but notable:** the default rule's time since its last carrier separates almost as well (0.746). Turnovers come after long stretches with no carrier (median 3.1 s), touches within a fraction of a second of one. The model gets it as `rule_carrier_age_s`.

**Positives only (descriptive, 545 rows: 231 turnovers, 314 touches):** here it's the other way round. Shape separates best: `home_max_x`, `diff_past_halfway` and `home_centroid_x` are all at about 0.80–0.81. On a touch before a shot, the new team (the defenders) sits deep in its own half (`home_max_x` median −27 m, nobody past halfway). On a turnover it's spread upfield. The old-team contact share separates less here (0.648). So near shots, *where* the touch happens matters more than *how long* the old team had the ball. A tree model can use both; no single hand rule can. That's the case for the learned model, not proof that it will pass the gate.

## What it means

- **The check passes on a contact feature, and its definition was fixed before the run.** Per 03, the feature contract `pfeat-v1` is frozen as written, and the next step is the pilot on the workstation CPU.
- A separation of 0.755 on one feature says a tree has something to build on. It doesn't predict the gate. The scored-row gate (five bars, including positives ≤ 3,862) is still the decision.
- The positives view says the model has to combine contact history with where the play is. That's what the gate's positives bar will test.

## Next

- Freeze `pfeat-v1`. The rest of the build (model, nested fits, plumbing, tests per 03), then the pilot: one match's extraction (measured here at about 1.5 s per match, 87 s for all 64) and `outer_0` probe + refit on the workstation CPU.
