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

## Result

(filled in after the single run)
