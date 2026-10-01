# Learned possession: smoothing sweep after the v1 gate (2026-10-01)

v1 (`lgbm-v1-nested5x4-s1`) failed 03's gate on churn alone: 69,357 changes vs the 18,182 limit, with the four row bars passing easily (overall 150,748 vs 253,445; first 3 s 90,441 vs 131,585; positives 1,450 vs 3,862; late rate 0.0278 vs 0.0551). Report in `data/runs/possession-gate-lgbm-v1-nested5x4-s1/`. This is the diagnostic behind the one allowed revision. No gate rows, no per-fold results, no shot metrics.

## Data and method

- **Rows:** inner-OOF learned state only (`state_inferred_age_*_outer<k>_inner-oof<j>.parquet`, 256 contexts, every match 4 times). The gate's outer `test` roles aren't read.
- **Scoring:** `prediction.possession_gate`'s own helpers (`native_rows`, `scored_grid`, `match_rows`, `tally`, `churn`), so rows, buckets and churn are counted exactly as at the gate. Margin 0 / dwell 0 / no hold reproduces v1's stored `possession_team` exactly.
- **Bars, relative:** the inner-OOF pool has 4× the gate's rows, so each bar is read as the ratio it encodes, on the same rows: overall ≤ 0.90 × default, first 3 s ≤ 0.85 × default, positives ≤ 1.00 × default, late rate ≤ default + 0.010, churn ≤ 1.25 × PFF.
- **Proxy check:** unsmoothed inner-OOF lands where the gate did: churn 4.79 × PFF (gate 4.77), overall 0.539 × default (gate 0.535).
- Script: `scripts/possession_smoothing_sweep.py`, about 6 min on the workstation. It resets at each period start and calls away at p ≤ 0.5 − margin, exactly as 03's Revision v1h specs the rule. The first two runs carried the team across periods and used p < 0.5 − margin. That moved some numbers in the third or fourth decimal and changed no verdict. The table is from the rerun.

## Where the churn comes from (v1, inner-OOF)

- 12% of changes sit on a fallback/model boundary, so the fallback rows aren't the main cause.
- It's flicker: 31% of possession spells last under 0.2 s, 52% under 0.5 s, 61% under 1 s. p hovers around 0.5 and the hard label follows it.

## Smoothing (all causal)

- **Margin:** switch only when p crosses 0.5 + m (to home) or 0.5 − m (to away); in between, keep the current team.
- **Dwell:** the new team has to be wanted for d consecutive grid ticks. At 10 Hz, 0.1 s is the same as none.
- **Hold:** on fallback rows, keep the model's last team instead of the 2D rule's. Fallback rows are about 40% of grid rows and none of the scored rows, so this only changes churn and what later rows carry in from those stretches.

## Results (ratios as above; pass = all five relative bars)

| margin | dwell s | hold | overall | first 3 s | positives | late − default | churn | pass |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | no | 0.539 | 0.587 | 0.394 | −0.017 | 4.79 | no |
| 0 | 0.5 | no | 0.619 | 0.778 | 0.401 | −0.022 | 1.68 | no |
| 0.1 | 0.5 | no | 0.652 | 0.871 | 0.347 | −0.026 | 1.33 | no |
| 0.2 | 0 | no | 0.562 | 0.708 | 0.310 | −0.025 | 1.60 | no |
| 0.2 | 0 | yes | 0.557 | 0.704 | 0.310 | −0.025 | 1.30 | no |
| 0.2 | 0.2 | yes | 0.590 | 0.786 | 0.303 | −0.028 | 1.08 | yes |
| 0.25 | 0 | no | 0.582 | 0.759 | 0.283 | −0.027 | 1.38 | no |
| **0.25** | **0** | **yes** | **0.574** | **0.755** | **0.284** | **−0.028** | **1.10** | **yes** |
| 0.25 | 0.2 | no | 0.626 | 0.843 | 0.282 | −0.029 | 1.20 | yes |
| 0.3 | 0 | no | 0.611 | 0.818 | 0.251 | −0.029 | 1.19 | yes |
| 0.3 | 0 | yes | 0.602 | 0.814 | 0.253 | −0.030 | 0.94 | yes |
| 0.35 | 0 | yes | 0.642 | 0.882 | 0.245 | −0.033 | 0.81 | no |

Full grid in the script's CSV output.

## What it means

- Churn and first-3-s disagreement are the two bars that pull against each other. Every bit of smoothing delays real turnovers too.
- Margin is cheaper than dwell: at the same churn, dwell costs more first-3-s accuracy. Dwell is left out.
- Hold is free on the scored rows (slightly better on overall) and takes about 20% off churn at every margin.
- Without hold, no setting passes with more than about 0.04 of room on its tightest bar. With hold, margin 0.25 has room on both: churn 1.10 (limit 1.25) and first 3 s 0.755 (limit 0.85).
- **Proposed revision:** margin 0.25, no dwell, hold through fallback, under a new model ID. Same booster, features and probabilities as v1, so no retraining: only the label from p changes. Needs 03's "no smoothing, minimum dwell or hysteresis in v1" and fallback lines revised first.
