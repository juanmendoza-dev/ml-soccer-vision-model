# Vision sensitivity test (2026-09-28)

How much the LightGBM baseline loses when PFF tracking is degraded the way vision fails (05 "Vision sensitivity test", detection review W9). 34 runs, H = 5, same frozen folds and τ rule, each compared with the clean `lgbm-held-2026-09-27` (PR-AUC **0.296**) by `evaluation.compare`. Degradation code at `36b40e7`..`60e7711`; runs are stamped `60e7711`, `ff41236` or `eb2ddd0`, and nothing under `prediction/` or `evaluation/` changed between them. `python3 scripts/sens_table.py` rebuilds the table below from the runs' `compare.md`.

This measures sensitivity. It doesn't say real vision output looks like this; that needs paired footage (07 #4).

## Findings
- **Everything at the target levels at once costs −0.070 PR-AUC** (0.296 → 0.226, 5/5 folds worse). That's still well above the floor on clean data (0.177), so a vision system that meets the detection review's targets keeps most of what the baseline has.
- **Train on degraded data.** Fitting on clean data and predicting degraded folds (`--degrade-arm test`) costs −0.086 for the combined arm vs −0.070 when training is degraded too. Same for geometry loss (−0.028 vs −0.018) and player noise (−0.021 vs −0.013). Ball misses are the exception (−0.015 vs −0.017, within noise): the held ball already makes a gap look the same in both. So the model that runs on vision should be trained with these degradations switched on.
- **Seed noise is small.** `ball_miss` 0.1 with a second seed: −0.0156 vs −0.0170. Every arm except `id_fragment` and `no_ball_z` is worse on 5/5 folds, and the smallest of them (−0.006) is several times the seed gap.
- **Losses don't add up.** The single arms at target sum to about −0.10, the combined arm is −0.070. Use the per-arm numbers as a ranking, not as shares of the total.
- **Rank on PR-AUC.** Miss rate and false alarms per match move with τ and swing either way (`camera_drift` 0.93 "wins" on miss rate by paying +1.0 false alarms per match), as 05 warned.

**At the target levels** (ΔPR-AUC, mean over folds, all 5/5 folds worse unless noted)

| arm | target | ΔPR-AUC |
|---|---|---|
| `geom_loss` (no homography) | 17% of time | −0.018 |
| `ball_miss` | 10% of time | −0.017 (seed 2: −0.016) |
| `player_noise` | 0.93 m | −0.013 |
| `team_flip` | 5% (6% realized) | −0.012 |
| `ball_noise` | 1 m | −0.010 |
| `camera_drift` | 0.93 m | −0.008 |
| `ball_false` | 5% of time | −0.0075 |
| `team_unknown` | 5% (6% realized) | −0.007 |
| `player_miss` | 10% of time | −0.006 |
| `no_ball_z` | always | −0.002 (4/5) |
| `id_fragment` | 2 s track life | −0.001 (2/5) |

**Sweeps** (ΔPR-AUC at each level)

| arm | target | worse | worst |
|---|---|---|---|
| `geom_loss` | 0.17: −0.018 | 0.3: −0.042 | 0.5: **−0.082** |
| `ball_miss` | 0.1: −0.017 | 0.25: −0.035 | 0.5: **−0.070** |
| `ball_noise` | 1 m: −0.010 | 2 m: −0.026 | 4 m: −0.049 |
| `camera_drift` | 0.93 m: −0.008 | 2 m: −0.018 | 4 m: −0.039 |
| `ball_false` | 0.05: −0.0075 | 0.1: −0.013 | 0.2: −0.029 |
| `team_flip` | 0.06: −0.012 | 0.12: −0.016 | 0.23: −0.026 |
| `team_unknown` | 0.06: −0.007 | 0.12: −0.010 | 0.23: −0.022 |
| `player_noise` | 0.93 m: −0.013 | 2 m: −0.024 | |
| `player_miss` | 0.1: −0.006 | 0.2: −0.010 | 0.4: −0.023 |

(`team_*` levels are the realized shares; the crowding term near the ball adds about 20% on top of the nominal level.)

- **Geometry availability is the biggest lever, and it gets worse faster than linearly.** 17% of time without a homography costs −0.018, 30% costs −0.042, 50% costs −0.082, more than every arm at target combined. 17% is the one measured number (smoke04), so this is where vision is today, not a hypothetical.
- **Rejecting vs accepting a bad homography.** Losing 17% of frames costs the same as 2 m of drift on every frame (both −0.018), and 30% lost (−0.042) is about 4 m of drift (−0.039). So the acceptance thresholds shouldn't be tightened past the point where the frames they reject have errors under ~2 m. Tune them against this trade, not for zero error.
- **The ball is the second lever.** Misses are about linear, −0.014 per 10% of time. At 2× the target error, ball noise (−0.026) and false balls (−0.029 at 0.2) are each bigger than any team or player arm at the same multiple. Ball position is half the model's gain (lgbm review), and it shows.
- **"Unknown" beats a wrong team at every level:** −0.007 vs −0.012, −0.010 vs −0.016, −0.022 vs −0.026. When the kit cluster is unsure, emitting no team is cheaper than guessing (W5).
- **Players matter less.** Missing 40% of player-time costs −0.023, less than 25% of ball-time missing. Player position noise at 2 m (−0.024) is similar. Detector fine-tuning for players is the lowest priority of the measured failures.
- **ID fragmentation and ball height don't matter to this baseline.** As 05 says, its features barely read track history (IDs only reach `carrier_speed` and `carrier_vgoal`, about 5% of gain). The attack-building features and the temporal GNN will read a lot more, so rerun `id_fragment` on those before treating tracking as solved.

## What this means for Phase 2
In order:
1. **Homography availability and accuracy where attacks happen (W4).** Keep time without geometry at or under the measured 17%, and tune the acceptance thresholds (03 Homography acceptance, still untuned) on the reject-vs-drift trade above.
2. **Ball: temporal association and reset on cuts (F8, D2/W2).** The roadmap made this conditional on the ball being the bottleneck. It's the second largest, with geometry the first, so do it. Target recall ≥ 90% and precision ≥ 95% hold the loss near −0.017 and −0.0075.
3. **Team "unknown" option (W5).** Cheap, and better than a wrong guess at every level.
4. **Train the vision-deployed model with degradations on.** It recovers about 0.016 of the combined loss. This is a prediction task, not a vision one.
5. **Later:** player detection fine-tuning (W8), ID/tracking quality once the temporal models exist (rerun `id_fragment` then).

## All runs

| run | degradation | arm | realized | PR-AUC | ΔPR-AUC (mean over folds) | folds worse | Δmiss rate | Δfalse / match |
|---|---|---|---|---|---|---|---|---|
| sens-ball_false-0.05 | ball_false:0.05 | both | ball_false 0.0503 | 0.289 | -0.0075 | 5/5 | +0.010 | +0.46 |
| sens-ball_false-0.1 | ball_false:0.1 | both | ball_false 0.1003 | 0.283 | -0.0131 | 5/5 | +0.024 | -0.02 |
| sens-ball_false-0.2 | ball_false:0.2 | both | ball_false 0.2018 | 0.267 | -0.0285 | 5/5 | +0.031 | +0.59 |
| sens-ball_miss-0.1 | ball_miss:0.1 | both | ball_miss 0.0978 | 0.279 | -0.0170 | 5/5 | +0.011 | +0.54 |
| sens-ball_miss-0.1-s2 | ball_miss:0.1 (seed 2) | both | ball_miss 0.0982 | 0.281 | -0.0156 | 5/5 | -0.040 | +1.73 |
| sens-ball_miss-0.1-test | ball_miss:0.1 | test | ball_miss 0.0978 | 0.281 | -0.0147 | 5/5 | +0.015 | +0.02 |
| sens-ball_miss-0.25 | ball_miss:0.25 | both | ball_miss 0.2469 | 0.261 | -0.0349 | 5/5 | +0.048 | -0.21 |
| sens-ball_miss-0.5 | ball_miss:0.5 | both | ball_miss 0.5031 | 0.227 | -0.0695 | 5/5 | +0.056 | +0.41 |
| sens-ball_noise-1 | ball_noise:1 | both | ball_noise 1.0017 | 0.286 | -0.0102 | 5/5 | +0.010 | +0.47 |
| sens-ball_noise-2 | ball_noise:2 | both | ball_noise 2.0035 | 0.270 | -0.0258 | 5/5 | +0.061 | +0.19 |
| sens-ball_noise-4 | ball_noise:4 | both | ball_noise 4.0069 | 0.248 | -0.0493 | 5/5 | +0.080 | +1.02 |
| sens-camera_drift-0.93 | camera_drift:0.93 | both | camera_drift 0.9343 | 0.288 | -0.0077 | 5/5 | -0.030 | +1.00 |
| sens-camera_drift-2 | camera_drift:2 | both | camera_drift 2.0093 | 0.279 | -0.0182 | 5/5 | -0.003 | +0.85 |
| sens-camera_drift-4 | camera_drift:4 | both | camera_drift 4.0185 | 0.257 | -0.0393 | 5/5 | +0.018 | +0.81 |
| sens-geom_loss-0.17 | geom_loss:0.17 | both | geom_loss 0.172 | 0.278 | -0.0178 | 5/5 | -0.011 | +0.89 |
| sens-geom_loss-0.17-test | geom_loss:0.17 | test | geom_loss 0.172 | 0.269 | -0.0280 | 5/5 | +0.011 | -0.36 |
| sens-geom_loss-0.3 | geom_loss:0.3 | both | geom_loss 0.3035 | 0.254 | -0.0419 | 5/5 | +0.001 | +1.10 |
| sens-geom_loss-0.5 | geom_loss:0.5 | both | geom_loss 0.5077 | 0.213 | -0.0822 | 5/5 | +0.045 | +0.16 |
| sens-id_fragment-2 | id_fragment:2 | both | id_fragment 1.9988 | 0.295 | -0.0009 | 2/5 | -0.003 | +0.74 |
| sens-no_ball_z | no_ball_z | both | no_ball_z 1 | 0.294 | -0.0020 | 4/5 | -0.003 | +0.30 |
| sens-player_miss-0.1 | player_miss:0.1 | both | player_miss 0.1004 | 0.290 | -0.0063 | 5/5 | +0.019 | -0.16 |
| sens-player_miss-0.2 | player_miss:0.2 | both | player_miss 0.2012 | 0.286 | -0.0098 | 5/5 | +0.021 | +0.01 |
| sens-player_miss-0.4 | player_miss:0.4 | both | player_miss 0.4041 | 0.273 | -0.0229 | 5/5 | +0.019 | +0.56 |
| sens-player_noise-0.93 | player_noise:0.93 | both | player_noise 0.93 | 0.283 | -0.0128 | 5/5 | +0.029 | +0.23 |
| sens-player_noise-0.93-test | player_noise:0.93 | test | player_noise 0.93 | 0.276 | -0.0209 | 5/5 | -0.000 | +1.43 |
| sens-player_noise-2 | player_noise:2 | both | player_noise 2 | 0.272 | -0.0243 | 5/5 | +0.060 | +0.68 |
| sens-target | target (all) | both | see run.json | 0.226 | -0.0696 | 5/5 | +0.050 | +1.45 |
| sens-target-test | target (all) | test | see run.json | 0.211 | -0.0859 | 5/5 | +0.070 | +0.20 |
| sens-team_flip-0.05 | team_flip:0.05 | both | team_flip 0.0601 | 0.284 | -0.0120 | 5/5 | +0.005 | +1.07 |
| sens-team_flip-0.1 | team_flip:0.1 | both | team_flip 0.1193 | 0.280 | -0.0159 | 5/5 | +0.063 | -0.07 |
| sens-team_flip-0.2 | team_flip:0.2 | both | team_flip 0.2339 | 0.269 | -0.0261 | 5/5 | +0.037 | +1.11 |
| sens-team_unknown-0.05 | team_unknown:0.05 | both | team_unknown 0.0605 | 0.289 | -0.0069 | 5/5 | -0.000 | +0.42 |
| sens-team_unknown-0.1 | team_unknown:0.1 | both | team_unknown 0.1195 | 0.285 | -0.0099 | 5/5 | +0.016 | +0.44 |
| sens-team_unknown-0.2 | team_unknown:0.2 | both | team_unknown 0.2346 | 0.273 | -0.0221 | 5/5 | +0.043 | +0.56 |
