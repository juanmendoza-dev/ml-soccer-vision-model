# 05 — Prediction Model

## Goal
At each frame t, output P(shot in (t, t+H]) and P(goal in (t, t+H]).

## Labels
- `shot_within_H`: 1 if the attacking team shoots within H seconds after t.
- Default **H = 5 s**; also evaluate H = 3 s.
- Only frames in open play with a team in possession. Exclude dead-ball frames.
- Attacking team = `possession_team`; flip coordinates so it always attacks +x.
- Training uses provider `ball_state` and `possession_team` (02).

## Resampled frames and labels
`prediction/resample.py` turns a game state match (02) into a 10 Hz table with labels. It isn't part of 02: only prediction code reads it, and the schema's tables stay untouched.

    python -m prediction.resample [--games 10502 ...] [--gamestate DIR] [--out DIR]

### Output
`data/processed/<match_id>/`:
- `frames_10hz.parquet`: one row per 10 Hz grid point.
- `objects_10hz.parquet`: 02's `objects` rows for the native frame each grid point uses.
- `resample_report.json`: rows in/out, grid points skipped, eligible share, positives per label, and open-play shots with no positive frame (listed, with why).

`frames_10hz`:
| Column | Type | Notes |
|---|---|---|
| match_id | str | |
| period | int | |
| t_s | float | Period time on the 0.1 s grid (`k / 10`) |
| frame_id | int | Native frame used: the latest one at or before `t_s` |
| timestamp_s | float | That frame's native time, ≤ `t_s` |
| home_attacks_positive_x, ball_state, possession_team, ball_carrier_id, view_polygon | | Copied from the native frame (02) |
| flipped | bool/null | True if coordinates were rotated so `possession_team` attacks +x. null when `possession_team` is null (not rotated) |
| eligible | bool | `ball_state = alive` and `possession_team` set. Uses nothing after t, so features and inference may use it |
| all_estimated | bool | Every player and goalkeeper `visible = False` (or none present). Training drops these; they're scored as null (see Off-camera players) |
| label_set_play_phase | bool/null | 02's `frames.set_play_phase` at the native frame. Label-side only: never a feature (02) |
| label_mask_h5, label_mask_h3 | bool | `eligible`, not in a set-play phase (null counts as not), and no non-open-play shot or goal by `possession_team` in (t, t+H]. **Looks ahead** |
| label_shot_h5, label_shot_h3 | bool/null | Open-play shot by `possession_team` in (t, t+H]. null where the mask is false |
| label_goal_h5, label_goal_h3 | bool/null | Open-play goal for `possession_team` in (t, t+H]. null where the mask is false |

`objects_10hz` has 02's `objects` columns plus `period`, `t_s` and `x_att, y_att, vx_att, vy_att`: rotated 180° (x → −x, y → −y, same for velocity; `z` unchanged) where `flipped` is true, otherwise a copy. Keyed by (`period`, `t_s`, `object_id`), since a sparse native rate could put the same native frame under two grid points.

### Resampling
- The grid is built per period on `timestamp_s`: `t_s = k / 10`, from the first multiple at or after the period's first frame to its last frame. All comparisons are in integer microseconds, so float error can't pull in the wrong frame.
- Each grid point takes the **latest native frame at or before it**. No interpolation: interpolating uses a later frame.
- **Gaps:** a grid point is skipped (and counted) if its frame is older than 1.5 native intervals, from the declared `match.native_fps` (02). That decides from the past only, so it doesn't depend on whether the next frame exists. The interval used to be the median over the whole match, which let later frames change whether an earlier row exists (detection review D8); on all 66 converted games the declared rate gives byte-identical output. Periods are never bridged because the grid is per period.
- Positions and velocities are the native ones, unsmoothed. `vx`/`vy` are already causal (02).

### Labels
- Event time is the `timestamp_s` of the event's frame. The window is **(t, t+H]** on `t_s`, in the same period, and is cut at the period end.
- Open play = `set_piece = open_play` and `set_play_phase` not true (02). A null `set_piece` counts as not open play.
- Shots: `event_type = shot` rows, any outcome (a disallowed goal was still a shot). Goals: `event_type = goal` rows with `outcome` in {goal, own_goal}. Both are credited to the event's `team`, which for own goals is the scoring team (02).
- Only the team in possession at t counts. A shot by the other side after a turnover is a 0.
- **Mask, not 0:** a window containing a non-open-play shot or goal (penalty, direct free kick, set-play-phase shot) by the possession team is neither a clean positive nor a clean negative, so the frame is masked. Masked and ineligible frames stay in the table with null labels.
- **Set-play phases are masked whether or not they end in a shot** (02 `frames.set_play_phase`; schema 0.5). Otherwise every corner or free kick that comes to nothing would train as a clean 0 and the model could learn "set-piece shape → no shot". An open-play shot just after a phase keeps the lead-up frames that fall outside the phase.

### Leakage
- Every column not prefixed `label_` uses frames ≤ t only. `tests/test_resample.py` alters frames, objects and events after t and checks those columns don't change.
- **PFF's ESTIMATED ball uses later frames (checked 2026-09-27).** 25.6% of scored rows have a ball PFF marks as not visible (ESTIMATED). The spec PDF is unreadable, so this was measured on 13 games (8,422 ESTIMATED runs with a VISIBLE ball on both sides). The estimated path isn't a straight line between detections, but it lands on the next detection: for runs of ≥ 1 s where the ball moved > 5 m (2,261 runs), the last estimated position is 0.5 m from the next visible one (median, p90 1.4 m; 85% under 1 m). That's no worse than the jump at the start of the run, and it doesn't grow with run length (0.4 m after 30 m, 3+ s gaps). A past-only estimate can't know where the ball reappears, so **any feature built on an ESTIMATED ball leaks**, and vision can't reproduce it live.
  - Models read the ball through `ball_source` (recorded in `run.json`): **`held`** (default) uses VISIBLE balls only and carries the last one forward within the period, with `ball_age_s` = seconds since it was seen. That's what vision's ball extrapolation will feed the model. **`raw`** is the ESTIMATED ball as-is, kept only to measure how much the leak inflates results.
  - Velocities too: 02's `vx`/`vy` difference over 0.2 s, so a visible frame just after an ESTIMATED run uses an estimated position. Features recompute velocity on the 10 Hz grid from visible positions only.
  - PFF's ESTIMATED players are already dropped from features (broadcast view, below), and they're probably filled the same way.
- The flip follows `possession_team` at each frame, so coordinates jump 180° when possession changes. A model reading a window of frames (temporal GNN) should re-rotate the whole window using the anchor frame's possession: keep `x`/`y`, or undo with `flipped`.

## Inference
- Predict only on frames where `ball_state` is not dead and `possession_team` is set; otherwise output null (the overlay holds or greys out the meter).
- `ball_state = null` counts as not dead, so vision gaps don't blank the meter. Report how often this happens.

## Leakage rules
- Inputs use frames `<= t` only.
- Split train/val/test **by match**.
- Report performance separately for lead times (see 07); a model that only fires 0.2 s before the shot is not useful.

## Models (build in order)
0. **Floor:** logistic regression on ball distance + angle to goal. Any model that can't beat this isn't learning anything. Done (`prediction/floor.py`, `python -m prediction.cv --model floor`): PFF pooled OOF PR-AUC **0.177** at H = 5 on the held ball (ROC-AUC 0.901, 0.159 at H = 3), and **0.167** with the leaky ESTIMATED ball (`ball_source=raw`, see Leakage; the original run) (base rate 0.025, ROC-AUC 0.918, folds 0.167 ± 0.007), 0.149 at H = 3; well calibrated. At 3 false alarms per match it misses 1,151 of 1,154 shots, so the alarm numbers to beat are essentially nothing (`Docs/reviews/floor-2026-09-27.md`). Features: `ball_dist` and `ball_angle` (goal-mouth angle) from `prediction/features.py`; rows with no ball get the training base rate.
1. **Baseline:** gradient boosting (LightGBM) on hand features. This is the bar the GNNs must clear. Done (`prediction/lgbm.py`, `python -m prediction.cv --model lgbm`, `Docs/reviews/lgbm-2026-09-27.md`):
   - **Features** (`prediction/features.py`, 29, `FEATURES_VERSION` keys a per-match cache in `data/processed/<id>/`, which the resampler deletes whenever it rewrites a match): ball position, angle, height, speed and velocity toward goal, `ball_age_s`, change in goal distance over 1 and 3 s, seconds in the final third over the last 5 s, possession age; the likely carrier (nearest VISIBLE attacker, since PFF has none): distance to ball, speed, velocity toward goal, goal distance; pressure (nearest defender, defenders within 5 m); shooting lane (defenders and keeper in the ball-to-posts triangle, keeper off the line and to the ball); attackers/defenders goal side of the ball and in the box; visible counts. VISIBLE players only, held ball (Leakage). Pitch control is left for later.
   - **Model:** fixed parameters (learning rate 0.05, 31 leaves, ≥ 500 rows per leaf), early stopping on 15% of the fit's training matches, then a refit on all of them. No class weights.
   - **Results, held ball (PFF pooled OOF):** PR-AUC **0.296** at H = 5 (floor on the same held ball 0.177; ROC-AUC 0.931), **0.298** at H = 3 (floor 0.159). Better than the floor on **5/5 folds** for PR-AUC, ROC-AUC, Brier and miss rate. Calibrated. At 3 false alarms per match it catches 174 of 1,154 shots (2.5 false alarms per match), but the median lead is 0.7 s, so **00's ≥ 2 s isn't met**: p two seconds before a shot is 0.16 (median) against τ ≈ 0.6.
   - **What it uses** (gain): ball position 0.54, likely carrier 0.19, defenders 0.13, ball motion/history 0.07.
   - With the leaky ESTIMATED ball (`raw`): PR-AUC 0.309, no better alarms.
2. **Frame GNN:** one graph per frame. Nodes = players + ball (+ goals); node features = position, velocity, team, dynamic + profile features (04); edges = all pairs or k-nearest, edge features = distance, relative velocity. Built with `unravelsports` SoccerGraphConverter.
3. **Temporal GNN:** last 2–3 s of frames (at 10 Hz) through a GNN backbone, then a GRU/T-GCN over time. Follows the SoccerAI approach.

## Off-camera players
Vision only sees players inside the broadcast frame. All training tracking has all 22 (PFF estimated, SkillCorner extrapolated, IDSSE and Metrica optical). Training only on full data would teach the model to rely on players it won't see at inference.
- Every node gets a `visible` feature (02 `objects.visible`).
- **Training view:** by default, train on what a broadcast would show:
  - PFF (primary): drop players with `visibility = ESTIMATED` (maps to `visible = False`). No camera footprint, so the visible players are the view. Frames where every player is ESTIMATED (cutaways, 06) are dropped from training and scored as null.
  - SkillCorner: drop players with `is_detected = False` (maps to `visible = False`). Its `view_polygon` is the real camera footprint.
  - Full-pitch sources (IDSSE, Metrica): drop players outside a camera footprint borrowed from a SkillCorner frame with a similar ball position.
- The ball carrier and ball are always kept if the source has them, since the broadcast camera follows the ball.
- Compare **full** vs. **broadcast view** on the SkillCorner folds (see 07 #5; PFF's ESTIMATED positions are too poor for a full-view arm). If broadcast view costs a lot, vision output will too.

## Vision sensitivity test
How much the baseline loses when PFF tracking is degraded the way vision fails (detection review W9). It decides which Phase 2 vision work is worth doing. It measures sensitivity. It doesn't show that real vision output looks like this (07 #4 does that, once paired footage exists).

**Degradations** (`prediction/degrade.py`) are applied to `objects_10hz` in memory when a match is loaded. `data/processed` is never rewritten.
- **Causal and deterministic.** Every random draw is a hash of (seed, degradation, match, period, grid tenth, object). Episodes and correlated noise run forward in time only. So a degraded row depends only on rows at or before it, and a match always degrades the same way for a given seed (default `20260928`).
- **Episodes** (gaps, false balls, flips) start with a fixed chance on each grid row. Lengths are geometric with a set mean. The start chance is set so the expected share of time covered equals the severity. It's never "pick exactly f of the match", because that would use the whole match. The realized share is recorded in `run.json`.
- **Correlated noise** is AR(1) per axis with stationary σ and a correlation time.
- Anything that depends on where the ball is (drift center, "near the ball") uses the **clean held VISIBLE ball**, never an ESTIMATED one. That's where the camera and the crowd actually are.
- **Labels and scored rows never change.** `eligible`, `all_estimated`, `possession_team`, `ball_state` and labels come from `frames_10hz` as before. So every run is scored on the same rows, and a row that loses all its objects stays scored, with NaN features. Possession and ball state stay provider values (inferred state is 07 #6).
- **Held ball only.** `--degrade` with `ball_source=raw` is an error.
- **No feature cache for degraded runs.** Features take ~8 s for all 64 games, so it isn't needed, and a degraded run can't read or overwrite the clean `features_v1_held` cache.

**Severity levels.** The first pass puts each arm at the detection review's proposed acceptance targets (§4). These are engineering targets, not measured vision numbers, so the question is "is the target good enough?" Worse levels are swept only for arms that hurt. The one measured number is homography acceptance on smoke04: 310 of 375 match frames, so 17% lost.

| Arm | Vision failure | What it does | Severity | Target level | Sweep |
|---|---|---|---|---|---|
| `ball_miss` | Ball not detected (small, blurred, occluded) | Episodes, mean 2 s: the VISIBLE ball goes not visible. The held ball covers the first 1 s | share of time | 0.10 (ball recall ≥ 90%) | 0.25, 0.5 |
| `ball_false` | Distractor ball: boots, heads, lines, spare balls (D2) | Episodes, mean 0.5 s: the VISIBLE ball moves by one offset per episode, 5–30 m in a random direction | share of time | 0.05 (ball precision ≥ 95%) | 0.1, 0.2 |
| `ball_noise` | Ball localization, airborne ball through a ground homography | AR(1), τ = 1 s, on the ball | σ, m | 1 | 2, 4 |
| `no_ball_z` | Vision has no ball height (02: `z` null) | `z` → null | none | always | none |
| `camera_drift` | Homography error, shared by the whole frame | AR(1), τ = 3 s: an offset plus a scale error around the ball (offset σ / 20 per m) on every object. The offset σ is severity / 1.3, so the RMS error per axis of visible objects comes out at the severity (1.25–1.32× the offset on 8 games) | error per axis, m | 0.93 | 2, 4 |
| `geom_loss` | Rejected homography: nothing gets projected | Episodes, mean 1 s: every object goes not visible | share of time | 0.17 (smoke04) | 0.3, 0.5 |
| `player_miss` | Missed player detections | Per-player episodes, mean 1 s: not visible | share of time | 0.10 | 0.2, 0.4 |
| `player_noise` | Foot-anchor and box error | Per-player AR(1), τ = 0.5 s | σ, m | 0.93 | 2 |
| `team_flip` | Wrong kit cluster | Per-player episodes, mean 3 s, covering f everywhere plus another f within 10 m of the ball (crowding) | share of time | 0.05 (team accuracy > 95%) | 0.1, 0.2 |
| `team_unknown` | W5's "unknown" team instead of a guess | Same process, `team` → null (counted as neither side) | share of time | 0.05 | 0.1, 0.2 |
| `id_fragment` | Track breaks (occlusion, 1 s lost-track buffer, cuts) | Per-player new `object_id` at a constant rate. Player velocities need the same ID 0.5 s apart | mean track life, s | 2 | 1, 0.5 |

- 0.93 m per axis is the review's "≥ 90% of players within 2 m" target: for 2D Gaussian error, P(r < 2) = 0.9.
- **Combined arm:** every arm at its target level at once. `camera_drift` and `player_noise` each take 0.66 m, so together they still make 0.93. This is the headline. Arms always apply in the table's order, whatever the command-line order.

**Protocol**
- `python -m prediction.cv --model lgbm --degrade <arm>:<severity> [--degrade ...] [--degrade-arm both|test] [--degrade-seed N]`. The arms, seed and realized severity go into `run.json` config.
- **Primary arm (`both`):** degrade training and test matches. Vision errors can be simulated in training, so this is the deployable setting.
- **Secondary arm (`test`):** fit and choose τ on clean data, then predict degraded held-out folds. This is the cost of not simulating errors. It's run for the combined arm and the arms that hurt most.
- Same folds, same τ rule, H = 5 only (H = 3 tracked H = 5 closely for LightGBM). Each run is compared with the clean `lgbm-held-2026-09-27` using `evaluation.compare`. At the lane commit the clean features recompute to that run's cache exactly on all 64 games.
- **Ranking:** by mean ΔPR-AUC over folds, plus how many folds get worse. Miss rate and false alarms per match are secondary. They're noisy at this budget: raw vs held ball swung fold 0's miss rate from 0.75 to 0.89 on +0.01 PR-AUC.
- **Seed noise:** `ball_miss` at its target level is also run with a second seed, to show how big a Δ is just noise.
- **Caveat:** the ranking is for this baseline's features. ID fragmentation can only reach `carrier_speed` and `carrier_vgoal` (about 5% of gain), and ball history is about 7%. The temporal GNN and the planned attack-building features will lean much more on tracks and history. So a small loss here doesn't clear tracking work.

**Results (2026-09-28, `Docs/reviews/sensitivity-2026-09-28.md`)**
- Every arm at target together: PR-AUC 0.296 → **0.226** (−0.070, 5/5 folds). Fit clean and predict degraded: −0.086. So a model that runs on vision output gets trained with the degradations on.
- Biggest single losses at target: `geom_loss` −0.018, `ball_miss` −0.017, `player_noise` −0.013, `team_flip` −0.012, `ball_noise` −0.010. `id_fragment` and `no_ball_z` are noise for this baseline.
- `geom_loss` grows faster than linear (0.3: −0.042, 0.5: −0.082). Per affected frame, rejecting a homography costs more than keeping it with 2–4 m of error (17% rejected ≈ 2 m on every frame), so acceptance thresholds lean permissive; the cutoff is set on the vision benchmark.
- `team_unknown` costs less than `team_flip` at the same share, but a real unknown option abstains on some correct teams too. At 2× the share it's about even, so it's only worth it if abstentions land mostly on would-be flips.

## xG model
### Feature rule
Every xG feature must be (a) computable at frame t from game state (02), before any shot happens, and (b) present in the training source. Anything only known once the shot is taken is out.

| Feature | Known at t? | StatsBomb 360 | Wyscout | Use |
|---|---|---|---|---|
| Distance, angle to goal | Yes | Yes | Yes | Yes |
| Defenders in the shooting cone | Yes | Yes (freeze frame) | No | Yes |
| Goalkeeper position / distance off line | Yes | Yes (freeze frame) | No | Yes |
| Nearest defender distance (pressure) | Yes | Yes (freeze frame) | No | Yes |
| Body part (foot / head) | **No** | Yes | Yes | **No** |
| Shot technique (volley, lob, ...) | **No** | Yes | Partly | **No** |
| Play pattern (open play / set piece) | Yes | Yes | Yes | Filter only: train on open play, matching 05's labels |

### Training data
- **Primary: StatsBomb 360 open data.** 300 men's matches have 360 freeze frames (~7,500 shots, estimated at ~25/match; checked 2026-09-25). Freeze frames only include players visible on the broadcast, like our vision output.
- **Wyscout (defcon CSV):** location-only xG (distance + angle). A baseline and sanity check, not the production model.
- Convert StatsBomb coordinates (120 × 80 yards, origin top-left) to 02's meters.
- **Leakage:** PFF World Cup 2022 is in every CV fold, so StatsBomb's World Cup 2022 (the same 64 matches) is always dropped from xG training.
- **Check calibration on our own data:** apply the xG model at the shot frame to PFF shots (129 goals in the 51 tracked games, 06) and SkillCorner shots (61 goals), and compare. Report it; don't retrain on it.

### Combining with P(shot)
xG at the carrier's current position is an approximation. The shot usually happens later, from somewhere closer to goal. Build in order:
1. **v1: carrier position.** P(goal) = P(shot) × xG(ball carrier at t). Tends to underestimate P(goal) early in an attack, since the carrier is still far from goal. Measure how much by comparing to xG at the actual shot frame.
2. **v2: shooter-weighted.** If the node-level "who will shoot" head exists, P(goal) = Σ_i P(player i shoots) × xG(player i at t). Still uses positions at t, but covers runners who aren't on the ball.
- A direct P(goal within H) model is out: even PFF + SkillCorner give ~200 goals with tracking, far too few to train it.

## Class imbalance
- Weighted loss or focal loss; evaluate with PR-AUC, not accuracy.

## Acceptance criteria
- LightGBM baseline beats the distance + angle floor.
- Temporal GNN beats the LightGBM baseline on PR-AUC in grouped cross-validation (pooled out-of-fold, and on most folds), confirmed on the IDSSE external test set (see 07).
- Calibration error acceptable after (optional) temperature scaling.

## Open questions
- Predict "which player will shoot" as a node-level head too?
- Add uncertainty (MC dropout / Bayesian head, as in Goka et al. 2023)?
