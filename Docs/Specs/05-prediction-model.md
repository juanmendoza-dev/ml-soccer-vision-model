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
   - **Features** (`prediction/features.py`, 29, `FEATURES_VERSION` keys a per-match cache in `data/processed/<id>/`): ball position, angle, height, speed and velocity toward goal, `ball_age_s`, change in goal distance over 1 and 3 s, seconds in the final third over the last 5 s, possession age; the likely carrier (nearest VISIBLE attacker, since PFF has none): distance to ball, speed, velocity toward goal, goal distance; pressure (nearest defender, defenders within 5 m); shooting lane (defenders and keeper in the ball-to-posts triangle, keeper off the line and to the ball); attackers/defenders goal side of the ball and in the box; visible counts. VISIBLE players only, held ball (Leakage). Pitch control is left for later.
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
