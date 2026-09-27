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
| label_mask_h5, label_mask_h3 | bool | `eligible`, and no non-open-play shot or goal by `possession_team` in (t, t+H]. **Looks ahead** |
| label_shot_h5, label_shot_h3 | bool/null | Open-play shot by `possession_team` in (t, t+H]. null where the mask is false |
| label_goal_h5, label_goal_h3 | bool/null | Open-play goal for `possession_team` in (t, t+H]. null where the mask is false |

`objects_10hz` has 02's `objects` columns plus `period`, `t_s` and `x_att, y_att, vx_att, vy_att`: rotated 180° (x → −x, y → −y, same for velocity; `z` unchanged) where `flipped` is true, otherwise a copy. Keyed by (`period`, `t_s`, `object_id`), since a sparse native rate could put the same native frame under two grid points.

### Resampling
- The grid is built per period on `timestamp_s`: `t_s = k / 10`, from the first multiple at or after the period's first frame to its last frame. All comparisons are in integer microseconds, so float error can't pull in the wrong frame.
- Each grid point takes the **latest native frame at or before it**. No interpolation: interpolating uses a later frame.
- **Gaps:** a grid point is skipped (and counted) if its frame is older than 1.5 native intervals. That decides from the past only, so it doesn't depend on whether the next frame exists. Periods are never bridged because the grid is per period.
- Positions and velocities are the native ones, unsmoothed. `vx`/`vy` are already causal (02).

### Labels
- Event time is the `timestamp_s` of the event's frame. The window is **(t, t+H]** on `t_s`, in the same period, and is cut at the period end.
- Open play = `set_piece = open_play` and `set_play_phase` not true (02). A null `set_piece` counts as not open play.
- Shots: `event_type = shot` rows, any outcome (a disallowed goal was still a shot). Goals: `event_type = goal` rows with `outcome` in {goal, own_goal}. Both are credited to the event's `team`, which for own goals is the scoring team (02).
- Only the team in possession at t counts. A shot by the other side after a turnover is a 0.
- **Mask, not 0:** a window containing a non-open-play shot or goal (penalty, direct free kick, set-play-phase shot) by the possession team is neither a clean positive nor a clean negative, so the frame is masked. Masked and ineligible frames stay in the table with null labels.
- The mask only sees set plays that end in a shot. A set-play phase that ends without one stays labelled 0.

### Leakage
- Every column not prefixed `label_` uses frames ≤ t only. `tests/test_resample.py` alters frames, objects and events after t and checks those columns don't change.
- The flip follows `possession_team` at each frame, so coordinates jump 180° when possession changes. A model reading a window of frames (temporal GNN) should re-rotate the whole window using the anchor frame's possession: keep `x`/`y`, or undo with `flipped`.

## Inference
- Predict only on frames where `ball_state` is not dead and `possession_team` is set; otherwise output null (the overlay holds or greys out the meter).
- `ball_state = null` counts as not dead, so vision gaps don't blank the meter. Report how often this happens.

## Leakage rules
- Inputs use frames `<= t` only.
- Split train/val/test **by match**.
- Report performance separately for lead times (see 07); a model that only fires 0.2 s before the shot is not useful.

## Models (build in order)
0. **Floor:** logistic regression on ball distance + angle to goal. Any model that can't beat this isn't learning anything.
1. **Baseline:** gradient boosting (LightGBM) on hand features — ball distance/angle to goal, defenders in shooting cone, carrier speed, pitch control near the box. This is the bar the GNNs must clear.
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
