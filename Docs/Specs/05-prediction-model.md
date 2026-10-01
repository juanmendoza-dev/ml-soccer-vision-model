# 05 — Prediction Model

## Goal
At each frame t, output P(shot in (t, t+H]) and P(goal in (t, t+H]).

## Labels
- `shot_within_H`: 1 if the attacking team shoots within H seconds after t.
- Default **H = 5 s**; also evaluate H = 3 s.
- Only frames in open play with a team in possession. Exclude dead-ball frames.
- Attacking team = `possession_team`; flip coordinates so it always attacks +x.
- Training uses provider `ball_state` and `possession_team` (02). Labels always do; the model's inputs can take stage 8's possession instead (Inferred possession, below).

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
   - **Lead-time features (v2, tested 2026-09-28: not adopted).** The baseline's alarms come 0.7 s before the shot because p only climbs once the ball is near goal (median p 0.05 five seconds out). v2 adds features that see an attack *building* over 2–5 s. `features_version = "2"` is v1's 29 columns unchanged plus the ones below, cached as `features_v2_<ball_source>.parquet`. v1 stays the default until v2 wins on most folds, so the sensitivity runs still reproduce.
     - **Anchor frame for history.** Every older row is measured toward the anchor row's goal, and "attackers" are the anchor row's `possession_team`, even across a turnover. Each quantity is computed in pitch coordinates for both teams and both ends, and the anchor row picks its own (team, end). Rows with a null `possession_team` have no attacking team, so team features there are NaN (the model doesn't predict on them).
     - **Windows** cover the last n grid rows, the anchor included, and never cross a period or a grid gap (`seg`). Possession runs (`poss_*`) are consecutive rows with the same non-null `possession_team` in one `seg`. `possession_s` (v1) only resets at periods, but PFF has no grid gaps inside a period (130 segs = 130 periods), so the two agree on PFF.
     - **Same inputs as v1:** the held VISIBLE ball, velocities only between two fresh sightings 0.5 s apart, and VISIBLE players only. Goalkeepers are left out of the near-box counts and the defensive line. The likely carrier is the anchor team's visible player nearest the held ball, as in v1.
     - **Known v1 flaw, left in place:** `ball_final_third_s` flags each older row with *its own* flip. So after a turnover, time the ball spent in the old attacker's final third counts for the new attacker, and null-possession rows count toward +x. v1 columns stay byte-identical so that v2 − v1 is a clean additive test. `poss_final_third_s` is the anchor-correct replacement.

     | feature | definition (anchor frame) | NaN when |
     |---|---|---|
     | `ball_dist_change_5s` | goal distance now − 5 s ago | no ball at either end |
     | `ball_dist_max_5s`, `ball_dist_min_5s` | farthest / closest the ball was from goal over the last 5 s | no ball in the window |
     | `ball_vgoal_mean_2s`, `ball_vgoal_mean_5s` | mean speed toward goal over the rows with a fresh velocity | no fresh velocity in the window |
     | `ball_box_s` | seconds of the last 5 s with the ball in the box | never (0 without a ball) |
     | `poss_start_dist` | goal distance of the first ball seen in this possession | no ball seen yet in the run |
     | `poss_min_dist` | closest the ball has been to goal in this possession | same |
     | `poss_final_third_s` | seconds of this possession with the ball in the final third | no team |
     | `poss_advance_rate` | (`poss_start_dist` − `ball_dist`) / max(possession age, 1 s), m/s | either is NaN |
     | `carrier_goal_dist_change_3s`, `_5s` | the carrier's goal distance now − the anchor team's carrier then | no carrier at either end |
     | `att_near_box`, `def_near_box` | outfield players in or near the box: x ≥ 26 m (10 m in front of the box), \|y\| ≤ 25 m | no team (0 with no visible players) |
     | `near_box_diff` | `att_near_box` − `def_near_box` | same |
     | `att_near_box_change_3s`, `def_near_box_change_3s` | count now − count 3 s ago (the anchor team's attackers / defenders then) | no row 3 s back |
     | `def_line_dist` | 52.5 − x of the deepest visible outfield defender | no visible outfield defender |
     | `def_line_change_3s` | `def_line_dist` now − 3 s ago (negative: the line is dropping) | no line at either end |
     | `ball_line_gap` | line x − ball x (negative: the ball is past the deepest defender) | no line or no ball |
     | `att_beyond_line` | visible attackers deeper than that defender | no line |
     - **Tests** (`tests/test_features.py`): every v2 feature is unchanged when frames after t change; scrambling every ESTIMATED position changes none of them; a mirrored scene with swapped teams gives identical v2 features; hand-built windows that contain a turnover and a null-possession row. On all 64 games, the v1 columns inside v2 must equal the `features_v1_held` cache.
     - **Test protocol:** `python -m prediction.cv --model lgbm --features-version 2 --horizons h5 h3`, compared with `lgbm-held-2026-09-27` using `evaluation.compare`. v2 − v1 is the ablation (gain share isn't one). Reported:
       - PR-AUC and calibration
       - at the τ budget: shots caught, median lead, and the share of all 1,154 open-play shots with ≥ 2 s lead (v1: 3 / 1,154)
       - median p at 5 / 2 / 1 s before open-play shots, for both runs on the same shots and rows (`scripts/lead_time.py`); diagnostic only, not the test
     - **Results (`Docs/reviews/lead-time-2026-09-28.md`): v2 doesn't help, so v1 stays the baseline.**
       - PR-AUC **0.287 vs 0.296** at H = 5 (v2 better on 0/5 folds) and 0.294 vs 0.298 at H = 3 (0/5). ROC-AUC +0.001, still calibrated.
       - Median p before shots is unchanged (5 s: 0.048 vs 0.051, 2 s: 0.158 vs 0.158, same shots and rows). Ranking of positives 2–5 s before the shot isn't better either (1/5 folds).
       - Alarms: at matched false alarms (pooled, descriptive) v2 catches fewer shots (127 vs 172 at ≤ 3 per match). Its inner τ went over budget on the held-out folds (4.6 false alarms per match). Lead ≥ 2 s: 1 of 1,154 vs 3.
       - The v2 features take 0.24 of the gain, mostly from the carrier and defence groups, without better ranking. So the last 5 s summarized this way adds no ranking that v1 doesn't already have. Lead time goes to the temporal GNN next.
2. **Frame GNN** (designed 2026-09-28; `prediction/graphs.py` builds the graphs, `prediction/gnn.py` is the model, `python -m prediction.cv --model gnn`). One graph per 10 Hz grid row and a small message-passing network. Same folds, τ rule and run format as LightGBM, so `evaluation.compare` and `scripts/lead_time.py` read it unchanged. Code and tests on the M1, full CV on the workstation (roadmap, 09).
   - **Not unravelsports.** Its converter smooths each track's velocity with a centered Savitzky–Golay filter over the whole period (7 frames for players, 3 for the ball), so the velocity at t reads frames after t. It takes a kloppy dataset, which has no `visible` flag, so it would read PFF's ESTIMATED positions, and it orients every frame by its own ball-owning team. The graphs have to follow the leakage rules below, so they're built from `objects_10hz` the way the hand features are. unravelsports and torch-geometric leave the `prediction` extra (both resolved on Windows, so they can come back if a later model needs them).
   - **Nodes,** in the attacking frame (pitch x, y and velocities times the row's sign, −1 where `flipped`; a row with no team isn't rotated and has no attackers):
     - every VISIBLE player and goalkeeper on the row (referees are left out), and
     - the held VISIBLE ball (`ball_source = held`: the last visible ball, carried forward up to 1 s within the seg), when there is one.
     - At most 23 nodes: the ball plus 22 players. PFF never has more than 22 visible players on a row (Metrica has 23 on 3 rows). Past 22, the players nearest the held ball are kept.
     - Order: the ball first, then players by distance to the ball (no ball: by goal distance). The network doesn't depend on node order; a fixed order just makes the tensors comparable in tests (a mirrored scene gives the same tensor).
     - No goal nodes: the goal is at (52.5, 0) in the attacking frame, so each node carries its distance and angle to it.
     - No separate `visible` feature (Off-camera players): every player node is visible, and the ball's freshness is `ball_age_s`.
     - No profile features (04) yet. They come with the profiles ablation.

     | node feature | players | ball |
     |---|---|---|
     | `x`, `y` (÷ 52.5, ÷ 34) | position | held position |
     | `vx`, `vy` (÷ 10), `has_vel` | same `object_id` VISIBLE on this row and 0.5 s earlier in the same seg, as the hand features; else 0 with `has_vel` = 0 | between two fresh sightings 0.5 s apart, as `ball_speed` |
     | `is_ball`, `is_att`, `is_def`, `is_gk` | team vs `possession_team` (no team or no possession: neither); keeper flag | `is_ball` |
     | `goal_dist` (÷ 52.5), `goal_angle` (÷ π) | to (52.5, 0), the goal-mouth angle of the hand features | same |
     | `ball_dist` (÷ 52.5), `has_ball` | distance to the held ball; 0 and `has_ball` = 0 without one | 0, 1 |
     | `ball_age_s` | 0 | seconds since the ball was seen (0–1) |
     | `ball_z` (÷ 3) | 0 | held height; 0 when null (vision) |

   - **Edges:** fully connected, with self-loops. With ≤ 23 nodes a dense 23 × 23 is cheaper than building kNN lists. Edge features are computed on the device from the node features: dx, dy, distance, relative velocity (dvx, dvy) and a flag for both nodes having a velocity.
   - **Graph-level extras: used.** The row's 29 v1 hand features go in as one global vector: standardized on the fit's training rows, clipped to ±5, NaN → 0 plus a missing flag per feature. The GNN then starts from what LightGBM has (possession age and ball history included), and the question is whether the graph adds to it. A neural net on tabular features often trails gradient boosting, so this doesn't guarantee a match.
   - **Arms, and which claim each one supports.** The hybrid is the default and the one run first. The other two are optional follow-ups, same folds and horizons:

     | run | flags | compared with | supports the claim |
     |---|---|---|---|
     | hybrid (graph + v1 globals) | default | LightGBM | "the GNN beats the baseline" |
     | globals-only net | `--gnn-layers 0` (same net and globals, no message passing or node readout) | hybrid | "**the graph** adds something". Without this arm a hybrid win could just be a neural net reading the v1 features differently |
     | graph-only | `--gnn-globals none` | LightGBM, hybrid | "the graph alone learns what the hand features know" / what the globals add |
   - **Implementation: dense and padded, in plain PyTorch.** A match's graphs are one `(rows, 23, 15)` float16 array plus a node count, and a batch is a slice of it. There are no PyG `Data` objects (3.9M of them would be slow) and no DataLoader workers (Windows spawns processes).
     - Network: 3 message-passing layers, width 64, about 120k parameters. Each pair gets a message from both nodes and the edge. Messages are combined by an attention-weighted mean plus a plain sum (counts matter: defenders in the box), then a residual update with LayerNorm.
     - Readout: mean, max and sum over the nodes, the ball node's embedding and the global vector, then an MLP to one logit.
     - LayerNorm, never BatchNorm: a row's p must not depend on the other rows in its batch.
     - Batches of similar node counts: each run of 64 batches is sorted by node count, and every batch is padded only to its own largest graph. Padding is masked, so a row's p doesn't change. PFF's scored rows average 14.6 nodes (median 15), so this cuts the node pairs 2.25× compared with padding to 23, and ran about 2× faster on the M1.
   - **Training:**
     - Plain log loss, no class weights, as LightGBM: p has to stay calibrated for the alarms and for P(goal) = P(shot) × xG (this replaces the old weighted/focal loss line, see Class imbalance).
     - The training rows are LightGBM's (label mask true, not `all_estimated`), subsampled: epoch e takes the rows whose grid tenth k has (k + e) mod 4 = 0. An epoch is a quarter of the rows, and every 4 epochs cover all of them (10 Hz rows are near duplicates). **Prediction covers every grid row** (alarms run over all rows, 07), null where 05 says not to predict.
     - Early stopping holds out 15% of the fit's training matches (whole matches, seeded, like LightGBM's `es_share`). After every epoch it takes the log loss on all their training rows, with patience 4 and at most 40 epochs (10 passes). The best epoch's weights are kept, and **there is no refit** on all the matches. A refit would double the cost, and training on 85% of the matches handicaps the GNN, not LightGBM.
     - Then one temperature T is fitted on the early-stopping matches' rows (1 parameter, log loss), and p = sigmoid(logit / T). It changes calibration only, never the ranking. T is recorded per fold.
     - AdamW, learning rate 1e-3, weight decay 1e-4, batch 512, gradient clipping at 1, dropout 0.1. Fixed parameters, not tuned (as LightGBM).
   - **Leakage** (the hand features' rules): VISIBLE players only, the held VISIBLE ball, velocities only between VISIBLE sightings, and no history past that 0.5 s, rotated with the anchor row's sign. The globals are the v1 features, already tested causal. The standardization and T come from training matches only.
   - **Tests** (`tests/test_graphs.py`; the GNN cases are in `tests/gnn_cases.py`, and `tests/test_gnn.py` runs them in their own pytest process, since on macOS torch and LightGBM can't share one, 09):
     - graphs and p don't change when frames after t change;
     - scrambling every ESTIMATED position changes nothing;
     - a mirrored match with the teams swapped gives the same graphs and p;
     - a row's p doesn't depend on the other rows in its batch;
     - the GNN learns a toy signal through `run_cv`;
     - every prediction is out of fold, including against the early-stopping matches.
   - **Scale:** about 3.9M grid rows over 64 games.
     - Graphs are cached per match as `graphs_v2_held.npz` (v2 added the temporal GNN's `team`, `sign` and `poss`, nodes unchanged) next to the feature caches. The key is separate, so `features_v1/v2` are never touched, and the resampler deletes both when it rewrites a match.
     - All 64 games in float16 take about 2.7 GB of RAM, and about 4 GB with the frames and v1 features. That's well under the workstation's 32 GB (and fine for the M1's tests).
     - VRAM: the per-pair tensors are about 2 GB at batch 512, under the 2060's 6 GB. The peak is recorded.
     - Run time on the 2060 isn't measured yet. The one-fold timing run measures it before a full run.
   - **Smoke check** (M1, MPS, 2026-09-29; `scripts/gnn_smoke.py`). Fitted on 10503 and 10504, with 10502 for early stopping, and predicted 10505. Not a result: three matches can't be compared with LightGBM.
     - Training loss went from 0.127 to 0.052. Early-stopping loss was best at epoch 5 (0.111 → 0.098), then rose until patience stopped the fit at epoch 9.
     - On 10505: PR-AUC 0.34 (base rate 0.020), ROC-AUC 0.95. Median p before its 15 open-play shots was 0.03 at 5 s, 0.29 at 2 s and 0.42 at 0.2 s.
     - Mean p was 0.028 against a 0.020 base rate, and the top decile predicted 0.24 but saw 0.16 (T = 1.14). So it's a bit overconfident on this one match. Watch calibration in the full run.
     - Speed: 2,800 training rows/s on MPS (early-stopping passes included), and 58,572 rows predicted in 2 s.
   - **Device and determinism:** `--device auto|cuda|mps|cpu`, where auto means cuda, then mps, then cpu. Every random draw is seeded: torch, the early-stopping matches and the shuffling. CUDA runs with deterministic algorithms where it has them. CPU is bitwise reproducible; CUDA and MPS come out close, not bitwise. The device, its name, the peak GPU memory and the timings go into `run.json`.
   - **Timing run:** `--one-fold N` runs outer fold N only (its inner fit, τ and outer fit) and skips the final τ. It saves that fold's matches only and marks the run partial (`partial` in `run.json`, a "PARTIAL RUN" report). Then it prints an estimate of the full run: about 5.5 × one fold per horizon, plus the one-off data loading. The first run on a machine also builds the graph caches; that time is reported separately and left out of the estimate. It's a timing and a sanity check, not a result.
   - **Protocol:** `python -m prediction.cv --model gnn --horizons h5 --run-id gnn-frame-<date>-h5`, then the same with `h3`. Compared with `lgbm-held-2026-09-27` by `evaluation.compare` (per fold, "wins" means most folds) and `scripts/lead_time.py` (lead time, matched false alarms, ranking by time to shot). The GNN has no gain shares (`coef` is empty), so lead_time.py's gain table shows zeros for it.
3. **Temporal GNN** (designed 2026-09-29; `python -m prediction.cv --model tgnn`, same `GNNModel` with `steps` > 1, the network is `TemporalGNN` in `prediction/gnn_net.py`). The frame GNN sees one row. This model sees the last 2.5 s of rows, so it can rank the rows 2–5 s before a shot on how the attack is moving, which is where LightGBM and the v2 features fell short (Lead-time features above). Code and tests on the M1, runs on the 2060, like the frame GNN.
   - **Window:** `steps` = 6 grid rows spaced `step_rows` = 5 apart (0.5 s): t − 2.5 s, t − 2.0 s, …, t. Both are params, so the one-fold timing run can shrink them without code edits (`--gnn-param steps=4`, also `batch_size`, `stride` and the rest of `gnn.PARAMS`). A step is used only if its row is in the same match and period and its tenth is exactly `step_rows` × lag below the anchor's (so no grid gap in between). Otherwise the step is masked. A valid step means no gap between it and the anchor, so a gap masks every step before it: masked steps are always the oldest ones, as history never crosses a seg in the hand features. The only holes in the middle are empty graphs (nothing visible), which the GRU skips. On 10502's training rows 0.07% of steps are masked and 3.7% are empty.
   - **Re-rotated to the anchor** (Leakage above). Each stored graph is in its own row's frame, so every step's nodes are turned into the anchor row's frame:
     - flip factor = the step row's sign × the anchor's sign. Where it's −1: x, y, vx, vy change sign, and `goal_dist` and `goal_angle` are recomputed. They're left as stored where it's +1, so the anchor step is exactly the frame GNN's input. `ball_dist`, the ball's age and height don't change.
     - `is_att` / `is_def` come from each node's team against the **anchor's** `possession_team`, so after a turnover the old attackers are defenders in the older steps. A step with a null `possession_team` gets its teams the same way (it has no rotation of its own, sign +1).
     - That needs per-node team and per-row sign and possession, which the frame graphs didn't keep. `GRAPHS_VERSION` 2 adds `team` (rows × 23, int8: +1 home, −1 away, 0 ball, padding or no team), `sign` and `poss` (+1 home, −1 away, 0 none) to the cache. `nodes` are unchanged, so the frame GNN reads the same inputs.
   - **Windows aren't stored.** They're gathered from the `GraphStore` per batch (6 × 3.9M graphs would be about 16 GB). The re-rotation runs in numpy in float32 after the gather.
   - **Network:**
     - Each step's graph goes through the frame GNN's encoder and message passing (3 layers, width 64, shared by all steps). The readout is the same (mean, max, sum, ball node).
     - Each step's readout goes through a linear layer to width 64, then a GRU runs over the steps, oldest first. A step with no nodes (masked, or nothing visible on its row) keeps the previous hidden state.
     - The head reads the GRU's last state, the anchor's own readout and the global vector (the anchor row's v1 features), then an MLP to one logit. The anchor readout is there so the model starts from what the frame GNN sees.
     - No node links across time. Each step's graph is order-free, so a player isn't followed from step to step. Track identity only reaches the model through `has_vel` (the same `object_id` 0.5 s back).
     - The same class is written separately from `FrameGNN`, which isn't refactored: changing its modules would shift its init and so its results.
   - **Training:** the frame GNN's rules (training rows, stride-4 epochs, log loss, early stopping on whole matches, no refit, temperature) with **batch 128**. A batch is 128 × 6 graphs, about as many pairs as the frame GNN's 512 at 1.5×, which stays under the 2060's 6 GB. Batches are bucketed by the largest node count in each window. `--gnn-layers 0` is rejected (no graph means no temporal model); `--gnn-globals none` works.
   - **Leakage:** every step is at or before t, in the same match and period, built from the frame graphs (already causal, ESTIMATED-free). The anchor row decides the rotation and the teams. The globals are the anchor's v1 features.
   - **Tests** (`tests/test_graphs.py` for the window gather, numpy only; `tests/gnn_cases.py` for the model):
     - a window never reads another match, another period or across a grid gap, and a gap masks every step before it;
     - a turnover and a null-possession row inside a window: older steps are rotated into the anchor frame and their teams follow the anchor's possession; a mirrored match gives the same windows;
     - the anchor step equals the frame graph exactly;
     - p doesn't change when frames after t change, or when ESTIMATED positions are scrambled; a mirrored match gives the same p; a row's p doesn't depend on its batch;
     - it learns a toy signal through `run_cv`, and every prediction is out of fold;
     - it learns a label only the window shows (did the ball get closer to goal between 1.0 and 0.5 s ago, on a random-walk ball): PR-AUC 0.64 against 0.37 for the frame GNN without globals, which sits at the base rate (0.38).
   - **Cost:** about 6× the frame GNN per row. Not measured. If the one-fold timing run is too slow on the 2060: `steps` 4 (last 1.5 s), then stride 8.
   - **Protocol:** `--model tgnn --horizons h5 --run-id gnn-temporal-<date>-h5`, then `h3`. The comparison that matters is **temporal vs frame GNN** (same globals, paired by fold with `evaluation.compare`, plus lead time and ranking by time to shot from `scripts/lead_time.py`): that's the claim "history helps". Beating LightGBM alone doesn't show it.
   - **Vision sensitivity (`id_fragment` rerun).** Since tracks aren't followed across steps, fragmentation reaches this model through `has_vel` only, much as it reached LightGBM. So the rerun will measure less than "the temporal model leans on tracks" suggests. `--degrade` works with the GNNs (Vision sensitivity test).

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
- **Labels and scored rows never change.** `eligible`, `all_estimated`, `possession_team`, `ball_state` and labels come from `frames_10hz` as before. So every run is scored on the same rows, and a row that loses all its objects stays scored, with NaN features. Possession and ball state stay provider values (inferred state is 07 #6, Inferred possession below).
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
- **Caveat:** the ranking is for this baseline's features. ID fragmentation can only reach `carrier_speed` and `carrier_vgoal` (about 5% of gain), and ball history is about 7%. The temporal GNN will lean much more on tracks and history, so a small loss here doesn't clear tracking work: rerun `id_fragment` on it. The attack-building features (v2) were tested and not adopted (Lead-time features above), so they don't need a rerun.
- **GNNs** (`--model gnn|tgnn --degrade ...`, added 2026-09-29): the graphs are built from the same degraded objects as the global features. The degradation is deterministic per (seed, match), so running it once for the features and once for the graphs gives the same objects (the features pass their scored rows, which only change the stats; `tests/test_degrade.py` checks that). Degraded graphs are built in memory and never cached (about 1 s a match on the M1), so a degraded run can't read or overwrite the clean `graphs_v2_held` cache. With `--degrade-arm test`, two graph stores are loaded: the clean one for every fit and τ choice, and a degraded one swapped in before each held-out prediction (about 2.7 GB each, fine in 32 GB). `id_fragment` renames `object_id`, so it reaches the graphs through `has_vel` only (no node follows a track across the temporal GNN's steps). Compared with the clean run of the same GNN, not with LightGBM.

**Results (2026-09-28, `Docs/reviews/sensitivity-2026-09-28.md`)**
- Every arm at target together: PR-AUC 0.296 → **0.226** (−0.070, 5/5 folds). Fit clean and predict degraded: −0.086. So a model that runs on vision output gets trained with the degradations on.
- Biggest single losses at target: `geom_loss` −0.018, `ball_miss` −0.017, `player_noise` −0.013, `team_flip` −0.012, `ball_noise` −0.010. `id_fragment` and `no_ball_z` are noise for this baseline.
- `geom_loss` grows faster than linear (0.3: −0.042, 0.5: −0.082). Per affected frame, rejecting a homography costs more than keeping it with 2–4 m of error (17% rejected ≈ 2 m on every frame), so acceptance thresholds lean permissive; the cutoff is set on the vision benchmark.
- `team_unknown` costs less than `team_flip` at the same share, but a real unknown option abstains on some correct teams too. At 2× the share it's about even, so it's only worth it if abstentions land mostly on would-be flips.

## Inferred possession (07 #6)
The model trains and scores on PFF's `possession_team`, but live it would get stage 8's (03, `vision/state.py`), which agrees with PFF on 77.9% of frames (`Docs/reviews/stage8-2026-09-29.md`). This arm trains the same model on the same folds with its **inputs** built from stage 8's possession, and keeps everything that's scored on PFF's. The Δ against the provider run is what stage 8 alone costs, before any vision error.

    python -m prediction.cv --model lgbm --possession inferred [--horizons h5 h3] [--run-id ID]

**What changes: the inputs only.**
- Per match, `vision.state.infer` runs on the native game state (`data/gamestate/<match>/frames.parquet` + `objects.parquet`, 02), exactly as `vision.state_check` calls it: VISIBLE ball and players only, default `StateConfig`. Stage 8's rules and defaults aren't touched.
- Its `possession_team` is joined onto the 10 Hz grid by `frame_id` (every grid row keeps the native frame it uses), and `flipped` is recomputed from it with the resampler's own rule (`resample.flipped_expr`, the one `add_frame_flags` uses, against the grid's `home_attacks_positive_x`).
- Only that `(period, t_s, possession_team, flipped)` goes into `match_features`. So the attacking-frame rotation, the attacker/defender split (every player feature), `possession_s`, and v2's possession runs all follow stage 8.
- **Unused on purpose:** stage 8's `ball_state` and `ball_carrier_id`. No feature reads `ball_state` (it's in the loaded table for the alarms only), and the carrier features take the nearest visible attacker, not `ball_carrier_id`. Inferred ball state is out of scope for this run: it's 45% null, and all it could change is `eligible`, which is label-side here (below). Where ball state matters is the alarm rule, and that's a later question (07 #6).
- **Where it's called from:** `prediction/possession.py` is the one place prediction calls into `vision/`, and only `vision.state.infer` (03 says stage 8 is written to run on dataset tracking). `resample.py` keeps its no-`vision/` rule.

**What stays PFF's.** `eligible`, `all_estimated`, `label_mask_*`, `label_shot_*` and the table's own `possession_team`, `flipped` and `ball_state` are the provider values, read from `frames_10hz` as before. So:
- Every arm is trained and scored on the same rows as the provider run (the shots really happened), and `evaluation.compare` is paired row for row.
- `predictable()` uses PFF's `eligible`, so p is null on the same rows.
- τ selection and the alarm rule read `possession_team` and `ball_state` from the table, so they stay on PFF's too. Only the model's inputs differ between the two arms (07 #6).

**Where they disagree.** On a scored row where stage 8 says the other team has the ball, the model sees the pitch rotated toward the other goal and the teams' roles swapped, while the label asks about PFF's team. That mismatch is the cost being measured, not a bug.

**Nulls.** Stage 8's possession is null on 0.1% of frames (the start of a period until someone first carries the ball). A null does what a null PFF possession already does in `match_features`: `flipped` null, no rotation (sign +1), nobody is an attacker or a defender (counts 0, carrier and pressure features NaN), `possession_s` NaN. The difference is that PFF-null rows are never scored, while an inferred null can land on a PFF-eligible row, so here it is scored and trained on, with those features.

**Leakage.** A grid row at `t_s` uses the native frame at or before it (Resampling). `infer` is causal: a frame's output depends on native frames up to it in `(period, timestamp_s)` order and on VISIBLE objects only (03, tested in `tests/test_state.py`). So the inferred possession on a row uses native frames `<= t_s`, and PFF's ESTIMATED positions (05 Leakage) never reach it. Native frames between grid points (and in skipped gaps) do feed the state machine; they're all in the past too. `tests/test_possession.py` changes objects after t and checks inferred possession and every feature at rows `<= t` stay the same.

**Caches.** Both are separate from the provider ones and never read or overwrite them:
- Inferred state per match: `data/processed/<match>/state_inferred_<key>.parquet` (`frame_id`, `possession_team`), where `<key>` is the first 10 hex of sha1 of the sorted `StateConfig` JSON.
- Features: `features_v<version>_<ball_source>_pinf_<key>.parquet`, same key.
- The inferred state comes from the game state, not the resampled tables, so it can't tell when a match was reconverted. `prediction.resample` deletes it with the feature and graph caches when it rewrites a match, and a reconversion is always followed by a resample.

**Run metadata** (`run.json` config): `possession: "inferred"`, `state_config` (every `StateConfig` value), and over the scored rows of `horizons[0]`: the share where inferred ≠ PFF possession, where `flipped` differs, and where inferred is null. Plus the time stage 8 took and the time the features took, apart. A provider run leaves these keys out, so its `run.json` config looks like the earlier runs'.

**Scope.**
- `--model lgbm` is the run. `--model floor` goes through the same `load_match` and works too (its ball features rotate with `flipped`), but isn't part of 07 #6.
- `--model gnn|tgnn` with `--possession inferred` is an error: the graphs are built from `frames_10hz`'s possession in `prediction.graphs` and cached separately, and wiring them is its own change.
- `--degrade` with `--possession inferred` is an error. Stage 8 would run on clean native tracking while the features see degraded objects, which isn't anything vision produces. The live combination (stage 8 on degraded tracking) needs `degrade` at the native rate, which it doesn't do.
- H = 5 and H = 3 in one run, like `lgbm-held-2026-09-27`, with the same features version (1), ball source (held), seed and folds.

**Result (2026-09-29, `Docs/reviews/possession-inferred-2026-09-29.md`).** PR-AUC 0.296 → **0.258** at H = 5 (−0.038, 5/5 folds), 0.298 → 0.260 at H = 3. Stage 8 disagrees with PFF on 14.2% of scored rows, and there the model can't rank at all (0.026 vs 0.237). There, PFF's team goes on to shoot about 3× as often as stage 8's, so stage 8 is mostly the one that's wrong. Where they agree the model is still worse (−0.014), since it trained with the frame reversed on 14% of rows. Stage 8 needs work before a model runs on it (07 #6).

### Learned possession (proposed 2026-09-30, not built)
The full contract is in [03, stage 8](03-vision-pipeline.md), “Learned possession”: 159 causal features (contract `pfeat-v1`), a 2D rule with no height input, mirrored training and symmetrized probabilities, strict nested fits, timing fallback and fixed gates. It needs no change to schema 0.6. The rule-only path above and stale arms below keep their old behavior.

**Plumbing.** Add `--possession-model <model-id>`, setting `StateConfig.possession_model`; None stays the default and is omitted from `to_dict()`. The default `config_key` stays **`e737054b5d`**. Supported IDs start with `lgbm-v1-nested5x4-s1` (full rows), or `-s2` / `-s4` (03's timing-only stride fallbacks, in that order). Models live in `vision/`; `prediction/possession.py` is the only bridge. Prediction consumes possession output, not the learned features or weights.
- `prediction.cv.run` retains the canonical provider labels/row index and shots table. In learned mode it supplies `run_cv` with a fold data provider instead of reusing one feature table per match. `load_data`/`load_match` receive an explicit `(outer, role, source fold/fit ID)` context, validated through `prediction/possession.py`. Assemble one context at a time (outer 0–4, then final) and run both horizons on it: the learned branch puts contexts outside horizons, 6 assemblies instead of 12. Only `test`-role loads add to `possession_stats`, and `training_rows(...)` reads the canonical provider table. `vision.possession_model.predict_match` returns a match's learned grid table; `prediction/possession.py` writes the caches below. The existing shared-table path, loop order included, remains unchanged when no model is selected.
- Let F_j be fold j's matches in the frozen `folds.json`, U all 64. In outer k, test matches use possession from a fit on `U − F_k`; a training match in F_j uses `inner-oof` possession from a fit on `U − (F_k ∪ F_j)`. Five test fits plus twenty inner fits (25 logical fits backed by 15 trainings: `outer_k_inner_j` and `outer_j_inner_k` share one model, 03), shared by both horizons. All early stopping stays inside each allowed match set. No row's possession fit sees its own match, and no model supplying outer k sees F_k's PFF possession.
- Outer k's shot-model τ fit, τ validation (`inner_split(k)`), own early stopping and refit all use k's `inner-oof` features. Only its held-out prediction uses `test`. Final τ keeps `inner_split(None)`, using `final` features for **every** match: a match in F_j reuses possession from test fit j. The eventual all-64 shot fit uses those OOF features too. No extra possession fits for final τ; the all-64 live possession model never supplies CV inputs. As 03 explains, this protects the outer test fold, but does not additionally exclude all τ-validation matches from other training matches' possession fits.
- Learned output joins by `(match_id, period, integer grid tick)`, with native frame_id checked; rule-only `swap_possession` keeps its frame_id join. Replace only the feature-input possession/flip. Provider labels, eligibility, all_estimated, returned possession/flip/ball_state, predictable rows, alarms and τ bookkeeping stay unchanged. Check exact keys/order before attaching features or writing predictions. Aggregate run statistics from outer `test` roles once, not repeated inner tables.

**Caches.** Under `data/processed/<match>/`, `<key>` is the unchanged hash function applied to the non-default config. Define `<context>` as `outer<k>_test` or `outer<k>_inner-oof<j>`; j is the match's fold (redundant, asserted on load). The `final` role reads `outer<j>_test` for a match in F_j, recorded as an alias in the run's provenance, not a separate file.
- `state_inferred_age_<key>_<context>.parquet`: grid keys/frame_id, learned possession, unchanged 2D-rule carrier/ball state/age, rule possession, p_home, fallback and fit_id. The complete column contract is in 03. It is not the old native-frame state file.
- `features_v1_held_pinf_<key>_<context>.parquet`: the ordinary v1 shot features rebuilt for that possession. Each new state/feature file has a same-stem `.json` with context and input/grid/model/output hashes; feature provenance also hashes the state it used. Missing/mismatched provenance is an error, never a reason to fall back to old caches.
- `data/vision_cache/<match>/possession_inputs_pfeat-v1.parquet` and `.json`: vision's reusable 159-column inputs and their provenance, keyed by feature contract so every stride version shares them. Prediction does not read the feature matrix.
- `data/models/possession/<model-id>/`: sealed CV `manifest.json` mapping the 25 logical IDs to 15 stored trainings, `fits/outer_<k>/` (5) and `fits/pair_<a>_<b>/` (10), each `model.txt` + `fit.json`; later `live_all64/model.txt` + its own `fit.json`. Models are prepared by `vision.possession_train`, never implicitly fitted by `prediction.cv`.
- Every learned cache records the SHA256 of its native frames/objects and grid, and a read recomputes and refuses on a mismatch. That's the real guard: vision inputs come from `data/gamestate`, and a reconvert doesn't force a resample. As an extra, `prediction/resample.py` changes to remove `features_v*.parquet`/`.json`, `graphs_v*.npz`, `state_inferred_*.parquet`/`.json` and `data/vision_cache/<m>/possession_inputs_*` on resample. It keeps models/runs; changed native/grid/training hashes make an artifact unusable until a new version is fitted. No learned cache can collide with the unsuffixed old caches.

**Run metadata.** Keep `possession: "inferred"`, `state_config` and `possession_scored` (outer-test rows of horizons[0]). Add learned-only model/scheme/stride, feature version and 2D-rule policy, sealed artifact/fold hashes, per-context fit/cache hashes, expanded fit/early-stopping assignments or their immutable manifest reference, symmetrization/threshold, fallback counts, and fit/inference/feature timings plus peak memory. Keep test, inner and final counts separate. Provider and old inferred runs acquire no learned keys; their config, cache bytes, features and stable metadata must reproduce unchanged. Wall times and run timestamps are not byte comparisons.

**Scope and tests.** The one learned arm is `--model lgbm --possession inferred --possession-model <id> --features-version 1 --ball-source held --horizons h5 h3`, after 03's disagreement gate. Reject model selection with provider possession, floor/GNN/TGNN, raw ball, v2/v3, degradation, stale-possession unknown/stale-after, non-default rule overrides, wrong/unfrozen folds, absent context or a live artifact in a CV role. Existing flags keep their semantics without model selection. Test those errors at the CLI and library boundary, nested provenance/exclusions, all τ contexts, full-key joins, separate caches/invalidation and byte-identical default/provider/old-inferred paths. 03 lists the causal, visibility, formula, mirror and streaming tests. No shot PR-AUC tuning.

### Stale possession (07 #6 follow-up, 2026-09-29)
Stage 8's possession is the team of the latest confirmed carrier, carried forward until the other team gets one. How long ago that carrier was seen says how far to trust it. `scripts/possession_staleness.py` measured it on the scored rows of all 64 matches (`Docs/reviews/possession-stale-2026-09-29.md`). **Carrier age** is the seconds since the last native frame of the same period with a non-null `ball_carrier_id` from stage 8. It's 0 on a carrier frame and null before the period's first carrier. Disagreement with PFF climbs steeply with it: 2.1% at 0 s, 4–5% up to 1 s, 7% at 1–2 s, 15% at 2–3 s, 25% at 3–5 s, 37% at 5–10 s, and 51.5% past 10 s. So the model side gets two cheap arms before stage 8's rules change.

**Carrier age as an input.** It reads only 02's `ball_carrier_id` as stage 8 fills it, from native frames at or before the row's frame, so it's causal and something vision produces live. Stage 8's state cache now holds it too (Caches below).

**Arm S: `poss_carrier_age_s` (features v3).**
- `features_version = "3"` is v1's 29 columns unchanged plus `poss_carrier_age_s`: the carrier age at the row's native frame, NaN before the period's first carrier. v1 and v2 and their caches stay byte-identical.
- It always comes from stage 8's carrier (default `StateConfig`), whichever possession the other inputs use. PFF has no carrier (02), and the feature describes the tracking, not whose ball it is. So v3 needs `gamestate_dir` and runs stage 8 in both arms.
- Runs: `--possession inferred --features-version 3` (does it fix stage 8?) and `--possession provider --features-version 3` (does the feature help anyway?).

**Arm U: unknown when stale (`--stale-possession unknown --stale-after S`).**
- On the inferred path only, stage 8's possession is set to null on every grid row whose carrier age is over **S = 10 s**. Then there's no flip, nobody is an attacker or defender, and `possession_s` is NaN, as for any null possession (Nulls above). When a carrier is confirmed again, age drops to 0 and possession comes back. `possession_s` then restarts at 0 even if it's the same team: after an unknown stretch, how long the team has had it isn't known.
- **S is fixed before any run.** It's where disagreement crosses about 50% (51.5% past 10 s, 37% at 5–10 s), which is where "unknown" stops losing to a guess. It was picked from the disagreement column only, which is stage 8 against PFF's possession on every match. The shot-rate and PR-AUC columns of the same table weren't used. This is the same mild use of all 64 matches as stage 8's own tuning, not tuning on 07 #6 results. S isn't changed after the run.
- Rows past 10 s are 11.3% of scored rows, 5.5% of positives and 41% of the disagreeing rows.
- **Not neutral on the ball.** Null possession means no rotation (sign +1), so the ball features (`ball_x`, `ball_dist`, `ball_angle`, `ball_vgoal`, about half the gain) are measured toward the +x goal. On a stale row that's close to a coin flip about direction, not "unknown". Only the player features and `possession_s` really become unknown. The arm is kept as defined: it's what a null possession does in `match_features` today, and the sensitivity test's `team_unknown` worked the same way.
- Labels, `eligible`, `all_estimated` and the scored rows stay PFF's, so these rows are still trained on and scored, just with the "unknown" inputs.
- `--stale-possession unknown` needs `--possession inferred` and a positive `--stale-after`. `--stale-after` needs `--stale-possession unknown`. Anything else is an error, and `none` (the default) is the old inferred path.
- **S+U** (both at once) is run only if S and U each help on their own.

**Result (2026-09-29, `Docs/reviews/possession-stale-2026-09-29.md`).** Neither arm helps. H = 5 PR-AUC: arm S 0.259 (−0.037 vs provider, 0/5 folds), arm U 0.255 (−0.040, 0/5), against 0.258 for inferred v1. Provider + v3 is 0.296 (−0.0006, 3/5 folds), so the feature doesn't help on its own either, and S+U isn't run. Carrier age can't turn a wrongly rotated frame around. Arm U's unknown rows rank as badly as wrong ones (PR-AUC 0.024 vs 0.159 for the provider), because the ball features still point at the +x goal, as expected above. 85% of the disagreeing rows are PFF changing possession while stage 8 hasn't followed, so the fix goes in stage 8's rules (03).

**Rule changes (2026-09-30).** Faster nearest-player rules (`carrier_min_s` 0.1, the new `team_near_s`) didn't pass the scored-row bar, so 07 #6 wasn't rerun (stage 8 review). The learned model proposed in 03 stage 8 comes next. It requires the per-outer-fold assembly and context-specific caches in "Learned possession" above; one shared feature table per match is not enough.

**Caches.**
- Stage 8's state per match: `data/processed/<match>/state_inferred_age_<key>.parquet` (`frame_id`, `possession_team`, `carrier_age_s`). The old `state_inferred_<key>.parquet` holds possession only. It's never read again, and a resample deletes both (the `state_inferred_*` glob).
- Features: v3 provider is `features_v3_<ball_source>_s8_<key>.parquet`, since it depends on stage 8's config. v3 inferred is `features_v3_<ball_source>_pinf_<key>.parquet`. Arm U adds `_unk<S>` to the inferred name (`features_v1_held_pinf_<key>_unk10.parquet`). The v1/v2 provider names and `features_v1_held_pinf_<key>` stay as they are.
- `<key>` is `config_key(StateConfig)`. A change to stage 8's rules has to change it too (a new config field for the new rule), or the old state and features would be reused (03 stage 8).

**Run metadata.** An arm U run adds `stale_possession: {"arm": "unknown", "after_s": S}` to the `run.json` config. `possession_scored` also gets `stale_unknown_share`: the share of scored rows the arm made unknown. `inferred_null_share` then counts every null the model sees, these included. v3 is recorded as `features_version`, as v2 was.

**Tests** (`tests/test_possession.py`):
- carrier age by hand: 0 while carried, growing while the ball is loose, reset at the period start
- v1 provider and v1 inferred unchanged by the new code, and v3's first 29 columns equal v1's
- past-only: objects after t change nothing at rows ≤ t, carrier age included
- arm U nulls exactly where age > S, with labels, `eligible` and scored rows equal to the provider's
- every new cache is separate from the old ones
- refused flag combinations

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
- **Primary: StatsBomb 360 open data.** 300 men's matches have 360 freeze frames: 7,589 shots, 4,507 in the training population (open play, no set-play phase, 360 frame, World Cup 2022 dropped) with 508 goals (counted 2026-10-01, `Docs/reviews/xg-pgoal-2026-10-01.md`). Freeze frames only include players visible on the broadcast, like our vision output.
- **Wyscout (defcon CSV):** location-only xG (distance + angle). A baseline and sanity check, not the production model.
- Convert StatsBomb coordinates (120 × 80 yards, origin top-left) to 02's meters.
- **Leakage:** PFF World Cup 2022 is in every CV fold, so StatsBomb's World Cup 2022 (the same 64 matches) is always dropped from xG training.
- **Check calibration on our own data:** apply the xG model at the shot frame to PFF shots (1,154 open-play shots outside set-play phases with 126 goals on all 64, counted 2026-10-01) and SkillCorner shots (61 goals), and compare. Report it; don't retrain on it.

### Combining with P(shot)
xG at the carrier's current position is an approximation. The shot usually happens later, from somewhere closer to goal. Build in order:
1. **v1: carrier position.** P(goal) = P(shot) × xG(ball carrier at t). Tends to underestimate P(goal) early in an attack, since the carrier is still far from goal. Measure how much by comparing to xG at the actual shot frame.
2. **v2: shooter-weighted.** If the node-level "who will shoot" head exists, P(goal) = Σ_i P(player i shoots) × xG(player i at t). Still uses positions at t, but covers runners who aren't on the ball.
- A direct P(goal within H) model is out: even PFF + SkillCorner give ~200 goals with tracking, far too few to train it.

### xG v1 build (spec 2026-10-01)
Everything below fixes how xG is trained and applied so the StatsBomb side and the game-state side can't drift apart. The Wyscout baseline and SkillCorner check wait for those sources.

**Inputs are the shot model's own columns.** xG reads eight columns that `prediction/features.py` already computes on every grid row: `ball_dist`, `ball_angle`, `lane_defenders`, `gk_in_lane`, `gk_off_line`, `gk_ball_dist`, `press_dist` and `def_within_5m`. They are already causal, VISIBLE-only and in the attacking frame. Defenders are the team not in possession, the keeper is the defending `goalkeeper`, and the location is the held ball. No new feature code runs at inference.
- **Why the ball, not a carrier.** PFF has no `ball_carrier_id`, and stage 8 names a carrier on only about a quarter of alive frames. The shot model's own carrier is the visible attacker nearest the ball, so the ball is the shooter's position whenever someone is on it. StatsBomb's shot `location` is the shooter's position at the strike, so both sides are measured at the ball. A row without a held ball has no xG and no P(goal).

**The training adapter (StatsBomb 360 → the same columns).** `prediction/xg.py` turns each StatsBomb shot with a 360 frame into one row of those eight columns, using `features.py`'s own `goal_distance`, `goal_angle`, `in_lane`, `GOAL_X` and `POST_Y`. No second geometry.
- **Coordinates, anchored at the goal.** StatsBomb's 120 × 80 yards always attack +x, with origin at the top-left, so +y is the shooter's left. That makes `X = 52.5 − (120 − x) × 0.9144` and `Y = (40 − y) × 0.9144`, in meters from the goal, not stretched to 105 × 68. The 8-yard goal comes out 7.32 m wide, as in `features.py`, and the penalty spot 11 m out. A stretch would get both wrong.
- **Players.** 360 `freeze_frame` entries with `teammate = False` are defenders, `keeper = True` among them is the keeper, and `actor` is the shooter. Teammates count for nothing here, as in the eight columns. The frame already holds only on-camera players, like VISIBLE. `visible_area` is never a feature, since `view_polygon` is null on PFF (03).
- **Parity test.** A hand-built frame goes through `features.py` on a synthetic game state and through the adapter as the equivalent StatsBomb record; the eight columns match to float tolerance. A test converts the posts and the penalty spot.

**Same shot population as the labels.** P(shot) counts open-play shots outside set-play phases (02, Labels), so xG trains on the closest StatsBomb match:
- `shot.type = Open Play` (drops penalties, direct free kicks and corners, kick-offs)
- not a set-play phase: no corner, and no free kick taken at least 17.5 m past halfway in the attacking direction, by the shooter's team in the same period within the 10 s before the shot. This is 02's PFF proxy, read from StatsBomb's own pass events (`pass.type` Corner / Free Kick, their location and timestamp).
- men's competitions with 360 data, minus FIFA World Cup 2022 (competition 43, season 106): the same 64 matches as PFF.

**Model.** LightGBM binary on the eight columns with plain log loss (Class imbalance). Missing values stay NaN; a keeper off camera is NaN `gk_off_line`. Parameters, fixed and not searched: learning rate 0.05, 7 leaves, at least 100 shots per leaf, feature and bagging fraction 0.8 every round, L2 1, seed 20260927, `deterministic`. At most 1,000 rounds, early stopping after 50 on held-out matches.
- **Baseline.** Logistic regression on `ball_dist` and `ball_angle` (the location-only xG).
- **CV.** 5 folds grouped by match: `numpy.random.default_rng(20260927)` permutes the sorted match IDs and deals them round-robin. Inside each training split the shot model's own `LGBMModel` holds out 15% of matches for early stopping and refits on the whole split; the baseline is `prediction.floor.LogisticFloor`. Both read the shot table under the shot model's column names, so no second training loop exists. Report pooled and per-fold log loss and Brier, calibration in 10 quantile bins, and the same for the baseline. `statsbomb_xg` is shown as a reference column, never a target or input: it uses body part.
- **Choice, fixed now.** The production xG is LightGBM if its pooled out-of-fold log loss beats the baseline's and it wins on at least 4 of 5 folds; otherwise the baseline. Nothing else is tuned.
- **Final model.** Fit the chosen model on every training shot the same way as inside a fold (`LGBMModel.fit`: its own 15% match hold-out for early stopping, then a refit on all of them). Store it under `data/models/xg/xg-v1/` as `model.txt` (or coefficients), plus a `manifest.json`: shot and match IDs, the population filter, the parameters, CV results, the source file hashes and the git commit.

**Checks on held-out data (report only, never retrain).**
1. **World Cup 2022 on StatsBomb 360.** The same tournament as PFF, never trained on. Same population filter. Log loss, Brier, calibration and goals vs summed xG, next to `statsbomb_xg` on the same shots.
2. **PFF on game state.** For each PFF open-play shot outside set-play phases, read the eight columns from the provider-possession `features_v1_held` cache at the latest grid row at or before the shot's frame, and apply the final model. Report goals vs summed xG, calibration in 5 bins (too few goals for 10), and how often the row has no held ball. PFF's ball at the shot frame is a different measurement from StatsBomb's location, which is the point of this check.

**P(goal) v1.** `P(goal within H)(t) = P(shot within H)(t) × xG(t)` on every scored row with a held ball. P(shot) is a run's out-of-fold prediction, so P(goal) is out of fold too. xG never saw World Cup 2022.
- `python -m prediction.pgoal <run>` writes `pgoal.parquet` (the run's rows plus `xg` and `p_goal`) and `pgoal.md` into the run folder.
- Scored against 05's `label_goal_h5` / `label_goal_h3`: PR-AUC, calibration (pooled bins, then summed P(goal) vs goals), and next to the same with xG alone and P(shot) alone. Descriptive: there's no bar to pass yet.
- **The v1 underestimate.** For every open-play shot, xG(t) at 5, 2 and 1 s before it vs xG at the shot's own grid row: the median ratio, and how it splits by goal vs no goal. That's how much "xG at the ball now" undershoots the shot that follows, which v2 is for.
- The first run is on `lgbm-held-2026-09-27` (provider possession, the baseline everything else compares to).

## Class imbalance
- **No class weights and no focal loss** (decided 2026-09-28; this replaces "weighted or focal loss"). Plain log loss keeps p calibrated, which the alarms and P(goal) = P(shot) × xG both need. LightGBM showed it works at a ~2.5% base rate: the top decile predicts 0.186 and sees 0.185. Reweighting inflates p, and undoing that is another calibration step to get right. The GNNs use the same loss.
- Evaluate with PR-AUC, not accuracy.

## Acceptance criteria
- LightGBM baseline beats the distance + angle floor.
- Temporal GNN beats the LightGBM baseline on PR-AUC in grouped cross-validation (pooled out-of-fold, and on most folds), confirmed on the IDSSE external test set (see 07).
- Calibration error acceptable after (optional) temperature scaling.

## Open questions
- Predict "which player will shoot" as a node-level head too?
- Add uncertainty (MC dropout / Bayesian head, as in Goka et al. 2023)?
