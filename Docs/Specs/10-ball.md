# 10 — Ball

Truth, tracking and detection for the ball (03 stage 5), from `Docs/reviews/ball-research-2026-10-02.md`. 03 stage 5 and 07 "Ball labels" / "Ball score" point here. Same rules as everywhere:
- vision output is causal;
- pixels stay inside `vision/` (the detections-cache exception, 03 Diagnostics);
- every bench score is a replay;
- no schema change.

## Why (short)
- **PFF's ball can't be the truth.** Projected into the image it's within 25 px of the real ball on 84% of VISIBLE frames, with ~1 m biases lasting 5–15 s. It's ESTIMATED, 44 px off, on the 31–39% of live frames with aerial or lost balls. It can't say a ball is hidden.
- **It's good for:** pre-placing labels, flagging bad labels, a loose automatic score, and auto-labels where a detection agrees with it.
- **The ball model is the limit.** A candidate is near the ball on ~70% of PFF-VISIBLE frames. A better picker adds 1–3 pt; a bigger input, under 1 pt.

## 1. Truth

### 1a. Verified labels (the truth for targets)
07 "Ball labels" as now: every 5th source frame outside the marks, `[x, y]` / `"none"` / `"unsure"`, keyed by source frame index. Same file and format. What changes is how they're made.

### 1b. PFF reference (`vision/ball_truth.py`, bench only)
For clips with a `match_id`. Built once per clip, cached in `data/vision_bench/ball_truth/<clip_id>.parquet` (gitignored; recomputed when the video hash, manifest sync or code version changes).
- **Frames:** the 07 label frames (every 5th source frame, outside the marks). Decoded sequentially from the video like `scripts/ball_click.py`, never by seeking.
- **Camera:** PnLCalib run on each of those frames (the run's weights and thresholds, 03's per-call checks). Not the run's held camera: between calls it adds error (median 19 vs 9 px against clicks on vb02). ~0.4 s a frame on the 2060, ~2.5 min a minute of clip.
- **PFF ball:** the two PFF ball rows around `video_s + sync offset` (07 sync), linearly interpolated. No extra shift: measured best within ±1 frame on all three clips (review, Timing). Both rows must exist and be one PFF frame apart, else `no_pff`.
- **Projection:** 02 → TV (`to_02`) → PnLCalib world `(x, −y, −z)` → `K R [I | −C]`, at **z + 0.11 m** (the ball's center; review, Height).
- **Columns:**
  - `src`, `camera_ok`, the camera (as `camera.parquet`);
  - `pff_visible`, `pff_x`, `pff_y`, `pff_z`, `pff_speed`;
  - `u`, `v` (projection in source pixels), `diam_px` (0.22 m at that depth);
  - `kind`:
    - `pff`: VISIBLE, camera ok, in front of the camera, inside the image;
    - `estimated`: either PFF row ESTIMATED;
    - `off_image`;
    - `no_camera`;
    - `no_pff`.
- **Never an input to vision.** It reads PFF's ESTIMATED rows, and future frames through interpolation, which is fine for a scorer and forbidden in the pipeline.

### 1c. Label tool: pre-placement and flags (`scripts/ball_click.py`)
- **`--assist`:** on each frame, a suggestion ring at the ball candidate nearest the PFF projection, within 40 px on `pff` frames and 60 px on `estimated` ones, at any confidence.
  - Candidates come from the clip's `balls.parquet`, or from `--candidates FILE` (an offline high-recall run of a newer model, same columns).
  - PFF's projection is drawn too: solid when VISIBLE, dashed when ESTIMATED.
  - **Enter** accepts the suggestion. A click, space and `u` work as now.
  - A frame with no suggestion needs a click or a key, as now.
  - Labels are what the person chose. The suggestion is never saved unless accepted.
- **`--flag`:** the label frames where a saved `[x, y]` is more than 30 px from a `pff` projection, or a `"none"` falls on a `pff` frame with a candidate within 25 px of it. Opened like `--frames`, for a second look. On vb02's first pass this lists the speck clicks (240–280), the trailing clicks (580–610), 665, 1670, 1685, 1750, plus PFF's bias stretches (710–915, 1205–1215), where the click is right. So a flag is a prompt, never an automatic rejection.
- **Auto-accept (`--assist`, on by default):** a `pff` frame is labeled without being shown when its suggestion:
  - has confidence ≥ 0.5;
  - is within 15 px of the projection;
  - has no other candidate within 40 px of the projection.

  Saved as `[x, y]` at the candidate's box center, listed in the label file's per-clip `auto` list (source frame indices) so it can be told apart and redone.
  - Measured on vb02's good clicks: 82 of 147 `pff` frames qualify, and 81 are within 15 px of the click (median 3.8 px; the other one is 18 px).
  - **Spot check:** the tool shows a random 10% of auto-accepted frames for a normal decision, and reports the share changed. Above 3%, auto-accept is off for that clip.
- **Expected effort, today's candidates:**
  - frames left for a person: vb01 236 of 290, vb02 187 of 280 (already labeled; only the flags need a look), vb03 126 of 360;
  - on those, the suggestion is right on ~53% of `pff` frames and 26% of `estimated` ones (one key press);
  - it all improves with the detector, so relabel after step 5 costs less.

### 1d. Scores (07)
- **Ball score:** on verified labels, as in 07 now (R = 15 px, recall, precision, miss buckets). This is the score the targets apply to.
- **PFF score** (new, printed under it, never against the targets):
  - on `pff` frames, a hit is a vision ball row with pitch x/y within **40 px** of the projection;
  - recall is hits over `pff` frames, precision is hits over vision rows on `pff` frames;
  - also printed: rows on `estimated` frames (no hit/miss), the ceiling (any candidate within 40 px), and recall at 25 px;
  - its job is paired comparisons (sweeps, A vs B) and clips without labels;
  - on vb02, with a camera solved on each frame (as specced), its verdict agreed with the clicks' on 92.5% of frames at 40 px, recall 3.4 pt lower (76.2 vs 79.6%). At 25 px: 87.8%, 10.9 pt lower.
- **Agreement check, per labeled clip:**
  - the share of good `[x, y]` labels on `pff` frames within 25 px of the projection (vb02 84.4%);
  - verdict agreement at 40 px.
  - A clip far below that is a sync or camera problem, and the PFF score is withheld for it.

## 2. Stage 5 tracker (`vision/ball.py`, `BallTrack`)
One class, as now, shared by pipeline and replay. All new fields go in `VisionConfig`. Defaults for old runs reproduce today's rule, set in `replay.run_config` like `pnl_blind_kp`.

### 2a. Candidates
- The ball model's detections at conf ≥ `track_min_conf` (0.1), as `balls.parquet` stores them.
- Dropped before picking:
  - **no pitch position:** off the pitch by more than **`ball_cand_margin_m`** (2 m; today a candidate may sit 10 m out, `max_off_pitch_m`). Board logos and the net project several meters out. **Caution:** a ball in the air projects through the ground homography to a point behind it. A ball 1 m up near the far touchline lands a few meters out, and one 3 m up lands much further. So this filter, the size filter and the pitch-space gate can drop a real airborne ball. Measured on the 73 good vb02 clicks that fall on PFF-ESTIMATED (mostly aerial) frames: no loss (recall 89.0% for `"max"`, 91.8% for gate + size + margin). But vb02's highest balls are in its guessed ranges, so they aren't in that sample.
  - **wrong size:** box width outside `[ball_size_lo × w_exp, ball_size_hi × w_exp + 6 px]`, with `w_exp` the width of 0.22 m at that pitch point through the frame's homography. **0.5 and 2.0.** Catches the bottom-of-frame specks (too small for the near touchline) and head- or boot-sized boxes. The +6 px allows motion blur.
- Off by default for old runs (`ball_cand_margin_m = max_off_pitch_m`, `ball_size_lo = 0`, `ball_size_hi = inf`).

### 2b. Association: `ball_picker = "gate"` (old runs: `"max"`)
State: the last detection's time, pitch position, pitch velocity, box and confidence (as now).
- **With a track:**
  - prediction `p = xy + v · dt`, gate radius `ball_gate_m + ball_gate_mps · dt` (**3 m + 25 m/s**);
  - candidates inside the gate at conf ≥ **`ball_gate_conf` (0.15)** compete on `conf − 0.02 · distance_m`;
  - outside the gate, only a candidate at conf ≥ **`ball_reacq_conf` (0.5)** restarts the track (full-frame reacquisition: a bad track can't lock out the ball);
  - otherwise no detection: extrapolate as now.
- **Without a track:** the most confident candidate ≥ `min_det_conf` (0.3).
- **Velocity:** from the previous detection, as now. Above 40 m/s (faster than any kick; a jump between two objects), keep the position, set the velocity to zero.
- **Why these values:** replay on vb01–vb03 (review, R2 table). Gate 2–5 m and 20–35 m/s give the same result; reacquire 0.4–0.6 is within 0.6 pt. Fixed, not swept further.

### 2c. Resets (D1, W2)
Each clears the track:
- a new match segment (exists);
- invalid geometry, or a detection without a pitch position (exists);
- history older than `ball_max_gap_s` (exists);
- a period change: by construction, since one run is one period (`VisionConfig.period`) and a new run starts a new `BallTrack`;
- a confirmed cut, once W3 provides one (03 Open questions).

### 2d. Gaps
- Extrapolate at the last velocity for at most `ball_max_gap_s`, `interpolated = True`. **0.5 s since 2026-10-05** (was 1 s): with the gate and the speed rule, 1 s of extrapolation cost 2.1 pt of precision on verified aerial (PFF-ESTIMATED) frames; at 0.5 s every clip gains on both scores (ball fix review, Phase C). Old runs keep their 1 s from `run.json`.
- **Carrier hold** (`ball_carrier_hold`, default off), after stage 8 is wired into the pipeline:
  - if the 2D rule had a carrier on the previous frame, a missing ball is placed at that carrier's position instead of extrapolated, `interpolated = True`, for at most `ball_max_gap_s`;
  - the carrier comes from the 2D rule that reads only `interpolated = False` balls (03 stage 8, Time and visibility contract), so a held ball can't keep its own carrier;
  - the historical rule (`vision/state.py` `infer`) reads every visible ball. On vision output it must skip `interpolated = True` balls before this is turned on (a `StateConfig` change with its own key, 05 Caches);
  - scored on the PFF score's `pff` frames that fall in a fill.

### 2e. Cache and replay
- `balls.parquet` is unchanged: replay already has every candidate, the homographies and the view. The size filter reads the frame's homography, as replay has it.
- The carrier hold needs stage 8's 2D rule over the replayed people. Replay runs it.
- **Exactness:** the run config's replay reproduces the run's ball rows (the bench's replay check). An old run (no new fields) replays as `"max"` with no filters and matches its cache.

### 2f. Tests
- **The fake-stage pipeline test**, extended:
  - a distractor with a higher score outside the gate is ignored while the track is fresh;
  - a candidate ≥ 0.5 outside the gate restarts the track;
  - a fast kick (30 m/s) stays in the gate;
  - a speck smaller than half the expected width is dropped;
  - a logo 5 m off the pitch is dropped;
  - a ball on the touchline is kept;
  - reset on segment, period and invalid geometry;
  - no velocity across a reset.
- **Causal:** changing candidates after t changes no row at or before t.
- **Replay:** the run config reproduces the cache; an old `run.json` replays as `"max"`.

### 2g. Ball in the air (`ball_air_ratio`, added 2026-10-05)

- **Problem:** a ball in the air is projected through the ground homography to a point ~14 m too far from the camera (median, p90 24 m; ball fix review, Phase D measurements). The goal model then reads a confident, fresh, wrong ball; on the demo the meter falls to 0.3% where PFF has 2.2–2.6%.
- **No height is written.** Height from the box size (depth = fx × 0.22 / (w / 1.38)) is right on average but 6 m off in the plane at the median (p90 17 m) and 4× worse than the ground plane on the ground; `z` stays null.
- **Airborne flag:** a detection with a pitch position is airborne when its box width is at least `ball_air_ratio` × `expected_width` at its ground projection (the 2a function). The rule works at **1.5**, but the default is **`inf`** (off) for new runs until §4 step 5's fine-tune re-derives the ratio on its boxes (user's call, 2026-10-06): at 1.5, vb01 and vb02 have no usable ball on 23% and 29% of ball frames, so the meter blanks there. The demo runs it as the `air15` variant (`--set ball_air_ratio=1.5`). Old runs replay as `inf` (never airborne), like the 2a–2b fields. Measured as a classifier of PFF z > 1 m: recall 95%, precision 33%, 32% of ground detections flagged (vb02 most: its boxes run 1.5× the ball against 1.32–1.38 elsewhere).
- **Extrapolated rows** take the airborne flag of the detection they extrapolate. A detection that isn't airborne clears it. Without geometry there's no ratio, so no flag.
- **Output:** an airborne row is written `visible = False, interpolated = True` with its ground-projected x/y, as 02 writes PFF's ESTIMATED ball: its position is a guess.
  - The goal model's held ball (05) and stage 8 read only visible balls. So they hold the last ground ball (≤ `BALL_HOLD_S`, 1 s), which is the input the model was trained on for PFF's aerial ball, and give no carrier from the wrong spot. A hold longer than 1 s leaves no ball, so there's no xG or P(goal) there, as with PFF.
  - A false flag on a ground ball costs one held row.
- **Association is unchanged.** The gate still follows the ground projection, which stays continuous in the air.
- **Scores:** 07's ball score and PFF score read the row's box and `has_xy`, so they don't change. The bench prints the share of ball rows written airborne.
- **Tests** (2f):
  - a box ≥ the ratio × expected width is airborne, one under it isn't;
  - extrapolation inherits the flag, and a ground detection clears it;
  - no flag without geometry;
  - the writer gives an airborne row `visible = False, interpolated = True` with x/y;
  - an old `run.json` replays with no airborne rows;
  - the fake-stage pipeline test has one airborne frame.
- **Done when:** on the demo variant, the aerial rows (2613–2622, 2667–2688) no longer sit at ~0.3%; ball score and PFF score are unchanged on every clip; the replay check passes on all four caches.

## 3. Detector
The ball model's recall is the ceiling (~70% of PFF-VISIBLE frames, review). Input size doesn't move it (1280 → 1920: +0.5 pt, false candidates ×2.6).

### 3a. Auto-labels (`scripts/ball_autolabel.py`)
- **Footage:** the six 2022 matches with PFF that aren't on the bench: 3857 KOR–POR, 10507 BRA–KOR, 3816 KSA–ARG (unstitched), 10510 BRA–CRO, 10508 MAR–ESP, 10514 ARG–CRO (stitched: per piece). 5 min each.
  - **Never 10517, 10511, 3854** (bench matches), nor any future bench match. The script refuses manifest matches.
- **Sync, by the ball:**
  - start from the scoreboard (±1 s), or for a stitched piece its range in `stitched_segments.csv`;
  - then sweep the offset ±1.5 s in PFF-frame steps. The pick maximizes the share of PFF-VISIBLE moving-ball frames (> 5 m/s) with a ball-model candidate ≥ 0.5 within 15 px of the projection;
  - on vb01–vb03 this peaks at the manifest's offset to within one frame (review, Timing). The script checks it there first and refuses to run if it doesn't.
- **Frames:** every 3rd frame of live wide play: PnLCalib accepts a camera; frames whose PFF players are all ESTIMATED (cutaways) are skipped.
- **Per frame:**
  - PnLCalib camera;
  - PFF ball projection (1b);
  - the current ball model at conf ≥ 0.05.
- **Labels:**
  - **positive:** PFF VISIBLE, the candidate nearest the projection is within 25 px, and **no other candidate is within 40 px**. Box = that candidate's box. PFF drifts up to ~1 m for seconds (review). In those stretches the nearest candidate can be a boot while the ball sits 35 px away. Projections are first corrected by the median offset of confident (≥ 0.5) candidates within 40 px over ±1 s (excluding ±0.2 s around the frame), which took vb02's agreement from 84.4% to 88.4% within 25 px;
  - **hard negatives:** candidates ≥ 0.3 more than 60 px from a VISIBLE projection. Only on frames that also have a positive, so the frame's one ball is known;
  - **left out:** frames with an ESTIMATED ball, frames with a VISIBLE ball and no candidate within 25 px (a missed ball must not become background), frames with a second ball-like candidate ≥ 0.5 within 3 m of the touchline (spare balls).
  - **missed balls:** a random 300 of the frames with a VISIBLE ball and no candidate go through `ball_click --assist`, and only verified boxes (click → box of `diam_px`) join. These teach the model what it misses now.
- **Output:** YOLO format under `data/ball_train/` (gitignored), with a manifest: matches, offsets, counts, git commit.

### 3b. Fine-tune
- **Start:** the current `football-ball-detection.pt` (YOLOv8x), imgsz 1280.
- **Mix:** the auto-labels + the Roboflow set it was trained on (v2), so it doesn't forget other broadcasts.
- **Validation for early stopping:** a held-out match from the six, never the bench.
- **Hardware:** the 2060 at batch 2–4 (09), else Colab.
- **Weights:** new file name, hashed into `run.json` `weights` as now.
- **Live:** a YOLOv8s/m at 1280 from the same data. Measured on the same bench before it replaces anything live.

### 3c. Done when
On fresh bench runs with the new weights:
- the verified-label ball score improves on recall and precision on every clip, against the current weights on the same labels;
- the PFF-score ceiling rises;
- false candidates per frame (≥ 0.3, > 60 px from a `pff` projection) don't rise;
- people and geometry scores are unchanged.

A model that doesn't beat the current one is a recorded result (detection review W8), not a reason to move the benchmark.

## 4. Build order
Each step is a commit series with its own bench check.

1. **Done 2026-10-05** (`Docs/reviews/ball-fix-2026-10-05.md`, Phase A). **PFF reference + PFF score** (`vision/ball_truth.py`, `vision.bench` prints the PFF score and the agreement check).
   - Check: on vb02, agreement within 25 px ≥ 80% (review: 84.4%); PFF-score recall at 40 px within 4 pt of the click recall (review: 76.2 vs 79.6%).
   - Done when the bench prints both scores for all three clips and the replay check still passes.
2. **Done 2026-10-05** (`Docs/reviews/ball-fix-2026-10-05.md`, Phase B). **Label tool** (`--assist`, `--flag`).
   - The user re-reviews vb02's flagged frames (`--flag`), then labels vb01 and vb03 with `--assist`. About 290 + 360 frames, around half one key press.
   - Done when all three clips have verified labels with no unresolved flags, and 07's ball score runs on all three.
3. **Done 2026-10-05** (ball fix review, Phase C), with `ball_max_gap_s` 0.5 s at the user's choice: as specced the config lost 2.1 pt aerial precision. **Tracker** (2a–2c, 2e–2f; carrier hold waits for stage 8 in the pipeline).
   - Check by replay, both scores, all three clips. The gate + filters config must:
     - beat `"max"` on pooled recall and precision on verified labels;
     - not lose more than 1 pt on any clip;
     - **not lose on the verified labels of PFF-ESTIMATED frames** (the aerial ones the PFF score can't see).
   - If the filters lose there, apply them only when no candidate ≥ `ball_gate_conf` is inside the gate.
   - Done when it's the default for new runs and old runs replay exactly.
4. **Auto-label sync check** on vb01–vb03 (3a sync), then auto-labels for the six matches.
   - Done when the manifest lists six matches' offsets, each with its peak share, and the counts of positives, negatives and left-out frames.
5. **Missed-ball sample** (300 frames, `--assist`), then the **fine-tune** (3b).
   - Done per 3c.
6. **Carrier hold** (2d), after stage 8 runs inside `VisionPipeline`.
   - Check by replay on the PFF score's filled frames: median distance to the projection lower than extrapolation's.
3b. **Ball in the air** (2g; ball fix plan Phase D, design approved by the user 2026-10-05).
   - Done per 2g. Built; the default stays `inf` until step 5 re-derives `ball_air_ratio` on the fine-tuned boxes, then it's switched on if the bench clips keep their ball.
7. **Live ball model** (smaller YOLO, TensorRT fp16, rate). Measured against 09's budget.
