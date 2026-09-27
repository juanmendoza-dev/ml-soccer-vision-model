# 03 — Vision Pipeline

## Goal
Turn broadcast video into game state (02) for each frame.

## Inputs / Outputs
- **In:** video file (mp4), optional team rosters.
- **Out:** `match.parquet`, `objects.parquet`, `frames.parquet`, `players.parquet` in schema v0.4.

## Stages
0. **View gate** — decides per frame whether it's a usable match view, before any GPU work. Broadcasts cut to ads, studio, crowd and close-ups; detection on those wastes time and produces junk.
   - **Grass share:** downscale to 160×90, share of pixels in a green HSV range. Below `min_grass` (~0.3) → `other`. ~1 ms on CPU, every frame.
   - **Keypoints:** when stage 4 runs, fewer than `min_keypoints` (~4) pitch keypoints found → `other`, even if green (close-ups on grass, green ads).
     The last count holds until stage 4 runs again, so failures on keypoint frames add up across the frames between them. While `other`, a green frame gets a keypoint probe every `keypoints_every` frames (feeds the gate only, no homography), so a view comes back on grass and lines, not grass alone.
   - **Hysteresis:** switch to `other` after `off_after_s` (~0.5 s) of failing frames; back to `match` after `on_after_s` (~1 s) of passing frames. Causal: uses frames `<= t` only.
   - **While `other`:** stages 1–7 don't run. Game state still gets a `frames` row with `ball_state`, `possession_team`, `ball_carrier_id` and `view_polygon` null (02: can't decide → null), and no `objects` rows.
   - **Back to `match`:** reset the tracker (IDs don't survive a cut). Keep the team assignment fit; refit if the break was longer than `refit_after_s` (~120 s, e.g. half-time). A refit matches its clusters to the old ones by kit color, so `home_cluster` keeps pointing at the same kit.
   - **Replays pass the gate** (grass, valid keypoints) and get tracked as if live. See open questions.
   - Thresholds are config values, tuned on labeled broadcast clips.
1. **Detection** — YOLOv8 fine-tuned on players, goalkeepers, referees, ball. Start from Roboflow's pretrained soccer weights.
2. **Tracking** — ByteTrack (via `supervision`) for stable `object_id`s. People down to `track_min_conf` (~0.1) go to the tracker, whose second pass keeps existing tracks alive on weak detections; the ball and kit-color samples need `min_det_conf` (new tracks need ~0.35, inside supervision). A lost track is dropped after `lost_track_s` (~1 s) of wall time, whatever the frame skip.
3. **Team assignment** — SigLIP crop embeddings → UMAP → KMeans(k=2), as in roboflow/sports. Goalkeepers assigned by nearest team centroid.
4. **Pitch homography** — pitch keypoint model → per-frame homography → pixel to meters. Smooth over time to reduce jitter.
5. **Ball tracking** — dedicated detector at higher input resolution; interpolate short gaps (< 1 s) and mark `interpolated=True`.
6. **Jersey OCR** — read numbers over multiple frames, vote per track, link to roster → `player_id`. Borrow approach from sn-gamestate.
7. **Velocities** — backward differences on trailing-smoothed positions (causal, frames `<= t` only; see 02 `vx, vy`). A centered window would leak future frames into the features.
8. **Game state inference** — fills `possession_team`, `ball_carrier_id`, `ball_state` and `view_polygon` (02). Works on pitch coordinates only, not pixels, so the same code runs on dataset tracking for testing.
   - **Ball carrier:** the player nearest the ball, within ~1.5 m, with ball height < ~1 m where known, for at least ~3 consecutive frames at 10 Hz. Otherwise null (loose ball, pass in flight).
   - **Possession team:** the carrier's team. It persists through passes and loose balls until a player from the other team becomes carrier. Null at the start of a period until someone first controls the ball.
   - **Ball state:** dead when the ball is past a touchline or goal line; dead while the ball is stationary at a restart spot (corner arc, penalty spot, touchline) with players set up; null when the ball hasn't been seen for > 2 s or there is no valid homography (replays, close-ups); otherwise alive.
   - **View polygon:** the four image corners projected through the frame's homography.
   - All thresholds are config values, tuned on dataset tracking (see acceptance criteria).

## Requirements
- Runs on M1 with `--device mps` for short clips; RTX 2060 for full matches.
- Configurable frame skip for detection (tracker fills gaps).
- Stage 0 runs on every frame, including skipped ones, so a cut is caught within `off_after_s`.
- Every stage can cache its output so later stages rerun without redoing detection.

## Streaming API
One stateful object, one frame at a time. Offline is a loop over it, so live (soccer-live-overlay) and offline share code, and every stage is causal by construction.

    pipe = VisionPipeline(config, stages)
    for frame_id, t, image in video:
        vf = pipe.step(frame_id, t, image)      # VisionFrame
        writer.add(vf)                           # game state (02) + caches
    writer.close()

- `VisionFrame`: `frame_id`, `t`, `view` (stage 0), `homography_ok`, `objects` (object_id, class, team, pitch x/y or null, confidence, `tracked_only`, display box), `ball` (same, or null), `view_polygon`.
- Stages are injected (detector, tracker, team assigner, keypoint model), so tests run with fakes and live mode can swap in smaller models.
- Live forms of stages that are batch-style in roboflow/sports:
  - **Teams:** fit on a warmup window (the first `team_warmup_s` of `match` frames), then assign. `team = null` until fitted.
  - **Ball gaps:** extrapolated forward from the last velocity, `interpolated=True`, never filled from later frames.
  - **Homography:** smoothed over a trailing window only.

## Pitch template
The keypoint model is roboflow/sports' `football-pitch-detection` (32 keypoints). Its template (`SoccerPitchConfiguration`) is 120 × 70 m in centimetres, origin at a corner, y growing toward the near touchline, with a 20.15 m deep penalty box. We keep its **keypoint order** (that's what the weights predict) but put each landmark at its **real** position in 02's frame: meters, center origin, 105 × 68, penalty area 16.5 × 40.32, goal area 5.5 × 18.32, penalty spot 11, center circle 9.15. The homography then outputs 02 coordinates directly, nothing gets rescaled.
- Keypoint 1 (roboflow `(0, 0)`) is the far-left corner as seen on TV → `(-52.5, 34)` in the TV frame (+x right, +y toward the far touchline, as in 02).
- Keypoints 9 / 22 (penalty spots) → `(∓41.5, 0)`; 31 / 32 (center circle on the halfway line) → `(∓9.15, 0)`.
- The corner assignment is read from roboflow's radar drawing (y down = toward the camera). Check it on the first real clip: the center spot and a penalty spot should land where they are on screen.
- Keypoints below `min_keypoint_conf` are dropped; at least 4 are needed for a homography (stage 0 counts them).

## Teams and direction
Clustering gives cluster 0 / 1, and the homography gives the TV frame. 02 needs `team` as home/away and +x toward the goal home attacks in period 1. Two config inputs:
- `home_cluster`: which cluster is home (picked from sample crops at warmup, or a flag). Until set, `team = null`. Never guessed.
- `home_attacks_tv_right_p1`: if false, positions are rotated 180° (x → -x, y → -y) for the whole match, so +x is always home's period-1 attack. `home_attacks_positive_x` is true in odd periods, false in even ones.
- Later: infer direction from which end each team's goalkeeper stands at during warmup.
- `period` and the period start come from config too (live: set at kickoff and half-time).

## Display output (pixel exception)
Pixels never leave `vision/` as data, but the live overlay draws rings on the video, so it needs screen positions. The one exception: `VisionFrame` carries each object's box as fractions (0–1) of the frame, for **display only**. Same status as the detections cache below, which 08's debug mode draws from. Game state, prediction and evaluation never read it.

## Diagnostics
Debugging works from cached data after the run, not from extra logging during it. Nothing here adds work to the inference loop beyond writing the cache.

### Detections cache
`data/vision_cache/<match_id>/detections.parquet`, one row per tracked object per frame. Vision-internal: pixels are allowed here and never leave `vision/`.

| Column | Type | Notes |
|---|---|---|
| match_id, frame_id | | Same frame_ids as game state |
| object_id | str | Same as `objects.object_id` |
| class | enum | player, goalkeeper, referee, ball (detector class, before team assignment) |
| team_cluster | int/null | Stage 3 kit cluster (0 / 1) before the home/away mapping; null until teams are fitted, and for referees and the ball |
| x1, y1, x2, y2 | float | Box in pixels |
| det_confidence | float | Detector score; null for tracker-filled frames |
| tracked_only | bool | True if the tracker filled this frame, no detection |
| pitch_x, pitch_y | float/null | Box anchor through the homography; null without a valid homography |
| homography_ok | bool | Per frame (repeated on each row) |
| homography_err_m | float/null | Mean keypoint reprojection error in meters, per frame |

`view.parquet` next to it, one row per frame (stage 0), so gate thresholds can be tuned after a run:

| Column | Type | Notes |
|---|---|---|
| match_id, frame_id | | |
| view | enum | match, other (after hysteresis) |
| grass_share | float | 0–1 |
| keypoints_found | int/null | null on frames where stage 4 didn't run |

Plus `run.json` next to it: config, git commit, model weights hash, video file hash, per-stage wall time, and the 02 validator errors (`vision.run` exits nonzero if there are any; a clip with no match view fails). With the video, this is enough to redraw any moment of the run.

- Anomaly checks (ID switches, ball gaps, homography jumps, ...) are scripts over this cache, written when a real problem shows up. Not part of the pipeline.
- The overlay renderer's debug mode (08) draws boxes, IDs and teams for a frame range from this cache.

## Acceptance criteria
- Output passes the schema validator in `tests/`.
- On SoccerNet-GSR validation clips: positions within ~2 m for most visible players (measure with sn-trackeval GS-HOTA as a secondary metric).
- Team assignment correct on > 95% of player-frames on sample clips.
- Stage 8, run on PFF, SkillCorner and IDSSE tracking and compared to provider values: possession team matches on ≥ 90% of frames where the provider has a value; frames the provider marks dead are labeled dead or null ≥ 90% of the time. Starting targets; revisit after the first run.

## Open questions
- Is Roboflow's pretrained ball detector good enough, or is ball fine-tuning needed?
- Replays pass stage 0. Detect them later from the broadcaster's replay transition graphic, or with a shot-type classifier trained on SoccerNet camera-shot labels?
- Throw-ins and goal kicks happen partly off camera; is the ball-state rule enough, or does it need a small classifier (e.g. on SoccerNet action spotting labels)?
