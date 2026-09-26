# 03 — Vision Pipeline

## Goal
Turn broadcast video into game state (02) for each frame.

## Inputs / Outputs
- **In:** video file (mp4), optional team rosters.
- **Out:** `match.parquet`, `objects.parquet`, `frames.parquet`, `players.parquet` in schema v0.4.

## Stages
1. **Detection** — YOLOv8 fine-tuned on players, goalkeepers, referees, ball. Start from Roboflow's pretrained soccer weights.
2. **Tracking** — ByteTrack (via `supervision`) for stable `object_id`s.
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
- Every stage can cache its output so later stages rerun without redoing detection.

## Diagnostics
Debugging works from cached data after the run, not from extra logging during it. Nothing here adds work to the inference loop beyond writing the cache.

### Detections cache
`data/vision_cache/<match_id>/detections.parquet`, one row per tracked object per frame. Vision-internal: pixels are allowed here and never leave `vision/`.

| Column | Type | Notes |
|---|---|---|
| match_id, frame_id | | Same frame_ids as game state |
| object_id | str | Same as `objects.object_id` |
| class | enum | player, goalkeeper, referee, ball (detector class, before team assignment) |
| x1, y1, x2, y2 | float | Box in pixels |
| det_confidence | float | Detector score; null for tracker-filled frames |
| tracked_only | bool | True if the tracker filled this frame, no detection |
| pitch_x, pitch_y | float/null | Box anchor through the homography; null without a valid homography |
| homography_ok | bool | Per frame (repeated on each row) |
| homography_err_m | float/null | Mean keypoint reprojection error in meters, per frame |

Plus `run.json` next to it: config, git commit, model weights hash, video file hash, per-stage wall time. With the video, this is enough to redraw any moment of the run.

- Anomaly checks (ID switches, ball gaps, homography jumps, ...) are scripts over this cache, written when a real problem shows up. Not part of the pipeline.
- The overlay renderer's debug mode (08) draws boxes, IDs and teams for a frame range from this cache.

## Acceptance criteria
- Output passes the schema validator in `tests/`.
- On SoccerNet-GSR validation clips: positions within ~2 m for most visible players (measure with sn-trackeval GS-HOTA as a secondary metric).
- Team assignment correct on > 95% of player-frames on sample clips.
- Stage 8, run on PFF, SkillCorner and IDSSE tracking and compared to provider values: possession team matches on ≥ 90% of frames where the provider has a value; frames the provider marks dead are labeled dead or null ≥ 90% of the time. Starting targets; revisit after the first run.

## Open questions
- Is Roboflow's pretrained ball detector good enough, or is ball fine-tuning needed?
- How to handle replays and close-ups (detect and skip frames with no valid homography)?
- Throw-ins and goal kicks happen partly off camera; is the ball-state rule enough, or does it need a small classifier (e.g. on SoccerNet action spotting labels)?
