# 03 — Vision Pipeline

## Goal
Turn broadcast video into game state (02) for each frame.

## Inputs / Outputs
- **In:** video file (mp4), optional team rosters.
- **Out:** `objects.parquet`, `frames.parquet`, `players.parquet` in schema v0.1.

## Stages
1. **Detection** — YOLOv8 fine-tuned on players, goalkeepers, referees, ball. Start from Roboflow's pretrained soccer weights.
2. **Tracking** — ByteTrack (via `supervision`) for stable `object_id`s.
3. **Team assignment** — SigLIP crop embeddings → UMAP → KMeans(k=2), as in roboflow/sports. Goalkeepers assigned by nearest team centroid.
4. **Pitch homography** — pitch keypoint model → per-frame homography → pixel to meters. Smooth over time to reduce jitter.
5. **Ball tracking** — dedicated detector at higher input resolution; interpolate short gaps (< 1 s) and mark `interpolated=True`.
6. **Jersey OCR** — read numbers over multiple frames, vote per track, link to roster → `player_id`. Borrow approach from sn-gamestate.
7. **Velocities** — finite differences on smoothed positions.

## Requirements
- Runs on M1 with `--device mps` for short clips; RTX 2060 for full matches.
- Configurable frame skip for detection (tracker fills gaps).
- Every stage can cache its output so later stages rerun without redoing detection.

## Acceptance criteria
- Output passes the schema validator in `tests/`.
- On SoccerNet-GSR validation clips: positions within ~2 m for most visible players (measure with sn-trackeval GS-HOTA as a secondary metric).
- Team assignment correct on > 95% of player-frames on sample clips.

## Open questions
- Is Roboflow's pretrained ball detector good enough, or is ball fine-tuning needed?
- How to handle replays and close-ups (detect and skip frames with no valid homography)?
