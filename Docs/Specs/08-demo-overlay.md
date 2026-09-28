# 08 — Demo Overlay

## Goal
A video that makes the model's output obvious to someone who knows nothing about ML.

## Elements
- **Danger meter:** vertical bar showing P(goal in next H s), color shifts green → red.
- **Likely shooter:** highlight ring on the player with the highest node-level threat (if node head exists).
- **Minimap:** top-down pitch with player dots, team colors, ball.
- **Event markers:** flash when an actual shot/goal happens, so viewers can see the lead time.
- **Ball marker:** highlight ring/glow on the ball instead of a raw detection box, and a short velocity arrow off `vx, vy` (02). Rendering only, no new inference.
- **Possession panel:** rolling possession % (time each team has had the ball, from `possession_team`) and territorial/attacking-third % (share of time each team spends with the ball in each pitch third, from `pitch_x`). Aggregation over existing game state, no new inference; not the same signal as the danger meter and shouldn't be framed as a second P(goal)-style metric.
- **Sprint highlight:** color a player's ring by `vx, vy` magnitude (jogging vs. sprinting). Rendering only.
- **Ball trail:** short fading line over the ball's last ~1 s of positions, alongside the velocity arrow. Rendering only.
- **Confidence/uncertainty tint:** dim or dash a player's ring when `confidence` is low or `interpolated=True` (guessed/off-camera position), so viewers can see when the model is working from a guess. Rendering only.
- **Event ticker:** short text flashes from `events.parquet` (corner, free kick, ...) along the bottom, beyond the existing shot/goal flash. Rendering only.
- **Shooting-lane cone:** the ball-to-posts triangle used as a LightGBM feature (05: defenders/keeper in the shooting lane), shaded by how open it is. Draws an existing model feature; no new inference.
- **Offside line:** horizontal line at the second-to-last defender's x-position. Pure geometry off team + x-positions already in `objects.parquet`; no new inference. Needs solid homography accuracy on the defensive line specifically.
- **Pitch control / space heatmap:** shade the pitch by which team's players are geometrically closer to each patch of grass (Voronoi diagram over player positions). Computational geometry over existing positions, CPU-only, no new inference; extends the possession/territorial work above from a number into a picture.

## Modes
- **Offline:** render from saved game state + predictions (any machine).
- **Debug:** same renderer, for a frame range (`--frames 1200-1500`): boxes, track IDs, teams and minimap from the detections cache (03 Diagnostics). For finding where and when vision went wrong; never published.
- **Live:** run on RTX 2060 workstation; detection every 2nd–3rd frame, smaller YOLO model. Frame budget and stages that need changes for live are in 09.

## Footage
| Use | Footage | Can it be published? |
|---|---|---|
| Development and testing | SoccerNet clips (GSR, tracking) | No. Their NDA restricts redistribution; internal only |
| Private demos (showing people directly) | Pro broadcast highlights, downloaded for personal use | No. Copyrighted; never uploaded |
| Public demo / write-up | Footage recorded yourself: a local or amateur match, filmed from a high sideline spot, with players' consent | Yes |
| Possible later | PFF-linked broadcast | Only if PFF's terms allow it (unknown until access, see 06) |

- Self-recorded footage looks different from broadcast (camera height, zoom, kit colors, pitch markings). Expect to fine-tune detection and pitch keypoints on a few hundred labeled frames from it (03).
- Self-recorded matches have no provider tracking, so the public demo shows the overlay only. The accuracy numbers in the write-up come from 07, not from this footage.
- Keep a record of which footage each rendered clip used, so nothing restricted gets published by accident.

## Acceptance criteria
- 30–60 s clip including at least one goal, where the meter visibly rises before the shot.
- The public version of that clip uses self-recorded footage.
