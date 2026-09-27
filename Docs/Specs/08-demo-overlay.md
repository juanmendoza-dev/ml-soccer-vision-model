# 08 — Demo Overlay

## Goal
A video that makes the model's output obvious to someone who knows nothing about ML.

## Elements
- **Danger meter:** vertical bar showing P(goal in next H s), color shifts green → red.
- **Likely shooter:** highlight ring on the player with the highest node-level threat (if node head exists).
- **Minimap:** top-down pitch with player dots, team colors, ball.
- **Event markers:** flash when an actual shot/goal happens, so viewers can see the lead time.

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
