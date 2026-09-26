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
- **Live:** run on RTX 2060 workstation; detection every 2nd–3rd frame, smaller YOLO model.

## Acceptance criteria
- 30–60 s clip including at least one goal, where the meter visibly rises before the shot.
