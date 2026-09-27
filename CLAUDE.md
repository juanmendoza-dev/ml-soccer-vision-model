# Project: Soccer Goal Predictor

Read `Docs/Specs/` before writing code. Start with `Docs/Specs/00-overview.md`, then `Docs/Specs/02-game-state-schema.md`.

## Rules
- All components read/write the game state format in `Docs/Specs/02-game-state-schema.md`. Do not invent new formats; propose schema changes in that spec first.
- **No leakage:** a prediction at frame `t` may only use data from frames `<= t`. Train/val/test splits are by match, never by frame.
- Coordinates are pitch meters, never pixels, once they leave the vision pipeline. One exception: display-only boxes (0–1 fractions) for the live overlay, never read by prediction (03 Display output).
- Keep vision (`vision/`) and prediction (`prediction/`) as separate packages that only share the schema.

## Hardware
Full workstation specs and live feasibility: `Docs/Specs/09-hardware.md`.
- MacBook Pro M1: coding, prediction model training, offline vision inference on short clips.
- Workstation (RTX 2060, 32GB RAM): YOLO fine-tuning, full-video vision inference, live demo.
- Google Colab: fallback for jobs that exceed the 2060's VRAM.
