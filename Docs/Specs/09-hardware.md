# 09 — Hardware

## Goal
Record what the workstation actually is and whether it can run the vision pipeline (03) live on a broadcast feed (08 Live mode). Training is secondary here.

## Workstation (as of 2026-09-27)
| Part | Spec |
|---|---|
| Board | MSI MS-7C51 |
| OS | Windows 11 Home, build 10.0.26200, x64 |
| CPU | Intel Core i5-11400F, 6 cores / 12 threads, 2.6 GHz base. No integrated GPU (the "F"), so no Quick Sync |
| GPU | NVIDIA GeForce RTX 2060, 6 GB VRAM, Turing (compute capability 7.5, `sm_75`) |
| GPU driver | 32.0.16.1088 (nvidia-smi 610.88), supports CUDA up to 13.3 |
| RAM | 32 GB (2x Corsair 16 GB @ 3200 MHz), ~23 GB free at idle |
| Storage | WD 500 GB SSD (WDS500G2B0C) |
| Python | 3.13.5, 64-bit (system install) |

The M1 MacBook Pro is covered in 01.

## Setup notes
- **Python version:** 01 pins Python 3.11 via `uv`. `uv` downloads its own 3.11, so the 3.13 system install doesn't matter. Don't build the venv on 3.13.
- **PyTorch + CUDA:** the pip wheels bundle their own CUDA runtime. The driver only has to be new enough for that runtime, and 610.88 is. Pick the CUDA wheel from the PyTorch install page and don't try to match "13.3". After installing, check:
  ```python
  import torch
  torch.cuda.is_available()          # True
  "sm_75" in torch.cuda.get_arch_list()  # True; Turing support
  ```
- **Video decode/encode:** no Quick Sync, so decoding runs on the CPU (6 cores handle 1080p30 H.264) or on the 2060's NVDEC. Encoding the overlay output, e.g. for OBS, should use NVENC so the CPU stays free.
- **Disk:** 500 GB is tight once SoccerNet downloads, raw match video (2–4 GB per 1080p match) and `data/vision_cache/` pile up. Keep raw video on external or homelab storage (01 open question).

## First vision run (workstation)
```
git clone <this repo> && cd <repo>
uv sync --extra dev --extra vision   # pyproject sends Windows torch to the cu128 index
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_arch_list())"   # True, includes sm_75

# roboflow/sports weights (their examples/soccer/setup.sh, needs gdown)
git clone https://github.com/roboflow/sports ../sports && bash ../sports/examples/soccer/setup.sh

uv run python -m vision.run --video <demo clip> --weights-dir ../sports/examples/soccer/data --match-id demo1 --max-frames 750
uv run python -m demo.debug --video <demo clip> --cache data/vision_cache/demo1 --out demo1_debug.mp4
```
- Watch `demo1_debug.mp4`: rings colored by kit cluster, IDs, minimap. Then rerun `vision.run` with `--home-cluster 0|1` to get home/away in the game state.
- The progress lines print fps. That's the first real number for the live budget below.
- Keep clips and outputs outside iCloud or other synced folders (08: footage is private).

## Live feasibility

### Frame budget
Broadcast runs at 25–30 fps, so each frame gets **33–40 ms**. With 08's detection on every 2nd–3rd frame (tracker fills the rest), each detection gets **66–120 ms**. Tracking, the game state writer and the overlay still run every frame.

### Per-stage estimates
These are unmeasured estimates for a 2060 with FP16 TensorRT exports. Plain PyTorch is roughly 2–3x slower. Replace them with the benchmark numbers (roadmap Phase 3).

| Stage (03) | Live approach | Rough cost |
|---|---|---|
| 0. View gate | Every frame, CPU; skips everything below on ads/studio/close-ups | ~1 ms |
| 1. Player detection | YOLOv8s at 640–960, every 2nd–3rd frame | ~5–10 ms |
| 5. Ball detection | Small model at higher resolution, or tiled crops near the last ball position | ~10–25 ms; the most expensive stage |
| 4. Pitch keypoints | Every ~5th frame, trailing-window smoothing | ~10–20 ms when it runs |
| 2. Tracking (ByteTrack) | Every frame, CPU | < 2 ms |
| 3. Team assignment | Fit during a warmup window, then classify only new tracks | Occasional SigLIP batch |
| 6. Jersey OCR | Separate async worker, low rate, outside the main loop | Doesn't count against the frame budget |
| 8. Game state inference | Every frame, pitch coordinates only | Negligible |
| Predictor (05) | 10 Hz, small GNN | Negligible |

VRAM isn't the limit: detector, ball model, keypoint model and SigLIP together need well under 6 GB in FP16 at batch size 1. The limit is GPU time per frame. The full offline setup (large models, every frame, full resolution) won't reach real time on this card. Expect single-digit fps for that.

### Verdict
**Live is feasible with the reduced setup 08 already assumes**: a smaller YOLO, frame skip, TensorRT FP16, keypoints and OCR at reduced rates. The benchmark has to confirm it before the live config is fixed. If the ball stage doesn't fit, lower its rate first (the tracker and the < 1 s gap rule cover short misses), then lower the processing resolution.

### Stages that aren't live-safe as specced
- **Ball gap interpolation (03 stage 5):** interpolating a gap of up to 1 s needs the frames after the gap. Live mode extrapolates forward instead and marks those frames `interpolated=True` (see Live app).
- **Homography smoothing (03 stage 4):** must use a trailing window only. A centered window leaks future frames, and live mode can't see them anyway.
- **Team assignment (03 stage 3):** the offline method clusters crops from the whole video. Live mode fits KMeans on the first ~N seconds and then assigns new tracks to the nearest centroid. Refit after half-time kit/side changes if needed.
- **Jersey OCR (03 stage 6):** voting over frames works incrementally. `player_id` shows up a few seconds after a track appears, and stays null until then.
- **The predictor is already fine:** the no-leakage rule (CLAUDE.md) means it only uses frames `<= t`, so it runs online without changes.

## Training (brief)
YOLOv8n/s/m fine-tuning at 640 fits in 6 GB at batch size ~8–16. A ball detector at 1280 needs batch size ~2–4 or Colab (01).

## Live app
Live mode lives in its own repo, `soccer-live-overlay`, which will install this one as a dependency. It never trains or defines formats.
- **Input:** screen capture of the match already playing, via `dxcam` on the workstation. It captures a monitor region, so the player runs fullscreen. DRM-protected players can capture as black frames; check the streaming service first.
- **Output:** predictions over a local websocket to an OBS browser source.
- **Ball gaps:** extrapolated, never buffered. The overlay sits on the video, so ~1 s of delay would show.
- **Recording:** every live session is recorded (video, per-frame timing, predictions, shot/goal markers) for replays and backtests. Format in that repo's `Docs/Specs/02-session-recording.md`.

## Open questions
- The live config (model sizes, input resolution, per-stage rates) comes from the benchmark.
