# Handoff prompt: ball build, steps 1–3 of 10-ball (2026-10-02)

Paste everything below the line into a fresh Claude Code session in this repo, on the workstation (RTX 2060, Windows 11). The job is **building**: 10-ball §4 steps 1–3 (PFF reference and score, the label tool, the tracker). Steps 4–7 (auto-labels, fine-tune, carrier hold, live model) come after, in their own sessions.

---

You're picking up the ball work in the Soccer Goal Predictor repo (`C:\Users\superCookie\Desktop\ml-soccer-vision-model`). You start with no memory of earlier sessions. Everything you need is in the files below, and the files win if anything here contradicts them (say so when it happens).

## Read first, in this order
1. `CLAUDE.md`. In short:
   - no leakage: vision output is causal (frame t uses frames ≤ t; the bench scorer may look ahead, the pipeline never);
   - pitch meters outside `vision/`;
   - `vision/` and `prediction/` share only the schema;
   - spec before code.
2. **`Docs/Specs/10-ball.md`**, all of it. This is the plan you're building.
3. `Docs/reviews/ball-research-2026-10-02.md`: why each choice was made, with the numbers to reproduce.
4. `Docs/Specs/07-evaluation.md`, "Vision benchmark (W0)" through "Ball score" and the new PFF score.
5. `Docs/Specs/03-vision-pipeline.md`: stage 5, Streaming API → "Ball gaps", Diagnostics (`balls.parquet`, `camera.parquet`, replay), Pitch calibration → PnLCalib.
6. Code:
   - `vision/ball.py` (`BallTrack`), `vision/replay.py` (`replay_ball`, `frame_homographies`, `cached_camera`);
   - `vision/bench.py` (`ball_frames`, `ball_headline`, `Clip`), `vision/calib.py`, `vision/stages.py` (`PnLCalibCamera`), `vision/config.py`;
   - `scripts/ball_click.py`; `tests/test_vision_*.py`.

## What's established (don't redo it)
- **Bench:** vb01–vb03, caches in `data/vision_cache/<clip>/` on the current defaults, with `balls.parquet` and `camera.parquet`. The replay check covers ball rows and passes.
- **Projection chain, verified on people** (9–11 px median on call frames): 02 → TV via `to_02` → PnLCalib world `(x, −y, −z)` → `K R [I | −C]`.
  - Project the ball at **z + 0.11 m**.
  - Interpolate PFF's ball to the frame's exact time.
  - Add **no** extra time shift.
- **Camera:** solve one on each label frame. The run's held camera is worse between calls. PnLCalib on the 930 extracted label JPGs took 0.4 s a frame. The tool spec says to decode from the video sequentially. Check that the two agree: on run call frames the JPG camera matched the run's to < 1 px.
- **Numbers to reproduce** (review, R1):
  - vb02 good clicks within 25 px of the projection: **84.4%** (147 frames, median 11.9 px);
  - label frames by kind: vb01 197 pff / 93 estimated; vb02 170 / 110; vb03 340 / 2 / 18 no camera;
  - stage 5 verdict agreement with clicks at 40 px: 95.2%.
- **Bad vb02 labels** ("good" excludes them): 240–280 (specks), 575–610 (trailing), plus guesses at 1060–1080, 1505–1590, 1610–1650, 1665–1690, 1705–1725.
- **Picker replay** (review, R2 table): today's rule 70.2% recall / 73.6% precision on PFF at 25 px, 82.7% / 87.9% on the good clicks. Gate + size + 2 m margin: 72.0 / 74.5 and 85.5 / 90.0.
- **Don't run two `vision.run`s at once on the 2060** (6 GB; 67 min instead of ~10). The same goes for any GPU job next to a run.

## Steps for this session
1. **PFF reference + PFF score** (10-ball 1b, 1d):
   - `vision/ball_truth.py` builds `data/vision_bench/ball_truth/<clip>.parquet`;
   - `vision.bench` prints the PFF score (40 px; also 25 px and the ceiling) and the agreement check under the ball score;
   - tests on a synthetic camera (a known ball lands on a known pixel; the z + 0.11 m; `kind` cases);
   - check against the review's numbers.
2. **Label tool** (10-ball 1c): `scripts/ball_click.py --assist` and `--flag`. It's interactive, so the user runs it.
   - Give the exact PowerShell command without a leading `!`: `$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; .venv\Scripts\python.exe scripts\ball_click.py --clip vb02-ned-arg --flag`.
   - The user fixes vb02's flags, then labels vb01 and vb03 with `--assist`. Don't wait for it to finish before step 3: step 3 is scored on both truths.
3. **Tracker** (10-ball 2a–2c, 2e, 2f):
   - new `VisionConfig` fields, old runs replay as `"max"` exactly;
   - the gate, the filters and the period reset in `BallTrack`;
   - tests;
   - replay on all three clips, both scores, pooled and per clip;
   - adopt per 10-ball §4 step 3's rule. New defaults apply to new runs; no fresh GPU run is needed to score, since every score is a replay.
4. Write the results into a review (`Docs/reviews/ball-build-<date>.md`), tick the roadmap boxes, and update 10-ball where reality differed.

## Environment
- **Python:** `.venv/Scripts/python.exe` (Git Bash) with `PYTHONUTF8=1`. Not plain `uv run`, which syncs without extras and can drop torch.
- **Bench:** `PYTHONUTF8=1 .venv/Scripts/python.exe -m vision.bench [--clip ID] [--set F=V]`.
- **Tests:** `.venv/Scripts/python.exe -m pytest -q tests/test_vision_*.py` (~45 s).
- **PnLCalib weights:** `C:/Users/superCookie/Desktop/PnLCalib` (prefix `SV_FT_WC14`). Ball weights: `../sports/examples/soccer/data/football-ball-detection.pt`.
- **Label frames:** `data/vision_bench/ball_frames/<clip>/<src>.jpg`. Footage paths: `data/vision_bench/videos.json` (gitignored); footage in `C:\footage\wc2022\`, never commit it.
- **Bash-tool gotcha:** long `python - <<'EOF'` heredocs with code have failed to parse here. Write scripts to the scratchpad and run the file.

## Working rules (the user's)
- Concise, direct writing: numbers, short bullets, no filler.
- Commit and push in small logical steps with short, casual, human messages and no AI attribution. Commits are signed already; never disable signing. End every turn with a clean, pushed tree (a Stop hook enforces it).
- Don't touch `prediction/` or the 02 schema.
- Ask before deleting files or anything destructive.
- End with a short, plain-language summary: what you built, what the numbers say, what's next.
