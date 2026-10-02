# Handoff prompt: view-gate quick win, then the ball (2026-10-02)

Paste everything below the line into a fresh Claude Code session in this repo, on the workstation (RTX 2060, Windows 11). Two jobs, in order: (A) a quick tune of the view gate's `on_after_s`, then (B) the ball work: labels, a ball score on the bench, then fixes.

---

You're picking up the vision work in the Soccer Goal Predictor repo (`C:\Users\superCookie\Desktop\ml-soccer-vision-model`). Geometry was just fixed: stage 4 is now PnLCalib, and the bench is at 90% of players within 2 m. The next two jobs are (A) a small view-gate tune and (B) the ball: make it measurable, then make it better. You start with no memory of earlier sessions. Everything you need is in the files below, and the files win if anything here contradicts them (say so when it happens).

## Read first, in this order
1. `CLAUDE.md`: project rules. Schema lives in 02; no leakage (frame `t` uses frames `<= t` only); pitch meters, not pixels, once data leaves `vision/`; `vision/` and `prediction/` share only the schema; **spec before code**.
2. `Docs/reviews/pnlcalib-bench-2026-10-02.md`: the latest bench results and what's left. Read all of it.
3. `Docs/Specs/03-vision-pipeline.md`: stage 0 (view gate, lines ~11–19), stage 5 (ball, line ~28), Streaming API → "Ball gaps", Pitch calibration → PnLCalib (skim), Diagnostics (detections cache, `view.parquet`, `camera.parquet`, replay).
4. `Docs/Specs/07-evaluation.md`, section "Vision benchmark (W0)": manifest, marks, scored frames, scorecard, the replay check, "Not scored yet: the ball … waits for the ball-click labels".
5. `Docs/reviews/detection-improvement-spec-2026-09-27.md`: the detection review. Find D2/W2 (ball association), W3 (cut detector) and the ball targets.
6. `Docs/reviews/sensitivity-2026-09-28.md`: what ball errors cost the goal predictor. `ball_miss` 10% of time −0.017 PR-AUC, 50% −0.070; `ball_noise` 1 m −0.010, 4 m −0.049; `ball_false` 5% −0.0075.
7. `Docs/Specs/roadmap.md`, Phase 2 "4. Quality": the ball item ("reset on camera cuts, temporal candidate association instead of max confidence … recall ≥ 90%, precision ≥ 95%") and "Tune the stage 0 thresholds".
8. Code: `vision/view_gate.py`, `vision/pipeline.py` (`step`, `_ball_object`), `vision/stages.py` (`YoloDetector`, `BallAndPeopleDetector`), `vision/config.py`, `vision/bench.py`, `vision/replay.py`, `vision/run.py`, `vision/writer.py`.

## What's established (don't redo it)
- **Bench clips** (`data/splits/vision_benchmark.json`), World Cup 2022, PFF tracking as truth for people:
  - vb01 ARG–FRA, period 2, video from 52.0 s, 2040 frames, 5 close-up marks
  - vb02 NED–ARG, period 2, from 0.0 s, 1776 frames, 3 close-up marks
  - vb03 JPN–ESP, period 1, from 211.2 s, 2214 frames, no marks (stitched video, no cuts)
  - The clip → video map is `data/vision_bench/videos.json`; footage is in `C:\footage\wc2022\`. Never commit footage.
- **Current bench** (PnLCalib default config): within 2 m vb01 86.2%, vb02 87.8%, vb03 94.4%, pooled 90.0%, median 0.59 m; geometry missing 5.8% (view `other` 5.3%, homography rejected 0.5%); false live vb01 0.1 s, vb02 0.2 s; outfield teams 96.6–98.6%.
- **These numbers are replays with overrides.** The cached runs in `data/vision_cache/vb0*/` were made before the final defaults (they ran window 3, jump 5 m, no blind rule). Their `run.json` keeps those values, so the bench reproduces today's numbers with `--set homography_window=1 --set max_homography_jump_m=10 --set pnl_blind_kp=4`. Any fresh run uses the new defaults directly.
- **The gate is now the largest geometry loss.** View `other` is 10.6% of vb01's scored frames and 6.7% of vb02's. It's mostly the gate's `on_after_s` = 1.0 s wait after each close-up before switching back to `match` (vb02 22.2–23.0 s is one example). PnLCalib's gate leaves a close-up after the minimum `off_after_s` = 0.5 s, and the blind rule keeps positions out of that 0.5 s.
- **Ball today** (stage 5): roboflow's dedicated ball model (`football-ball-detection.pt`), the most confident detection over `min_det_conf` 0.3 per frame, extrapolated forward up to `ball_max_gap_s` 1 s (`interpolated=True`). History expires before reuse; there's no extrapolation without geometry. **Not done:** reset on camera cuts, and temporal association instead of max confidence (detection review D2/W2).
- **The ball isn't scored anywhere yet.** PFF can't say whether the ball is visible in the image, so the bench needs human ball labels in the image (07).

## Environment
- Run Python as `.venv/Scripts/python.exe ...` (Git Bash) with `PYTHONUTF8=1`. **Not plain `uv run`**: it syncs without extras and can drop torch. If deps change, `uv sync --all-extras`.
- Roboflow weights: `../sports/examples/soccer/data/` (ball, player, pitch `.pt`). PnLCalib weights: `C:/Users/superCookie/Desktop/PnLCalib` (`SV_FT_WC14_kp` / `_lines`).
- Fresh run of a clip (about 10–12 min each at ~3 fps):
  ```
  PYTHONUTF8=1 .venv/Scripts/python.exe -m vision.run --video "<path from videos.json>" \
    --weights-dir ../sports/examples/soccer/data --pnl-weights-dir C:/Users/superCookie/Desktop/PnLCalib \
    --match-id <clip_id> --start-s <start> --max-frames <frames> --period <period>
  ```
  This overwrites `data/vision_cache/<clip_id>/` and `data/gamestate/<clip_id>/`. Back them up first if you want to compare. Older roboflow runs sit in `<clip_id>-roboflow`.
- Score: `PYTHONUTF8=1 .venv/Scripts/python.exe -m vision.bench [--clip ID] [--set F=V] [--grid F=V1,V2]`. Every score is a replay, and the bench refuses a clip whose replay doesn't reproduce the run's cache. **Replay doesn't simulate the view gate**, so gate settings need fresh runs.
- Tests: `.venv/Scripts/python.exe -m pytest -q tests/test_vision_*.py` (~45 s). The full suite takes ~6 min. Some ruff errors in `prediction/` and `scripts/` predate this work; keep `vision/` and `tests/test_vision_*` clean (`ruff check`, `ruff format`).
- **Bash-tool gotcha:** long `python - <<'EOF'` heredocs with code have failed to parse here. Write scripts to the scratchpad and run the file. Interactive tools (OpenCV windows) need the user to run them: tell them the exact command, and suggest prefixing it with `!` so the output lands in the chat.

## Task A: view gate `on_after_s` (quick win; keep it to about an hour of compute)
- **Goal:** less view `other` after close-ups without letting close-ups through. `on_after_s` (default 1.0) trades directly against false live; `off_after_s` (0.5) stays.
- Try `on_after_s` 0.5 and 0.7 next to 1.0. Fresh runs on vb01 and vb02 only (vb03 has no cuts). Score each with `vision.bench` and report view `other`, within 2 m and false live per clip.
- **Pick rule:** the lowest `on_after_s` that keeps false live per clip at or under 0.5 s, and doesn't let through a mark the current setting keeps out. 8 marks on 2 clips is a small sample: say so, and don't go below 0.5.
- Check the gate's cost on marks per close-up, not only in total (match-view seconds per mark from `view.parquet`, as the bench review did).
- If it's adopted: change the default in `vision/config.py`, note it in 03 stage 0 and the bench review (a dated section), and tick the stage 0 item on the roadmap or narrow it.

## Task B: the ball
**B1, spec first** (07, plus 03 if the pipeline changes). Decide and write down:
- **Label format and location.** Per clip, every Nth scored frame (N = 3 or 5; justify it), with the ball's pixel center or "not visible" (off screen or hidden). Committed like the marks: a small JSON next to the manifest, or in it. Not in `data/vision_bench/` if that's gitignored (check). The labels carry the video hash, like the manifest.
- **Ball score in the bench:**
  - recall: labeled-visible frames where vision has a ball within R px of the click (R ~ ball size at 1080p, say 10–15 px; pick and justify it)
  - precision: vision ball rows on labeled frames that land within R of a visible ball
  - extrapolated / interpolated rows reported apart, since they're guesses
  - optionally the error in meters through the frame's homography
  - targets from the roadmap: recall ≥ 90%, precision ≥ 95%
- **Pixels.** Ball labels are pixels inside `vision/`, allowed under the detections-cache exception (03). They never leave vision.

**B2, the labeling tool** (`scripts/ball_click.py` or similar). Steps through the frames to label, left click = ball, a key = not visible, back/skip keys, saves as it goes, resumable. Show the previous label as a hint. The user does the clicking, so make it fast. Give the user the exact command, then wait for the labels before B3. Offer to label a single clip first.

**B3, the baseline:** score the current ball stage on the labeled clips, report it in a new dated review (`Docs/reviews/ball-bench-<date>.md`), and break the misses down: no detection at all, a detection elsewhere (a wrong max-confidence pick), or extrapolation drift.

**B4, the fixes**, in the order the baseline points to, each with a spec note in 03 first:
- reset ball history on a camera cut (the view gate's switch, or a jump in the camera)
- temporal association: pick the candidate nearest the predicted position, not the most confident, with a gate on distance
- lower `min_det_conf` for the ball, if misses are low-confidence true balls
- detector input resolution, or tiling near the last position
- fine-tuning the ball detector comes later, only if these fall short

Score each fix on the bench. Keep the replay check working: if a change needs reruns, rerun. Mind the no-leakage rule: live ball filling is extrapolation only, never later frames.

## Working rules (the user's)
- Concise, direct writing: numbers, short bullets, no filler. Match the tone of 03 and the reviews.
- Commit and push in small logical steps (spec, tool, scorer, each fix, review), with short, casual, human commit messages and no AI attribution. Commits are signed already; never disable signing. End every turn with a clean, pushed tree (a Stop hook enforces it).
- Don't touch `prediction/` or the 02 schema.
- Ask before deleting files or anything destructive.
- End with a short, plain-language summary for the user: what changed, the numbers, and what's next. The user prefers simple, organized bullets.
