# Handoff prompt: ball research, then a detailed ball spec (2026-10-02)

Paste everything below the line into a fresh Claude Code session in this repo, on the workstation (RTX 2060, Windows 11). The job is **research and a spec, not implementation**: find the best way to (1) measure the ball without hand clicks and (2) track the ball better, then write a detailed spec the next build session can follow.

---

You're picking up the ball work in the Soccer Goal Predictor repo (`C:\Users\superCookie\Desktop\ml-soccer-vision-model`). You start with no memory of earlier sessions. Everything you need is in the files below, and the files win if anything here contradicts them (say so when it happens).

**What the user wants:** real research (web sources, papers, repos, plus measurements on our own caches), then a super detailed spec. **No pipeline code this session.** Scratch scripts for measurements are fine (in the scratchpad, not the repo). The user asked for this after hand-labeling one clip: clicking is slow and error-prone, and there's most likely an open-source model that tracks the ball better. They also pointed out that possession has to survive gaps, since the ball can't be seen all the time.

## Read first, in this order
1. `CLAUDE.md`: project rules. Schema in 02; **no leakage** (frame `t` uses frames `<= t` only, which also rules out any live model that looks at frame `t+1`); pitch meters, not pixels, once data leaves `vision/`; `vision/` and `prediction/` share only the schema; **spec before code**.
2. `Docs/Specs/07-evaluation.md`, "Vision benchmark (W0)" through "Ball score": the manifest, scored frames, and the ball labels and ball score added 2026-10-02 (R = 15 px, miss buckets). It currently says PFF "can't say whether [the ball] is visible in the image". That claim is one of the things to test (below).
3. `Docs/Specs/03-vision-pipeline.md`: stage 5 (ball), Streaming API → "Ball gaps", stage 8 (possession, `vision/state.py`), Pitch calibration → PnLCalib (the full camera), Diagnostics (`camera.parquet`, `balls.parquet`, ball replay).
4. `Docs/Specs/06-data-sources.md`, PFF section: the ball is 3D (`x, y, z`) with its own `visibility` flag (VISIBLE / ESTIMATED); cutaways have an empty `balls` list.
5. `Docs/reviews/pnlcalib-bench-2026-10-02.md`, all of it, including the new "Gate `on_after_s`" section.
6. `Docs/reviews/detection-improvement-spec-2026-09-27.md`: D1/D2, W2 (ball as a temporal track), W3 (cuts), W8 (fine-tuning), and the ball targets (recall ≥ 90%, precision ≥ 95%).
7. `Docs/reviews/sensitivity-2026-09-28.md`: what ball errors cost the predictor (`ball_miss` 10% → −0.017 PR-AUC, 50% → −0.070; `ball_noise` 1 m → −0.010, 4 m → −0.049; `ball_false` 5% → −0.0075).
8. Possession: `Docs/reviews/stage8-2026-09-29.md`, `possession-inferred-2026-09-29.md`, `possession-learned-2026-10-01.md`, `possession-smoothing-2026-10-01.md` (skim the rest of the `possession-*` reviews). Stage 8's carrier rule and the learned possession model already hold possession through ball gaps. Read what they found before proposing anything there.
9. `Docs/Specs/09-hardware.md` (live budget) and `Docs/Specs/roadmap.md`, Phase 2 "4. Quality" (the ball item) and the detection review follow-ups.
10. Code: `vision/ball.py` (`BallTrack`, stage 5), `vision/pipeline.py`, `vision/replay.py` (`replay_ball`, `frame_homographies`), `vision/bench.py` (`ball_frames`, `ball_headline`, `ball_label_frames`), `vision/calib.py` + `vision/types.py` (`Camera`: PnLCalib's world is centered meters, y toward the near touchline, z down), `vision/stages.py` (`YoloDetector` at imgsz 1280, `BallAndPeopleDetector`), `scripts/ball_click.py`.

## What's established (don't redo it)
- **Bench:** vb01 ARG–FRA, vb02 NED–ARG, vb03 JPN–ESP (World Cup 2022, PFF tracking as truth). Clips in `data/splits/vision_benchmark.json`, footage paths in `data/vision_bench/videos.json` (gitignored; footage in `C:\footage\wc2022\`, never commit it).
- **Fresh runs on the current defaults** (`on_after_s` 0.5 adopted 2026-10-02; PnLCalib window 1, jump 10, blind 4). Caches in `data/vision_cache/vb0*/` with `balls.parquet`. Bench pooled: within 2 m **92.0%**, median 0.59 m, geometry missing 3.2%, false live 0.1 / 0.2 s. Older caches are kept as `<clip>-on1.0`, `-on0.5`, `-roboflow`.
- **`balls.parquet`** holds every ball candidate the ball model returned (conf ≥ 0.1). Stage 5 is `BallTrack` (`vision/ball.py`, still the most confident candidate ≥ `min_det_conf` 0.3, extrapolated ≤ 1 s). `vision.replay` reruns stage 5 from the candidates **exactly** (the bench's replay check covers ball rows and passes on all three clips). So association rules, thresholds and cut resets can be scored on CPU in seconds. Only a new detector, resolution or tiling needs a fresh run.
- **Ball labels** (`data/splits/vision_ball_labels.json`, 07 "Ball labels"): every 5th source frame outside the marks, keyed by source-video frame index. Only **vb02** is labeled (280 frames, first pass), and it has problems. An audit of all 280 crops found:
  - ~235 good: median 5 px from confident detections;
  - **zero frames marked "not visible"**: guesses were clicked where the ball couldn't be seen (1505–1590 a ball in the air, 1610–1650 / 1665–1690 inside player groups, 1060–1080 and 1705–1725 fast pans; at 1715 the ball is visible ~110 px from the click);
  - **240–280:** clicks on a small white speck, not the ball (the ball is at an orange player's feet upper right);
  - **575–610:** clicks trailing a rolling ball by 15–23 px.

  The user may have fixed these with `--frames` since: check `git log -- data/splits/vision_ball_labels.json` and re-audit if they did. vb01 and vb03 are unlabeled; the user would rather not click them.
- **vb02 first-pass ball score** (provisional, the labels above are flawed): recall 68.9%, precision 76.0%. Misses: view_other 9, no_geometry 1, wrong_pick 12, low_conf 8, drift 21, not_detected 36. Seen in the audit: the ball model picks **white specks near the bottom of the frame** (y ≈ 850–900) and **adidas board logos / the goal net** over the real ball at 0.5–0.8 confidence.
- **Ball box size** at 1080p: median 15–18 px wide, 5th percentile 10.5 px.
- **PFF time vs video:** cut-edge syncs are exact to a frame; kicks read 0.11 s early against PFF's ball (07 sync rung 3).
- **Don't run two `vision.run`s at once on the 2060.** It fills the 6 GB, spills to shared memory and takes 67 min instead of ~10. One at a time: ~3.1 fps.

## Research questions
Answer each with sources (links) and, where possible, a measurement on our caches.

**R1. Ball truth without hand labels (most important).** PFF tracked the ball in 3D from the same broadcast, with a `visibility` flag. The bench already trusts PFF's VISIBLE flag for players. With PnLCalib's full camera cached per call (`camera.parquet`), PFF's VISIBLE ball (`x, y, z`) can be projected into the image, airborne balls included.
- Work out the coordinate chain: PFF/02 meters → TV frame (`home_attacks_tv_right_p1`, period) → PnLCalib's world (centered, y toward the near touchline, z down) → pixels. Check it on people first: PFF VISIBLE players' feet projected vs vision's foot points.
- Measure on vb01–03: per scored frame with a VISIBLE PFF ball, the pixel distance from the projection to the nearest ball candidate and to stage 5's pick. Break it down by ball height `z` and speed, and against the sync (try ±1–3 frame offsets, since PFF's ball may lag).
- Measure against the vb02 hand labels (the good ones only, or the fixed ones): how often PFF-VISIBLE agrees with "visible in the image", and the projection-to-click distance. That tells whether 07's claim holds and what radius the PFF truth needs. Frames between camera calls use the held camera (every 5th frame), so also measure the error on call frames only.
- Decide: is projected PFF good enough to be the main ball truth (every frame, all clips, and any new 2022 clip for free), with a small hand-labeled set as its check? What radius, and which frames are excluded (ball off screen after projection, ESTIMATED, high `z`)?

**R2. Better ball detection and tracking, open source.** Survey and rank. For each: license, weights available, single-frame or multi-frame, **causal or not** (a model that needs frame `t+1` can only run with a delay; say how much), reported soccer results, speed on an RTX 2060 at 1080p, and the integration effort with `BallTrack` / `balls.parquet`. Verify each claim in the source; don't trust memory. Starting list, to check and extend:
- roboflow/sports' own soccer example: ball detection with `InferenceSlicer` (tiles) and its ball tracker (we already use its weights);
- WASB (sports ball detection baseline, BMVC 2023, includes soccer);
- TrackNet v2 / v3 / v4 and any soccer adaptation;
- FootAndBall;
- DeepBall and other soccer-specific detectors;
- YOLO fine-tuned on public soccer ball data. Check whether SoccerNet-Tracking really has ball boxes, plus ISSIA-CNR, Roboflow Universe sets and anything newer. Note licenses (some SoccerNet data needs an NDA).
- Also the cheap knobs on what we have: detector input size above 1280, tiles around the last position, `min_det_conf`, temporal association (nearest to the predicted position with a gate) instead of max confidence, reset on cuts (W2), and filters for the distractors seen (specks, logos, the net).

**R3. Tracking through gaps and possession.** How strong systems handle the ball when it's hidden or in the air (Kalman/alpha-beta, candidate graphs, player-ball interaction). Tie this to what stage 8 and the learned possession model already do. Use the sensitivity numbers to say how much ball quality the predictor needs. Don't re-run possession experiments; this is reading plus a short argument.

**R4. Live budget.** For the top candidates, what they cost per frame next to the current load (YOLO people + YOLO ball at 1280 + PnLCalib every 5th frame ≈ 3.1 fps offline; 09's live plan).

## Deliverables
1. **`Docs/reviews/ball-research-<date>.md`**: findings per question, with links and the R1 measurements (tables), a ranked comparison of R2 candidates, and a recommendation: which truth, which detector or tracker path, and in what order.
2. **The spec, in detail, where the project keeps specs** (spec before code):
   - **07:** the ball truth. If R1 holds, PFF projected as the main truth, with the hand labels as its check. Frames, radius, exclusions, sync handling, miss buckets, targets. Edit the "Ball labels" / "Ball score" text rather than adding a parallel one.
   - **03 stage 5:** the tracker design (association, gates, reset on cuts and segments, extrapolation, what's cached, replay exactness, causality), and the detector change if recommended (model, input size or tiles, cadence, how it's cached and replayed).
   - If it's too big for 03, a dedicated `Docs/Specs/10-ball.md` linked from 03 and 07.
   - The order of the build steps, each with its bench check and a done-when.
3. **Roadmap:** update Phase 2 "4. Quality" ball item and the detection review follow-ups to the new plan.
4. **A short handoff prompt** for the build session, in `Docs/handoff/`, like this one.

## Environment
- Run Python as `.venv/Scripts/python.exe ...` (Git Bash) with `PYTHONUTF8=1`. **Not plain `uv run`**: it syncs without extras and can drop torch.
- Bench: `PYTHONUTF8=1 .venv/Scripts/python.exe -m vision.bench [--clip ID] [--set F=V]` (prints the ball score where labels exist). Tests: `.venv/Scripts/python.exe -m pytest -q tests/test_vision_*.py` (~45 s).
- Useful for measurements: `replay.load`, `replay.load_balls`, `replay.frame_homographies`, `replay.cached_camera` (a `Camera` from a `camera.parquet` row), `bench.load_truth` / `align` / `sync_offset` (PFF frames and players for a clip; `load_truth` reads people only), `bench.load_ball_labels`. PFF game state is in `data/gamestate/<match_id>/` (02 format): the ball is `objects.parquet` rows with `object_type == "ball"`, columns `x, y, z, visible, interpolated` (ESTIMATED → `visible=False, interpolated=True`).
- Extracted label frames (1080p JPGs) are in `data/vision_bench/ball_frames/<clip>/<source frame>.jpg` for all three clips. Good for looking at crops.
- Web research: use WebSearch / WebFetch (load them with ToolSearch first).
- **Bash-tool gotcha:** long `python - <<'EOF'` heredocs with code have failed to parse here. Write scripts to the scratchpad and run the file.
- Interactive tools (OpenCV windows) need the user. Give them the exact PowerShell command without a leading `!` if they run it in their own terminal (`$env:PYTHONPATH='.'; $env:PYTHONUTF8='1'; .venv\Scripts\python.exe ...`).

## Working rules (the user's)
- Concise, direct writing: numbers, short bullets, no filler. Match the tone of 03, 07 and the reviews.
- Commit and push in small logical steps (each measurement script result into the review, each spec section, the roadmap, the handoff), with short, casual, human commit messages and no AI attribution. Commits are signed already; never disable signing. End every turn with a clean, pushed tree (a Stop hook enforces it).
- Don't touch `prediction/` or the 02 schema. No pipeline code this session.
- Ask before deleting files or anything destructive.
- End with a short, plain-language summary for the user: what you found, what you recommend, and what the build session will do. The user prefers simple, organized bullets.
