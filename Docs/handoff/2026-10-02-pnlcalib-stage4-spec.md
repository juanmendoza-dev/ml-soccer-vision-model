# Handoff prompt: spec PnLCalib as vision stage 4 (2026-10-02)

Paste everything below the line into a fresh Claude Code session in this repo, on the workstation (RTX 2060, Windows). The task is the **spec in 03**, plus one small experiment that the spec depends on. No pipeline code yet.

---

You're picking up a vision-geometry decision in the Soccer Goal Predictor repo (`C:\Users\superCookie\Desktop\ml-soccer-vision-model`). The goal is to write the plan in `Docs/Specs/03-vision-pipeline.md` for replacing stage 4 (pitch homography from roboflow's pitch keypoint model) with **PnLCalib**, a SoccerNet camera-calibration model. Code comes in a later session; this one writes the spec, runs one experiment that answers an open design question, and updates the roadmap.

## Read first, in this order
1. `CLAUDE.md` (project rules: schema in 02, no leakage, meters not pixels, `vision/` and `prediction/` only share the schema, spec before code).
2. `Docs/reviews/pnlcalib-2026-10-02.md`: **the evidence this task rests on.** Read all of it.
3. `Docs/Specs/03-vision-pipeline.md`: stage 0 (view gate, lines 11–18), stage 4 (line 27), "Pitch template" and "Homography acceptance" (~242–257), "Diagnostics" / detections cache / `keypoints.parquet` / replay (~268–311), the Streaming API (~225). The new plan has to fit these.
4. `Docs/Specs/07-evaluation.md`, section "Vision benchmark (W0)" (~107–160): how the bench scores, the replay check every score depends on, and the homography threshold sweep.
5. `Docs/reviews/vision-bench-2026-10-01.md`, the last section "Clip 3": why our current stage 4 fails (keypoints on a mowing stripe, a fit that passes its own check and is wrong).
6. Code you'll be specifying against: `vision/config.py` (stage 4 fields: `keypoints_every` 5, `min_keypoint_conf`, `ransac_m`, `min_inliers`, `max_homography_err_m`, `homography_window` 3, `homography_max_age_s`, `max_homography_jump_m`, `circle_kp_x_m`), `vision/pitch.py` (`fit_homography`, `HomographyFilter`), `vision/pipeline.py`, `vision/stages.py` (`YoloKeypoints`), `vision/replay.py`, `vision/bench.py`, `scripts/pnl_compare.py`.
7. `Docs/Specs/09-hardware.md` for the live budget on the 2060.

## What's established (don't redo it)
- **Bench clips** (`data/splits/vision_benchmark.json`), all World Cup 2022 with PFF tracking as truth: vb01 ARG–FRA (Lusail), vb02 NED–ARG (Lusail), vb03 JPN–ESP (Khalifa International, stitched video, synced by kicks). All three syncs were confirmed to within 0–2 PFF frames by an offset scan on PnLCalib positions.
- **Current stage 4 on the bench** (within 2 m / median): vb01 20.9% / 2.43 m, vb02 32.4% / 2.25 m, vb03 15.3% / 3.10 m. Target is ≥ 90% within 2 m.
- **PnLCalib on the same vision foot points (box bottom center) and the same PFF truth**, every 5th frame, no view gate, no temporal filter (`scripts/pnl_compare.py`): vb01 **81.7%** / 0.60 m, vb02 **91.2%** / 0.57 m, vb03 **84.1%** / 0.45 m. The pipeline on the same frames: 14.8 / 32.5 / 15.8%.
- **The remaining gap is no-fit frames**, not bad fits: vb01 47/290, vb02 6/280, vb03 35/360 sampled frames return no camera. They come in **runs of 3–5 s on ordinary wide play** (vb03 257.5–260.5 s and 281–285 s, vb01 78–83.3 s, vb02 22.2–23.0 s), including views with halfway and the circle visible. On vb03 our current stage 4 fits ~98% of frames, so PnLCalib is worse on availability there.
- **Speed on the 2060:** ~420 ms per call end to end. The two HRNetV2-W48 nets are 305 ms fp32 / 185 ms fp16 autocast; voting + PnL refinement ~60 ms. Peak GPU memory 1.9 GB. **Stage 4 already runs only every `keypoints_every` = 5 frames**, so offline that's ~0.08 s per frame on average, about a third of the current pipeline's ~0.25 s/frame. Live (30 fps) is a separate budget question for 09.
- **License/NDA: settled by the user.** PnLCalib is GPL-2.0 and its weights are trained on NDA-gated SoccerNet data. The repo is private, so neither is a concern, including importing or vendoring it. Don't raise it again.
- PnLCalib's output is a full camera: focal lengths, principal point, position, rotation (`projection_from_cam_params` builds the 3×4 P). The ground-plane homography is `P[:, [0, 1, 3]]` in its centered world frame (x − 52.5, y − 34). Mapping to our TV frame is **x same, y negated** (axes (1, −1)), consistent on all three clips.
- `circle_kp_x_m` (the 31/32 template shift) belongs to the roboflow model and goes away with it.

## Environment
- PnLCalib is cloned at `C:\Users\superCookie\Desktop\PnLCalib` (outside the repo) with weights `SV_kp` / `SV_lines` (base single-view). Its extra deps (shapely, lsq-ellipse) are in `PnLCalib\_deps`. Run with the project venv directly:
  `PYTHONPATH=".;C:/Users/superCookie/Desktop/PnLCalib/_deps" .venv/Scripts/python.exe scripts/pnl_compare.py vb03-jpn-esp 5 C:/Users/superCookie/Desktop/PnLCalib`
  (~3 min per clip; prints within 2 m, no-fit times and an offset scan.)
- Use `.venv/Scripts/python` directly, **not plain `uv run`**: that syncs without extras and can drop torch.
- Footage is in `C:\footage\wc2022\` (never commit it); the clip → path map is `data/vision_bench/videos.json`. Vision caches are in `data/vision_cache/<clip_id>/`.
- Finetuned weights are on the PnLCalib releases page: `https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/` + `SV_FT_WC14_kp`, `SV_FT_WC14_lines`, `SV_FT_TSWC_kp`, `SV_FT_TSWC_lines` (~265 MB each).

## Task
### Step 1: experiment (answers the no-fit question before the spec commits to a fallback)
- Download the WC14 and TSWC finetunes into the PnLCalib folder.
- `scripts/pnl_compare.py` hardcodes `SV_kp` / `SV_lines`. Add an optional argument for the weight names (small change; commit it).
- Run all three clips with the base weights (numbers above) and each finetune. Report within 2 m, median, no-fit count, and whether the no-fit runs shrink.
- Also try PnLCalib's own thresholds (`kp_threshold` 0.3434, `line_threshold` 0.7867 by default) a little lower on the no-fit frames only, if the finetunes don't close the gap. Keep it to a few settings: this is a feasibility check, not a sweep to tune on 3 clips.
- Write the results into `Docs/reviews/pnlcalib-2026-10-02.md` as a new section (dated, same style), and commit.

### Step 2: the spec in 03
Write the stage 4 replacement plan into `03-vision-pipeline.md`. Edit the existing stage 4 line and the "Pitch template", "Homography acceptance" and "Diagnostics" sections rather than appending a parallel document. Keep the old behavior described only as far as old caches and runs still need it (old `run.json` must still replay, as with `circle_kp_x_m`). The spec must decide, with reasons:
1. **Integration.** Import PnLCalib from an external path, vendor the needed modules into `vision/`, or a thin wrapper class in `vision/stages.py` next to `YoloKeypoints` (e.g. `PnLCalibCamera`). Where the weights live and how they're hashed into `run.json` (the run already records weight hashes). How the extra deps (shapely, lsq-ellipse) enter `pyproject.toml` (the `vision` extra).
2. **What stage 4 outputs per call:** the full camera vs only the ground homography. A full camera is needed if anything later uses height (ball in the air), so say whether to keep it. Pixel → pitch stays a ground homography for people's feet.
3. **Cadence:** keep `keypoints_every`, or decouple calibration cadence from the gate's keypoint probe. Note the view gate currently uses stage 4's keypoint count (`min_keypoints`), so the gate needs a replacement signal: PnLCalib's keypoint/line counts, or "a camera was found".
4. **No-fit handling**, from step 1's result: hold the last camera up to an age limit (`homography_max_age_s` exists), interpolate between calls, fall back to the old roboflow fit, or accept the loss. Runs are 3–5 s, so hold-last alone likely isn't enough; say what is.
5. **Temporal filtering and acceptance:** what replaces `HomographyFilter` (window average of homographies, jump check). Averaging camera parameters, or homographies? What "rejected fit" means now: a sanity check on camera height/focal/position, consistency with the previous camera, and PnLCalib's own reprojection error. This is where a fixed-camera-position-per-match (PTZ) constraint can live, if useful. Keep it simple; per-match camera position is a candidate, not a requirement.
6. **Cache and replay (07 depends on this):** what goes into the cache per stage 4 call so `vision.replay` can redo acceptance, filtering and projection **without the GPU** and reproduce the run exactly. Candidates: PnLCalib's raw keypoints + lines (lets you redo voting on CPU), or the per-call camera parameters (cheaper, but acceptance can't revisit voting). Keep the bench's replay check (`replay must reproduce the run's detections cache`) intact. Define the new `keypoints.parquet` columns or a new `camera.parquet`, with a schema table like the existing ones.
7. **Config:** new `VisionConfig` fields (calibration backend, weights, thresholds, cadence, max age) and which old ones become backend-specific. Old `run.json` without the new fields must replay as the roboflow backend.
8. **Speed:** offline estimate from the numbers above, plus the live plan for 09 at a line's depth (fp16 autocast, fewer calls, a smaller input). Don't benchmark TensorRT here.
9. **Acceptance criteria for the implementation session:** the bench numbers to expect (all three clips through `python -m vision.bench`, homography-rejected share reported as now), the replay check passing, tests to add. Make the success bar concrete: e.g. pooled within 2 m ≥ 80% with geometry missing not worse than now (10.4% pooled), and a stated plan if no-fit runs keep it below that.
10. **Open questions** that remain, listed at the end of the section.

Also:
- 07 "Homography threshold sweep": update which fields get swept under the new backend.
- Roadmap (`Docs/Specs/roadmap.md`): Phase 2, the geometry item that ends "Next: plan PnLCalib as stage 4 in 03 ...". Mark the plan written, and add the implementation as the next item with the acceptance bar.
- Don't touch `prediction/` or the 02 schema. Coordinates leaving vision stay in pitch meters.

## Working rules (the user's)
- Concise, direct writing in the specs and reviews: numbers, short bullets, no filler. Match the existing tone of 03 and the reviews.
- Commit and push in small logical steps as you go (script change, review section, 03, 07, roadmap), with short, casual, human-sounding commit messages and no AI attribution. Commits are signed already; never disable signing. End the session with a clean, pushed tree.
- Spec before code: no code changes in this session except the weights argument in `scripts/pnl_compare.py`.
- If something in this prompt contradicts what you find in the files, trust the files and say so.

## Done when
- The step 1 results are in the PnLCalib review.
- 03 has the stage 4 plan, covering points 1–10.
- 07 and the roadmap are updated.
- Everything is committed and pushed.
- You end with a short summary for the user: what the spec decided, the step 1 numbers, and what the implementation session should do first.
