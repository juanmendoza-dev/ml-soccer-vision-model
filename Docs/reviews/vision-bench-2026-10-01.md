# Vision benchmark, first clip (2026-10-01)

First real run of the W0 benchmark (07 "Vision benchmark (W0)"). One clip, so this is a baseline and a check that the tooling works on real footage, not a result to tune on. Marks checked by the user 2026-10-01.

## Clip
`vb01-arg-fra`: PFF 10517 (Argentina v France, final), second half, video 1:00–2:00 (match clock 65:26–66:26), 1080p30. Vision ran from 0:52 (8 s pre-roll), roboflow weights, `--detect-every 1`, RTX 2060 at 5.0 fps.

- **Sync on cut edges:** five tight live close-ups. Their 10 edges give offsets of 1167.44–1167.52 s against PFF's switches to and from all-`ESTIMATED` frames, so it's the world feed and the sync is good to about a frame. The scoreboard clock agrees to 1.5 s. `--offset-check` puts the best offset 0.23 s earlier; worth a look on the next clips.
- **Direction:** with the right goal on screen, PFF's visible players are at x ≈ +45, so home attacks TV right in period 1.
- **Replay check:** the replay reproduced the run's detections cache (the bench refuses a clip otherwise).

## Baseline (current thresholds)
| | |
|---|---|
| Scored frames | 1,327 of 1,800 (the rest are close-ups) |
| Geometry missing | 24.5%: view `other` 16.1%, homography rejected 10.1% of match view |
| People within 2 m (misses count) | 14.0% of 23,155 PFF visible player-frames |
| Matched error, median / p90 | 2.81 m / 4.43 m |
| Within 2 m on frames with geometry | 17.8% |
| Unmatched vision rows per frame | 4.37 |
| Outfield team accuracy / coverage | 52.4% / 99.4% |
| Keeper team accuracy / coverage | 87.5% / 54.7% |
| False live | 1.5 s of marked close-up |

## What it shows
1. **Team clustering failed on this clip.** 98% of player rows (21,140 of 21,479 with a cluster) landed in cluster 1, both teams alike. So team accuracy is a coin flip. That's `KitColorTeams`, not the scorer: navy France against white and sky-blue Argentina should separate, so the fit is splitting on something else (grass, lighting, referees).
2. **Positions are about 2–3 m off, both ways.** Flipping x or y makes it worse, so orientation is right. Against the nearest PFF player, vision's error is about 2 m in each axis (median absolute), with a −1.1 m bias in y (toward the near touchline). Plots of four frames show it near the halfway line too, not only far out. Candidates: keypoint/homography accuracy on this stadium, the foot anchor, and what PFF's position means (centre of mass vs feet). One clip can't separate them.
3. **The view gate drops 16% of unmarked wide frames.** That's more than the rejected homographies, and no acceptance threshold moves it (07). Example: video 1:16, a wide shot with the halfway line and centre circle has no geometry.
4. The 17% target on rejected homographies is met here (10.1%), but within 2 m is far below the 90% target. **Threshold tuning isn't the bottleneck on this clip;** geometry accuracy and teams are.

## Diagnosis (same clip, after the user checked the marks)
**Teams.** `KitColorTeams` uses mean Lab chroma (a, b) only. Its own docstring says kits that differ mainly in lightness need L back, and navy vs white/sky blue is that case. On 350 torso crops matched to PFF players (≤ 2.5 m), a 2-means fit on standardized features:

| Feature | Agreement with PFF team |
|---|---|
| (a, b) mean (current) | 56.9% |
| (L, a, b) mean | 85.4% |
| (L, a, b) median | 85.4% |
| L histogram + (a, b) | 85.4% |

These are per crop; per-track votes come on top.

**Positions: the homography is wrong as a whole on each frame, not the detections.**
- Removing one shift per frame: median 2.80 → 1.95 m (within 2 m: 26% → 52%). Removing one affine per frame: 0.52 m (88%). So boxes and feet are fine, and each frame's mapping is off by a stretch.
- The per-frame affine from vision to PFF has median a_xx 0.87 and a_yy 0.945 (p10–p90: 0.78–1.02 and 0.88–1.05). Vision's picture is stretched about 13% along the pitch and 5% across, and the amount changes per frame. One global affine leaves 2.39 m, so it's not one constant (pitch size, PFF units).
- The fit's own check can't see it. Accepted fits have a 0.41 m mean inlier error (p90 0.65 m), but a homography has 8 parameters and these fits use 5–11 inliers (median 8). The fit can pass close to every keypoint and still be wrong elsewhere.
- Not the cause: inlier count or spread (error is flat across quartiles of the inlier hull area, 2.5–2.8 m), lag (`homography_window = 1`: 2.81 → 2.70 m), distance from the camera (flat across the width), the error threshold (`max_homography_err_m = 0.5` makes it worse).

## Next
- User checks the marks; then the other four 2022 clips and the three geometry-only clips, before reading anything into these numbers.
- If items 1 and 2 hold across clips, they go ahead of the threshold sweep: kit clustering, and a look at where the 2 m comes from. Tackle the y bias first, as the cheapest lead.

## Fix 1: teams (2026-10-01)
03 stage 3 now: mean CIELAB (L, a, b) of the shirt pixels, and each track's team is the majority of all its votes so far (was chroma only, fixed by the first 5 votes). Offline test `scripts/team_crops.py`: the pipeline's own warmup fit, tracks labeled by majority PFF team after removing each frame's affine (109 tracks). The first labeling, without the affine step, mislabeled 6 tracks (white Argentina shirts labeled away, navy France labeled home), checked by eye.

| vb01 outfield | Old | New |
|---|---|---|
| Offline, per crop | 53% | 97.4% |
| Offline, track team | 52% | 97.6% |
| Rerun, rows on PFF-labeled tracks | 53.9% | 97.5% |
| Bench `team accuracy` (coverage 99–100%) | 52.4% | 83.0% |
| Bench keepers (accuracy / coverage) | 87.5% / 55% | 93.3% / 91% |

- **The bench understates it.** Its team score pairs PFF and vision players per frame by position, and with positions ~2.8 m off it pairs neighbours across teams. The rerun's rows on tracks labeled after the affine step are 97.5% right. The bench number should rise to match once positions are fixed (fix 2).
- smoke03 (red Spain vs blue Japan, no truth): both kits split cleanly by eye. Any L weight from 0.1 to 1.0 gives the same vb01 result; scaling each channel by its warmup spread was dropped because it splits off a few bright outliers (synthetic smoke03 test).
- Geometry numbers are unchanged by the rerun, as expected (same detections and keypoints).
- Still to do: confirm on a second clip.

## Fix 2, step a: is the template wrong? (2026-10-01)
`scripts/keypoint_check.py`: on 147 vb01 frames with ≥ 8 matched players, a homography fitted from the players' feet to their PFF positions (median residual ≤ 0.6 m) says where each detected keypoint really is. Keypoints within 20 m of a matched player, TV frame, offset = PFF minus template:

| # | Landmark | n | dx | dy | spread |
|---|---|---|---|---|---|
| 14–17 | Halfway line (far touchline, circle far, circle near, near touchline) | 90–106 | −0.05 to 0.13 | −1.27, 0.86, 1.67, 3.17 | 0.65–1.51 |
| 31 / 32 | Circle left / right of the center spot | 99 / 115 | **+2.13 / −1.75** | 1.18 / 1.54 | 1.1 / 1.2 |
| 18 / 19 | R box front, far | 119 / 100 | −6.97 / −6.01 | 0.73 / 0.05 | 3.5 / 4.7 (erratic) |
| 20–24, 26–28 | R box, spot, 6-yd box, goal line | 8–49 | +0.45 to +1.85 | −0.84 to −3.0 | 0.7–1.8 |

- **31/32 are a real, steady offset:** the model's "circle left/right" points sit about 7.2 m from the center spot, not 9.15 m, while the halfway points beside them land where they should in x. The template forces them out to 9.15 m, which stretches the middle of the pitch along x.
- Replayed on the bench (no rerun needed):

| Template | Within 2 m | Median / p90 | Homography rejected |
|---|---|---|---|
| Current | 14.0% | 2.81 / 4.43 m | 10.1% |
| 31/32 at ±7.2 m | 20.9% | 2.43 / 4.24 m | 10.1% |
| 31/32 dropped | 11.7% | 2.87 / 4.50 m | 17.6% |
| Every landmark with n ≥ 20 moved to its measured spot (fitted on this clip: an upper bound, not a fix) | 30.8% | 1.75 / 3.91 m | 15.0% |

- **The template isn't the main cause.** Even moved to measured positions on the same clip, the median stays at 1.75 m, against 0.52 m once each frame's affine is removed. Each landmark wanders about 1 m from frame to frame (the spread column), and a free 8-parameter fit on ~8 such points bends the pitch differently on each frame.
- Points 18/19 are erratic (6–7 m off, wide spread): likely confused with another landmark on far-side views.
- One clip, and the 31/32 offset was measured on it: confirm on a second clip before changing the template. The PFF players' fit is itself a ruler with ~0.5 m noise and extrapolates near the goal line; the right-end offsets (+1–2 m in x) may be partly that.

## Fix 2, step a applied: 31/32 at ±7.2 m (2026-10-01, candidate)
03 Pitch template: 31/32 now sit at `(∓circle_kp_x_m, 0)`, default 7.2 m, where the model puts them. It's a `VisionConfig` value, so it goes into `run.json` and the replay check still holds; a `run.json` without it ran on 9.15 (vb01's run). `scripts/keypoint_check.py` now measures against the real landmarks (31/32 at 9.15) whatever the config says, so the next clip's table reads like the one above (rerun on vb01: same numbers).

`python -m vision.bench --clip vb01-arg-fra --grid circle_kp_x_m=9.15,7.2`:

| circle_kp_x_m | Within 2 m | Median / p90 | With geometry: within 2 m, median | Rejected | Unmatched / frame |
|---|---|---|---|---|---|
| 9.15 (vb01's run) | 14.0% | 2.81 / 4.43 m | 17.8%, 2.81 m | 10.1% | 4.37 |
| 7.2 (new default) | 20.9% | 2.43 / 4.24 m | 26.5%, 2.43 m | 10.1% | 3.87 |

- Same as the monkeypatched replay above, so the wiring is right.
- Still a candidate: measured and scored on the same clip. Confirm on clip 2 (NED–ARG, 10511) with `keypoint_check` (do 31/32 sit near ±7.2 again?) and the bench grid.
- Far from the 90% target either way: the per-frame fit is the bigger problem (step b, PTZ camera model).

## Clip 2: vb02-ned-arg, and 31/32 confirmed (2026-10-01)
`vb02-ned-arg`: PFF 10511 (Netherlands v Argentina, QF), second half, video 0:08–0:59.2 (clock 78:19–79:10), 1080p30. Vision ran from 0:00 (8 s pre-roll), on the new default (`circle_kp_x_m` 7.2), RTX 2060 at 4.1 fps. Home (Netherlands) attacks TV right in period 1: with the TV-left goal on screen, PFF's visible players are at x ≈ −40.
- **Sync on cut edges:** three live close-ups and the cut at 59.27 s. Their 7 edges give offsets 1992.57–1992.65 s against PFF's all-`ESTIMATED` switches, every one a hard cut (frames either side checked). The scoreboard agrees to 1.6 s. Marks checked by the user 2026-10-01.
- `--offset-check`: best offset 0.53 s earlier than the sync (vb01: 0.23 s earlier). Two clips, same sign. Scored with the sync shifted by −0.23 / −0.53 s: vb01 (run config, 9.15) 14.0% → 15.0% / 15.3%, vb02 32.4% → 35.8% / 36.8%, median about 0.1 m better. So timing is a small part of the 2+ m, not the main cause. The cut edges agree to a frame, so the sync stays as marked; it may be PFF's positions lagging the video slightly.

**keypoint_check on vb02** (235 frames, offsets against the real landmarks):

| # | Landmark | n | dx | dy | spread |
|---|---|---|---|---|---|
| 14–17 | Halfway line | 77–139 | −0.22 to −0.01 | −1.64 to 0.84 | 0.73–1.13 |
| 31 / 32 | Circle left / right | 153 / 114 | **+1.17 / −1.64** | −0.86 / −1.88 | 0.95 / 1.15 |
| 10 / 11 | L box front far / at 6-yd far | 192 / 192 | +1.82 / +1.89 | 0.99 / −0.77 | 4.0 / 3.9 (erratic) |
| 2–9, 12–13 | L goal line, 6-yd box, spot, box front near | 34–131 | −1.58 to −0.56 | −1.93 to 2.48 | 0.9–1.5 |

- **31/32 sit inside 9.15 again:** at about −8.0 / +7.5 m (vb01: −7.0 / +7.4). The halfway points land right in x again.
- **The "box front far" pair is erratic on both clips:** 18/19 on vb01's right end, 10/11 (their mirror) on vb02's left end, spread ~4 m. Same failure, so it's the model on that landmark, not one stadium. A robust per-frame solve should drop them (step b).

**Bench, `--grid circle_kp_x_m`, per clip** (within 2 m / median):

| circle_kp_x_m | vb01 | vb02 |
|---|---|---|
| 7.0 | 21.2% / 2.41 m | 32.6% / 2.24 m |
| **7.2** | **20.9% / 2.43 m** | **32.4% / 2.25 m** |
| 7.5 | 20.1% / 2.46 m | 31.0% / 2.32 m |
| 7.8 | 18.8% / 2.54 m | 29.2% / 2.37 m |
| 8.2 | 16.5% / 2.68 m | 28.2% / 2.41 m |
| 9.15 | 14.0% / 2.81 m | 25.3% / 2.54 m |

Homography rejected moves 0.4–1.1 pt across the grid on either clip, no trend. 7.0 and 7.2 are within 0.3 pt on both, so 7.2 stays (measured on vb01, not picked on this grid). **31/32 at 7.2 is confirmed on a second clip, marks checked by the user.** 6.5 is worse on both (vb01 20.0%, vb02 30.8%), so 7.0–7.2 is a real optimum. But on vb02 keypoint_check measures 31/32 at 7.5–8.0 m while the bench prefers 7.2: the value also soaks up some of the per-frame stretch. Grid it again once the PTZ solver exists.

**vb02 scorecard at the new default:**

| | |
|---|---|
| Scored frames | 1,399 (the rest are close-ups) |
| Geometry missing | 6.4%: view `other` 2.1%, homography rejected 4.4% of match view |
| People within 2 m (misses count) | 32.4% of 23,792 PFF visible player-frames |
| Matched error, median / p90 | 2.25 m / 3.93 m |
| Outfield team accuracy / coverage | 89.5% / 100% (home cluster picked by agreement; the pairing caveat above) |
| Keeper team accuracy / coverage | 61.3% / 99% |
| False live | 2.9 s of 4.6 s of marked close-up |

Pooled, both clips: within 2 m 23.3%, median 2.46 m, homography rejected 7.1%, geometry missing 15.6%.
- vb02 is an easier clip (fewer cuts, most of it midfield), so it's not a like-for-like comparison with vb01.
- Teams hold on a second clip (89.5% even with neighbours paired across teams). Keepers at 61% are new: worth a look once positions are better, since keepers take their cluster from position.
- False live 2.9 s: the close-ups here are 1.1–2.2 s, about the gate's switch time (0.5 s off, 1 s on), so most of each one stays `match`. That's the cut detector's job (roadmap W3), not a threshold.
- Positions are still ~2.3 m off on the easy clip. Next is step b, the PTZ camera model.

## Clip 3: vb03-jpn-esp, kick sync, and a stadium the keypoints get wrong (2026-10-02)
`vb03-jpn-esp`: PFF 3854 (Japan v Spain, group stage), first half, video 3:45–4:45 (clock 4:40–5:40), Khalifa International Stadium (vb01 and vb02 were both at Lusail). Video 11 is stitched from YouTube pieces; the clip is inside piece 6 (211.17–300 s). Vision ran from 211.2 s (13.8 s pre-roll, starting right after the stitch), RTX 2060 at 4.0 fps, default config (`circle_kp_x_m` 7.2). Japan (home) attacks TV right in period 1.

**No cut edges to sync on.** Every PFF all-`ESTIMATED` run in P1 0–370 s falls in a gap between pieces: the stitching cut out each cutaway, so the clock jumps (3:10 → 3:14, 4:20 → 4:25) and a piece is wide live play only. No shots before 6:00 either. So 07 gets a third sync rung, kicks (`scripts/kick_times.py`):
- PFF's ball moves every other frame and is smoothed, so a kick is a 6 → 9 → 12 → 14 m/s ramp over ~0.2 s. The kick is put at the last slow update before it.
- **Checked on vb02 against its cut edges first:** 3 kicks at 1992.44 / 1992.56 / 1992.49 s (spread 0.12 s) vs 1992.57–1992.65 from the cut edges. Kicks read **0.11 s early**, so vb03 gets +0.11 s to be comparable with the cut-edge clips.
- **vb03:** 3 kicks at 55.46 / 55.48 / 55.43 s (spread 0.05 s), each where PFF puts it in the image (circle top, far side left of halfway, circle left edge). Offset 55.57 s. The scoreboard agrees to 1 s.
- No marks: the scoreboard steps cleanly through the clip and there are no cuts (scene detect over 140–300 s finds none). False live can't be scored on this clip.

**Scorecard:**

| | vb03 |
|---|---|
| Geometry missing | 2.2% (view `other` 0%, homography rejected 2.2%) |
| People within 2 m | **15.3%** |
| Matched error, median / p90 | 3.10 m / 4.59 m |
| Unmatched vision rows per frame | 7.19 (vb02: 2.76) |
| Outfield team accuracy / coverage | 73.0% / 100% (home cluster picked by agreement) |
| Keepers | none on screen (all midfield) |
| `circle_kp_x_m` 9.15 vs 7.2 | 15.0% / 3.12 m vs 15.3% / 3.10 m |

All three clips at 7.2: vb01 20.9%, vb02 32.4%, vb03 15.3%; pooled 22.1% within 2 m, median 2.59 m, homography rejected 5.1%.

**It's the keypoints, not the sync or PFF:**
- The offset doesn't matter: from −6 to +3 s within 2 m stays at 10–15% (peak 15% at −1 to +0.5 s, where the kicks put it). `--offset-check` runs to the edge of its ±1 s window because the curve is flat (slow midfield possession), not because the sync is off. Flipping x or y drops it to 1% / 10%.
- PFF is consistent: 140 frames pass keypoint_check's feet → PFF fit (median residual ≤ 0.6 m, its acceptance filter), and PFF's extents for 3854 match the other matches (players ±51 × ±33.7 m).
- keypoint_check: every landmark is off by **+3 to +8 m in y** with ~1 m spread (vb01/vb02: mostly ≤ 2 m). Box front far / at 6-yd far (10/11) are +5 / +7 m, and the pen spot is +6.8 m.
- Overlaying a frame (250 s, frame 1165) shows why. The pipeline used keypoints 1, 10, 11, 14–16 and 31 there and accepted the fit at 0.27–0.28 m error (run config: ransac 2.0 m, window 3). A full-res crop has no paint at 10's or 11's pixels (294, 288 / 125, 373), only the edge of a diagonal **mowing stripe**. A short piece of paint at the frame's left edge (~5, 325) may be the real box line. Point 1 (far goal-line corner) can't be on screen with halfway at x ≈ 1450. Refitting those points (9.15 template) tilts the halfway line and lifts the circle off the painted one. The fit passes its own check and is wrong, as on vb01 but worse. This pitch has strong diagonal and straight stripes.
- 31/32 here: circle left +0.18 m off 9.15, circle right −2.06 m. So the 7.2 shift isn't a fixed property of the template. On vb01/vb02 the model pulled 31/32 in by ~2 m; here it doesn't on one side. 7.2 still doesn't hurt (15.0 → 15.3%).

**What it means:**
- Geometry is the problem again, on a new stadium and more so. A per-frame fit from 5–8 keypoints can't reject a pair of confident wrong points that agree with each other.
- Keep vb03 in the set as the hard case. **Later the same day:** PnLCalib gets 84.1% within 2 m here, against 15.8% for our fit, on the same foot points. Replacing the keypoint model beats a PTZ solver on top of it (`Docs/reviews/pnlcalib-2026-10-02.md`).
- Teams 73% is likely downstream of positions (pairs land on the wrong player when positions are 3 m off, as on vb01 before the fix). Recheck once geometry is better.
