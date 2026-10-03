# Ball research (2026-10-02)

Research only, no pipeline code. Asked after the first ball labels (vb02): clicking is slow and error-prone, a better open-source ball model probably exists, and possession has to survive ball gaps. Four questions from the handoff (`Docs/handoff/2026-10-02-ball-research.md`): R1 ball truth without clicks, R2 better detection and tracking, R3 gaps and possession, R4 live cost. Measurements are scratch scripts on the vb01–vb03 caches (current defaults), plus PnLCalib and the ball model rerun on the 930 extracted label frames. Nothing here changed the caches.

## Summary
- **PFF's ball is a good locator, not a pixel truth.** Projected through a camera solved on the same frame, PFF's VISIBLE ball lands within 25 px of a good hand click on 84% of frames (median 12 px). It drifts by up to 25–36 px (~1 m) for 5–10 s at a time, while PFF's players on the same frames project to within 7–11 px. That's too loose to score a 90% recall target at a ball-sized radius.
- **PFF's VISIBLE flag means "on the ground and tracked", not "in the image".** On these clips VISIBLE balls are never above 1.45 m. Airborne balls are ESTIMATED (median z 2.6 m), and so are some grounded balls PFF lost. On ESTIMATED frames the ball is often plainly visible, and PFF's estimate is 44 px off (median). They're 31–39% of the label frames on vb01/vb02.
- **So 07's claim half holds:** PFF can't say where an airborne ball is in the image, and it can't say a ball is hidden. It can say a grounded ball is on screen, and roughly where.
- **What PFF is good for:**
  1. Pre-placing labels so most frames take one key press instead of a click.
  2. Finding bad labels automatically: it flagged all of vb02's speck clicks and the trailing clicks.
  3. An automatic score at a loose radius (40 px) for paired sweeps on any 2022 clip.
  4. Auto-labels for fine-tuning: PFF and a detection agreeing.
- **The detector is the bottleneck, not the picker.** Our ball model has a candidate near PFF's ball on about 70% of PFF-VISIBLE frames.
  - Smarter association (gate on predicted position, size and off-pitch filters) adds 1–2 pt.
  - Larger input (1536, 1920) adds under 1 pt and doubles or triples false candidates.
  - Tiles (roboflow's 640 slices) don't help either and cost 4× the time.
- **Open-source models don't transfer as-is.** WASB, TrackNet v2–v4, FootAndBall and DeepBall are trained on fixed cameras (ISSIA-CNR) or racket sports, at 512 × 288 input. Ours is YOLOv8x trained on broadcast frames. WASB's 3-frame window is the only multi-frame idea worth keeping. It looks up to 2 frames ahead (67 ms delay) unless retrained to output only its last frame.
- **Recommendation:** verified labels made fast with PFF pre-placement as the truth that decides targets. The PFF auto-score for sweeps. Then fine-tune the ball model on PFF-agreement auto-labels from the six non-bench 2022 matches plus mined hard negatives. A gated tracker with cut reset on top.

## R1. PFF's ball as truth

### Method
- **Chain:** 02 meters → TV frame (`to_02` is its own inverse; all three clips have `home_attacks_tv_right_p1` true) → PnLCalib's world `(x, −y, −z)` → pixels with `P = K R [I | −C]` from `camera.parquet`.
- **Time:** PFF's ball is interpolated to the vision frame's exact time (`video_s + offset`), between the two PFF frames around it.
- **Cameras compared:**
  - the run's call on that frame;
  - the held camera (the pipeline's);
  - an interpolated camera;
  - a camera solved on each label frame from the extracted JPGs (PnLCalib WC14, same checks; 930 frames, 0.4 s each).
- **References:**
  - nearest ball candidate in `balls.parquet`;
  - stage 5's row;
  - vb02's clicks. The audit's bad ranges are excluded as "good clicks": 240–280, 575–610, 1060–1080, 1505–1590, 1610–1650, 1665–1690, 1705–1725 (source frames).

### The chain is right: people first
PFF VISIBLE players' feet (z = 0) against vision's foot points, one-to-one within 60 px, on scored match-view frames:

| | Call frames: matched / median / p90 | Held camera: median / p90 |
|---|---|---|
| vb01 | 96.9% / 10.7 px / 30.1 px | 13.6 / 37.0 px |
| vb02 | 96.7% / 8.8 / 23.5 | 11.4 / 30.6 |
| vb03 | 95.6% / 10.7 / 32.7 | 12.5 / 34.8 |

About 0.12 of a box height: foot-point ambiguity plus PFF's own position error. The camera on its own frame is better than a held one.

### Timing: PFF's ball is on the players' clock
Sweep of a time shift added to PFF on moving balls (> 5 m/s), scored by the median distance to the nearest confident candidate:
- **Best shift:** vb01 0 to +0.01 s, vb02 +0.02 to +0.04 s, vb03 −0.01 to +0.01 s. Whole PFF frames: 0 / +1 / 0.
- **What it means:** PFF's ball doesn't lag its players. vb03's sync already carries the +0.11 s kick correction (07 sync rung 3), and its best shift is 0, so the correction was right: the kicks were read early by eye, not a different PFF clock.
- **Scoring:** truth uses shift 0 with interpolation in time. The residual is a third of a frame, which at 20+ px per frame matters only for the fastest balls.

### Height: project the ball's center, not its ground point
PFF gives z = 0 for a ball on the grass. The image shows its center, 0.11 m up.

| Projected at | Median offset (candidate − projection) | Median distance |
|---|---|---|
| z | du +2.0 px, dv −8.0 px | 11.8 px |
| z + 0.11 m | du +1.9, dv −2.7 | 9.4 px |
| z + 0.2 m | du +1.8, dv +2.0 | 10.3 px |

z + 0.11 m from here on. vb01 keeps a +5 px horizontal offset, which its people show too (+4.6 px), so that's the camera.

### What VISIBLE means
PFF ball rows on unmarked frames of the three clips:
- **VISIBLE:** z median 0.0 m, p95 0.21 m, p99 0.46 m, max 1.45 m.
- **ESTIMATED:** z median 2.6 m, p95 7.5 m, max 9.2 m.

ESTIMATED runs inside live wide play (view `match`):

| Clip | Runs | Longest | What's in them |
|---|---|---|---|
| vb01 | 17 | 4.2 s (z up to 8.2 m) | three lofted balls (2246–2504, 3274–3394), short dropouts |
| vb02 | 10 | 7.6 s (920–1148, z up to 6.8 m) | a long aerial phase with headers; 752–792 a **grounded** ball PFF lost (z 0.3) |
| vb03 | 1 | 0.3 s | nothing airborne in this minute |

Looked at by eye (overlays drawn on the extracted label frames; source frame numbers):
- **vb02 1000:** the ball at a header, plainly visible, ESTIMATED, estimate 42 px off.
- **vb02 760:** a grounded ball rolling clear, ESTIMATED, estimate 132 px off.
- **vb02 1520:** a lofted ball against the crowd, estimate ~50 px from a white blob that may be the ball. The click there was a guess 247 px away.

ESTIMATED positions are interpolated with later frames (05, Leakage). That's fine for truth, but they aren't accurate: on 42 ESTIMATED frames where a click and a candidate agree, the estimate is 44 px off (median), p90 126 px.

**Label frames by PFF state** (every 5th source frame, unmarked, camera solved on the frame):

| | Label frames | PFF VISIBLE, in the image | ESTIMATED | No camera |
|---|---|---|---|---|
| vb01 | 290 | 197 (68%) | 93 | 0 |
| vb02 | 280 | 170 (61%) | 110 | 0 |
| vb03 | 360 | 340 (94%) | 2 | 18 |

### How close VISIBLE is to the real ball
**Against the good vb02 clicks** (147 VISIBLE frames, camera solved on the frame, center height):

| | Median | p90 | ≤ 15 px | ≤ 20 | ≤ 25 | ≤ 30 |
|---|---|---|---|---|---|---|
| All 147 | 11.9 px | 35.0 | 63.9% | 76.9% | 84.4% | 87.8% |
| src 0–660 (48) | 10.8 | | | | 100% | |
| src 660–1000 (43) | 20.6 | | | | 62.8% | |
| src 1000–1310 (22) | 17.5 | | | | 77.3% | |
| src 1310–1800 (34) | 7.0 | | | | 94.1% | |

- **The bad stretches are PFF's ball, not the camera.** People in the same stretches project to 7.0 / 8.5 / 10.8 / 8.8 px. No time shift helps either: the best shift for 660–1000 is 0.
- **The error is a slow horizontal bias,** median du −14.7 px in 660–1000 and +16.0 px in 1000–1310.
- **The same holds on all three clips.** Median offset of the nearest confident (≥ 0.5) candidate within 80 px, per 5 s block:
  - vb01 reaches +25 to +27 px for 15 s (src 2550–3150); in two of those frames (3130, 3425) the ball model has the ball at 0.86–0.87, 31–39 px from PFF;
  - vb02 −21 to −36 px (src 750–1000);
  - vb03 stays within ±15 px.
- **Removing it with confident detections from ±1 s around the frame** (excluding ±0.2 s) only goes from 84.4% to 88.4% within 25 px. Part of the error is per-frame.
- **Agreement looked better on the run's call frames:** 85 good clicks, median 9.2 px, 96.5% within 25 px. That's not because of the camera (the camera solved on the JPG matches the run's call to < 1 px, median 0). vb02's call frames happen to fall in the good stretches.

**Scoring agreement.** Stage 5 hit or miss under PFF at radius R, against hit or miss under the click at 15 px, on the 147 good VISIBLE frames (held/interpolated camera):

| R | Recall vs clicks | Recall vs PFF | Same verdict |
|---|---|---|---|
| 15 px | 79.6% | 60.5% | 79.6% |
| 25 px | 79.6% | 70.1% | 87.8% |
| 40 px | 79.6% | 77.6% | 95.2% |

At 40 px PFF reads 2 pt low. At 25 px, 10 pt low.

With a camera solved on each frame (what the spec builds), it's 92.5% at 40 px (recall 76.2 vs 79.6%), and 87.8% at 25 px. A truth that moves recall by 2–10 pt can't decide a 90% target. It can rank two settings scored on the same frames, since the error is the same for both.

### Bad labels it finds
VISIBLE frames where the click is more than 30 px from PFF's projection (32 of 170):
- **all nine speck clicks at 240–280:** the click is 765–1196 px from PFF, and a candidate sits 3–21 px from PFF, on the real ball;
- **the trailing clicks at 580–610:** 36–50 px;
- **likely bad clicks:** 665, 1670, 1685, 1750;
- **PFF's own bias stretches:** 710–915 and 1205–1215, where the click is right.

So the flag list needs a look, not automatic rejection. 1215 is a still ball at the keeper's feet that PFF puts 83 px away.

### Pre-placed labels
Suggestion: the candidate nearest PFF's projection within 40 px, at any confidence.
- **VISIBLE frames (147):** it's the clicked ball (within 6 px) on 53%; another candidate on 25%; none on 22%.
- **ESTIMATED frames (73):** right on 26%.
- **Frames with a suggestion:** vb01 132 of 290 label frames, vb02 134 of 280, vb03 287 of 360.

The rest needs a click or "none". With today's detector, about half the frames become one key press. A better detector raises that, since the suggestion is only as good as the candidates.

**Auto-accept.** Some suggestions are safe without a look. The rule: confidence ≥ 0.5, within 15 px of PFF's projection, no other candidate within 40 px.
- **On vb02's good VISIBLE clicks:** 82 of 147 qualify. 81 of them are within 15 px of the click (median 3.8 px, 97.6% within 10 px); the other is 18 px.
- **Frames that still need a person:**

| Clip | Label frames | Auto-accepted | Left |
|---|---|---|---|
| vb01 | 290 | 54 | 236 |
| vb02 | 280 | 93 | 187 |
| vb03 | 360 | 234 | 126 |

On the frames left, half of the `pff` suggestions take one key press. The number left shrinks as the detector improves.

### Verdict
- **No.** PFF projected can't be the main truth for the targets:
  - it misplaces the real ball by over 25 px on ~16% of VISIBLE frames, in bias stretches of 5–15 s;
  - it has nothing usable on the 31–39% of live frames that are ESTIMATED, which is where aerial balls and headers are;
  - it can't mark a hidden ball.
- **It replaces most of the clicking work instead:**
  - the verified labels (07) stay the truth, but a frame shows a pre-placed suggestion, and a good one takes one key;
  - PFF flags labels that disagree with it, for a second look;
  - an automatic PFF score at 40 px runs on every label frame of any 2022 clip with no labels, for paired sweeps and new clips. Its absolute numbers are reported as PFF's, never against the 90 / 95% targets;
  - auto-labels for fine-tuning (R2) use PFF and a detection agreeing, not PFF alone.

## R2. Better detection and tracking

### What limits us now
Our ball model: roboflow/sports' `football-ball-detection.pt`, **YOLOv8x trained at imgsz 1280 on 1,966 Roboflow Universe broadcast images** for 50 epochs (val mAP50 0.924, mAP50-95 0.571; [training notebook](https://raw.githubusercontent.com/roboflow/sports/main/examples/soccer/notebooks/train_ball_detector.ipynb)). Ultralytics is AGPL-3.0.

**Ceiling: is there any candidate near the ball?** The run's `balls.parquet` (conf ≥ 0.1) has a candidate within 25 px of PFF's projection on 70.4% of PFF-VISIBLE label frames (vb01 54%, vb02 69%, vb03 80%).
- PFF's bias pulls this down, most on vb01. With the local bias correction it's 65.5 / 77.1 / 84.1%.
- **The same holds on the clicks.** On vb02's 220 good clicks, a candidate is within 15 px on 85.5% (≥ 0.3: 81.4%), against stage 5's 82.7% recall. So the picker is close to what the candidates allow, on clean truth too.
- **Misses seen by eye:**
  - balls at a player's feet (vb01 2695, vb02 835);
  - a ball at a player's knee (vb03 8310: the model's top pick is a spare ball by the touchline at 0.50);
  - blur at 20–30 m/s;
  - small far balls.

**Pickers and filters, replayed from `balls.parquet`** (CPU, exact replay of today's stage 5 for "max"). Scored on PFF truth at 25 px (707 VISIBLE label frames) and on vb02's good clicks at 15 px (220):

| Variant | Recall (PFF) | Precision (PFF) | vb01 / vb02 / vb03 | vb02 clicks recall / precision |
|---|---|---|---|---|
| max confidence ≥ 0.3 (now) | 70.2% | 73.6% | 53.8 / 68.8 / 80.3% | 82.7% / 87.9% |
| max ≥ 0.2 | 70.6% | 73.1% | | 83.6% / 88.0% |
| max ≥ 0.4 | 70.0% | 72.8% | | 82.7% / 87.1% |
| roboflow buffer (nearest to the mean of the last 20 picks) | 69.9% | 72.6% | | 82.7% / 87.1% |
| gate: predicted pitch position ± (3 m + 25 m/s × gap), reacquire ≥ 0.5 | 70.7% | 73.2% | | 85.5% / 90.0% |
| gate, reacquire ≥ 0.6 | 71.3% | 73.8% | | 84.5% / 89.0% |
| gate 2 + 20 dt, or 5 + 35 dt | 70.7% | 73.2% | | 85.5% / 90.0% |
| max + box width 0.5–2× the expected ball | 70.0% | 72.8% | | 83.2% / 87.6% |
| max + candidates ≤ 2 m off the pitch | 71.1% | 74.0% | | 83.2% / 87.6% |
| gate + size + 2 m margin | **72.0%** | **74.5%** | 55.3 / 70.6 / 82.4% | **85.5% / 90.0%** |

- **The picker is worth 1–3 pt.** The gate is the best rule on both truths, and its settings barely matter.
- **Airborne balls.** The off-pitch and size filters and the pitch-space gate work through the ground homography, so an airborne ball sits too far out. They were checked on the 73 good vb02 clicks on PFF-ESTIMATED (mostly aerial) frames. Recall/precision there:
  - `"max"` 89.0 / 91.5%;
  - gate 91.8 / 91.8%;
  - size or margin alone 90.4 / 90.4%;
  - all three 91.8 / 91.8%.

  No loss on this sample. The highest balls are in vb02's guessed ranges, though, so 10-ball's adoption rule also requires no loss on verified ESTIMATED frames.
- **Recall can't pass the ceiling,** and the ceiling is the detector.
- **Precision against PFF is low partly because of PFF's bias.** Against the clicks it's 88–90%.

**Input size and tiles** (the same model on the 707 VISIBLE label frames, RTX 2060, one job on the GPU):

Columns:
- **ceiling:** any candidate ≥ 0.1 within 25 px of PFF;
- **top-1:** the most confident candidate ≥ 0.3 is within 25 px;
- **false top-1:** it's ≥ 0.3 and further;
- **false/frame:** candidates ≥ 0.3 more than 25 px away.

PFF's bias costs every row the same, so compare rows, not the absolute numbers.

| Setting | Ceiling | vb01 / vb02 / vb03 | Top-1 | False top-1 | False/frame | ms a frame (median) |
|---|---|---|---|---|---|---|
| full frame, imgsz 1280 (now) | 72.1% | 52.8 / 78.2 / 80.3% | 68.7% | 19.1% | 0.53 | 114 |
| 1280, fp16 | 71.4% | 51.3 / 78.2 / 79.7% | 68.0% | 19.8% | 0.54 | 62 |
| 1536 | 72.4% | 54.8 / 77.6 / 80.0% | 69.9% | 20.7% | 0.86 | 162 |
| 1920 (native) | 72.6% | 57.9 / 75.9 / 79.4% | 67.0% | 23.8% | 1.40 | 250 |
| 8 tiles 640 at imgsz 640 (native scale, roboflow's example) | 72.1% | 56.9 / 76.5 / 78.8% | 67.8% | 24.9% | 1.51 | 446 |
| 6 tiles 960 at imgsz 1280 (upscaled) | 71.1% | 57.9 / 74.1 / 77.4% | 53.5% | 38.6% | 2.51 | 1,210 |

- **Nothing beats 1280.** Larger inputs and tiles find a few more balls on vb01 (+4–5 pt) and lose some on vb02/vb03, while false candidates triple to quintuple. Tiles cost 4–10× the time.
- **The model was trained at 1280 on downscaled broadcast frames.** At native scale the ball looks bigger than anything it learned, so a resolution change only pays after training at that scale.
- **fp16 costs 0.7 pt for half the time.**
- **When it finds the ball it's sure of it:** 97% of the found balls score ≥ 0.3 at 1280. The misses are balls with no candidate at all, not low scores. That's also why `min_det_conf` 0.2–0.4 changes nothing in the picker table.

### Open-source survey
Verified in each source (links). "Causal" means it can emit frame t from frames ≤ t.

| Model | License, weights | Frames | Trained on | Reported soccer result | Fit here |
|---|---|---|---|---|---|
| **roboflow/sports** ball model + `InferenceSlicer` + `BallTracker` ([main.py](https://raw.githubusercontent.com/roboflow/sports/main/examples/soccer/main.py), [ball.py](https://raw.githubusercontent.com/roboflow/sports/main/sports/common/ball.py)) | AGPL-3.0 (ultralytics) / MIT (supervision); weights public | 1, causal | 1,966 broadcast frames | val mAP50 0.924 on its own split | What we run. Their example tiles 640 × 640 at imgsz 640, NMS 0.1. Their tracker picks the detection nearest the mean of the last 20 positions (above: no gain) |
| **WASB** ([repo](https://github.com/nttcom/WASB-SBDT), [paper, BMVC 2023](https://arxiv.org/abs/2311.05237)) | MIT; soccer weights in MODEL_ZOO | 3 in, 3 heatmaps out; `step 3` windows don't overlap, so frame t can use t+1 and t+2: **not causal, ≤ 2 frames (67 ms) late**. A causal variant would keep only the last output (untested) | ISSIA-CNR / D'Orazio: 6 **fixed** 1080p cameras, 4 train / 2 test clips, re-annotated | F1 88.3, AP 83.6 (TrackNetV2 86.6 / 77.2, DeepBall 44.5 / 26.3); 55.7 fps (step 3) on a V100 server | Input 512 × 288: our 12 px ball would be 3 px. Fixed cameras don't pan or zoom. Needs retraining on broadcast; its online tracker is a gate around the last position (`max_disp` 300 px) plus best score, i.e. what we tested |
| **TrackNetV2** (in WASB's zoo) | weights in WASB's zoo | 3, same windowing | ISSIA (WASB's run) | F1 86.6 | as WASB, 11.3 M params |
| **TrackNetV3** ([repo](https://github.com/qaz812345/TrackNetV3)) | MIT | 3 + a background image + a 16-frame inpainting rectifier | badminton | none | Background median assumes a fixed camera; the rectifier is non-causal. Not for a panning broadcast |
| **TrackNetV4** ([paper](https://arxiv.org/abs/2409.14543), [code](https://github.com/AIKnowU/tracknet-v4-pytorch)) | code public | 3 frames, 512 × 288, motion attention from frame differences | tennis, badminton | none | Frame differences mix camera motion with ball motion in a pan |
| **FootAndBall** ([repo](https://github.com/jac99/FootAndBall), [paper](https://arxiv.org/abs/1912.05445)) | MIT; weights in the repo | 1, causal | ISSIA-CNR cameras 1–4 + SoccerPlayerDetection | not in the repo | Very small and fast, but fixed-camera training. A retrain candidate if YOLO is too slow live |
| **DeepBall** (in WASB's zoo) | | 1 | ISSIA | F1 44.5 | no |
| **YOLO fine-tuned on soccer balls** | ultralytics AGPL-3.0 | 1, causal | | ours: above | The path below |

**Training data for a fine-tune:**
- **Our own 2022 footage with PFF (best fit):**
  - six non-bench matches with PFF tracking: 3857, 10507, 3816, 10510, 10508, 10514, 5 min each, ~54k frames;
  - same broadcast style, stadiums and ball as the bench. The bench matches (10517, 10511, 3854) stay out entirely;
  - auto-labels: a candidate within 25 px of a PFF-VISIBLE projection;
  - hard negatives: confident candidates more than 60 px from a PFF-VISIBLE ball (specks, logos, the net, heads, boots), minus anything that looks like a second ball (spare balls are balls; leave them unlabeled rather than negative);
  - missed balls (PFF VISIBLE, no candidate) can't be boxed from PFF alone, given its bias. They go through the pre-placement tool on a sample.
- **[SoccerNet-Tracking](https://arxiv.org/abs/2204.06918):**
  - 200 × 30 s clips, 1080p 25 fps, Swiss league broadcast main camera, 215,156 ball boxes;
  - boxes are keyframed with linear interpolation in between, so airborne and occluded balls are interpolated guesses;
  - the [sn-tracking README](https://github.com/SoccerNet/sn-tracking) downloads it with `pip install SoccerNet`, no password mentioned. Videos in general are NDA-gated on Hugging Face ([soccer-net.org/data](https://www.soccer-net.org/data)), and the license isn't stated in the repo. **Check both before use.** Good for scale and stadium variety.
- **The [Roboflow set](https://universe.roboflow.com/roboflow-jvuqo/football-ball-detection-rejhg)** our model came from (v2, 1,966 / 121 images): keep it in the mix so the fine-tune doesn't forget. License not checked (page returned 403).
- **ISSIA-CNR:** fixed cameras; not useful for broadcast.

### Ranking
1. **Fine-tune our YOLOv8x ball model** on PFF-agreement auto-labels from the six 2022 matches, with mined hard negatives and the Roboflow set, at 1280 (the table above: no tiles, no larger input). Causal, drops into `BallTrack` and `balls.parquet` with no format change; the only thing that attacks the 70% ceiling. Effort: a labeling script, a training run (Colab or the 2060 at batch 2–4, 09), a fresh bench run.
2. **Gated tracker + filters + reset on cuts in `BallTrack`** (+1–3 pt, mostly precision; W2). Replayable from `balls.parquet`, CPU only.
3. **Carry the ball with its carrier through short gaps** (R3), replayable.
4. **WASB retrained on broadcast at a higher input size**, as a second opinion on a crop around the predicted ball. Only if 1–3 fall short. Causal only with a last-frame output.
5. TrackNetV3/V4, FootAndBall, DeepBall: not for this footage.

## R3. Tracking through gaps, and possession
- **What strong systems do:**
  - offline, a global optimization over player-ball interaction ([Maksai et al., What players do with the ball](https://arxiv.org/abs/1511.06181), a mixed integer program over the sequence). It's not causal, so not for live;
  - online, simple and causal: WASB's tracker gates candidates around the last position and reports no improvement from Kalman or particle filters ([paper](https://arxiv.org/abs/2311.05237)). Its code ([`trackers/online.py`](https://github.com/nttcom/WASB-SBDT)) drops candidates more than 300 px from the last position and takes the best score.
- **Our replay agrees:** the gate is worth 1–3 pt, and its parameters don't matter.
- **What already handles gaps here:**
  - stage 8 holds the carrier through ball gaps up to `carrier_gap_s` (0.5 s), and holds possession until another team's carrier, with no time limit;
  - ball state turns null only after `ball_lost_s` (2 s);
  - the learned possession model (v1h, now stage 8's possession for vision output) reads team shape when the ball is missing, holds the ball at most 1 s, never reads interpolated balls, and falls back to the rule only when no players are visible (03 stage 8).
  - **So possession is built to survive gaps.** v1h passed all five gate bars on PFF's VISIBLE-only inputs, where the ball is missing on many scored rows (stage 8 review: 30.5% disagreement for the old rule with no visible ball vs 8.7% with one). Nothing in possession changes for the ball work.
- **What gaps cost the predictor** (sensitivity review): ball misses −0.014 PR-AUC per 10% of ball-time missing (10% −0.017, 50% −0.070), ball noise 1 m −0.010, 4 m −0.049, false balls 5% −0.0075.
  - Today's recall is ~70% against PFF and ~83% against the good vb02 clicks. So 17–30% of ball-visible time is missed, around −0.025 to −0.04 by that rate. That's before the ESTIMATED (aerial) frames, which no number here scores yet.
  - Recall at 90% holds the miss cost near −0.014. Precision from ~74–88% to 95% takes the false-ball cost from about −0.013 to −0.007.
- **Gap fill near a carrier.** A ball that disappears at a player's feet is the commonest gap seen (R2 misses). Today stage 5 extrapolates it at its last velocity for up to 1 s, which drifts. Proposed: if stage 8 has a carrier when the ball disappears, hold the ball at the carrier's feet (pitch position), `interpolated = True`, for up to `ball_max_gap_s`.
  - It's causal: it uses the carrier on frame t−1 for frame t. It's scored with the PFF auto-score on the frames it fills.
  - **It mustn't feed itself.** The carrier comes from the 2D rule that learned possession runs, which reads only balls with `interpolated = False` (03, Time and visibility contract). A held ball can't keep its own carrier alive.
  - **The historical rule is the catch.** It (`vision/state.py` `infer`) reads every `visible` ball, interpolated ones included, so on vision output a held ball would feed it. The spec makes the rule skip interpolated balls on vision output, or limits the hold to runs that use the 2D rule.
- **What the predictor needs:** the targets stand (recall ≥ 90%, precision ≥ 95% on usable live frames). Measured on verified labels, PFF-ESTIMATED frames included.

## R4. Live budget
- **Today offline:** people YOLO + ball YOLOv8x at 1280 + PnLCalib every 5th frame ≈ 3.1 fps on the 2060 (bench review), with PnLCalib alone ~85 ms a frame averaged (03 Speed).
- **Ball model, measured here** (median per frame, batch 1, fp32 unless noted):

| Setting | ms a frame | Frames a second (ball model alone) |
|---|---|---|
| YOLOv8x, 1280, fp32 (now) | 114 | 8.8 |
| YOLOv8x, 1280, fp16 | 62 | 16 |
| YOLOv8x, 1536 / 1920 | 162 / 250 | 6.2 / 4.0 |
| 8 tiles 640 | 446 | 2.2 |

The times include ultralytics' pre- and post-processing, batch 1, after warm-up, one job on the GPU.

- **09's live budget:** 33–40 ms a frame, detection every 2nd–3rd frame (66–120 ms each). The ball stage gets 10–25 ms there (09).
  - YOLOv8x at 1280 is several times that. Live needs a smaller ball model (YOLOv8s/m fine-tuned on the same auto-labels), TensorRT fp16, a lower rate, or a crop around the predicted ball with a periodic full-frame refresh (W2).
  - The auto-label pipeline serves all of these, which is another reason to build it first.
- **WASB-sized heatmap models** run 55 fps on a V100 at 512 × 288. At a broadcast-useful input (crop or ≥ 1280 wide) their cost is unmeasured.
- **The gate and the carrier hold are CPU and negligible.**

## Recommendation, in order
1. **Truth (07):** verified labels stay the truth for targets. They're made with PFF pre-placement, and PFF flags disagreements. The PFF auto-score (40 px) is added for sweeps and unlabeled clips. Fix vb02's flagged labels with the new tool, then label vb01 and vb03 the same way.
2. **Tracker (03 stage 5):** the gate, size and margin filters, reset on cuts and segments, carrier hold. Replay only, scored on both truths.
3. **Detector:** auto-labels from the six non-bench matches, fine-tune, fresh bench runs. Then a smaller model for live.

Spec: `Docs/Specs/10-ball.md` (linked from 03 stage 5 and 07). Build order and done-when there.
